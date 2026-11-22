# ------------------------------------------------------------------------------
# Archivo: 05_text_finetune.py
# Proyecto: Taller 3 - Big Data y Machine Learning
#
# Descripcion:
#   Fine-tuning de un modelo tipo BERT (MiniLM multilingue o BETO) para
#   predecir el log(precio) a partir del titulo y la descripcion del anuncio.
#   Sobre el modelo preentrenado se agrega una "cabeza" de regresion (promedio
#   de los vectores de las palabras + una capa lineal) y se entrenan TODOS los
#   pesos con una tasa de aprendizaje pequena.
#
#   La prediccion se usa como variable para los modelos de R, por eso es
#   out-of-fold: se usan los mismos 5 folds espaciales de 04_text_features.py
#   (KMeans sobre lat/lon + GroupKFold). Para cada fold se entrena un modelo
#   con los otros 4, se predice el fold que se dejo por fuera y tambien test.
#   La prediccion final de test es el promedio de los 5 modelos.
#
#   Cada fold se guarda apenas termina, asi que el script se puede correr por
#   partes (p. ej. un fold por sesion de Colab) y retoma donde iba.
# Input:
#   01_data/01_raw/train.csv y test.csv.
# Output:
#   01_data/02_text/finetune/<modelo>/fold_<k>_{oof,test}.csv  (por fold)
#   01_data/02_text/text_finetune_<modelo>_{train,test}.csv    (al completar
#     los 5 folds; columnas property_id y txt_ft_<modelo>_lnp)
#   03_outputs/02_text/metricas_finetune_<modelo>.csv           (MAE por fold)
# Uso (en Colab con GPU):
#   python 02_code/05_text_finetune.py --model minilm --folds 0     # 1 fold
#   python 02_code/05_text_finetune.py --model beto                 # 5 folds
#   python 02_code/05_text_finetune.py --model minilm --sample 400 --epochs 1
#                                                    # prueba rapida en CPU
# ------------------------------------------------------------------------------

import argparse
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 2026

# ------------------------------------------------------------------------------
# (1) Rutas y parametros
# ------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DIR_RAW = ROOT / "01_data" / "01_raw"
DIR_TEXT = ROOT / "01_data" / "02_text"
DIR_OUT = ROOT / "03_outputs" / "02_text"

# Atajos para los modelos; tambien se puede pasar cualquier id de Hugging Face
MODELOS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "beto": "dccuchile/bert-base-spanish-wwm-uncased",
}

parser = argparse.ArgumentParser(description="Fine-tuning de BERT/MiniLM para el precio")
parser.add_argument("--model", default="minilm", help="minilm, beto o un id de Hugging Face")
parser.add_argument("--folds", default="all", help="'all' o lista de folds, p. ej. '0' o '0,1'")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=None,
                    help="Tasa de aprendizaje (por defecto 5e-5 MiniLM, 3e-5 BETO)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-len", type=int, default=192, help="Tokens maximos por anuncio")
parser.add_argument("--n-folds", type=int, default=5)
parser.add_argument("--n-clusters", type=int, default=60)
parser.add_argument("--sample", type=int, default=0, help="Si > 0, usa n filas (prueba)")
args = parser.parse_args()

nombre_hf = MODELOS.get(args.model, args.model)
etiqueta = args.model if args.model in MODELOS else re.sub(r"[^A-Za-z0-9]+", "_", args.model)
lr = args.lr if args.lr else (5e-5 if "MiniLM" in nombre_hf else 3e-5)
sufijo = f"_s{args.sample}" if args.sample > 0 else ""
DIR_FOLDS = DIR_TEXT / "finetune" / f"{etiqueta}{sufijo}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USAR_FP16 = DEVICE == "cuda"  # precision mixta: mas rapido en GPU


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


fijar_semilla(SEED)
print(f"Modelo: {nombre_hf} | dispositivo: {DEVICE} | lr: {lr} | epocas: {args.epochs}")

# ------------------------------------------------------------------------------
# (2) Datos, texto y folds espaciales (igual que en 04_text_features.py)
# ------------------------------------------------------------------------------

train = pd.read_csv(DIR_RAW / "train.csv")
test = pd.read_csv(DIR_RAW / "test.csv")
if args.sample > 0:
    train = train.sample(min(args.sample, len(train)), random_state=SEED)
    test = test.sample(min(args.sample, len(test)), random_state=SEED)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


def limpiar_texto(titulo, descripcion):
    t = f"{titulo if isinstance(titulo, str) else ''}. " \
        f"{descripcion if isinstance(descripcion, str) else ''}"
    t = t.lower()
    t = re.sub(r"\s*\b([aeiou])acute\b\s*", r"\1", t)
    t = re.sub(r"\s*\bntilde\b\s*", "n", t)
    t = t.replace("²", "2")
    return re.sub(r"\s+", " ", t).strip()


texto_train = [limpiar_texto(a, b) for a, b in zip(train["title"], train["description"])]
texto_test = [limpiar_texto(a, b) for a, b in zip(test["title"], test["description"])]
y_train = np.log(train["price"].to_numpy())

coords = train[["lat", "lon"]].fillna(train[["lat", "lon"]].median()).to_numpy()
grupos = KMeans(n_clusters=min(args.n_clusters, len(train) // 10), random_state=SEED,
                n_init=10).fit_predict(coords)
folds = list(GroupKFold(n_splits=args.n_folds).split(coords, groups=grupos))
folds_correr = range(args.n_folds) if args.folds == "all" else \
    [int(f) for f in args.folds.split(",")]

# ------------------------------------------------------------------------------
# (3) Modelo: BERT + promedio de tokens + capa lineal
# ------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(nombre_hf)


class BertRegresion(nn.Module):
    def __init__(self, nombre):
        super().__init__()
        self.bert = AutoModel.from_pretrained(nombre)
        self.dropout = nn.Dropout(0.1)
        self.cabeza = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **_):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        promedio = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean pooling
        return self.cabeza(self.dropout(promedio)).squeeze(-1)


def lotes(textos, y=None, barajar=False):
    """Genera lotes tokenizados (el relleno se ajusta a cada lote)."""
    idx = np.arange(len(textos))
    if barajar:
        np.random.shuffle(idx)
    for i in range(0, len(idx), args.batch_size):
        b = idx[i:i + args.batch_size]
        enc = tokenizer([textos[j] for j in b], padding=True, truncation=True,
                        max_length=args.max_len, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        yb = torch.tensor(y[b], dtype=torch.float32, device=DEVICE) if y is not None else None
        yield enc, yb


@torch.no_grad()
def predecir(modelo, textos):
    modelo.eval()
    salida = []
    for enc, _ in lotes(textos):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USAR_FP16):
            salida.append(modelo(**enc).float().cpu().numpy())
    return np.concatenate(salida)


def entrenar_fold(k, idx_tr, idx_va):
    """Entrena con idx_tr y devuelve predicciones (log precio) del fold y de test.
    Un 10% de idx_tr se reserva para escoger la mejor epoca, asi el fold que se
    predice no influye en ninguna decision del entrenamiento."""
    fijar_semilla(SEED + k)
    rng = np.random.default_rng(SEED + k)
    idx_tr = rng.permutation(idx_tr)
    n_es = max(1, int(0.1 * len(idx_tr)))
    idx_es, idx_fit = idx_tr[:n_es], idx_tr[n_es:]

    # Se estandariza el log(precio) para que la red entrene estable
    mu, sd = y_train[idx_fit].mean(), y_train[idx_fit].std()
    textos_fit = [texto_train[i] for i in idx_fit]
    y_fit = (y_train[idx_fit] - mu) / sd
    textos_es = [texto_train[i] for i in idx_es]

    modelo = BertRegresion(nombre_hf).to(DEVICE)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    pasos = args.epochs * math.ceil(len(textos_fit) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
    scaler = torch.amp.GradScaler("cuda", enabled=USAR_FP16)
    perdida = nn.SmoothL1Loss()  # Huber: robusta a precios extremos

    mejor_mae, mejor_estado = np.inf, None
    for ep in range(args.epochs):
        modelo.train()
        inicio, suma, n = time.time(), 0.0, 0
        for enc, yb in lotes(textos_fit, y_fit, barajar=True):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USAR_FP16):
                loss = perdida(modelo(**enc).float(), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            suma, n = suma + loss.item() * len(yb), n + len(yb)
        pred_es = predecir(modelo, textos_es) * sd + mu
        mae_es = np.mean(np.abs(np.exp(pred_es) - np.exp(y_train[idx_es])))
        print(f"  fold {k} epoca {ep + 1}: perdida {suma / n:.4f} | "
              f"MAE interno {mae_es / 1e6:,.1f} M | {time.time() - inicio:.0f} s")
        if mae_es < mejor_mae:
            mejor_mae = mae_es
            mejor_estado = {a: b.detach().cpu().clone() for a, b in modelo.state_dict().items()}

    modelo.load_state_dict(mejor_estado)
    pred_va = predecir(modelo, [texto_train[i] for i in idx_va]) * sd + mu
    pred_te = predecir(modelo, texto_test) * sd + mu
    del modelo
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return pred_va, pred_te


# ------------------------------------------------------------------------------
# (4) Correr los folds (se salta los que ya estan guardados)
# ------------------------------------------------------------------------------

DIR_FOLDS.mkdir(parents=True, exist_ok=True)
for k in folds_correr:
    archivo_oof = DIR_FOLDS / f"fold_{k}_oof.csv"
    if archivo_oof.exists():
        print(f"Fold {k} ya existe, se salta.")
        continue
    idx_tr, idx_va = folds[k]
    print(f"Fold {k}: entrena {len(idx_tr)} | valida {len(idx_va)}")
    inicio = time.time()
    pred_va, pred_te = entrenar_fold(k, idx_tr, idx_va)
    mae = np.mean(np.abs(np.exp(pred_va) - np.exp(y_train[idx_va])))
    print(f"Fold {k} listo en {(time.time() - inicio) / 60:.1f} min | "
          f"MAE fold espacial: {mae / 1e6:,.1f} millones")
    pd.DataFrame({"property_id": train["property_id"].iloc[idx_va].to_numpy(),
                  "pred": pred_va, "mae_fold": mae}).to_csv(archivo_oof, index=False)
    pd.DataFrame({"property_id": test["property_id"], "pred": pred_te}).to_csv(
        DIR_FOLDS / f"fold_{k}_test.csv", index=False)

# ------------------------------------------------------------------------------
# (5) Juntar los folds cuando esten todos
# ------------------------------------------------------------------------------

hechos = [k for k in range(args.n_folds) if (DIR_FOLDS / f"fold_{k}_oof.csv").exists()]
print(f"Folds terminados: {hechos} de {args.n_folds}")
if len(hechos) == args.n_folds:
    col = f"txt_ft_{etiqueta}_lnp"
    oof = pd.concat([pd.read_csv(DIR_FOLDS / f"fold_{k}_oof.csv") for k in hechos])
    pred_test = np.mean([pd.read_csv(DIR_FOLDS / f"fold_{k}_test.csv")["pred"].to_numpy()
                         for k in hechos], axis=0)
    out_train = train[["property_id"]].merge(
        oof[["property_id", "pred"]].rename(columns={"pred": col}), on="property_id")
    out_test = pd.DataFrame({"property_id": test["property_id"], col: pred_test})
    out_train.to_csv(DIR_TEXT / f"text_finetune_{etiqueta}_train{sufijo}.csv", index=False)
    out_test.to_csv(DIR_TEXT / f"text_finetune_{etiqueta}_test{sufijo}.csv", index=False)

    y_oof = np.log(train.set_index("property_id").loc[out_train["property_id"], "price"])
    mae_total = np.mean(np.abs(np.exp(out_train[col].to_numpy()) - np.exp(y_oof.to_numpy())))
    metricas = pd.DataFrame({"fold": [str(k) for k in hechos],
                             "mae_fold": [pd.read_csv(DIR_FOLDS / f"fold_{k}_oof.csv")
                                          ["mae_fold"].iloc[0] for k in hechos]})
    metricas.loc[len(metricas)] = ["total", mae_total]
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(DIR_OUT / f"metricas_finetune_{etiqueta}{sufijo}.csv", index=False)
    print(f"Listo: {col} | MAE out-of-fold total: {mae_total / 1e6:,.1f} millones")
