# ------------------------------------------------------------------------------
# Archivo: 04_text_features.py
# Proyecto: Taller 3 - Big Data y Machine Learning
#
# Descripcion:
#   Construir predictores a partir del titulo y la descripcion de cada anuncio
#   usando un modelo tipo BERT pequeno (MiniLM multilingue de Hugging Face,
#   via sentence-transformers) y, como punto de comparacion, TF-IDF.
#   Se generan cuatro familias de variables:
#     (a) txt_pc*         Componentes principales de los embeddings (resumen
#                         no supervisado del contenido del texto).
#     (b) txt_sim_*       Similitud coseno entre cada anuncio y frases ancla
#                         (lujo, remodelar, vista, amenidades, ...). Son
#                         variables interpretables para las diapositivas.
#     (c) txt_bert_lnp    Prediccion out-of-fold del log(precio) con una
#                         regresion Ridge sobre los embeddings ("precio que
#                         sugiere el texto").
#     (d) txt_tfidf_lnp   Lo mismo pero con TF-IDF de palabras y bigramas.
#   Las predicciones (c) y (d) son out-of-fold en train con folds espaciales
#   (grupos de anuncios cercanos), para que el modelo final no vea el precio
#   de cada anuncio a traves de su propia variable de texto. En test se
#   predice con el modelo entrenado en todo train.
# Input:
#   01_data/01_raw/train.csv y 01_data/01_raw/test.csv (title, description,
#   price, lat, lon, property_id).
# Output:
#   01_data/02_text/text_features_train.csv y text_features_test.csv
#     (property_id + variables de texto; se unen por property_id).
#   01_data/02_text/cache/  embeddings en .npy (no se suben a git).
#   03_outputs/02_text/metricas_texto.csv  MAE de los modelos solo-texto.
#   03_outputs/02_text/palabras_precio.csv palabras con mayor y menor efecto
#     sobre el precio en el modelo TF-IDF.
# Uso:
#   python 02_code/04_text_features.py               # corrida completa
#   python 02_code/04_text_features.py --sample 2000 # prueba rapida
# ------------------------------------------------------------------------------

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

SEED = 2026

# ------------------------------------------------------------------------------
# (1) Rutas (relativas a la raiz del repositorio)
# ------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DIR_RAW = ROOT / "01_data" / "01_raw"
DIR_TEXT = ROOT / "01_data" / "02_text"
ARCHIVO_FOLDS = ROOT / "01_data" / "02_clean" / "folds.csv"  # lo crea 13_folds.R
DIR_CACHE = DIR_TEXT / "cache"
DIR_OUT = ROOT / "03_outputs" / "02_text"

# ------------------------------------------------------------------------------
# (2) Parametros
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Variables de texto con MiniLM/BERT")
parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="Modelo de sentence-transformers en Hugging Face")
parser.add_argument("--prefix", default="",
                    help="Prefijo para cada texto (p. ej. 'query: ' para modelos e5)")
parser.add_argument("--n-pca", type=int, default=10, help="Componentes principales")
parser.add_argument("--n-folds", type=int, default=5, help="Folds espaciales")
parser.add_argument("--n-clusters", type=int, default=60,
                    help="Grupos espaciales (KMeans sobre lat/lon) para los folds")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--max-seq-length", type=int, default=256)
parser.add_argument("--sample", type=int, default=0,
                    help="Si > 0, usa solo n filas de train y test (prueba rapida)")
args = parser.parse_args()

np.random.seed(SEED)

# Usa GPU si existe (p. ej. Google Colab); si no, CPU.
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")

# ------------------------------------------------------------------------------
# (3) Datos y limpieza del texto
# ------------------------------------------------------------------------------

train = pd.read_csv(DIR_RAW / "train.csv")
test = pd.read_csv(DIR_RAW / "test.csv")
if args.sample > 0:
    train = train.sample(min(args.sample, len(train)), random_state=SEED)
    test = test.sample(min(args.sample, len(test)), random_state=SEED)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Folds compartidos con los modelos de R (13_folds.R). Solo estan los anuncios
# unicos que dejo 01_cleaning.R: se descartan los duplicados de train, para que
# una copia del mismo anuncio no caiga en otro fold. Con --sample no se usan.
tabla_folds = None
if ARCHIVO_FOLDS.exists() and args.sample == 0:
    tabla_folds = pd.read_csv(ARCHIVO_FOLDS)
    train = train.merge(tabla_folds[["property_id", "fold_espacial"]],
                        on="property_id", how="inner").reset_index(drop=True)
    print(f"Folds de {ARCHIVO_FOLDS.name}: {len(train)} anuncios unicos")
else:
    print("AVISO: sin folds.csv, se usan folds de KMeans (corra 13_folds.R antes)")

# El texto viene sin tildes y con restos de HTML ("aacute", "ntilde"). Se
# aplica la misma correccion que en 01_cleaning.R. Se conservan los numeros
# (pisos, metros, parqueaderos), porque el modelo los puede aprovechar.

def limpiar_texto(titulo, descripcion):
    t = f"{titulo if isinstance(titulo, str) else ''}. " \
        f"{descripcion if isinstance(descripcion, str) else ''}"
    t = t.lower()
    t = re.sub(r"\s*\b([aeiou])acute\b\s*", r"\1", t)
    t = re.sub(r"\s*\bntilde\b\s*", "n", t)
    t = t.replace("²", "2")
    t = re.sub(r"\s+", " ", t).strip()
    return t

texto_train = [limpiar_texto(a, b) for a, b in zip(train["title"], train["description"])]
texto_test = [limpiar_texto(a, b) for a, b in zip(test["title"], test["description"])]
texto_todo = texto_train + texto_test
y_train = np.log(train["price"].to_numpy())

# ------------------------------------------------------------------------------
# (4) Embeddings con sentence-transformers (se guardan en cache)
# ------------------------------------------------------------------------------

def calcular_embeddings(textos, nombre):
    """Codifica los textos con el modelo; reutiliza el .npy si ya existe."""
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    etiqueta = re.sub(r"[^A-Za-z0-9]+", "_", args.model)
    sufijo = f"_s{args.sample}" if args.sample > 0 else ""
    archivo = DIR_CACHE / f"emb_{nombre}_{etiqueta}{sufijo}.npy"
    if archivo.exists():
        emb = np.load(archivo)
        if emb.shape[0] == len(textos):
            print(f"  Embeddings {nombre} leidos de cache: {emb.shape}")
            return emb
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(args.model, device=DEVICE)
    modelo.max_seq_length = args.max_seq_length
    inicio = time.time()
    emb = modelo.encode([args.prefix + t for t in textos], batch_size=args.batch_size,
                        show_progress_bar=True, normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)
    print(f"  Embeddings {nombre}: {emb.shape} en {time.time() - inicio:.0f} s")
    np.save(archivo, emb)
    return emb

print("(4) Embeddings")
emb_todo = calcular_embeddings(texto_todo, "todo")
emb_train = emb_todo[: len(train)]
emb_test = emb_todo[len(train):]

# ------------------------------------------------------------------------------
# (5) Variables no supervisadas
# ------------------------------------------------------------------------------

# (5.1) Componentes principales de los embeddings. No usan el precio, asi que
# se ajustan con train y test juntos.
print("(5) PCA y similitudes con frases ancla")
pca = PCA(n_components=args.n_pca, random_state=SEED)
pcs = pca.fit_transform(emb_todo)
print(f"  Varianza explicada por {args.n_pca} componentes: "
      f"{pca.explained_variance_ratio_.sum():.1%}")
cols_pca = [f"txt_pc{i + 1}" for i in range(args.n_pca)]

# (5.2) Frases ancla. Cada concepto se representa con el promedio de los
# embeddings de varias frases; la variable es la similitud coseno del anuncio
# con ese concepto. Se escriben sin tildes, como vienen los anuncios.
anclas = {
    "lujo": ["acabados de lujo", "apartamento exclusivo de alta gama",
             "excelentes acabados en marmol y madera", "propiedad premium"],
    "remodelar": ["para remodelar", "necesita reparaciones",
                  "inmueble para actualizar", "ideal para inversionista que quiera remodelar"],
    "estrenar": ["apartamento para estrenar", "proyecto nuevo", "obra nueva entrega inmediata"],
    "vista": ["vista panoramica", "excelente vista a los cerros",
              "piso alto con vista exterior", "terraza con vista a la ciudad"],
    "amenidades": ["conjunto con piscina gimnasio y bbq", "zonas comunes salon comunal",
                   "club house con piscina", "porteria 24 horas y parque infantil"],
    "parqueadero": ["parqueadero cubierto", "dos garajes y deposito", "parqueadero privado"],
    "ubic_premium": ["zona exclusiva del norte de bogota", "cerca al parque de la 93",
                     "en el corazon de chapinero alto", "sector rosales chico"],
    "oportunidad": ["precio negociable", "venta urgente", "excelente oportunidad de inversion",
                    "se recibe vehiculo como parte de pago"],
    "pequeno": ["apartaestudio", "ideal para estudiantes", "un ambiente pequeno"],
    "comercial": ["casa con local comercial", "uso mixto renta", "apto para oficina"],
}
cols_sim = []
sims = {}
if anclas:
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(args.model, device=DEVICE)
    for concepto, frases in anclas.items():
        e = modelo.encode([args.prefix + f for f in frases], normalize_embeddings=True)
        centro = e.mean(axis=0)
        centro = centro / np.linalg.norm(centro)
        col = f"txt_sim_{concepto}"
        sims[col] = emb_todo @ centro
        cols_sim.append(col)

# ------------------------------------------------------------------------------
# (6) Variables supervisadas out-of-fold con folds espaciales
# ------------------------------------------------------------------------------

# Grupos espaciales: KMeans sobre lat/lon. Con GroupKFold cada grupo queda
# entero en un fold, asi la validacion imita predecir una zona no vista
# (como Chapinero en test).
print("(6) Modelos solo-texto out-of-fold")
if tabla_folds is not None:
    id_fold = train["fold_espacial"].to_numpy()
    folds = [(np.where(id_fold != k)[0], np.where(id_fold == k)[0])
             for k in sorted(np.unique(id_fold))]
    args.n_folds = len(folds)
else:
    coords = train[["lat", "lon"]].fillna(train[["lat", "lon"]].median()).to_numpy()
    grupos = KMeans(n_clusters=min(args.n_clusters, len(train) // 10), random_state=SEED,
                    n_init=10).fit_predict(coords)
    folds = list(GroupKFold(n_splits=args.n_folds).split(coords, groups=grupos))


def oof_ridge(X_tr, X_te, alphas, nombre):
    """Escoge alpha por MAE (en pesos) con folds espaciales y devuelve la
    prediccion out-of-fold en train y la prediccion en test."""
    resultados = []
    mejor = None
    for alpha in alphas:
        oof = np.zeros(len(y_train))
        for idx_tr, idx_va in folds:
            m = Ridge(alpha=alpha).fit(X_tr[idx_tr], y_train[idx_tr])
            oof[idx_va] = m.predict(X_tr[idx_va])
        mae = np.mean(np.abs(np.exp(oof) - np.exp(y_train)))
        resultados.append({"modelo": nombre, "alpha": alpha, "mae_cv_espacial": mae})
        print(f"  {nombre} alpha={alpha:<6} MAE={mae / 1e6:,.1f} millones")
        if mejor is None or mae < mejor[1]:
            mejor = (alpha, mae, oof)
    final = Ridge(alpha=mejor[0]).fit(X_tr, y_train)
    return mejor[2], final.predict(X_te), final, pd.DataFrame(resultados)


# (6.1) Ridge sobre embeddings de MiniLM
oof_bert, test_bert, _, met_bert = oof_ridge(emb_train, emb_test, [0.03, 0.1, 0.3, 1, 3], "ridge_minilm")

# (6.2) Ridge sobre TF-IDF (palabras y bigramas). El vocabulario se ajusta con
# train y test juntos (no usa el precio).
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.85, sublinear_tf=True,
                        dtype=np.float32)
X_tfidf = tfidf.fit_transform(texto_todo)
X_tfidf_tr, X_tfidf_te = X_tfidf[: len(train)], X_tfidf[len(train):]
oof_tfidf, test_tfidf, ridge_tfidf, met_tfidf = oof_ridge(X_tfidf_tr, X_tfidf_te,
                                                          [0.1, 0.3, 1, 3, 10], "ridge_tfidf")

# Referencia: predecir siempre la mediana de train
mae_base = np.mean(np.abs(np.median(train["price"]) - train["price"]))
metricas = pd.concat([met_bert, met_tfidf, pd.DataFrame(
    [{"modelo": "mediana_train", "alpha": np.nan, "mae_cv_espacial": mae_base}])])

# (6.3) Palabras que mas suben y bajan el precio en el modelo TF-IDF
coef = pd.DataFrame({"termino": tfidf.get_feature_names_out(), "coef": ridge_tfidf.coef_})
palabras = pd.concat([coef.nlargest(40, "coef").assign(direccion="sube"),
                      coef.nsmallest(40, "coef").assign(direccion="baja")])

# ------------------------------------------------------------------------------
# (7) Exportar
# ------------------------------------------------------------------------------

feats = pd.DataFrame(pcs, columns=cols_pca)
for col in cols_sim:
    feats[col] = sims[col]
feats["txt_bert_lnp"] = np.concatenate([oof_bert, test_bert])
feats["txt_tfidf_lnp"] = np.concatenate([oof_tfidf, test_tfidf])
feats["txt_n_palabras"] = [len(t.split()) for t in texto_todo]
feats.insert(0, "property_id", pd.concat([train["property_id"], test["property_id"]]).to_numpy())

DIR_TEXT.mkdir(parents=True, exist_ok=True)
DIR_OUT.mkdir(parents=True, exist_ok=True)
sufijo = f"_s{args.sample}" if args.sample > 0 else ""
feats.iloc[: len(train)].to_csv(DIR_TEXT / f"text_features_train{sufijo}.csv", index=False)
feats.iloc[len(train):].to_csv(DIR_TEXT / f"text_features_test{sufijo}.csv", index=False)
metricas.to_csv(DIR_OUT / f"metricas_texto{sufijo}.csv", index=False)
palabras.to_csv(DIR_OUT / f"palabras_precio{sufijo}.csv", index=False)

print("(7) Listo")
print(f"  Variables: {feats.shape[1] - 1} -> {DIR_TEXT}")
print(f"  MAE mediana (referencia): {mae_base / 1e6:,.1f} millones")
print(metricas.groupby("modelo")["mae_cv_espacial"].min().div(1e6).round(1).to_string())