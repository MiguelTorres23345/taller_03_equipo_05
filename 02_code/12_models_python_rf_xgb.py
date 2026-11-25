# ------------------------------------------------------------------------------
# Archivo: 12_models_python_rf_xgb.py
# Proyecto: Taller 3 - Big Data y Machine Learning (precios de vivienda, Chapinero)
#
# Descripcion:
#   Random forest y XGBoost en Python con las variables de texto.
#     1. Une train_planas.csv (limpieza + estrato, valor de referencia, OSM, etc.)
#        con las variables de texto por property_id.
#     2. Validacion cruzada espacial: un fold por localidad, sin entrenar con lo
#        que este a menos de 300 m de la localidad que se valida.
#     3. Busca hiperparametros de RF y XGBoost minimizando el MAE.
#     4. Ablacion con tres conjuntos de variables:
#          A_base       las 19 variables de 05_modelos.R (sin texto)
#          B_texto      A + 23 variables de TF-IDF y MiniLM congelado
#          C_finetune   B + prediccion del MiniLM con fine-tuning
#     5. Entrena con todo train y exporta predicciones e importancias.
# Input (desde la raiz del repositorio):
#   01_data/clean/train_planas.csv y test_planas.csv
#   01_data/02_text/text_features_{train,test}.csv
#   01_data/02_text/text_finetune_minilm_{train,test}.csv
#   01_data/01_raw/fuentes_externas/localidades/Loca.shp (shapefile completo)
# Output:
#   03_outputs/03_python/ (tablas, figuras) y 03_outputs/03_python/submits/
# Uso:
#   python 02_code/12_models_python_rf_xgb.py                 (completo, sin GPU)
#   python 02_code/12_models_python_rf_xgb.py --sample 600    (prueba rapida)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Paquetes y rutas
# ------------------------------------------------------------------------------

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')   # solo guarda las figuras
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

ROOT = Path.cwd().parent if Path.cwd().name == '02_code' else Path.cwd()

parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=int, default=0,
                    help='prueba rapida: N anuncios de train y grilla minima (0 = completo)')
args = parser.parse_args()

# Bases de 02_variables_planas.R: 01_data/02_clean si ya se corrio el flujo; si no,
# la copia guardada en el repositorio (01_data/clean)
DIR_CLEAN = ROOT / '01_data' / '02_clean'
if not (DIR_CLEAN / 'train_planas.csv').exists():
    DIR_CLEAN = ROOT / '01_data' / 'clean'
DIR_TEXT = ROOT / '01_data' / '02_text'
# La prueba rapida escribe aparte para no pisar los resultados completos
DIR_OUT = ROOT / '03_outputs' / ('03_python_prueba' if args.sample else '03_python')
DIR_SUBMIT = DIR_OUT / 'submits'
DIR_SUBMIT.mkdir(parents=True, exist_ok=True)

SEED = 2026
BUFFER = 300          # metros excluidos alrededor de la localidad que se valida
N_ARBOLES_RF = 300    # arboles del random forest
print('Raiz:', ROOT)

# ------------------------------------------------------------------------------
# (2) Datos: base externa + variables de texto
# ------------------------------------------------------------------------------

train = pd.read_csv(DIR_CLEAN / 'train_planas.csv')
test = pd.read_csv(DIR_CLEAN / 'test_planas.csv')

# Variables de texto (06_text_features.py) y fine-tuning (09_text_finetune.py)
txt_tr = pd.read_csv(DIR_TEXT / 'text_features_train.csv')
txt_te = pd.read_csv(DIR_TEXT / 'text_features_test.csv')
ft_tr = pd.read_csv(DIR_TEXT / 'text_finetune_minilm_train.csv')
ft_te = pd.read_csv(DIR_TEXT / 'text_finetune_minilm_test.csv')

if args.sample:
    train = train.sample(args.sample, random_state=2026).reset_index(drop=True)
    test = test.head(args.sample).reset_index(drop=True)

train = train.merge(txt_tr, on='property_id', how='left').merge(ft_tr, on='property_id', how='left')
test = test.merge(txt_te, on='property_id', how='left').merge(ft_te, on='property_id', how='left')

VARS_TEXTO = [c for c in txt_tr.columns if c != 'property_id']
VARS_FT = [c for c in ft_tr.columns if c != 'property_id']
assert train[VARS_TEXTO + VARS_FT].notna().all().all(), 'Hay anuncios de train sin variables de texto'
assert test[VARS_TEXTO + VARS_FT].notna().all().all(), 'Hay anuncios de test sin variables de texto'
print('train:', train.shape, '| test:', test.shape)

# ------------------------------------------------------------------------------
# (3) Conjuntos de variables
# ------------------------------------------------------------------------------

VARS_BASE = ['surface_covered', 'bedrooms', 'bathrooms', 'property_type', 'month', 'year',
             'estrato', 'valor_ref', 'uso', 'incidentes', 'dist_est5', 'dist_est6',
             'prop_comercio', 'dist_parque', 'dist_restaurante', 'dist_universidad',
             'dist_financiero', 'dist_avenida', 'dist_tm']
CATEGORICAS = ['property_type', 'uso']

def preparar(df, ref):
    df = df.copy()
    for v in VARS_BASE:
        if v in CATEGORICAS:
            df[v] = df[v].fillna(ref[v].mode()[0]).astype(str)
        else:
            df[v] = df[v].fillna(ref[v].median())
    return df

tr = preparar(train, train)
te = preparar(test, train)

# Dummies con las mismas columnas en train y test
todo = pd.get_dummies(pd.concat([tr, te], keys=['tr', 'te']), columns=CATEGORICAS, dtype=float)
tr, te = todo.loc['tr'].reset_index(drop=True), todo.loc['te'].reset_index(drop=True)
dummies = [c for c in tr.columns if c.startswith(tuple(f'{v}_' for v in CATEGORICAS))]
base = [v for v in VARS_BASE if v not in CATEGORICAS] + dummies

CONJUNTOS = {
    'A_base': base,
    'B_texto': base + VARS_TEXTO,
    'C_finetune': base + VARS_TEXTO + VARS_FT,
}
y = (train['price'] / 1e6).to_numpy()
for k, v in CONJUNTOS.items():
    print(f'{k}: {len(v)} columnas')

# ------------------------------------------------------------------------------
# (4) Validacion cruzada espacial por localidad
# ------------------------------------------------------------------------------

loc = gpd.read_file(ROOT / '01_data' / '01_raw' / 'fuentes_externas' / 'localidades' / 'Loca.shp').to_crs(3116)
loc['geometry'] = loc.geometry.make_valid()
pts = gpd.GeoDataFrame(train[['property_id']], crs=4326,
                       geometry=gpd.points_from_xy(train['lon'], train['lat'])).to_crs(3116)
pts = gpd.sjoin_nearest(pts, loc[['LocNombre', 'geometry']], how='left')
pts = pts[~pts.index.duplicated()]          # empates: se deja la primera
localidad = pts['LocNombre'].to_numpy()

folds = []
for nombre in sorted(set(localidad)):
    idx_va = np.where(localidad == nombre)[0]
    poligono = loc.loc[loc['LocNombre'] == nombre].geometry.union_all()
    lejos = pts.geometry.distance(poligono).to_numpy() > BUFFER
    idx_tr = np.where(lejos & (localidad != nombre))[0]
    folds.append((nombre, idx_tr, idx_va))

print(pd.DataFrame([(n, len(a), len(b)) for n, a, b in folds],
                   columns=['localidad', 'n_entrena', 'n_valida']).to_string(index=False))

# ------------------------------------------------------------------------------
# (5) Resumen de errores y funcion de validacion
# ------------------------------------------------------------------------------

def resumen(pred):
    err = pred - y                                  # > 0: el modelo sobreestima
    chap = localidad == 'CHAPINERO'
    return {
        'MAE': np.mean(np.abs(err)),
        'MAE_chapinero': np.mean(np.abs(err[chap])) if chap.any() else np.nan,
        'error_medio': np.mean(err),                # sesgo: + sobreestima, - subestima
        'pct_sobreestima': np.mean(err > 0) * 100,
    }

def validar(crear_modelo, columnas):
    '''Predice cada anuncio con un modelo entrenado sin su localidad.
    Devuelve las predicciones fuera de muestra y el resumen de errores.'''
    X = tr[columnas].to_numpy(dtype=np.float32)
    pred = np.full(len(y), np.nan)
    for nombre, idx_tr, idx_va in folds:
        m = crear_modelo()
        m.fit(X[idx_tr], y[idx_tr])
        pred[idx_va] = m.predict(X[idx_va])
    return pred, resumen(pred)

# ------------------------------------------------------------------------------
# (6) Random forest: busqueda de hiperparametros
# ------------------------------------------------------------------------------

grid_rf = list(product([0.33, 0.5], [5, 20]))   # (max_features, min_samples_leaf)
if args.sample:
    grid_rf = grid_rf[:1]
res_rf = []
for mf, hoja in grid_rf:
    t0 = time.time()
    crear = lambda: RandomForestRegressor(n_estimators=N_ARBOLES_RF, max_features=mf,
                                          min_samples_leaf=hoja, n_jobs=-1, random_state=SEED)
    pred, r = validar(crear, CONJUNTOS['C_finetune'])
    res_rf.append({'max_features': mf, 'min_samples_leaf': hoja, **r, 'pred': pred})
    print(f'RF mf={mf} hoja={hoja}: MAE {r["MAE"]:.1f} M | Chapinero {r["MAE_chapinero"]:.1f} M '
          f'| {time.time() - t0:.0f} s')

mejor_rf = min(res_rf, key=lambda d: d['MAE'])
print('\nMejor RF:', {k: mejor_rf[k] for k in ['max_features', 'min_samples_leaf', 'MAE']})

# ------------------------------------------------------------------------------
# (7) XGBoost: busqueda de hiperparametros
# ------------------------------------------------------------------------------

grid_xgb = list(product([300, 600], [3, 6], [0.03, 0.1]))   # (n_estimators, max_depth, eta)
if args.sample:
    grid_xgb = grid_xgb[:1]
res_xgb = []
for n, d, eta in grid_xgb:
    t0 = time.time()
    crear = lambda: XGBRegressor(n_estimators=n, max_depth=d, learning_rate=eta,
                                 objective='reg:absoluteerror', subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=5,
                                 tree_method='hist', n_jobs=-1, random_state=SEED)
    pred, r = validar(crear, CONJUNTOS['C_finetune'])
    res_xgb.append({'n_estimators': n, 'max_depth': d, 'eta': eta, **r, 'pred': pred})
    print(f'XGB n={n} depth={d} eta={eta}: MAE {r["MAE"]:.1f} M | '
          f'Chapinero {r["MAE_chapinero"]:.1f} M | {time.time() - t0:.0f} s')

mejor_xgb = min(res_xgb, key=lambda d: d['MAE'])
print('\nMejor XGB:', {k: mejor_xgb[k] for k in ['n_estimators', 'max_depth', 'eta', 'MAE']})

# Tabla de la busqueda de ambos modelos
pd.concat([pd.DataFrame(res_rf).drop(columns='pred').assign(modelo='random_forest'),
           pd.DataFrame(res_xgb).drop(columns='pred').assign(modelo='xgboost')]
          ).to_csv(DIR_OUT / 'busqueda_hiperparametros.csv', index=False)

# ------------------------------------------------------------------------------
# (8) Ablacion A vs. B vs. C
# ------------------------------------------------------------------------------

def crear_rf():
    return RandomForestRegressor(n_estimators=N_ARBOLES_RF, max_features=mejor_rf['max_features'],
                                 min_samples_leaf=mejor_rf['min_samples_leaf'],
                                 n_jobs=-1, random_state=SEED)

def crear_xgb():
    return XGBRegressor(n_estimators=mejor_xgb['n_estimators'], max_depth=mejor_xgb['max_depth'],
                        learning_rate=mejor_xgb['eta'], objective='reg:absoluteerror',
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        tree_method='hist', n_jobs=-1, random_state=SEED)

MODELOS = {'random_forest': crear_rf, 'xgboost': crear_xgb}
MEJORES = {'random_forest': mejor_rf, 'xgboost': mejor_xgb}
filas = []
for nombre_m, crear in MODELOS.items():
    for conj, cols in CONJUNTOS.items():
        if conj == 'C_finetune':   # ya se valido en la busqueda
            r = resumen(MEJORES[nombre_m]['pred'])
        else:
            _, r = validar(crear, cols)
        filas.append({'modelo': nombre_m, 'variables': conj, **r})
        print(f'{nombre_m:13s} {conj:11s} MAE {r["MAE"]:.1f} M')

comparacion = pd.DataFrame(filas)
comparacion.to_csv(DIR_OUT / 'comparacion_cv_espacial.csv', index=False)
print(comparacion.round(1).to_string(index=False))

# ------------------------------------------------------------------------------
# (9) Figura de la ablacion
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))
piv = comparacion.pivot(index='variables', columns='modelo', values='MAE')
piv.plot.barh(ax=ax, color=['#7a9cc0', '#2f5d8a'])
ax.set_xlabel('MAE validación espacial (millones de pesos)')
ax.set_ylabel('')
ax.set_title('Efecto de las variables de texto')
ax.invert_yaxis()
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f', padding=3, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(DIR_OUT / 'comparacion_texto.png', dpi=200)
plt.close()

# ------------------------------------------------------------------------------
# (10) Entrenar con todo train y exportar para Kaggle
# ------------------------------------------------------------------------------

modelos_finales = {}
for nombre_m, crear in MODELOS.items():
    for conj, cols in CONJUNTOS.items():
        m = crear()
        m.fit(tr[cols].to_numpy(dtype=np.float32), y)
        modelos_finales[(nombre_m, conj)] = m
        precio = m.predict(te[cols].to_numpy(dtype=np.float32)) * 1e6
        archivo = DIR_SUBMIT / f'{nombre_m}_{conj}.csv'
        pd.DataFrame({'property_id': test['property_id'], 'price': precio}).to_csv(archivo, index=False)
        print(f'{archivo.name}: mediana {np.median(precio) / 1e6:,.0f} M')

# ------------------------------------------------------------------------------
# (11) Importancia de las variables
# ------------------------------------------------------------------------------

cols = CONJUNTOS['C_finetune']
imp = pd.DataFrame({
    'variable': cols,
    'random_forest': modelos_finales[('random_forest', 'C_finetune')].feature_importances_,
    'xgboost': modelos_finales[('xgboost', 'C_finetune')].feature_importances_,
})
imp['es_texto'] = imp['variable'].str.startswith('txt_')
imp.sort_values('xgboost', ascending=False).to_csv(DIR_OUT / 'importancia_variables.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, m in zip(axes, ['random_forest', 'xgboost']):
    top = imp.nlargest(15, m).iloc[::-1]
    ax.barh(top['variable'], top[m], color=np.where(top['es_texto'], '#2f5d8a', '#b9b5ab'))
    ax.set_title(m.replace('_', ' ').title())
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Top 15 variables (azul = variables de texto)')
plt.tight_layout()
plt.savefig(DIR_OUT / 'importancia_variables.png', dpi=200)
plt.close()
