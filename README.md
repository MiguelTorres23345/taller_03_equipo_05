# Taller 3 · Predicción de precios de vivienda en Chapinero

Equipo 05 · Big Data y Machine Learning · Universidad de los Andes

## Objetivo y resultado

Predecir el precio en COP de los avisos de vivienda de Chapinero (Bogotá) a partir de sus características, su ubicación y el texto de su descripción. La métrica es el error absoluto medio (MAE).

- **Modelo final:** XGBoost con 1000 rondas, profundidad 8, eta 0,01 y `min_child_weight` 20, con variables de texto y validación espacial por bloques de 400 m.
- **MAE público en Kaggle: 152,97 millones de pesos.**
- Validación espacial por bloques: MAE ≈ 114,6 millones en general y ≈ 147,9 millones en Chapinero. El random forest llega a ≈ 119,2 millones.

## Antes de empezar: datos

GitHub guarda código, no datos. Quien clone el repositorio debe tener estos archivos en `01_data/01_raw/`:

1. Descargar `train.csv`, `test.csv` y `submission_template.csv` de la competencia de Kaggle del Taller 3 y guardarlos con esos nombres en `01_data/01_raw/`.
2. Las capas distritales van en `01_data/01_raw/fuentes_externas/`, una carpeta por fuente, con todos los componentes de cada shapefile (`.shp`, `.dbf`, `.shx`, `.prj`, ...). Ya vienen en el repositorio.
3. Los CSV de texto van en `01_data/02_text/` (`text_features_*.csv` y `text_finetune_minilm_*.csv`). También vienen en el repositorio; ver la sección de Colab si se quieren regenerar.
4. El extracto de OpenStreetMap (Bogotá, BBBike) se descarga solo la primera vez que corre `02_variables_planas.R` (necesita internet). Si falla la descarga, bájelo a mano de <https://download.bbbike.org/osm/bbbike/Bogota/> (`Bogota.osm.pbf`), guárdelo como `01_data/01_raw/fuentes_externas/osm/bbbike_Bogota.osm.pbf` y vuelva a correr. El PBF que se uso tiene SHA-256 `fe14497e823059909fb2b2c89f2e745782eefc37673e5d93948e3d2c92389c3f`; las actualizaciones de OpenStreetMap pueden cambiar un poco las distancias.

### Fuentes externas

| Fuente | Qué aporta | Dónde queda |
|---|---|---|
| Properati (vía Kaggle) | Avisos de venta con precio, características, coordenadas y texto | `01_raw/train.csv`, `test.csv` |
| Secretaría Distrital de Planeación | Estratificación por manzana | `fuentes_externas/estratificacion/` |
| Catastro Distrital | Valor de referencia por m² (2020) y uso predominante por manzana | `fuentes_externas/valor_referencia/`, `uso_predominante/` |
| IDECA | Localidades de Bogotá e incidentes de seguridad por sector catastral | `fuentes_externas/localidades/`, `seguridad/` |
| TransMilenio | Estaciones troncales | `fuentes_externas/transmilenio/` |
| OpenStreetMap (BBBike) | Parques, restaurantes, universidades, bancos y avenidas | `fuentes_externas/osm/` |

### Coordenadas

Los datos vienen en WGS84 (EPSG 4326). Para medir distancias en metros se proyectan a MAGNA-SIRGAS / Colombia Bogotá (EPSG 3116).

## Requisitos

- **R 4.5** (probado con 4.5.2). Los paquetes que faltan se instalan solos con `pacman` al correr cada bloque. En Linux, instalar antes los compiladores y las bibliotecas de desarrollo de GDAL, GEOS, PROJ, SQLite y udunits-2 (requisitos de `sf`).
- **Python 3.11 o superior** (probado con 3.14) para `06_text_features.py` y `12_models_python_rf_xgb.py`:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     # Windows
.venv/bin/python -m pip install -r requirements.txt          # Linux o Mac
```

Para que `main.R` use ese entorno: `PYTHON_BIN=.venv/Scripts/python.exe Rscript main.R`.

- Las versiones de R usadas quedan en `04_documentation/sessionInfo_R.txt`, que escribe `main.R` al terminar el flujo final.

## Cómo ejecutar

Todo se corre desde la raíz del repositorio.

```bash
Rscript main.R                 # flujo final completo (equivale a: Rscript main.R final)
Rscript main.R xgboost         # solo el XGBoost final
Rscript main.R random_forest   # solo el random forest
```

Si falta `01_data/02_clean/train_planas.csv`, el flujo corre primero la limpieza (`cleaning`), que además genera los embeddings E5 (ver tiempos). Los bloques sueltos se piden por nombre: `folds`, `texto_python`, `modelos_texto`, `xgboost`, `random_forest`, `modelos_python`, `ablacion`, `figuras`.

Python: `main.R` usa, en este orden, la variable de entorno `PYTHON_BIN`, el entorno `.venv` de la raiz y el Python del PATH. Si el elegido no tiene los paquetes de `requirements.txt`, se detiene con un aviso.

Interruptores al inicio de `main.R`:

| Variable | Valor por defecto | Efecto |
|---|---|---|
| `correr_texto_python` | `FALSE` | `FALSE` usa los CSV de texto guardados en `01_data/02_text/`; `TRUE` regenera `text_features_*.csv` |
| `correr_modelos_python` | `TRUE` | `FALSE` omite el RF y el XGBoost en Python |

El fine-tuning (`09_text_finetune.py`) **nunca** corre desde `main.R`: ver la sección de Colab.

## Qué hace cada script

| Orden | Script | Objetivo | Resultado |
|---|---|---|---|
| 1 | `01_cleaning.R` | Quitar duplicados y completar área y baños con el texto | `01_data/02_clean/train_clean.csv`, `test_clean.csv` |
| 2 | `02_variables_planas.R` | Unir estrato, valor de referencia, uso, seguridad, TransMilenio y OSM | `01_data/02_clean/train_planas.csv`, `test_planas.csv` |
| 3 | `13_folds.R` | Crear los folds que usan todos los modelos | `01_data/02_clean/folds.csv` |
| 4 | `06_text_features.py` | Variables de texto (TF-IDF y MiniLM sin ajustar) | `01_data/02_text/text_features_*.csv` |
| 5 | `09_text_finetune.py` | Fine-tuning de MiniLM (Colab con GPU) | `01_data/02_text/text_finetune_minilm_*.csv` |
| 6 | `14_models_text.R` | Random forest y XGBoost con texto, validación por bloques de 400 m | `03_outputs/04_modelos_texto/` |
| 7 | `12_models_python_rf_xgb.py` | RF y XGBoost en Python, ablación A/B/C del texto | `03_outputs/03_python/` |
| 8 | `15_ablacion_fuentes.R` | Ablación por fuente de datos con el mejor XGBoost | `03_outputs/06_ablacion/` |
| 9 | `16_figuras_slides.R`, `17_figura_validacion.R`, `18_figura_bloques.R` | Figuras de las diapositivas | `03_outputs/figuras_slides/` y `03_outputs/mapas/` |

Scripts de modelos anteriores, que se conservan y se piden con `Rscript main.R todo` o por bloque: `03_models_raw_data.R` (`arboles`), `08_models_nn_raw_data_base_r.R` (`nn`), `08_models_clean_data.R` (`modelos_limpios`), `10_models_regresion_basica.R`, `11_models_regresion_log.R` y `09_models_text_svd.R` (`xgboost_text_svd`). `05_modelos.R` es la versión previa de `14_models_text.R` y no está conectada a `main.R`.

### Si solo quiero...

| Quiero | Comando |
|---|---|
| El XGBoost final | `Rscript main.R xgboost` |
| El random forest | `Rscript main.R random_forest` |
| Las figuras de las slides | `Rscript main.R figuras` |
| La ablación por fuentes | `Rscript main.R ablacion` |
| Solo los modelos en Python | `python 02_code/12_models_python_rf_xgb.py` |

## De dónde sale cada figura y tabla de las slides

| Figura o tabla | Script | Archivo |
|---|---|---|
| Mapa de train y test | `16_figuras_slides.R` | `03_outputs/figuras_slides/mapa_train_test.png` y `mapa_train_test_zoom.png` |
| Bloques de la validación espacial | `16_figuras_slides.R` | `03_outputs/figuras_slides/mapa_bloques_cv.png` |
| Distancia de lo validado al entrenamiento | `16_figuras_slides.R` | `03_outputs/figuras_slides/distancia_validacion.png` |
| Ablación por fuentes (figura) | `16_figuras_slides.R` y `15_ablacion_fuentes.R` | `figuras_slides/ablacion_fuentes.png`, `06_ablacion/ablacion_fuentes.png` |
| Ablación por fuentes (tabla) | `15_ablacion_fuentes.R` | `03_outputs/06_ablacion/ablacion_fuentes.csv` |
| Tres paneles de validación (azar, localidad con buffer, bloques) | `17_figura_validacion.R` | `03_outputs/mapas/validacion_espacial.png` |
| Un fold de bloques de 400 m | `18_figura_bloques.R` | `03_outputs/mapas/validacion_bloques.png` |
| MAE e hiperparámetros del XGBoost y del RF finales | `14_models_text.R` | `03_outputs/04_modelos_texto/resultados_xgb_bloques_coord_texto_ft.csv` y `resultados_rf_bloques_coord_texto_ft.csv` |
| Efecto del texto (A, B, C) y búsqueda de hiperparámetros en Python | `12_models_python_rf_xgb.py` | `03_outputs/03_python/comparacion_cv_espacial.csv`, `busqueda_hiperparametros.csv`, `importancia_variables.csv` y sus `.png` |
| Métricas de las variables de texto | `06_text_features.py`, `09_text_finetune.py` | `03_outputs/02_text/metricas_texto.csv`, `palabras_precio.csv`, `metricas_finetune_minilm.csv` |

## Colab y Hugging Face (fine-tuning de MiniLM)

`09_text_finetune.py` ajusta todos los pesos de MiniLM (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, de Hugging Face) para predecir el logaritmo del precio a partir del título y la descripción. Se corrió en **Google Colab con GPU** porque los computadores del equipo no tienen GPU y en CPU el ajuste tarda días. **El código completo está en este repositorio**; Colab solo aportó la GPU. El flujo local usa los CSV ya generados y no repite el ajuste.

Para volver a correrlo:

1. Subir a Drive la carpeta `taller3/` con `02_code/09_text_finetune.py`, `01_data/01_raw/train.csv` y `test.csv` (y `01_data/02_clean/folds.csv` si se quieren los mismos folds que los modelos).
2. En Colab: Entorno de ejecución, Cambiar tipo de entorno, T4 GPU.
3. Instalar dependencias: `pip install -r requirements.txt` (en Colab basta `pip install sentence-transformers`).
4. Ejecutar `python 02_code/09_text_finetune.py --model minilm`. Hay un paso a paso en `02_code/colab_texto.ipynb`.
5. Descargar `01_data/02_text/text_finetune_minilm_train.csv` y `text_finetune_minilm_test.csv` y copiarlos a la misma ruta del repositorio.

Para comprobar que el script funciona sin GPU: `python 02_code/09_text_finetune.py --model minilm --sample 200 --epochs 1 --folds 0`. No pide token de Hugging Face.

`06_text_features.py` usa MiniLM sin ajustar y corre en CPU.

## Nota sobre los folds del texto

Los CSV de texto que están en el repositorio tienen 38.644 filas (incluyen anuncios duplicados) y se calcularon con folds de KMeans sobre las coordenadas. `13_folds.R` deja 33.769 anuncios únicos. Si existe `01_data/02_clean/folds.csv`, `06_text_features.py` y `09_text_finetune.py` lo usan; si no, usan KMeans. Regenerar el texto con los folds nuevos exige repetir el fine-tuning en Colab. El modelo final reportado se entrenó con el texto calculado con los folds de KMeans, que es lo que hay en `01_data/02_text/`; `14_models_text.R` usa solo las 33.769 filas que coinciden por `property_id`.

## Tiempos de ejecución

Medidos en un computador con Windows 11, 12 hilos, sin GPU y R 4.5.2. Primera corrida desde cero.

| Paso | Tiempo |
|---|---|
| `cleaning`: limpieza, variables planas y embeddings E5 (incluye la descarga de ONNX Runtime y de los pesos de E5, unos 0,5 GB) | 56 min |
| `13_folds.R` | 6 s |
| `14_models_text.R` (random forest y XGBoost con validación espacial) | 144 min |
| `12_models_python_rf_xgb.py` | 62 min |
| `15_ablacion_fuentes.R` | 13 min |
| `16`, `17` y `18` (figuras) | 4 min |
| **Total del flujo final** | **unas 4 h 40 min** |

`17_figura_validacion.R` llega a usar cerca de 18 GB de memoria RAM durante unos minutos.

Pruebas rápidas en CPU, con muestras: `12_models_python_rf_xgb.py --sample 600` tarda 1,5 min; `06_text_features.py --sample 300`, 2 min; `09_text_finetune.py --sample 200 --epochs 1 --folds 0`, 5 min. El fine-tuning completo no se corre en local (ver Colab).

Si solo se necesita el modelo final: `Rscript main.R xgboost`. Tarda menos que los 144 min del bloque completo (no se midió por separado; la grilla de XGBoost es de 24 combinaciones por 15 particiones).

## Estructura de carpetas

```
01_data/
  01_raw/              datos de Kaggle y capas externas (fuentes_externas/)
  02_text/             variables de texto (CSV)
  02_clean/            bases limpias y folds (se generan; no se versionan)
  clean/               copia de las bases planas del equipo
02_code/               scripts numerados por etapa
03_outputs/            resultados, figuras y archivos para Kaggle
04_documentation/      enunciado y sessionInfo de R
main.R                 punto de entrada
requirements.txt       entorno de Python
```

## Flujo anterior (bloques de `todo`)

El flujo previo del repositorio sigue disponible: `Rscript main.R todo` corre `arboles`, `nn`, `modelos_limpios`, `regresion_basica` y `xgboost_text_svd`, con E5 y una red neuronal. Su comprobación de referencia fue un XGBoost con TF-IDF/SVD con MAE de 131,14 millones en validación y 10.286 predicciones. Pide `sbert`, `onnxr` y `reticulate`, y descarga ONNX Runtime y los pesos de E5 la primera vez. Las salidas están en `03_outputs/01_models_raw_data` a `06_models_regresion_log`.
