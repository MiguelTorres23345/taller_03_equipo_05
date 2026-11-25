# Taller 3 · Precios de vivienda en Chapinero

**Pregunta de predicción.** ¿Qué tan bien podemos predecir el precio en COP de los avisos de vivienda de Chapinero usando sus características, su ubicación y su texto? Comparamos un punto de partida tabular, redes con transferencia y modelos enriquecidos. El archivo de entrega de este flujo corresponde al XGBoost con TF-IDF y SVD.

## Preparación

1. Abra una terminal en la raíz del repositorio. La réplica local se verificó con R 4.5.3. En Linux, instale los compiladores y las bibliotecas de desarrollo de GDAL, GEOS, PROJ, SQLite y udunits-2 antes de instalar los paquetes de R. [Requisitos de sf](https://cran.r-project.org/package=sf)
2. Verifique que `train.csv`, `test.csv` y `submission_template.csv` estén en `01_data/01_raw/`. Si su copia del repositorio carece de esos archivos, descárguelos de la competencia de Kaggle del Taller 3 y use esos nombres. Las capas distritales se leen desde `01_data/01_raw/fuentes_externas/`. Conserve juntos los componentes de cada shapefile.
3. Los cuatro CSV de texto del equipo están en `01_data/02_text/`. Para regenerarlos, ejecute `python 02_code/06_text_features.py` y `python 02_code/09_text_finetune.py --model minilm` en un entorno con PyTorch, sentence-transformers, transformers, pandas y scikit-learn. Son insumos de la limpieza. Las predicciones de texto entrenadas con otros folds quedan en la base para trazabilidad y fuera de los predictores de nuestros árboles.
4. Para replicar los resultados del extracto histórico de OpenStreetMap, ubique el `bbbike_Bogota.gpkg` del paquete de insumos del equipo en `01_data/01_raw/fuentes_externas/osm/`. El PBF original en esa carpeta también sirve para regenerar el extracto. Su SHA-256 es `fe14497e823059909fb2b2c89f2e745782eefc37673e5d93948e3d2c92389c3f`. Si falta el extracto local, el código descarga el disponible en BBBike. Las actualizaciones de OpenStreetMap pueden cambiar las distancias y predicciones.

`00_setup.R` instala los paquetes requeridos por el bloque. ONNX y los pesos de E5 se descargan al generar esos vectores por primera vez. Para TF-IDF/SVD, reticulate usa un Python compatible o prepara un entorno con NumPy 2.2.6, scikit-learn 1.9.1 y threadpoolctl 3.8.0. La réplica local usó Python 3.11.9, XGBoost 3.2.1.1, sf 1.1-0 y sbert 0.6.0. En una máquina nueva, las instalaciones y descargas requieren acceso a CRAN, Hugging Face y al repositorio de paquetes de Python. [Entorno administrado de reticulate](https://rstudio.github.io/reticulate/reference/py_require.html)

## Ejecución

```bash
Rscript main.R todo
```

Este comando ejecuta todos los bloques en orden. Si faltan las bases enriquecidas, la limpieza las prepara automáticamente usando los embeddings de la red. Para reconstruirlas explícitamente, ejecute `Rscript main.R cleaning`. Para regenerar o verificar solo el archivo de embeddings, ejecute `Rscript main.R nn_embeddings`. Las bases y los embeddings se guardan en `01_data/02_clean/`. La huella de los embeddings se calcula con los textos y la versión del codificador, de forma que un cambio de folds puede reutilizarlos.

| Comando desde la raíz | Script | Salida de referencia |
|---|---|---|
| `Rscript main.R arboles` | [03_models_raw_data.R](02_code/03_models_raw_data.R) | `03_outputs/01_models_raw_data/resumen.xlsx` y un CSV por familia |
| `Rscript main.R nn` | [08_models_nn_raw_data_base_r.R](02_code/08_models_nn_raw_data_base_r.R) | `03_outputs/02_models_nn_raw_data_base_r/resumen.xlsx`, predicciones y pesos de tres arquitecturas |
| `Rscript main.R cleaning` | [01_cleaning.R](02_code/01_cleaning.R) y [02_variables_planas.R](02_code/02_variables_planas.R) | `01_data/02_clean/train_clean.rds` y `test_clean.rds`, más las bases tabulares y geográficas en CSV |
| `Rscript main.R modelos_limpios` | [08_models_clean_data.R](02_code/08_models_clean_data.R) | `03_outputs/03_models_clean_data/resumen.xlsx` (incluye folds y predictores) y un CSV por familia |
| `Rscript main.R regresion_basica` | [10_models_regresion_basica.R](02_code/10_models_regresion_basica.R) | `03_outputs/05_models_regresion_basica/resumen.xlsx` y un CSV por modelo |
| `Rscript main.R regresion_log` | [11_models_regresion_log.R](02_code/11_models_regresion_log.R) | `03_outputs/06_models_regresion_log/resumen.xlsx` y un CSV por modelo |
| `Rscript main.R xgboost_text_svd` | [09_models_text_svd.R](02_code/09_models_text_svd.R) | `03_outputs/04_models_text_svd/` con el CSV elegido y `resumen.txt` |

`Rscript main.R` equivale a `Rscript main.R xgboost_text_svd`. Para replicar a un directorio de trabajo separado, se pueden indicar las carpetas de bases y salidas como segundo y tercer argumentos, por ejemplo `Rscript main.R todo 01_data/02_clean replicas`. Las tablas de resultados de la entrega se pueden localizar por bloque usando la columna de salida. El resumen identifica los hiperparámetros y las métricas exactas de su CSV.

## Validación y grillas

- El bloque crudo y la red usan cinco folds convencionales con semilla 2026. La red usa títulos y descripciones codificados con E5 más diez variables tabulares. Se ajustan tres arquitecturas, con salida directa y capas ocultas de 32 o 64 unidades respectivamente.
- Los árboles con datos enriquecidos usan cinco folds de bloques de 400 m. Cada familia de árboles, tanto cruda como enriquecida, tiene cuatro candidatos y cinco folds. Las grillas son una revisión de las corridas anteriores. Se fijan los parámetros secundarios y se varían dos parámetros principales por familia.
- La regresión lineal, Elastic Net y CART usan los mismos predictores y folds espaciales que los árboles enriquecidos. Elastic Net compara tres alphas (0, 0,5 y 1) y cuatro intensidades relativas de penalización (0,30, 0,10, 0,03 y 0,01). El CART usa una configuración fija.
- `regresion_log` repite los tres ajustes cuadráticos con el logaritmo del precio y los mismos folds espaciales. Los pronósticos se devuelven a COP con la exponencial y Elastic Net se selecciona por MAE en COP. Este bloque se ejecuta por separado de `todo`.
- Para el XGBoost enriquecido se prueban 200 o 400 rondas y peso mínimo del nodo de 20 o 30, con profundidad 6 y tasa 0,05. Esta selección se hace con los predictores enriquecidos que incluyen E5. El ganador se pasa al XGBoost con TF-IDF/SVD, que valida esa configuración con el vocabulario y la proyección ajustados dentro de cada fold. Si se ejecuta solo el último bloque, el principal vuelve a calcular la selección de cuatro candidatos. En `todo` reutiliza la que acaba de calcular en el bloque de datos limpios.
- Los resúmenes reportan pérdida absoluta asimétrica con peso 2/3 al sobreestimar y 1/3 al subestimar, MAE general y MAE por lado. La pérdida y los MAE están en COP. Para Kaggle, use el CSV de predicción identificado por el modelo y sus hiperparámetros. La etapa de estimación escribe exclusivamente las columnas `property_id` y `price` en el orden de la plantilla. Cada precio es numérico y cada ID aparece una sola vez.

La generación inicial de E5 y la validación de las tres redes son las etapas de mayor costo. En la corrida local de referencia, generar E5 tomó cerca de 97 minutos y ajustar y validar las redes tomó aproximadamente una hora. Los archivos derivados y las cachés se ignoran en nuevas incorporaciones a Git.


## Comprobación de referencia

Con las bases de la entrega, la grilla de cuatro candidatos con E5 eligió 400 rondas, tasa 0,05, profundidad 6 y peso mínimo del nodo 30. El XGBoost con TF-IDF/SVD obtuvo una pérdida asimétrica de 52,11 millones de COP y un MAE de 131,14 millones de COP en validación. La salida final contiene 10.286 predicciones y dos columnas. En la reproducción local, la diferencia máxima frente al CSV de entrega fue de 0,50 COP.
