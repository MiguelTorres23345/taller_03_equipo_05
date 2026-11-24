# ------------------------------------------------------------------------------
# Archivo: main_modelo_final.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Ejecutar solo el modelo final o reproducir su validacion.
# Input: Bases RDS de 01_data/02_clean o del respaldo, y plantilla original.
# Output: CSV de entrega y resumen.txt; la seleccion historica sale por consola.
# Uso: Rscript main_modelo_final.R [modelo_final|cv_final|seleccion_final]
#      Puede indicar la carpeta de bases limpias y la de salida como argumentos
#      segundo y tercero. Ejecutar desde la raiz del taller.
# Dependencias: R con readr, xgboost y reticulate. El modelo final usa Python
#               con numpy, scikit-learn y threadpoolctl.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir las rutas del modelo
# ------------------------------------------------------------------------------

argumentos <- commandArgs(trailingOnly = TRUE)
modo <- if (length(argumentos)) {
  match.arg(argumentos[1L], c("modelo_final", "cv_final", "seleccion_final"))
} else {
  "modelo_final"
}

dir_codigo <- "02_code"
dir_limpio <- file.path("01_data", "02_clean")
dir_salida <- file.path("03_outputs", "07_models_text_svd")
bases <- if (modo == "seleccion_final") "train_clean.rds" else c("train_clean.rds", "test_clean.rds")
if (length(argumentos) >= 2L) {
  dir_limpio <- argumentos[2L]
} else if (!all(file.exists(file.path(dir_limpio, bases)))) {
  respaldo <- file.path("..", "00_back_up", "03_taller_3", "01_data", "02_clean")
  if (all(file.exists(file.path(respaldo, bases)))) dir_limpio <- respaldo
}
if (length(argumentos) >= 3L) dir_salida <- argumentos[3L]
archivo_train <- file.path(dir_limpio, "train_clean.rds")
archivo_test <- file.path(dir_limpio, "test_clean.rds")
archivo_plantilla <- file.path("01_data", "01_raw", "submission_template.csv")

# ------------------------------------------------------------------------------
# (2) Ejecutar el bloque solicitado
# ------------------------------------------------------------------------------

if (modo == "seleccion_final") {
  source(file.path(dir_codigo, "10_seleccion_xgboost_final.R"), encoding = "UTF-8")
  seleccionar_xgboost_historico(archivo_train)
} else {
  source(file.path(dir_codigo, "09_models_text_svd.R"), encoding = "UTF-8")
  ejecutar_xgboost_final(archivo_train, archivo_test, archivo_plantilla, dir_salida, modo)
}
