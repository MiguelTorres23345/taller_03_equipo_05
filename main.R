# ------------------------------------------------------------------------------
# Archivo: main.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion:
#   Definir las rutas y el orden de ejecucion del taller.
# Input:
#   Ejecutar desde la raiz del taller. CSV originales en 01_data/01_raw/
#   y script 02_code/00_setup.R.
# Output:
#   Rutas del proyecto, semilla y paquetes cargados en la sesion.
#   No exporta archivos.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir la raiz del proyecto
# ------------------------------------------------------------------------------

root <- "."

# ------------------------------------------------------------------------------
# (2) Definir rutas y semilla
# ------------------------------------------------------------------------------

dir_codigo <- file.path(root, "02_code")
dir_datos_raw <- file.path(root, "01_data", "01_raw")
dir_resultados <- file.path(root, "03_outputs")
dir_documentacion <- file.path(root, "04_documentation")

archivo_train <- file.path(dir_datos_raw, "train.csv")
archivo_test <- file.path(dir_datos_raw, "test.csv")
archivo_plantilla <- file.path(dir_datos_raw, "submission_template.csv")
set.seed(2026)

# ------------------------------------------------------------------------------
# (3) Ejecutar los scripts en orden
# ------------------------------------------------------------------------------

scripts <- c("00_setup.R")
for (script in scripts) source(file.path(dir_codigo, script), encoding = "UTF-8")
