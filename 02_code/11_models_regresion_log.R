# ------------------------------------------------------------------------------
# Archivo: 11_models_regresion_log.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Repetir regresion lineal, Elastic Net y CART con log del precio.
# Input: RDS, plantilla y rutas de main.R, y ajustes de 10_models_regresion_basica.R.
# Output: Tres CSV en COP, modelos y resumen.xlsx en 06_models_regresion_log.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Activar el ajuste en logaritmos y la validacion por MAE en COP
# ------------------------------------------------------------------------------

usar_log <- TRUE
dir_modelos_regresion <- dir_modelos_regresion_log
source(file.path(dir_codigo, "10_models_regresion_basica.R"), local = TRUE,
       encoding = "UTF-8")
