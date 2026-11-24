# ------------------------------------------------------------------------------
# Archivo: main.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion:
#   Definir las rutas y el orden de ejecucion del taller.
# Input:
#   Ejecutar desde la raiz del taller. CSV originales en 01_data/01_raw/, capas
#   externas en las subcarpetas de 01_data/, predicciones de texto en
#   01_data/02_text/, rasgos de imagen en 01_data/imagenes/ y scripts en 02_code/.
# Output:
#   Rutas del proyecto, semilla y paquetes cargados en la sesion. Cada script
#   exporta sus propios archivos (bases en 01_data/clean/, predicciones y
#   tablas en 03_outputs/).
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir la raiz del proyecto
# ------------------------------------------------------------------------------

# Si la sesion de R quedo en 02_code (por ejemplo al abrir un script suelto), la
# raiz es la carpeta de arriba
root <- if (basename(normalizePath(getwd())) == "02_code") ".." else "."

# ------------------------------------------------------------------------------
# (2) Definir rutas y semilla
# ------------------------------------------------------------------------------

dir_codigo <- file.path(root, "02_code")
dir_datos_raw <- file.path(root, "01_data", "01_raw")
dir_texto <- file.path(root, "01_data", "02_text")
dir_externos <- file.path(root, "01_data")
dir_clean <- file.path(root, "01_data", "clean")
dir_imagenes <- file.path(root, "01_data", "imagenes")
dir_resultados <- file.path(root, "03_outputs")
dir_submit <- file.path(dir_resultados, "submits")
dir_mapas <- file.path(dir_resultados, "mapas")
dir_documentacion <- file.path(root, "04_documentation")

archivo_train <- file.path(dir_datos_raw, "train.csv")
archivo_test <- file.path(dir_datos_raw, "test.csv")
archivo_plantilla <- file.path(dir_datos_raw, "submission_template.csv")
set.seed(2026)

# Verificacion: la raiz esta bien definida
stopifnot("No se encuentra train.csv, abrir el proyecto taller_03_equipo_05.Rproj o fijar el directorio en la raiz del taller" =
            file.exists(archivo_train))

# ------------------------------------------------------------------------------
# (3) Ejecutar los scripts en orden
# ------------------------------------------------------------------------------

# 04_modelos_imagen.R usa los rasgos de la red convolucional (01_data/imagenes).
# Se quita de la lista si no se van a usar.
scripts <- c("00_setup.R", "01_cleaning.R", "02_variables_planas.R",
             "03_modelos.R", "04_modelos_imagen.R")
for (script in scripts) source(file.path(dir_codigo, script), encoding = "UTF-8")