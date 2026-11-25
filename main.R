# ------------------------------------------------------------------------------
# Archivo: main.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Ejecutar un bloque o reproducir todos los bloques en orden.
# Input: CSV originales, capas externas y predictores de texto del equipo.
# Output: Bases en 01_data/02_clean/ y resultados de cada bloque en 03_outputs/.
# Uso: Rscript main.R [todo|arboles|nn|nn_embeddings|cleaning|modelos_limpios|regresion_basica|regresion_log|xgboost_text_svd]
#      El segundo argumento cambia el directorio de bases y el tercero el de salidas.
#      Sin argumentos, ejecutar solo el flujo del XGBoost con TF-IDF/SVD.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir rutas y parametros comunes
# ------------------------------------------------------------------------------

argumentos <- commandArgs(trailingOnly = TRUE)
tarea <- if (length(argumentos)) argumentos[1L] else "xgboost_text_svd"
dir_codigo <- "02_code"
dir_datos_raw <- file.path("01_data", "01_raw")
dir_datos_clean <- if (length(argumentos) >= 2L) argumentos[2L] else file.path("01_data", "02_clean")
dir_datos_texto <- file.path("01_data", "02_text")
dir_fuentes <- file.path(dir_datos_raw, "fuentes_externas")
dir_osm <- file.path(dir_fuentes, "osm")
dir_resultados <- if (length(argumentos) >= 3L) argumentos[3L] else "03_outputs"
dir_dependencias <- ".deps_nn"
dir_modelos_raw <- file.path(dir_resultados, "01_models_raw_data")
dir_modelos_nn <- file.path(dir_resultados, "02_models_nn_raw_data_base_r")
dir_modelos_limpios <- file.path(dir_resultados, "03_models_clean_data")
dir_modelos_texto_svd <- file.path(dir_resultados, "04_models_text_svd")
dir_modelos_regresion <- file.path(dir_resultados, "05_models_regresion_basica")
dir_modelos_regresion_log <- file.path(dir_resultados, "06_models_regresion_log")

archivo_train <- file.path(dir_datos_raw, "train.csv")
archivo_test <- file.path(dir_datos_raw, "test.csv")
archivo_plantilla <- file.path(dir_datos_raw, "submission_template.csv")
archivo_train_clean <- file.path(dir_datos_clean, "train_clean.rds")
archivo_test_clean <- file.path(dir_datos_clean, "test_clean.rds")
crs_espacial_nn <- 3116L

# ------------------------------------------------------------------------------
# (2) Ejecutar preparacion y modelos
# ------------------------------------------------------------------------------

ejecutar <- function(bloque, seleccion = NULL) {
  script <- switch(bloque,
    arboles = "03_models_raw_data.R",
    nn = "08_models_nn_raw_data_base_r.R",
    nn_embeddings = "08_models_nn_raw_data_base_r.R",
    cleaning = "01_cleaning.R",
    modelos_limpios = "08_models_clean_data.R",
    regresion_basica = "10_models_regresion_basica.R",
    regresion_log = "11_models_regresion_log.R",
    seleccion_xgboost = "08_models_clean_data.R",
    xgboost_text_svd = "09_models_text_svd.R"
  )
  if (is.null(script)) stop("Bloque desconocido: ", bloque, call. = FALSE)

  if (bloque == "cleaning") {
    archivos_texto <- file.path(dir_datos_texto, c(
      "text_features_train.csv", "text_features_test.csv",
      "text_finetune_minilm_train.csv", "text_finetune_minilm_test.csv"
    ))
    faltantes <- archivos_texto[!file.exists(archivos_texto)]
    if (length(faltantes)) {
      stop("Faltan insumos de texto. Consulte el README: ",
           paste(faltantes, collapse = ", "), call. = FALSE)
    }
    archivo_embeddings_clean <- ejecutar("nn_embeddings")
  } else if (bloque %in% c("modelos_limpios", "regresion_basica", "regresion_log",
                             "seleccion_xgboost", "xgboost_text_svd") &&
             !all(file.exists(c(archivo_train_clean, archivo_test_clean)))) {
    ejecutar("cleaning")
  }
  if (bloque == "xgboost_text_svd" && is.null(seleccion)) {
    seleccion <- ejecutar("seleccion_xgboost")
  }

  message("Bloque: ", bloque)
  source(file.path(dir_codigo, "00_setup.R"), local = TRUE, encoding = "UTF-8")
  resultado <- source(file.path(dir_codigo, script), local = TRUE, encoding = "UTF-8")
  if (script == "08_models_clean_data.R") return(invisible(resultado$value))
  if (bloque == "nn_embeddings") return(generar_embeddings_nn()$archivo)
  if (bloque == "nn") ejecutar_nn(generar_embeddings_nn())
  if (bloque == "xgboost_text_svd") {
    ejecutar_xgboost_text_svd(archivo_train_clean, archivo_test_clean, archivo_plantilla,
                             dir_modelos_texto_svd, seleccion)
  }
  invisible(resultado$value)
}

if (tarea == "todo") {
  ejecutar("arboles")
  ejecutar("nn")
  seleccion <- ejecutar("modelos_limpios")
  ejecutar("regresion_basica")
  ejecutar("xgboost_text_svd", seleccion)
} else {
  ejecutar(tarea)
}
