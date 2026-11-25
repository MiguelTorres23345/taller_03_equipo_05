# ------------------------------------------------------------------------------
# Archivo: main.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Ejecutar un bloque o reproducir todo el flujo en orden.
# Input: CSV originales, capas externas y predictores de texto del equipo.
# Output: Bases en 01_data/02_clean/ y resultados de cada bloque en 03_outputs/.
# Uso: Rscript main.R [final|todo|arboles|arboles_corregido|nn|nn_embeddings|cleaning|modelos_limpios|
#                     regresion_basica|regresion_log|xgboost_text_svd|
#                     folds|texto_python|modelos_texto|xgboost|random_forest|
#                     modelos_python|ablacion|figuras|prueba]
#      Sin argumentos se ejecuta "final": el flujo que produce el modelo reportado
#      (XGBoost y random forest con validacion espacial por bloques de 400 m y
#      variables de texto), sus tablas y sus figuras. Orden de "final":
#        1. cleaning        01_cleaning.R y 02_variables_planas.R (solo si falta
#                           01_data/02_clean/train_planas.csv)
#        2. folds           13_folds.R
#        3. texto_python    06_text_features.py (solo si correr_texto_python)
#        4. modelos_texto   14_models_text.R
#        5. modelos_python  12_models_python_rf_xgb.py
#        6. ablacion        15_ablacion_fuentes.R
#        7. figuras         16, 17 y 18 (figuras de las slides)
#      "todo" conserva el flujo anterior (arboles, red neuronal, modelos limpios,
#      regresiones y XGBoost con TF-IDF/SVD). El segundo argumento cambia el
#      directorio de bases y el tercero el de salidas de esos bloques anteriores.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir rutas y parametros comunes
# ------------------------------------------------------------------------------

argumentos <- commandArgs(trailingOnly = TRUE)
tarea <- if (length(argumentos)) argumentos[1L] else "final"

# Raiz del proyecto. Los scripts 13 a 18 la leen con if (!exists("root")) root <- "."
root <- "."

# Interruptores del flujo "final"
# FALSE: usar los CSV de texto ya guardados en 01_data/02_text/ (valor por defecto).
# TRUE: regenerar text_features_*.csv con 06_text_features.py (corre en CPU; ver README).
# El fine-tuning (09_text_finetune.py) NUNCA se corre aqui: necesita GPU y se corrio en
# Google Colab (ver README y 02_code/colab_texto.ipynb).
correr_texto_python <- FALSE
# FALSE: omitir 12_models_python_rf_xgb.py (RF y XGBoost en Python, tarda varios minutos)
correr_modelos_python <- TRUE
# Interprete de Python para los bloques en Python. Orden de busqueda: variable de entorno
# PYTHON_BIN, el entorno .venv de la raiz del repositorio (ver README) y el PATH.
candidatos_python <- c(Sys.getenv("PYTHON_BIN"),
                       file.path(".venv", "Scripts", "python.exe"),
                       file.path(".venv", "bin", "python"),
                       unname(Sys.which(c("python", "python3"))))
candidatos_python <- candidatos_python[nzchar(candidatos_python) &
                                         (file.exists(candidatos_python) | candidatos_python == Sys.getenv("PYTHON_BIN"))]
python_bin <- if (length(candidatos_python)) candidatos_python[1L] else NA_character_

# Rutas de los bloques anteriores y del flujo final
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

# Verificacion: la raiz esta bien definida
stopifnot("No se encuentra train.csv, abrir el proyecto taller_03_equipo_05.Rproj o fijar el directorio en la raiz del taller" =
            file.exists(archivo_train))

# ------------------------------------------------------------------------------
# (2) Ejecutar preparacion y modelos
# ------------------------------------------------------------------------------

ejecutar <- function(bloque, seleccion = NULL) {
  script <- switch(bloque,
    arboles = "03_models_raw_data.R",
    arboles_corregido = "03_models_raw_data.R",
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

# ------------------------------------------------------------------------------
# (3) Flujo final: folds, texto, modelos con texto, ablacion y figuras
# ------------------------------------------------------------------------------

archivo_train_planas <- file.path("01_data", "02_clean", "train_planas.csv")

correr_python <- function(script, argumentos_py = character()) {
  if (is.na(python_bin)) stop("No se encontro Python en el PATH. Defina PYTHON_BIN.", call. = FALSE)
  # Si el Python del PATH no tiene los paquetes, avisar antes de fallar a mitad del script
  prueba <- system2(python_bin, c("-c", shQuote("import pandas, sklearn, xgboost, geopandas")),
                    stdout = FALSE, stderr = FALSE)
  if (prueba != 0L) {
    stop("El Python '", python_bin, "' no tiene los paquetes de requirements.txt. ",
         "Instale el entorno (ver README) y use PYTHON_BIN=.venv/Scripts/python.exe",
         call. = FALSE)
  }
  message("Python: ", script)
  estado <- system2(python_bin, c(shQuote(file.path(dir_codigo, script)), argumentos_py))
  if (estado != 0L) stop("Fallo ", script, " (codigo ", estado, ")", call. = FALSE)
  invisible(estado)
}

ejecutar_final <- function(bloque) {
  inicio_bloque <- Sys.time()
  on.exit(message(sprintf("TIEMPO bloque %s: %.1f min", bloque,
                          as.numeric(difftime(Sys.time(), inicio_bloque, units = "mins")))),
          add = TRUE)
  scripts_r <- switch(bloque,
    folds = "13_folds.R",
    modelos_texto = "14_models_text.R",
    xgboost = "14_models_text.R",
    random_forest = "14_models_text.R",
    ablacion = "15_ablacion_fuentes.R",
    figuras = c("16_figuras_slides.R", "17_figura_validacion.R", "18_figura_bloques.R")
  )
  # Los bloques en R necesitan las bases planas de 02_variables_planas.R
  if (!is.null(scripts_r) && !file.exists(archivo_train_planas)) ejecutar("cleaning")
  message("Bloque: ", bloque)
  source_paquetes <- if (bloque %in% c("xgboost", "random_forest")) "modelos_texto" else bloque
  bloque_setup <- bloque
  bloque <- source_paquetes
  source(file.path(dir_codigo, "00_setup.R"), local = TRUE, encoding = "UTF-8")
  bloque <- bloque_setup
  if (!is.null(scripts_r)) {
    for (script in scripts_r) {
      message("Script: ", script)
      entorno <- new.env(parent = environment())
      # Rscript main.R xgboost | random_forest ajusta solo ese modelo de 14_models_text.R
      if (bloque == "xgboost") entorno$modelos_a_correr <- "xgb"
      if (bloque == "random_forest") entorno$modelos_a_correr <- "rf"
      source(file.path(dir_codigo, script), local = entorno, encoding = "UTF-8")
    }
  } else if (bloque == "texto_python") {
    correr_python("06_text_features.py")
  } else if (bloque == "modelos_python") {
    correr_python("12_models_python_rf_xgb.py")
  } else {
    stop("Bloque desconocido: ", bloque, call. = FALSE)
  }
  invisible(NULL)
}

guardar_sesion <- function() {
  dir.create(file.path("04_documentation"), showWarnings = FALSE)
  writeLines(capture.output(sessionInfo()), file.path("04_documentation", "sessionInfo_R.txt"))
}

# ------------------------------------------------------------------------------
# (4) Prueba rapida: el mismo flujo con una muestra pequena de los datos
# ------------------------------------------------------------------------------
# Rscript main.R prueba [n]   (n anuncios de train, 2000 por defecto)
# Copia una muestra de las bases a 03_outputs/prueba/ y corre alli folds, modelos,
# ablacion y figuras, asi que no toca 01_data/ ni los resultados reales. La limpieza
# (cleaning) no se prueba aqui porque no tiene modo muestra.

preparar_prueba <- function(n) {
  raiz <- file.path("03_outputs", "prueba")
  unlink(raiz, recursive = TRUE)
  dir.create(file.path(raiz, "01_data", "02_clean"), recursive = TRUE)
  dir.create(file.path(raiz, "01_data", "02_text"), recursive = TRUE)
  dir.create(file.path(raiz, "01_data", "01_raw", "fuentes_externas", "localidades"), recursive = TRUE)
  set.seed(2026)
  ids <- sample(read.csv(archivo_train_planas, colClasses = c(property_id = "character"))$property_id, n)
  for (nombre in c("train_planas", "train_clean")) {
    base <- read.csv(file.path("01_data", "02_clean", paste0(nombre, ".csv")),
                     colClasses = c(property_id = "character"), check.names = FALSE)
    write.csv(base[base$property_id %in% ids, ],
              file.path(raiz, "01_data", "02_clean", paste0(nombre, ".csv")), row.names = FALSE)
  }
  file.copy(file.path("01_data", "02_clean", c("test_planas.csv", "test_clean.csv")),
            file.path(raiz, "01_data", "02_clean"))
  file.copy(list.files(file.path("01_data", "02_text"), pattern = "^text_.*[.]csv$", full.names = TRUE),
            file.path(raiz, "01_data", "02_text"))
  carpeta_loc <- file.path("01_data", "01_raw", "fuentes_externas", "localidades")
  file.copy(list.files(carpeta_loc, full.names = TRUE),
            file.path(raiz, "01_data", "01_raw", "fuentes_externas", "localidades"))
  raiz
}

if (tarea == "prueba") {
  n_prueba <- if (length(argumentos) >= 2L) as.integer(argumentos[2L]) else 2000L
  root <- preparar_prueba(n_prueba)
  message("PRUEBA con ", n_prueba, " anuncios de train; salidas en ", root)
  ejecutar_final("folds")
  ejecutar_final("modelos_texto")
  if (correr_modelos_python) {
    correr_python("12_models_python_rf_xgb.py", c("--sample", as.character(min(n_prueba, 600L))))
  }
  ejecutar_final("ablacion")
  ejecutar_final("figuras")
  message("PRUEBA terminada sin errores")
} else if (tarea == "final") {
  if (!file.exists(archivo_train_planas)) ejecutar("cleaning")
  ejecutar_final("folds")
  if (correr_texto_python) ejecutar_final("texto_python")
  ejecutar_final("modelos_texto")
  if (correr_modelos_python) ejecutar_final("modelos_python")
  ejecutar_final("ablacion")
  ejecutar_final("figuras")
  guardar_sesion()
} else if (tarea %in% c("folds", "texto_python", "modelos_texto", "xgboost",
                          "random_forest", "modelos_python", "ablacion", "figuras")) {
  ejecutar_final(tarea)
} else if (tarea == "todo") {
  ejecutar("arboles")
  ejecutar("nn")
  seleccion <- ejecutar("modelos_limpios")
  ejecutar("arboles_corregido")
  ejecutar("regresion_basica")
  ejecutar("xgboost_text_svd", seleccion)
} else {
  ejecutar(tarea)
}
