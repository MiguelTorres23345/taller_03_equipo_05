# ------------------------------------------------------------------------------
# Archivo: 00_setup.R
# Proyecto: Taller 3 - Big Data y Machine Learning
#
# Descripcion:
#   Cargar pacman, instalar/cargar los demas paquetes y resolver conflictos
#   entre funciones con el mismo nombre.
# Input:
#   Bloque activo y ruta de dependencias de main.R. Internet si faltan paquetes.
#   ZIP opcionales de LibTorch y LibLantern en .deps_nn/descargas_torch/.
# Output:
#   Paquetes cargados y binarios de Torch y ONNX en .deps_nn/,
#   y preferencias de funciones definidas en la sesion. No importa bases.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Cargar pacman
# ------------------------------------------------------------------------------

modo_taller <- getOption("taller3.modo_nn", "arboles")
usar_nn <- modo_taller %in% c("preparar", "nn", "espacial")
if (usar_nn) {
  # Los paquetes nuevos del bloque de redes se instalan dentro del respaldo.
  libreria_nn <- file.path(dir_dependencias_nn, "R")
  dir.create(libreria_nn, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(libreria_nn, .libPaths()))
  Sys.setenv(TORCH_INSTALL = "0", CUDA = "cpu")
  repos <- getOption("repos")
  if (any(repos == "@CRAN@")) {
    repos[repos == "@CRAN@"] <- "https://cloud.r-project.org"
    options(repos = repos)
  }
}

if (!require("pacman", quietly = TRUE)) {
  if (usar_nn) {
    install.packages("pacman", type = .Platform$pkgType)
  } else {
    install.packages("pacman")
  }
  if (!require("pacman", quietly = TRUE)) {
    stop("00_setup.R no pudo instalar pacman. Revise el acceso a CRAN.",
         call. = FALSE)
  }
}

# ------------------------------------------------------------------------------
# (2) Cargar los demas paquetes
# ------------------------------------------------------------------------------

# p_load() instala los paquetes faltantes y carga los instalados.
if (usar_nn) {
  paquetes_nn <- c(
    "readr",       # Lectura y exportacion de CSV.
    "dplyr",       # Combinacion de textos y tablas.
    "openxlsx",    # Resumen de validacion en Excel.
    "callr"       # Instalacion y comprobacion de Torch en otra sesion de R.
  )
  if (identical(modo_taller, "espacial")) paquetes_nn <- c(paquetes_nn, "sf")
  # En Windows y macOS usar binarios para evitar compiladores e instaladores.
  por_instalar <- setdiff(paquetes_nn, basename(find.package(paquetes_nn, quiet = TRUE)))
  if (length(por_instalar)) install.packages(por_instalar, type = .Platform$pkgType)
  p_load(char = paquetes_nn, install = FALSE, update = FALSE)
  faltantes_nn <- setdiff(paquetes_nn, .packages())
  if (length(faltantes_nn)) {
    stop("No se pudieron cargar los paquetes de redes: ",
         paste(faltantes_nn, collapse = ", "), call. = FALSE)
  }
} else p_load(
  # (2.1) Datos y presentacion
  tidyverse,     # Lectura de CSV, limpieza, agregacion, uniones y graficos.
  readxl,        # Lectura de los archivos Excel de documentacion.
  writexl,       # Exportacion de tablas descriptivas a Excel.
  openxlsx,      # Exportacion de rankings con filtros y formatos de Excel.
  Matrix,        # Matrices dispersas para la ecuacion de ingreso.
  tidytext,      # Tokenizacion y TF-IDF de titulos y descripciones.
  
  # (2.2) Entrenamiento y modelos
  stats,         # lm() y glm(): modelo lineal, logit y probit (incluido en R).
  lava,          # Dependencia indirecta de caret que falta en el entorno actual.
  caret,         # Validacion cruzada, tuning y rebalanceo dentro de los folds.
  glmnet,        # Ridge, lasso y elastic net para regresion y clasificacion.
  MASS,          # Analisis discriminante lineal (LDA) y cuadratico (QDA).
  naivebayes,    # Naive Bayes; motor naive_bayes de caret.
  rpart,         # Arboles de clasificacion y regresion (CART).
  rpart.plot,    # Visualizacion de los arboles ajustados con rpart.
  ranger,        # Bagging y random forest con implementacion eficiente.
  randomForest,  # Implementacion alternativa de random forest.
  adabag,        # AdaBoost.M1 y bagging de arboles.
  gbm,           # Gradient boosting de arboles.
  xgboost,       # Extreme Gradient Boosting (XGBoost).
  lightgbm,      # Light Gradient Boosting Machine (LightGBM).
  nnet,          # Redes neuronales de una capa oculta para regresion.
  rsample,       # Remuestreo y folds; base de spatialsample.
  spatialsample, # Validacion cruzada espacial (bloques y localidades).
  
  # (2.3) Evaluacion
  pROC,          # Curvas ROC, AUC y evaluacion de umbrales.
  MLmetrics,     # Metricas de clasificacion; dependencia de prSummary de caret.
  Metrics,       # Metricas de regresion y clasificacion.
  
  # (2.4) Organizacion y replicabilidad
  rstudioapi,    # Acceso al proyecto y al editor cuando se usa RStudio.
  conflicted,    # Resuelve funciones homonimas entre paquetes.
  docstring,     # Consulta de documentacion escrita dentro de funciones.
  renv,          # Permite fijar/restaurar versiones cuando se valide el entorno.
  
  # (2.5) Recoleccion web y datos espaciales
  httr2,         # Descargas y consultas HTTP a APIs de datos abiertos.
  rvest,         # Extraccion de tablas y campos de paginas HTML.
  jsonlite,      # Lectura y conversion de respuestas y archivos JSON.
  sf,            # Capas de OSM y del Distrito, cruces y distancias.
  osmextract,    # Descarga y lectura del extracto de OpenStreetMap de Bogota.
  terra,         # Procesamiento de imagenes satelitales y otros raster.
  reticulate
)

# (2.6) Preparar las dependencias locales de las redes

preparar_torch_nn <- function(cache) {
  ruta_torch <- find.package("torch", quiet = TRUE)
  if (!length(ruta_torch)) {
    install.packages("torch", type = .Platform$pkgType)
    ruta_torch <- find.package("torch")
  }

  version <- read.dcf(file.path(ruta_torch, "DESCRIPTION"), fields = "Version")[1, 1]
  destino <- file.path(cache, paste0("torch-", version))
  descargas <- file.path(cache, "descargas_torch")
  dir.create(destino, recursive = TRUE, showWarnings = FALSE)
  dir.create(descargas, showWarnings = FALSE)
  origenes <- Sys.getenv(c("TORCH_URL", "LANTERN_URL"))
  callr::r(function(destino, descargas, origenes) {
    Sys.setenv(TORCH_HOME = destino,
               R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep))
    options(download.file.method = "libcurl", timeout = max(900, getOption("timeout")))
    if (!torch::torch_is_installed()) {
      # Usar los ZIP locales compatibles cuando esten disponibles.
      Sys.unsetenv(names(origenes))
      urls <- torch::get_install_libs_url()
      nombres_zip <- basename(unlist(urls, use.names = FALSE))
      for (i in seq_along(origenes)) {
        if (!nzchar(origenes[i])) {
          candidatos <- file.path(descargas, unique(c(URLdecode(nombres_zip[i]), nombres_zip[i])))
          origenes[i] <- c(candidatos[file.exists(candidatos)], "")[1]
        }
        if (nzchar(origenes[i])) {
          if (!grepl("^https?://", origenes[i])) {
            if (!file.exists(origenes[i])) {
              stop("No se encontro el ZIP indicado en ", names(origenes)[i],
                   ": ", origenes[i], call. = FALSE)
            }
            origenes[i] <- normalizePath(origenes[i], winslash = "/", mustWork = TRUE)
            message("Usando el ZIP local ", basename(origenes[i]), ".")
          }
        }
      }
      if (any(nzchar(origenes))) {
        do.call(Sys.setenv, as.list(origenes[nzchar(origenes)]))
      }

      # La instalacion temporal evita falsos permisos de escritura en OneDrive.
      instalacion <- tempfile("libtorch-")
      dir.create(instalacion)
      Sys.setenv(TORCH_HOME = instalacion)
      message("Instalando LibTorch y LibLantern para CPU.")
      tryCatch(
        torch::install_torch(.inform_restart = FALSE),
        error = function(e) {
          stop(conditionMessage(e), "\nZIP compatibles para instalar desde archivos en ",
               descargas, ":\n", paste(URLdecode(nombres_zip), collapse = "\n"),
               call. = FALSE)
        }
      )
      archivos <- list.files(instalacion, all.files = TRUE, full.names = TRUE,
                             no.. = TRUE)
      if (!all(file.copy(archivos, destino, recursive = TRUE, overwrite = TRUE))) {
        stop("R no pudo copiar los binarios de Torch a ", destino, call. = FALSE)
      }
      Sys.setenv(TORCH_HOME = destino)
      if (!torch::torch_is_installed(recheck = TRUE)) {
        stop("No se pudieron cargar LibTorch y LibLantern.")
      }
    }
  }, args = list(destino = destino, descargas = descargas, origenes = origenes),
  libpath = .libPaths(), show = TRUE)
  Sys.setenv(TORCH_HOME = destino)
  library(torch)
}

if (usar_nn) {
  cache_nn <- normalizePath(dir_dependencias_nn, winslash = "/", mustWork = TRUE)
  preparar_torch_nn(cache_nn)

  # sbert genera los embeddings de E5 con ONNX Runtime.
  if (!length(find.package("sbert", quiet = TRUE))) {
    install.packages("sbert", type = .Platform$pkgType)
  }
  if (!requireNamespace("sbert", quietly = TRUE)) {
    stop("00_setup.R no pudo cargar sbert. Revise la instalacion desde CRAN.",
         call. = FALSE)
  }
  # onnxr usa R_user_dir("onnxr", "data") para instalar y encontrar el runtime.
  Sys.setenv(R_USER_DATA_DIR = file.path(cache_nn, "runtime"))
  if (!onnxr::onnx_is_installed()) sbert::install_runtime(cuda = FALSE)
}

# ------------------------------------------------------------------------------
# (3) Resolver conflictos entre funciones
# ------------------------------------------------------------------------------

# Elegir dplyr cuando otros paquetes tengan funciones con el mismo nombre.
if (!usar_nn) {
  conflict_prefer("select", "dplyr")
  conflict_prefer("filter", "dplyr")
  conflict_prefer("lag", "dplyr")
  conflict_prefer("slice", "dplyr")  # xgboost y lightgbm tambien tienen slice()
}
