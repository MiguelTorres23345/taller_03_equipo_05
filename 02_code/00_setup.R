# ------------------------------------------------------------------------------
# Archivo: 00_setup.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Instalar y cargar las dependencias necesarias para el bloque.
# Input: Bloque y directorio de dependencias definidos en main.R.
# Output: Paquetes cargados y funciones para preparar ONNX y Python.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Cargar pacman
# ------------------------------------------------------------------------------

libreria <- file.path(dir_dependencias, "R")
dir.create(libreria, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(libreria, .libPaths()))
repos <- getOption("repos")
repos[repos == "@CRAN@"] <- "https://cran.rstudio.com"
options(repos = repos)
if (!require("pacman", quietly = TRUE)) {
  install.packages("pacman")
  library(pacman)
}

# ------------------------------------------------------------------------------
# (2) Cargar los demas paquetes
# ------------------------------------------------------------------------------

# readr y dplyr leen y preparan las tablas. openxlsx exporta los resumenes.
# ranger, xgboost, lightgbm y rpart ajustan arboles. glmnet ajusta Elastic Net.
# sf y osmextract leen las capas.
# sbert y digest identifican y generan los embeddings. reticulate ejecuta TF-IDF.
paquetes <- switch(if (bloque == "regresion_log") "regresion_basica" else bloque,
  arboles = c("readr", "dplyr", "openxlsx", "ranger", "xgboost", "lightgbm"),
  nn = c("readr", "dplyr", "openxlsx", "sbert", "digest"),
  nn_embeddings = c("readr", "dplyr", "sbert", "digest"),
  cleaning = c("readr", "dplyr", "sf", "osmextract"),
  modelos_limpios = c("readr", "dplyr", "openxlsx", "ranger", "xgboost", "lightgbm"),
  regresion_basica = c("readr", "dplyr", "openxlsx", "glmnet", "rpart"),
  seleccion_xgboost = c("dplyr", "xgboost"),
  xgboost_text_svd = c("readr", "xgboost", "reticulate")
)
p_load(char = paquetes, update = FALSE)
faltantes <- setdiff(paquetes, .packages())
if (length(faltantes)) stop("Faltan paquetes de R: ", paste(faltantes, collapse = ", "), call. = FALSE)

instalar_onnx_nn <- function() {
  tryCatch(sbert::install_runtime(cuda = FALSE), error = function(error) {
    if (.Platform$OS.type != "windows" || R.version$arch != "x86_64") stop(error)

    # Microsoft tambien distribuye la DLL x64 en su paquete de Maven Central.
    url <- paste0("https://repo.maven.apache.org/maven2/com/microsoft/",
                  "onnxruntime/onnxruntime/1.25.1/onnxruntime-1.25.1.jar")
    archivo <- tempfile(fileext = ".zip")
    on.exit(unlink(archivo), add = TRUE)
    message("Probando el binario de ONNX que publica Microsoft en Maven Central.")
    download.file(url, archivo, mode = "wb", method = "libcurl")
    destino <- file.path(tools::R_user_dir("onnxr", "data"), "lib")
    dir.create(destino, recursive = TRUE, showWarnings = FALSE)
    unzip(archivo, files = paste0("ai/onnxruntime/native/win-x64/",
                                 c("onnxruntime.dll", "onnxruntime_providers_shared.dll")),
          exdir = destino, junkpaths = TRUE)
    if (!onnxr::onnx_is_installed()) {
      stop("No se pudo preparar ONNX Runtime en ", destino, call. = FALSE)
    }
  })
}

preparar_onnx <- function() {
  Sys.setenv(R_USER_DATA_DIR = file.path(normalizePath(dir_dependencias, winslash = "/"), "runtime"))
  if (!onnxr::onnx_is_installed()) instalar_onnx_nn()
}

if (bloque == "xgboost_text_svd" && !reticulate::py_available(initialize = FALSE)) {
  reticulate::py_require(c("numpy==2.2.6", "scikit-learn==1.9.1", "threadpoolctl==3.8.0"),
                         python_version = ">=3.11,<3.13")
  if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
    interprete <- Sys.which(c("python", "python3"))
    interprete <- unname(interprete[nzchar(interprete)])[1L]
    prueba <- "import numpy, sklearn, threadpoolctl; assert (numpy.__version__, sklearn.__version__, threadpoolctl.__version__) == ('2.2.6', '1.9.1', '3.8.0')"
    if (!is.na(interprete) && system2(interprete, c("-c", shQuote(prueba)), stdout = FALSE,
                                      stderr = FALSE) == 0L) {
      reticulate::use_python(interprete, required = TRUE)
    } else {
      Sys.setenv(RETICULATE_PYTHON = "managed")
    }
  }
}

# ------------------------------------------------------------------------------
# (3) Resolver conflictos entre funciones
# ------------------------------------------------------------------------------

if ("package:conflicted" %in% search() && "package:dplyr" %in% search()) {
  conflicted::conflict_prefer("filter", "dplyr")
  conflicted::conflict_prefer("lag", "dplyr")
}
