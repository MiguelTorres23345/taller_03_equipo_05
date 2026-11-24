# ------------------------------------------------------------------------------
# Archivo: 09_models_text_svd.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Reproducir el XGBoost con TF-IDF corregido y SVD elegido para la entrega.
# Input: train_clean.rds y test_clean.rds de 01_cleaning.R y plantilla original.
# Output: CSV del modelo seleccionado y resumen.txt en 07_models_text_svd.
#          El modo cv_final recalcula solo la validacion del modelo seleccionado.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Identificar la configuracion seleccionada
# ------------------------------------------------------------------------------

ejecutar_xgboost_final <- function(archivo_train_clean, archivo_test_clean,
                                    archivo_plantilla, salida, modo = "modelo_final") {
  semilla <- 2026L
  hilos <- 4L
  componentes <- 32L
  lado_bloque <- 400L
  rondas <- 400L
  # Ganador del ranking xgboost_clean_data de 08_models_clean_data.R.
  # Esa grilla se evaluo con 438 predictores, incluidos los embeddings E5.
  hp <- list(objective = "reg:quantileerror", quantile_alpha = 1/3,
             tree_method = "hist", nthread = hilos, seed = semilla,
             eta = 0.05, max_depth = 6L, min_child_weight = 30L,
             subsample = 0.85, colsample_bytree = 0.9, lambda = 1, max_bin = 128L)
  cv_seleccion <- c(perdida = 53413520.3635665, mae = 134560267.998439)
  cv_final <- c(perdida = 52110825.71443468, mae = 131136753.58969337)
  nombre <- sprintf("[model]XGBoost_tfidf_svd%d_corregido_nrounds_%d_eta_%s_depth_%d_min_child_%d.csv",
                    componentes, rondas, format(hp$eta, decimal.mark = ".", trim = TRUE),
                    hp$max_depth, hp$min_child_weight)

  # ----------------------------------------------------------------------------
  # (2) Preparar los predictores y el texto
  # ----------------------------------------------------------------------------

  rutas <- c(archivo_train_clean, archivo_test_clean, archivo_plantilla)
  if (!all(file.exists(rutas))) {
    stop("Faltan insumos del modelo final: ",
         paste(rutas[!file.exists(rutas)], collapse = ", "),
         ". Prepare las bases limpias antes de correr el modelo.", call. = FALSE)
  }
  train <- readRDS(archivo_train_clean)
  test <- readRDS(archivo_test_clean)
  tipo_id <- readr::cols(property_id = readr::col_character())
  plantilla <- readr::read_csv(archivo_plantilla, col_types = tipo_id, show_col_types = FALSE)
  stopifnot(!anyDuplicated(train$property_id), !anyDuplicated(test$property_id),
            !anyDuplicated(plantilla$property_id), all(is.finite(train$price)),
            setequal(test$property_id, plantilla$property_id))

  numericas <- c(
    "surface_covered", "bedrooms", "bathrooms", "month", "year", "lat", "lon",
    "coord_x", "coord_y", "estrato", "valor_ref", "incidentes", "indice_seg",
    "dist_est5", "dist_est6", "prop_comercio", "dist_parque", "dist_restaurante",
    "dist_universidad", "dist_financiero", "dist_avenida", "dist_tm",
    paste0("txt_pc", seq_len(10L)),
    paste0("txt_sim_", c("lujo", "remodelar", "estrenar", "vista", "amenidades",
                         "parqueadero", "ubic_premium", "oportunidad", "pequeno", "comercial")),
    "txt_n_palabras"
  )
  categoricas <- c("property_type", "uso")
  columnas <- c(numericas, categoricas, "title", "description")
  n <- nrow(train)
  filas_train <- seq_len(n)
  filas_test <- n + seq_len(nrow(test))
  precio <- train$price
  base <- rbind(train[, columnas], test[, columnas])
  x <- as.matrix(base[numericas])
  storage.mode(x) <- "double"
  x[!is.finite(x)] <- NA_real_
  for (variable in categoricas) {
    nivel <- sort(unique(enc2utf8(as.character(train[[variable]]))), method = "radix", na.last = NA)
    valor <- enc2utf8(as.character(base[[variable]]))
    valor[!valor %in% nivel] <- "sin_dato"
    categorias <- c(nivel, "sin_dato")
    indicadores <- 1 * outer(valor, categorias, "==")
    colnames(indicadores) <- paste(variable, make.names(categorias, unique = TRUE), sep = "_")
    x <- cbind(x, indicadores)
  }
  stopifnot(ncol(x) == 54L)
  texto <- function(z) ifelse(is.na(z), "", z)
  if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
    python <- Sys.which(c("python", "python3"))
    if (any(nzchar(python))) reticulate::use_python(python[nzchar(python)][1L], required = TRUE)
  }
  py <- reticulate::import_main()
  reticulate::py_set_attr(py, "documentos", enc2utf8(paste(
    texto(base$title), texto(base$description), sep = ". "
  )))
  rm(base)
  invisible(gc())

  # Aplicar la correccion ortografica elegida y retirar precios explicitos.
  reticulate::py_run_string(r"----(
import html, re, unicodedata
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

tokens_pat = re.compile(r"(?u)[^\W_]+")

erratas = {
    "aprtamento": "apartamento", "apartameto": "apartamento", "apartamnto": "apartamento",
    "apartamneto": "apartamento", "apartemento": "apartamento",
    "habitacones": "habitaciones", "abitaciones": "habitaciones",
    "parquedero": "parqueadero", "parqueadreo": "parqueadero", "parquadero": "parqueadero",
    "parqueadreos": "parqueaderos", "remodeldo": "remodelado", "remodelda": "remodelada",
    "remodellado": "remodelado", "terrasa": "terraza", "ilumincion": "iluminacion",
    "ilumunacion": "iluminacion", "conjuto": "conjunto", "conjuno": "conjunto",
    "acensor": "ascensor", "ascenzor": "ascensor", "extrenar": "estrenar",
}

def normalizar(texto):
    texto = unicodedata.normalize("NFKC", html.unescape(texto)).lower()
    texto = re.sub(r"(?:[$\u20ac\u00a3]\s*|\bcop\s*)\d[\d., ]*", " ", texto)
    texto = re.sub(r"\b\d[\d,.]*\s*(?:millones|mill[o\u00f3]n)\b", " ", texto)
    texto = re.sub(r"\s*\b([aeiou])acute\b\s*", r"\1", texto)
    texto = re.sub(r"\s*\bntilde\b\s*", "n", texto)
    return " ".join(erratas.get(token, token) for token in tokens_pat.findall(texto))

textos = [normalizar(doc) for doc in documentos]
del documentos

def proyectar(ajuste, validacion, componentes, semilla):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=.85,
                             max_features=15000, sublinear_tf=True, dtype=np.float32)
    svd = TruncatedSVD(n_components=componentes, algorithm="randomized",
                       n_iter=5, random_state=semilla)
    with threadpool_limits(limits=2):
        a = tfidf.fit_transform([textos[i] for i in ajuste])
        b = tfidf.transform([textos[i] for i in validacion])
        return svd.fit_transform(a), svd.transform(b)
)----")

  # ----------------------------------------------------------------------------
  # (3) Ajustar el modelo seleccionado
  # ----------------------------------------------------------------------------

  predecir <- function(a, b) {
    proyecciones <- py$proyectar(as.list(a - 1L), as.list(b - 1L), componentes, semilla)
    x_a <- cbind(x[a, , drop = FALSE], proyecciones[[1L]])
    x_b <- cbind(x[b, , drop = FALSE], proyecciones[[2L]])
    ajuste <- xgboost::xgb.DMatrix(x_a, label = precio[a] / 1e6, weight = rep(1, length(a)),
                                  nthread = hilos)
    validacion <- xgboost::xgb.DMatrix(x_b, nthread = hilos)
    fit <- xgboost::xgb.train(hp, ajuste, nrounds = rondas, verbose = 0)
    pred <- 1e6 * predict(fit, validacion)
    stopifnot(all(is.finite(pred)))
    pred
  }
  origen_cv <- "CV registrada del modelo seleccionado"
  if (modo == "cv_final") {
    coords <- as.matrix(train[c("coord_x", "coord_y")])
    celdas <- floor(sweep(coords, 2L, apply(coords, 2L, min)) / lado_bloque)
    bloque <- paste(celdas[, 1L], celdas[, 2L], sep = "_")
    bloques <- sort(unique(bloque), method = "radix")
    set.seed(semilla)
    reparto <- setNames(sample(rep(seq_len(5L), length.out = length(bloques))), bloques)
    fold <- unname(reparto[bloque])
    error <- numeric(n)
    for (k in seq_len(5L)) {
      b <- which(fold == k)
      error[b] <- predecir(which(fold != k), b) - precio[b]
      message("Fold ", k, " | ", length(b), " avisos")
    }
    cv_final <- c(perdida = mean((2/3) * pmax(error, 0) + (1/3) * pmax(-error, 0)),
                   mae = mean(abs(error)))
    origen_cv <- "CV recalculada del modelo seleccionado"
  }

  # ----------------------------------------------------------------------------
  # (4) Exportar en el orden de la plantilla y guardar el resumen
  # ----------------------------------------------------------------------------

  dir.create(salida, recursive = TRUE, showWarnings = FALSE)
  if (modo == "modelo_final") {
    pred <- predecir(filas_train, filas_test)
    orden <- match(plantilla$property_id, test$property_id)
    resultado <- data.frame(property_id = plantilla$property_id, price = pred[orden])
    archivo_salida <- file.path(salida, nombre)
    if (file.exists(archivo_salida)) {
      anterior <- readr::read_csv(archivo_salida, col_types = tipo_id, show_col_types = FALSE)
      coincide <- identical(anterior$property_id, resultado$property_id) &&
        isTRUE(all(abs(anterior$price - resultado$price) <= 1))
      if (!coincide) {
        stop("El reajuste difiere del CSV elegido. Se conserva el archivo en ", archivo_salida,
             call. = FALSE)
      }
    } else {
      readr::write_csv(resultado, archivo_salida)
    }
  }
  writeLines(c(
    nombre,
    "XGBoost con correccion ortografica, TF-IDF de palabras y bigramas y 32 componentes SVD",
    sprintf("Avisos de entrenamiento %d | Avisos de test %d | Predictores %d",
            n, nrow(test), ncol(x) + componentes),
    "",
    "Seleccion historica de hiperparametros",
    "Origen 08_models_clean_data.R | XGBoost con E5 | 36 combinaciones",
    "Cinco folds espaciales de 400 m | criterio de seleccion perdida asimetrica",
    sprintf("Rondas %d | eta %.2f | profundidad %d | peso minimo del nodo %d",
            rondas, hp$eta, hp$max_depth, hp$min_child_weight),
    sprintf("Subsample %.2f | colsample_bytree %.2f | lambda %g | max_bin %d",
            hp$subsample, hp$colsample_bytree, hp$lambda, hp$max_bin),
    sprintf("Semilla %d | objetivo cuantil 1/3", semilla),
    sprintf("CV de la configuracion ganadora con E5 | perdida %.2f | MAE %.2f millones de COP",
            cv_seleccion[["perdida"]]/1e6, cv_seleccion[["mae"]]/1e6),
    "",
    origen_cv,
    sprintf("Cinco folds espaciales de 400 m | perdida %.2f | MAE %.2f millones de COP",
            cv_final[["perdida"]]/1e6, cv_final[["mae"]]/1e6)
  ), file.path(salida, "resumen.txt"))
  invisible(nombre)
}
