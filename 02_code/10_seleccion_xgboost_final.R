# ------------------------------------------------------------------------------
# Archivo: 10_seleccion_xgboost_final.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Reproducir exclusivamente la grilla historica de XGBoost de
#              08_models_clean_data.R del respaldo. Usa E5 y CV espacial.
# Input: train_clean.rds de la limpieza con las variables E5 incluidas.
# Output: Ranking de CV por consola y como objeto R. No exporta predicciones.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Leer la base y definir la configuracion historica
# ------------------------------------------------------------------------------

seleccionar_xgboost_historico <- function(archivo_train) {
  if (!file.exists(archivo_train)) {
    stop("Falta la base limpia de entrenamiento: ", archivo_train, call. = FALSE)
  }
  train <- readRDS(archivo_train)
  semilla <- 2026L
  hilos <- 4L
  n_folds <- 5L
  lado <- 400L
  grilla <- expand.grid(
    nrounds = c(200L, 400L), eta = c(0.01, 0.02, 0.05),
    max_depth = c(4L, 6L), min_child_weight = c(10L, 20L, 30L),
    subsample = 0.85, colsample_bytree = 0.9, lambda = 1, max_bin = 128L,
    KEEP.OUT.ATTRS = FALSE
  )

  # ----------------------------------------------------------------------------
  # (2) Reconstruir los 438 predictores y los cinco folds
  # ----------------------------------------------------------------------------

  numericas <- c(
    "surface_covered", "bedrooms", "bathrooms", "month", "year", "lat", "lon",
    "coord_x", "coord_y", "estrato", "valor_ref", "incidentes", "indice_seg",
    "dist_est5", "dist_est6", "prop_comercio", "dist_parque", "dist_restaurante",
    "dist_universidad", "dist_financiero", "dist_avenida", "dist_tm",
    paste0("txt_pc", seq_len(10L)), grep("^txt_sim_", names(train), value = TRUE),
    "txt_n_palabras", sprintf("e5_%03d", seq_len(384L))
  )
  x <- as.matrix(train[numericas])
  storage.mode(x) <- "double"
  x[!is.finite(x)] <- NA_real_
  for (variable in c("property_type", "uso")) {
    valor <- enc2utf8(as.character(train[[variable]]))
    niveles <- sort(unique(valor[!is.na(valor)]), method = "radix")
    valor[is.na(valor)] <- "sin_dato"
    categorias <- c(niveles, "sin_dato")
    indicadores <- 1 * outer(valor, categorias, "==")
    colnames(indicadores) <- paste(variable, make.names(categorias, unique = TRUE), sep = "_")
    x <- cbind(x, indicadores)
  }
  stopifnot(ncol(x) == 438L, all(is.finite(train$price)))
  coords <- as.matrix(train[c("coord_x", "coord_y")])
  stopifnot(all(is.finite(coords)))
  celdas <- floor(sweep(coords, 2L, apply(coords, 2L, min)) / lado)
  bloque <- paste(celdas[, 1L], celdas[, 2L], sep = "_")
  bloques <- sort(unique(bloque), method = "radix")
  stopifnot(length(bloques) >= n_folds)
  set.seed(semilla)
  reparto <- setNames(sample(rep(seq_len(n_folds), length.out = length(bloques))), bloques)
  fold <- unname(reparto[bloque])
  precio <- train$price
  rm(train)

  # ----------------------------------------------------------------------------
  # (3) Evaluar las 36 combinaciones y ordenar por perdida asimetrica
  # ----------------------------------------------------------------------------

  metricas <- vapply(seq_len(nrow(grilla)), function(i) {
    hp <- as.list(grilla[i, , drop = FALSE])
    rondas <- hp$nrounds
    hp$nrounds <- NULL
    hp <- c(list(objective = "reg:quantileerror", quantile_alpha = 1/3,
                 tree_method = "hist", nthread = hilos), hp)
    error <- numeric(length(precio))
    for (k in seq_len(n_folds)) {
      b <- which(fold == k)
      a <- which(fold != k)
      set.seed(semilla)
      datos <- xgboost::xgb.DMatrix(x[a, , drop = FALSE], label = precio[a]/1e6, nthread = hilos)
      ajuste <- xgboost::xgb.train(hp, datos, nrounds = rondas, verbose = 0)
      validacion <- xgboost::xgb.DMatrix(x[b, , drop = FALSE], nthread = hilos)
      error[b] <- 1e6 * predict(ajuste, validacion) - precio[b]
    }
    stopifnot(all(is.finite(error)))
    message("Combinacion ", i, " de ", nrow(grilla))
    c(perdida_asimetrica_cop = (2/3) * mean(pmax(error, 0)) + (1/3) * mean(pmax(-error, 0)),
      mae_cop = mean(abs(error)))
  }, numeric(2L))
  resultados <- cbind(grilla, as.data.frame(t(metricas)))
  ranking <- resultados[order(resultados$perdida_asimetrica_cop, resultados$mae_cop), ]
  print(ranking, row.names = FALSE)
  invisible(ranking)
}
