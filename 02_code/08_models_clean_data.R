# ------------------------------------------------------------------------------
# Archivo: 08_models_clean_data.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Comparar Random Forest, XGBoost y LightGBM en cinco folds
#   espaciales de 400 m, con perdida absoluta asimetrica.
# Input: Rutas de main.R y RDS de 01_cleaning.R con las variables enriquecidas.
# Output: Resumen XLSX con folds y predictores, y un modelo y CSV por familia en
#   03_outputs/03_models_clean_data/. El modo seleccion_xgboost devuelve
#   la configuracion ganadora de XGBoost para 09_models_text_svd.R.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir los hiperparametros y la validacion
# ------------------------------------------------------------------------------

semilla <- 2026L
n_folds <- 5L
lado_bloque_m <- 400L
hilos <- 4L
peso_sobre <- 2 / 3
peso_sub <- 1 - peso_sobre
solo_cv_xgboost <- bloque == "seleccion_xgboost"

# Cada lista forma el producto cartesiano de sus valores.
# Un escalar es fijo. Para iterarlo, sustituir el valor por un vector c(...).
configuraciones <- list(
  random_forest = list(
    num.trees = c(150L, 250L),          # Cantidad de arboles del bosque.
    mtry = c(4L, 6L),                   # Predictores candidatos por division.
    min.node.size = 5L,    # Tamano minimo del nodo para dividirlo.
    max.depth = 20L,            # Profundidad maxima de un arbol.
    sample.fraction = 0.8               # Fraccion de train muestreada por arbol.
  ),
  xgboost = list(
    nrounds = c(200L, 400L),            # Cantidad de rondas de boosting.
    eta = 0.05,          # Tasa de aprendizaje.
    max_depth = 6L,              # Profundidad maxima de un arbol.
    min_child_weight = c(20L, 30L), # Peso minimo para crear un nodo hijo.
    subsample = 0.85,                   # Fraccion de filas por ronda.
    colsample_bytree = 0.9,             # Fraccion de predictores por arbol.
    lambda = 1,                        # Regularizacion L2 de las hojas.
    max_bin = 128L                     # Cantidad maxima de intervalos por variable.
  ),
  lightgbm = list(
    nrounds = c(200L, 400L),            # Cantidad de rondas de boosting.
    learning_rate = 0.05, # Tasa de aprendizaje.
    max_depth = 6L,              # Profundidad maxima de un arbol.
    min_data_in_leaf = c(10L, 20L), # Minimo de observaciones en cada hoja.
    num_leaves = 64L,                  # Tope de hojas para permitir profundidad 6.
    bagging_fraction = 0.85,            # Fraccion de filas por muestreo.
    bagging_freq = 1L,                  # Muestrear filas en cada ronda.
    feature_fraction = 0.9,            # Fraccion de predictores por arbol.
    max_bin = 127L                     # Cantidad maxima de intervalos por variable.
  )
)

if (solo_cv_xgboost) configuraciones <- configuraciones["xgboost"]
grillas <- lapply(configuraciones, expand.grid, KEEP.OUT.ATTRS = FALSE)
parametros_nombre <- list(
  random_forest = c("num.trees", "mtry", "min.node.size", "max.depth"),
  xgboost = c("nrounds", "eta", "max_depth", "min_child_weight"),
  lightgbm = c("nrounds", "learning_rate", "max_depth", "min_data_in_leaf")
)

carpetas <- c(
  random_forest = "01_random_forest",
  xgboost = "02_xgboost",
  lightgbm = "03_lightgbm"
)

# ------------------------------------------------------------------------------
# (1.1) Ajustar y predecir con los arboles y calcular las metricas
# ------------------------------------------------------------------------------

# El cuantil 1/3 minimiza la perdida absoluta con pesos 2/3 al sobreestimar
# y 1/3 al subestimar. Ranger lo estima con un bosque cuantilico.
ajustar_arbol <- function(familia, datos, precios, parametros) {
  set.seed(semilla)

  if (familia == "random_forest") {
    argumentos <- list(
      x = datos, y = precios, quantreg = TRUE, na.action = "na.learn",
      oob.error = FALSE, num.threads = hilos, seed = semilla
    )
    return(do.call(ranger, c(argumentos, parametros)))
  }

  rondas <- parametros$nrounds
  parametros$nrounds <- NULL

  if (familia == "xgboost") {
    parametros <- c(list(
      objective = "reg:quantileerror", quantile_alpha = peso_sub,
      tree_method = "hist", nthread = hilos
    ), parametros)
    datos_xgb <- xgb.DMatrix(as.matrix(datos), label = precios, nthread = hilos)
    return(xgb.train(parametros, datos_xgb, nrounds = rondas, verbose = 0))
  }

  parametros <- c(list(
    objective = "quantile", alpha = peso_sub, num_threads = hilos,
    seed = semilla, deterministic = TRUE, force_col_wise = TRUE, verbosity = -1L
  ), parametros)
  datos_lgb <- lgb.Dataset(as.matrix(datos), label = precios)
  lgb.train(parametros, datos_lgb, nrounds = rondas, verbose = -1L)
}

predecir_arbol <- function(familia, modelo, datos) {
  switch(
    familia,
    random_forest = predict(
      modelo, datos, type = "quantiles", quantiles = peso_sub, num.threads = hilos
    )$predictions[, 1],
    xgboost = predict(modelo, xgb.DMatrix(as.matrix(datos), nthread = hilos)),
    lightgbm = predict(modelo, as.matrix(datos), params = list(num_threads = hilos))
  )
}

medir_error_cv <- function(error_cop) {
  sobre <- error_cop > 0
  sub <- error_cop < 0

  # Las contribuciones se dividen por el total de avisos y suman la perdida.
  # Cada MAE por lado se divide solo por los avisos con ese signo de error.
  perdida_sobre <- peso_sobre * mean(pmax(error_cop, 0))
  perdida_sub <- peso_sub * mean(pmax(-error_cop, 0))
  data.frame(
    perdida_asim_cop = perdida_sobre + perdida_sub,
    perdida_sobre_cop = perdida_sobre,
    perdida_sub_cop = perdida_sub,
    mae_cop = mean(abs(error_cop)),
    mae_sobre_cop = if (any(sobre)) mean(error_cop[sobre]) else NA_real_,
    mae_sub_cop = if (any(sub)) mean(-error_cop[sub]) else NA_real_,
    pct_sobre = 100 * mean(sobre),
    sesgo_cop = mean(error_cop)
  )
}

# ------------------------------------------------------------------------------
# (2) Leer las bases y preparar los mismos predictores para todos los modelos
# ------------------------------------------------------------------------------

rutas_limpias <- if (solo_cv_xgboost) archivo_train_clean else c(archivo_train_clean, archivo_test_clean)
if (!all(file.exists(rutas_limpias))) {
  stop("Primero ejecute Rscript main.R cleaning.", call. = FALSE)
}
train <- readRDS(archivo_train_clean)
stopifnot(all(is.finite(train$price)))
if (!solo_cv_xgboost) {
  test <- readRDS(archivo_test_clean)
  plantilla <- read_csv(archivo_plantilla, col_types = cols(property_id = col_character()),
                        col_select = property_id, show_col_types = FALSE)
  stopifnot(!anyNA(test$property_id), !anyNA(plantilla$property_id),
            !anyDuplicated(test$property_id), !anyDuplicated(plantilla$property_id),
            setequal(test$property_id, plantilla$property_id))
}

# Las tres predicciones supervisadas de texto quedan fuera de los regresores:
# se generaron con otra particion. Si se incorporan, hay que rehacerlas por fold.
numericas <- c(
  "surface_covered", "bedrooms", "bathrooms", "month", "year", "lat", "lon",
  "coord_x", "coord_y", "estrato", "valor_ref", "incidentes", "indice_seg",
  "dist_est5", "dist_est6", "prop_comercio", "dist_parque", "dist_restaurante",
  "dist_universidad", "dist_financiero", "dist_avenida", "dist_tm",
  paste0("txt_pc", seq_len(10L)), grep("^txt_sim_", names(train), value = TRUE),
  "txt_n_palabras", sprintf("e5_%03d", seq_len(384L))
)
categoricas <- c("property_type", "uso")
niveles <- lapply(train[categoricas], function(x) {
  sort(unique(enc2utf8(as.character(x[!is.na(x)]))), method = "radix")
})

armar_matriz <- function(base) {
  x <- as.matrix(base[numericas])
  storage.mode(x) <- "double"
  x[!is.finite(x)] <- NA_real_
  for (variable in categoricas) {
    valor <- enc2utf8(as.character(base[[variable]]))
    ausente <- !(valor %in% niveles[[variable]])
    valor[ausente] <- "sin_dato"
    categorias <- c(niveles[[variable]], "sin_dato")
    indicadores <- 1 * outer(valor, categorias, "==")
    colnames(indicadores) <- paste(variable, make.names(categorias, unique = TRUE), sep = "_")
    x <- cbind(x, indicadores)
  }
  x
}

x_train <- armar_matriz(train)
if (!solo_cv_xgboost) x_test <- armar_matriz(test)
precio <- train$price
y_arboles <- precio / 1e6

# ------------------------------------------------------------------------------
# (3) Asignar los bloques espaciales a los folds
# ------------------------------------------------------------------------------

coords <- as.matrix(train[c("coord_x", "coord_y")])
stopifnot(all(is.finite(coords)))
origen <- apply(coords, 2L, min)
celdas <- floor(sweep(coords, 2L, origen) / lado_bloque_m)
bloque <- paste(celdas[, 1L], celdas[, 2L], sep = "_")
bloques <- sort(unique(bloque), method = "radix")
stopifnot(length(bloques) >= n_folds)
set.seed(semilla)
fold_bloque <- setNames(sample(rep(seq_len(n_folds), length.out = length(bloques))), bloques)
fold <- unname(fold_bloque[bloque])
validaciones <- split(seq_len(nrow(train)), fold)
stopifnot(length(validaciones) == n_folds)

# ------------------------------------------------------------------------------
# (4) Validar, seleccionar y exportar un ganador por familia
# ------------------------------------------------------------------------------

hojas <- list(resumen = NULL)
if (!solo_cv_xgboost) {
  dir.create(dir_modelos_limpios, recursive = TRUE, showWarnings = FALSE)
  hojas$folds <- data.frame(property_id = train$property_id, bloque, fold)
  hojas$predictores <- data.frame(predictor = colnames(x_train))
}
ganadores <- list()
avance <- 0L
total <- n_folds * sum(vapply(grillas, nrow, integer(1)))
inicio_cv <- Sys.time()

for (familia in names(grillas)) {
  grilla <- grillas[[familia]]
  resultados <- vector("list", nrow(grilla))
  x_arbol <- if (familia == "random_forest") as.data.frame(x_train) else x_train
  if (!solo_cv_xgboost) {
    test_arbol <- if (familia == "random_forest") as.data.frame(x_test) else x_test
  }

  for (i in seq_len(nrow(grilla))) {
    inicio <- Sys.time()
    parametros <- as.list(grilla[i, , drop = FALSE])
    pred_cv <- rep(NA_real_, nrow(train))
    for (k in seq_along(validaciones)) {
      validacion <- validaciones[[k]]
      inicio_fold <- Sys.time()
      modelo <- ajustar_arbol(
        familia, x_arbol[-validacion, , drop = FALSE], y_arboles[-validacion], parametros
      )
      pred_cv[validacion] <- 1e6 * predecir_arbol(
        familia, modelo, x_arbol[validacion, , drop = FALSE]
      )
      avance <- avance + 1L
      segundos_fold <- as.numeric(difftime(Sys.time(), inicio_fold, units = "secs"))
      message(sprintf("[%d/%d | %.1f%%] %s | combinacion %d/%d | fold %d/%d | %.1f s",
                      avance, total, 100 * avance / total, familia, i, nrow(grilla),
                      k, n_folds, segundos_fold))
    }

    stopifnot(all(is.finite(pred_cv)))
    metricas <- medir_error_cv(pred_cv - precio)
    segundos <- as.numeric(difftime(Sys.time(), inicio, units = "secs"))
    resultados[[i]] <- cbind(grilla[i, , drop = FALSE], metricas, segundos_cv = segundos)
  }

  ranking <- bind_rows(resultados)
  ranking <- ranking[order(ranking$perdida_asim_cop, ranking$mae_cop), , drop = FALSE]
  ranking <- cbind(puesto = seq_len(nrow(ranking)), ranking)
  mejor <- ranking[1L, , drop = FALSE]
  if (familia == "xgboost") {
    seleccion_xgboost <- cbind(mejor, n_combinaciones = nrow(grilla), k_folds = n_folds,
                               lado_bloque_m = lado_bloque_m)
  }
  if (solo_cv_xgboost) {
    print(ranking, row.names = FALSE)
    next
  }
  parametros <- as.list(mejor[names(grilla)])
  iterados <- parametros_nombre[[familia]]
  valores <- vapply(parametros[iterados], format, character(1), trim = TRUE,
                    scientific = FALSE, decimal.mark = ".")
  sufijo <- paste(paste(iterados, valores, sep = "_", recycle0 = TRUE), collapse = "_")
  etiqueta <- paste0(familia, "_clean_data")
  nombre <- if (nzchar(sufijo)) paste(etiqueta, sufijo, sep = "_") else etiqueta
  salida <- file.path(dir_modelos_limpios, carpetas[[familia]])
  dir.create(salida, showWarnings = FALSE)

  message("Reajustando ganador ", familia, " con todo train.")
  entrada <- list(numericas = numericas, categoricas = categoricas, niveles = niveles,
                  predictores = colnames(x_train), parametros = parametros,
                  objetivo = c(peso_sobre = peso_sobre, peso_sub = peso_sub))
  entrada$escala_precio <- 1e6
  modelo <- ajustar_arbol(familia, x_arbol, y_arboles, parametros)
  pred_test <- 1e6 * predecir_arbol(familia, modelo, test_arbol)
  switch(familia,
         random_forest = saveRDS(modelo, file.path(salida, paste0(nombre, ".rds"))),
         xgboost = xgb.save(modelo, file.path(salida, paste0(nombre, ".ubj"))),
         lightgbm = lgb.save(modelo, file.path(salida, paste0(nombre, ".txt"))))
  saveRDS(entrada, file.path(salida, paste0(nombre, "_entrada.rds")))
  stopifnot(length(pred_test) == nrow(test), all(is.finite(pred_test)))
  orden <- match(plantilla$property_id, test$property_id)
  predicciones <- data.frame(property_id = plantilla$property_id, price = pred_test[orden])
  write_csv(predicciones, file.path(salida, paste0("[model]", nombre, ".csv")))

  ganadores[[familia]] <- cbind(
    modelo = familia, bloque = "clean_data", k_folds = n_folds, semilla = semilla,
    lado_bloque_m = lado_bloque_m, buffer_m = 0L, n_train = nrow(train),
    n_predictores = ncol(x_train), n_combinaciones = nrow(grilla),
    mejor[names(metricas)], hiperparametros = sufijo
  )
  hojas[[etiqueta]] <- head(ranking, 15L)
  resumen <- bind_rows(ganadores)
  resumen <- resumen[order(resumen$perdida_asim_cop, resumen$mae_cop), , drop = FALSE]
  hojas$resumen <- resumen

  # Actualizar el Excel al terminar cada familia para tener resultados parciales.
  write.xlsx(hojas, file.path(dir_modelos_limpios, "resumen.xlsx"), overwrite = TRUE)
  message(sprintf("Ganador %s | perdida COP %.0f | MAE COP %.0f", familia,
                  mejor$perdida_asim_cop, mejor$mae_cop))
}

# ------------------------------------------------------------------------------
# (5) Mostrar la comparacion final
# ------------------------------------------------------------------------------

if (!solo_cv_xgboost) {
  print(resumen[c("modelo", "perdida_asim_cop", "mae_cop", "hiperparametros")],
        row.names = FALSE)
}
message(sprintf("Duracion total de modelos: %.1f minutos",
                as.numeric(difftime(Sys.time(), inicio_cv, units = "mins"))))
invisible(seleccion_xgboost)
