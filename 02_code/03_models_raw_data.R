# ------------------------------------------------------------------------------
# Archivo: 03_models_raw_data.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion:
#   Comparar y ajustar Random Forest, XGBoost y LightGBM sobre los datos crudos.
# Input:
#   Rutas de main.R a train.csv, test.csv, submission_template.csv
#   y al directorio dir_modelos_raw.
#   Paquetes cargados por 00_setup.R.
# Output:
#   Fold y rankings en la sesion. En 03_outputs/01_models_raw_data/ se exportan
#   resumen.xlsx, los tres modelos ganadores y sus CSV en formato Kaggle.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir los hiperparametros sobre los que se va a iterar
# ------------------------------------------------------------------------------

semilla <- 2026L
n_folds <- 5L
hilos <- 4L
cuantil <- 1 / 3

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
    min_child_weight = c(10L, 20L), # Peso minimo para crear un nodo hijo.
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
# (2) Leer los datos crudos y definir los folds
# ------------------------------------------------------------------------------

train <- read_csv(archivo_train, show_col_types = FALSE)
tipo_id <- cols(property_id = col_character())
test <- read_csv(archivo_test, col_types = tipo_id, show_col_types = FALSE)
plantilla <- read_csv(archivo_plantilla, col_types = tipo_id, col_select = property_id,
                       show_col_types = FALSE)

# Se conservan los NA, ceros y valores extremos tal como llegan en los CSV.
# Usar solo variables estructuradas. El ID se reserva para la entrega.
# Ciudad y operacion son constantes. Titulo y descripcion se usaran en otro bloque.
columnas_numericas <- c(
  "month", "year", "surface_total", "surface_covered", "rooms",
  "bedrooms", "bathrooms", "lat", "lon"
)
x_train <- as.data.frame(train[columnas_numericas])
x_test <- as.data.frame(test[columnas_numericas])
x_train$apartamento <- as.integer(train$property_type == "Apartamento")
x_test$apartamento <- as.integer(test$property_type == "Apartamento")

# Ajustar en millones de COP y evaluar/exportar en COP.
y <- train$price / 1e6
orden_test <- match(plantilla$property_id, test$property_id)
stopifnot(!anyNA(y), !anyNA(test$property_id), !anyNA(plantilla$property_id),
          !anyDuplicated(test$property_id), !anyDuplicated(plantilla$property_id),
          setequal(test$property_id, plantilla$property_id), !anyNA(orden_test))

set.seed(semilla)
fold <- sample(rep(seq_len(n_folds), length.out = nrow(train)))
indices_validacion <- split(seq_len(nrow(train)), fold)

# ------------------------------------------------------------------------------
# (3) Ajustes, predicciones y perdida
# ------------------------------------------------------------------------------

# El cuantil 1/3 minimiza la perdida absoluta con pesos 2/3 al sobreestimar
# y 1/3 al subestimar. Ranger lo estima con un bosque cuantilico.
ajustar <- function(familia, datos, precios, parametros) {
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
      objective = "reg:quantileerror", quantile_alpha = cuantil,
      tree_method = "hist", nthread = hilos
    ), parametros)
    datos_xgb <- xgb.DMatrix(as.matrix(datos), label = precios, nthread = hilos)
    return(xgb.train(parametros, datos_xgb, nrounds = rondas, verbose = 0))
  }

  parametros <- c(list(
    objective = "quantile", alpha = cuantil, num_threads = hilos,
    seed = semilla, deterministic = TRUE, force_col_wise = TRUE, verbosity = -1L
  ), parametros)
  datos_lgb <- lgb.Dataset(as.matrix(datos), label = precios)
  lgb.train(parametros, datos_lgb, nrounds = rondas, verbose = -1L)
}

predecir <- function(familia, modelo, datos) {
  switch(
    familia,
    random_forest = predict(
      modelo, datos, type = "quantiles", quantiles = cuantil, num.threads = hilos
    )$predictions[, 1],
    xgboost = predict(modelo, xgb.DMatrix(as.matrix(datos), nthread = hilos)),
    lightgbm = predict(modelo, as.matrix(datos), params = list(num_threads = hilos))
  )
}

medir <- function(error_cop) {
  sobre <- error_cop > 0
  sub <- error_cop < 0

  # Las contribuciones se dividen por el total de avisos y suman la perdida.
  # Cada MAE por lado se divide solo por los avisos con ese signo de error.
  perdida_sobre <- (2 / 3) * mean(pmax(error_cop, 0))
  perdida_sub <- (1 / 3) * mean(pmax(-error_cop, 0))
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
# (4) Validar, seleccionar y exportar el ganador de cada familia
# ------------------------------------------------------------------------------

hojas <- list(resumen = NULL)
ganadores <- list()
total_ajustes <- n_folds * sum(vapply(grillas, nrow, integer(1)))
avance <- 0L

for (familia in names(grillas)) {
  grilla <- grillas[[familia]]
  resultados <- vector("list", nrow(grilla))

  for (i in seq_len(nrow(grilla))) {
    parametros <- as.list(grilla[i, , drop = FALSE])
    pred_cv <- numeric(nrow(train))
    inicio <- Sys.time()

    for (k in seq_len(n_folds)) {
      validacion <- indices_validacion[[k]]
      modelo <- ajustar(familia, x_train[-validacion, ], y[-validacion], parametros)
      pred_cv[validacion] <- predecir(familia, modelo, x_train[validacion, ])
      avance <- avance + 1L
      message(sprintf(
        "[%d/%d | %.1f%%] %s | combinacion %d/%d | fold %d/%d",
        avance, total_ajustes, 100 * avance / total_ajustes,
        familia, i, nrow(grilla), k, n_folds
      ))
    }

    stopifnot(all(is.finite(pred_cv)))
    # Cada aviso aporta un solo error fuera de su fold de entrenamiento.
    metricas <- medir((pred_cv - y) * 1e6)
    segundos <- as.numeric(difftime(Sys.time(), inicio, units = "secs"))
    resultados[[i]] <- cbind(grilla[i, , drop = FALSE], metricas, segundos_cv = segundos)
  }

  ranking <- bind_rows(resultados)
  ranking <- ranking[order(ranking$perdida_asim_cop, ranking$mae_cop), ]
  ranking <- cbind(puesto = seq_len(nrow(ranking)), ranking)
  mejor <- ranking[1, , drop = FALSE]
  parametros <- as.list(mejor[names(grilla)])

  # El nombre utiliza los valores de los hiperparametros principales del ganador.
  iterados <- parametros_nombre[[familia]]
  valores <- vapply(parametros[iterados], format, character(1),
                    trim = TRUE, scientific = FALSE, decimal.mark = ".")
  pares <- paste(iterados, valores, sep = "_", recycle0 = TRUE)
  sufijo <- paste(pares, collapse = "_")
  etiqueta <- paste0(familia, "_raw_data")
  nombre <- paste(c(etiqueta, pares), collapse = "_")
  salida <- file.path(dir_modelos_raw, carpetas[[familia]])
  dir.create(salida, recursive = TRUE, showWarnings = FALSE)

  modelo <- ajustar(familia, x_train, y, parametros)
  pred_test <- predecir(familia, modelo, x_test)
  stopifnot(length(pred_test) == nrow(test), all(is.finite(pred_test)))

  switch(
    familia,
    random_forest = saveRDS(modelo, file.path(salida, paste0(nombre, ".rds"))),
    xgboost = xgb.save(modelo, file.path(salida, paste0(nombre, ".ubj"))),
    lightgbm = lgb.save(modelo, file.path(salida, paste0(nombre, ".txt")))
  )
  predicciones <- data.frame(property_id = plantilla$property_id, price = pred_test[orden_test] * 1e6)
  write_csv(predicciones, file.path(salida, paste0(nombre, ".csv")))

  ganadores[[familia]] <- cbind(
    modelo = familia, bloque = "raw_data", k_folds = n_folds, semilla = semilla,
    n_combinaciones = nrow(grilla), mejor[names(metricas)], hiperparametros = sufijo
  )
  hojas[[etiqueta]] <- head(ranking, 15L)
  message(sprintf("Ganador %s | %s | perdida COP %.0f", familia, sufijo,
                  mejor$perdida_asim_cop))
}

# ------------------------------------------------------------------------------
# (5) Guardar el resumen y mostrar los tres ganadores
# ------------------------------------------------------------------------------

resumen <- bind_rows(ganadores)
resumen <- resumen[order(resumen$perdida_asim_cop, resumen$mae_cop), ]
hojas[["resumen"]] <- resumen

# Mantener las cifras como numeros, con COP enteros para la visualizacion.
write.xlsx(hojas, file.path(dir_modelos_raw, "resumen.xlsx"), overwrite = TRUE)

print(resumen[c("modelo", "perdida_asim_cop", "mae_cop", "hiperparametros")],
      row.names = FALSE)
