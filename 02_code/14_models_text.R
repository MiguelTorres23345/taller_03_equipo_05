# ==============================================================================
# Archivo: 14_models_text.R
# Proyecto: Problem Set 3 - Prediccion de precios de vivienda en Chapinero
# Descripcion:
#   Ultimo modelo de 05_modelos.R (seccion 12 en adelante): validacion espacial
#   por bloques pequenos SIN buffer, repetida con varias semillas, y coordenadas
#   del anuncio como predictores. Se agregan las variables de texto de las
#   descripciones: 23 variables de TF-IDF y MiniLM (06_text_features.py) y la
#   prediccion del MiniLM con fine-tuning (09_text_finetune.py).
#   Random forest (ranger) y XGBoost; los hiperparametros se escogen para
#   minimizar el MAE. Al final de cada modelo se predice el test y se exporta
#   el archivo para Kaggle.
# Input:
#   train_planas.csv y test_planas.csv (salida de 02_variables_planas.R),
#   text_features_{train,test}.csv y text_finetune_minilm_{train,test}.csv
#   (01_data/02_text) y el shapefile de localidades de Bogota (IDECA, Loca.shp),
#   que solo se usa para reportar el MAE por localidad.
# Output:
#   CSV de predicciones y tablas de resultados en 03_outputs/04_modelos_texto.
# ==============================================================================


# ------------------------------------------------------------------------------
# (1) Paquetes y rutas
# ------------------------------------------------------------------------------

library(dplyr)
library(readr)
library(sf)
library(caret)
library(ranger)
library(xgboost)
library(rsample)
library(spatialsample)

if (!exists("root")) root <- "."
dir_clean <- file.path(root, "01_data", "02_clean")
dir_text <- file.path(root, "01_data", "02_text")
dir_submit <- file.path(root, "03_outputs", "04_modelos_texto")
archivo_localidades <- file.path(root, "01_data", "01_raw", "fuentes_externas", "localidades", "Loca.shp")

# Lado (metros) de los bloques de la validacion espacial por bloques. Se escoge
# para que la distancia de lo que se valida al entrenamiento se parezca a la del
# test (mediana cerca de 96 m, percentil 90 cerca de 275 m). No lleva buffer
tamano_bloque <- 400

# Cantidad de veces que se reparten los bloques entre los folds (semillas distintas)
n_semillas <- 3

# TRUE agrega la prediccion del MiniLM con fine-tuning (conjunto C); FALSE usa
# solo las 23 variables de TF-IDF y MiniLM congelado (conjunto B)
usar_finetune <- TRUE

# Texto que se agrega al nombre de los archivos de salida
sufijo <- if (usar_finetune) "_coord_texto_ft" else "_coord_texto"

n_arboles <- 500
vars_base <- c("surface_covered", "bedrooms", "bathrooms", "property_type",
               "month", "year", "estrato", "valor_ref", "uso", "incidentes",
               "dist_est5", "dist_est6", "prop_comercio", "dist_parque",
               "dist_restaurante", "dist_universidad", "dist_financiero",
               "dist_avenida", "dist_tm", "coord_x", "coord_y")

dir.create(dir_submit, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------------
# (2) Importar las bases y las variables de texto
# ------------------------------------------------------------------------------

train_planas <- read_csv(file.path(dir_clean, "train_planas.csv"), show_col_types = FALSE)
test_planas <- read_csv(file.path(dir_clean, "test_planas.csv"), show_col_types = FALSE)

# Variables de texto por property_id. Las de train son predicciones fuera de
# muestra: el modelo de texto que predijo cada anuncio no vio su precio
fun_leer_texto <- function(.base) {
  read_csv(file.path(dir_text, paste0("text_features_", .base, ".csv")),
           show_col_types = FALSE) |>
    left_join(read_csv(file.path(dir_text, paste0("text_finetune_minilm_", .base, ".csv")),
                       show_col_types = FALSE), by = "property_id")
}
texto_train <- fun_leer_texto("train")
texto_test <- fun_leer_texto("test")

train_planas <- left_join(train_planas, texto_train, by = "property_id")
test_planas <- left_join(test_planas, texto_test, by = "property_id")

vars_texto <- setdiff(names(texto_train), "property_id")
if (!usar_finetune) vars_texto <- setdiff(vars_texto, "txt_ft_minilm_lnp")
vars_modelo <- c(vars_base, vars_texto)
stopifnot(!anyNA(train_planas[, vars_texto]), !anyNA(test_planas[, vars_texto]))

# El shapefile viene en grados (MAGNA). Todo se trabaja en el sistema MAGNA
# Colombia Bogota (EPSG 3116, metros)
localidades <- st_read(archivo_localidades, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(3116)


# ------------------------------------------------------------------------------
# (3) Funciones
# ------------------------------------------------------------------------------

# Moda de un vector
fun_moda <- function(.x) {
  names(which.max(table(.x)))
}

# Rellena los vacios con la mediana (numericas) o la moda (categoricas) calculadas
# solo con train, y deja las categoricas como factor con los mismos niveles
fun_preparar <- function(.db, .ref) {
  for (v in vars_modelo) {
    if (is.numeric(.ref[[v]])) {
      .db[[v]][is.na(.db[[v]])] <- median(.ref[[v]], na.rm = TRUE)
    } else {
      .db[[v]][is.na(.db[[v]])] <- fun_moda(.ref[[v]])
      .db[[v]] <- factor(.db[[v]], levels = sort(unique(.ref[[v]])))
    }
  }
  .db
}

# Metrica para caret: MAE de cada fold
fun_mae <- function(data, lev = NULL, model = NULL) {
  c(MAE = mean(abs(data$obs - data$pred)))
}

# Puntos de una base en metros
fun_puntos <- function(.db) {
  .db |>
    st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
    st_transform(3116)
}

# Distancia en metros de cada punto de .obj al punto mas cercano de .ref
fun_dist_cercana <- function(.obj, .ref) {
  cercano <- st_nearest_feature(.obj, .ref)
  as.numeric(st_distance(.obj, .ref[cercano, ], by_element = TRUE))
}

# MAE total, en Chapinero y sesgo a partir de las predicciones fuera de muestra
# de caret. Con folds repetidos cada anuncio aparece una vez por semilla
fun_resumen <- function(.pred) {
  .pred |>
    mutate(localidad = db_train$localidad[rowIndex], error = pred - obs) |>
    summarise(MAE = mean(abs(error)),
              MAE_chapinero = mean(abs(error[localidad == "CHAPINERO"])),
              error_medio = mean(error),
              pct_sobreestima = 100 * mean(error > 0))
}


# ------------------------------------------------------------------------------
# (4) Coordenadas y localidad de cada anuncio
# ------------------------------------------------------------------------------

puntos_train <- fun_puntos(train_planas)
puntos_test <- fun_puntos(test_planas)

# Coordenadas en metros (EPSG 3116) como predictores
train_planas$coord_x <- st_coordinates(puntos_train)[, 1]
train_planas$coord_y <- st_coordinates(puntos_train)[, 2]
test_planas$coord_x <- st_coordinates(puntos_test)[, 1]
test_planas$coord_y <- st_coordinates(puntos_test)[, 2]

# Localidad mas cercana a cada anuncio. Solo se usa para reportar el MAE por
# localidad (Chapinero es la mas parecida al test)
train_planas$localidad <- localidades$LocNombre[st_nearest_feature(puntos_train, localidades)]


# ------------------------------------------------------------------------------
# (5) Base de modelo
# ------------------------------------------------------------------------------

# El precio se modela en millones de pesos
db_train <- train_planas |>
  mutate(precio = price / 1e6) |>
  fun_preparar(train_planas)
db_test <- fun_preparar(test_planas, train_planas)

# Mismo orden de filas que los folds
stopifnot(nrow(db_train) == nrow(puntos_train),
          !anyNA(db_train[, vars_modelo]), !anyNA(db_test[, vars_modelo]))

# XGBoost recibe una matriz numerica, asi que las categoricas (property_type, uso)
# se pasan a dummies. Los niveles son los de train, asi que train y test quedan
# con las mismas columnas
fun_matriz <- function(.db) {
  matriz <- model.matrix(~ . - 1, data = .db[, vars_modelo])
  colnames(matriz) <- make.names(colnames(matriz))
  matriz
}

x_train <- fun_matriz(db_train)
x_test <- fun_matriz(db_test)
stopifnot(identical(colnames(x_train), colnames(x_test)))


# ------------------------------------------------------------------------------
# (6) Validacion espacial por bloques sin buffer
# ------------------------------------------------------------------------------

# Bloques cuadrados de tamano_bloque metros asignados al azar a 5 folds, sin
# buffer. El reparto se repite n_semillas veces con semillas distintas y todos los
# folds se juntan en una sola lista para caret. puntos_train tiene el mismo orden
# de filas que db_train
indices_bloques <- list()
indices_fuera_bloques <- list()
for (s in seq_len(n_semillas)) {
  set.seed(2025 + s)
  folds_s <- spatial_block_cv(
    puntos_train, v = 5, method = "random", cellsize = tamano_bloque,
    radius = NULL, buffer = NULL
  )
  folds_s_caret <- rsample2caret(folds_s)
  names(folds_s_caret$index) <- paste0("S", s, "_", names(folds_s_caret$index))
  names(folds_s_caret$indexOut) <- names(folds_s_caret$index)
  indices_bloques <- c(indices_bloques, folds_s_caret$index)
  indices_fuera_bloques <- c(indices_fuera_bloques, folds_s_caret$indexOut)
}
length(indices_bloques)

# Revision del tamano del bloque. Distancia (m) de cada anuncio que se valida al
# anuncio de entrenamiento mas cercano, comparada con la del test al train
prob <- c(0.10, 0.25, 0.50, 0.75, 0.90)
dist_test_train <- fun_dist_cercana(puntos_test, puntos_train)
dist_validacion <- unlist(lapply(1:5, function(k) {
  fun_dist_cercana(puntos_train[indices_fuera_bloques[[k]], ],
                   puntos_train[indices_bloques[[k]], ])
}))
print(rbind(test = quantile(dist_test_train, prob),
            validacion_bloques = quantile(dist_validacion, prob)))

ctrl_bloques <- trainControl(
  method = "cv",
  index = indices_bloques,
  indexOut = indices_fuera_bloques,
  summaryFunction = fun_mae,
  savePredictions = "final"
)


# ------------------------------------------------------------------------------
# (7) Random forest con validacion por bloques
# ------------------------------------------------------------------------------

# Misma grilla que la ultima version de 05_modelos.R
rf_grid <- expand.grid(
  mtry = c(5, 8, 11, 14),
  splitrule = "variance",
  min.node.size = c(40, 80)
)

set.seed(2026)
rf_bloques <- train(
  x = db_train[, vars_modelo],
  y = db_train$precio,
  method = "ranger",
  metric = "MAE",
  maximize = FALSE,
  trControl = ctrl_bloques,
  tuneGrid = rf_grid,
  num.trees = n_arboles,
  importance = "impurity"
)

rf_bloques
rf_bloques$bestTune

# Importancia de las variables
print(varImp(rf_bloques))

# Predicciones fuera de muestra con los mejores hiperparametros
pred_cv_rf_bloques <- rf_bloques$pred |>
  mutate(localidad = db_train$localidad[rowIndex])

# MAE total (millones de pesos) y por localidad
mae_rf_bloques <- mean(abs(pred_cv_rf_bloques$obs - pred_cv_rf_bloques$pred))
mae_rf_bloques

mae_localidad_rf_bloques <- pred_cv_rf_bloques |>
  group_by(localidad) |>
  summarise(n = n(), MAE = mean(abs(obs - pred)), .groups = "drop") |>
  arrange(desc(n))
print(mae_localidad_rf_bloques)

# Chapinero es la localidad mas parecida al test
filter(mae_localidad_rf_bloques, localidad == "CHAPINERO")

# Prediccion del test y exportar
predict_rf_bloques <- test_planas |>
  mutate(price = predict(rf_bloques, newdata = db_test[, vars_modelo]) * 1e6) |>
  select(property_id, price)

name_rf_bloques <- paste0("RF_bloques_mtry_", rf_bloques$bestTune$mtry,
                          "_nodo_", rf_bloques$bestTune$min.node.size, sufijo, ".csv")
write_csv(predict_rf_bloques, file.path(dir_submit, name_rf_bloques))

# Tabla de resultados para comparar con los demas modelos
resultados_rf_bloques <- data.frame(modelo = "RF_bloques", archivo = name_rf_bloques,
                                    mtry = rf_bloques$bestTune$mtry,
                                    min.node.size = rf_bloques$bestTune$min.node.size,
                                    fun_resumen(rf_bloques$pred))
write_csv(resultados_rf_bloques, file.path(dir_submit, paste0("resultados_rf_bloques", sufijo, ".csv")))
print(resultados_rf_bloques)


# ------------------------------------------------------------------------------
# (8) XGBoost con validacion por bloques
# ------------------------------------------------------------------------------

# Desde xgboost 3.0, el metodo "xgbTree" de caret falla ("ALTLIST classes must
# provide a Set_elt method") porque caret intenta escribir dentro del modelo. Este
# metodo es el mismo xgbTree (misma grilla y perdida cuadratica) pero guarda el
# modelo dentro de una lista, asi que funciona con xgboost 1.7 y con 3.x
fin_inclusivo <- local({
  # En xgboost >= 3 el fin de iterationrange es inclusivo; antes era exclusivo
  x0 <- matrix(seq_len(100) / 100, 50)
  b0 <- xgb.train(params = list(max_depth = 2, nthread = 1),
                  data = xgb.DMatrix(x0, label = x0[, 1]), nrounds = 2, verbose = 0)
  isTRUE(all.equal(predict(b0, x0), predict(b0, x0, iterationrange = c(1, 2))))
})

modelo_xgb <- getModelInfo("xgbTree", regex = FALSE)[[1]]
modelo_xgb$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  parametros <- list(objective = "reg:squarederror", eta = param$eta,
                     max_depth = param$max_depth, gamma = param$gamma,
                     colsample_bytree = param$colsample_bytree,
                     min_child_weight = param$min_child_weight, subsample = param$subsample)
  datos <- xgb.DMatrix(as.matrix(x), label = y)
  list(booster = xgb.train(params = parametros, data = datos,
                           nrounds = param$nrounds, verbose = 0))
}
# caret entrena solo el mayor nrounds y predice los menores con los primeros arboles
modelo_xgb$predict <- function(modelFit, newdata, submodels = NULL) {
  nueva <- as.matrix(newdata)
  out <- predict(modelFit$booster, nueva)
  if (!is.null(submodels)) {
    menores <- lapply(submodels$nrounds, function(n) {
      predict(modelFit$booster, nueva, iterationrange = c(1, if (fin_inclusivo) n else n + 1))
    })
    out <- c(list(out), menores)
  }
  out
}
modelo_xgb$varImp <- function(object, ...) {
  imp <- xgb.importance(model = object$booster)
  data.frame(Overall = imp$Gain, row.names = imp$Feature)
}

# Misma grilla que la ultima version de 05_modelos.R
xgb_grid <- expand.grid(
  nrounds = c(300, 600, 1000),
  max_depth = c(6, 8),
  eta = c(0.01, 0.03),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = c(20, 50),
  subsample = 0.8
)

set.seed(2026)
xgb_bloques <- train(
  x = x_train,
  y = db_train$precio,
  method = modelo_xgb,
  metric = "MAE",
  maximize = FALSE,
  trControl = ctrl_bloques,
  tuneGrid = xgb_grid
)

xgb_bloques
xgb_bloques$bestTune

# Importancia de las variables
print(varImp(xgb_bloques))

# Predicciones fuera de muestra con los mejores hiperparametros
pred_cv_xgb_bloques <- xgb_bloques$pred |>
  mutate(localidad = db_train$localidad[rowIndex])

# MAE total (millones de pesos) y por localidad
mae_xgb_bloques <- mean(abs(pred_cv_xgb_bloques$obs - pred_cv_xgb_bloques$pred))
mae_xgb_bloques

mae_localidad_xgb_bloques <- pred_cv_xgb_bloques |>
  group_by(localidad) |>
  summarise(n = n(), MAE = mean(abs(obs - pred)), .groups = "drop") |>
  arrange(desc(n))
print(mae_localidad_xgb_bloques)

# Chapinero es la localidad mas parecida al test
filter(mae_localidad_xgb_bloques, localidad == "CHAPINERO")

# Prediccion del test y exportar
predict_xgb_bloques <- test_planas |>
  mutate(price = predict(xgb_bloques, newdata = x_test) * 1e6) |>
  select(property_id, price)

name_xgb_bloques <- paste0("XGB_bloques_nrounds_", xgb_bloques$bestTune$nrounds,
                           "_depth_", xgb_bloques$bestTune$max_depth,
                           "_eta_", xgb_bloques$bestTune$eta, sufijo, ".csv")
write_csv(predict_xgb_bloques, file.path(dir_submit, name_xgb_bloques))

# Tabla de resultados para comparar con los demas modelos
resultados_xgb_bloques <- data.frame(modelo = "XGB_bloques", archivo = name_xgb_bloques,
                                     nrounds = xgb_bloques$bestTune$nrounds,
                                     max_depth = xgb_bloques$bestTune$max_depth,
                                     eta = xgb_bloques$bestTune$eta,
                                     min_child_weight = xgb_bloques$bestTune$min_child_weight,
                                     fun_resumen(xgb_bloques$pred))
write_csv(resultados_xgb_bloques, file.path(dir_submit, paste0("resultados_xgb_bloques", sufijo, ".csv")))
print(resultados_xgb_bloques)
