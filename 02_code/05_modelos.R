# ==============================================================================
# Archivo: 03_modelos.R
# Proyecto: Problem Set 3 - Prediccion de precios de vivienda en Chapinero
# Descripcion:
#   Modelos del taller para el precio: random forest (ranger) y XGBoost. Los
#   hiperparametros se escogen para minimizar el MAE. Cada modelo se corre con
#   dos validaciones cruzadas espaciales. Una deja una localidad por fold y
#   excluye del entrenamiento lo que esta a menos de distancia_buffer metros de
#   ella. La otra usa bloques pequenos sin buffer, repetidos con varias semillas,
#   para que la distancia entre lo que se valida y lo que se entrena se parezca a
#   la del test. Las coordenadas del anuncio entran como predictores. Al final de
#   cada modelo se predice el test y se exporta el archivo para Kaggle.
# Input:
#   train_planas.csv y test_planas.csv (salida de 02_variables_planas.R) y el
#   shapefile de localidades de Bogota (IDECA, Loca.shp).
#   Las columnas de seguimiento de la limpieza y los textos no entran al modelo.
# Output:
#   CSV de predicciones en la carpeta submits y tabla de resultados del modelo.
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

dir_datos <- "C:/Users/aeste/OneDrive/Documentos/10 Semestre/Big Data ML/Taller 3"
dir_clean <- file.path(dir_datos, "clean")
dir_submit <- file.path(dir_datos, "submits")
archivo_localidades <- file.path(dir_datos, "localidades", "Loca.shp")

# Metros que se excluyen alrededor de la localidad que se valida
distancia_buffer <- 300

# Lado (metros) de los bloques de la validacion espacial por bloques. Se escoge
# para que la distancia de lo que se valida al entrenamiento se parezca a la del
# test (mediana cerca de 96 m, percentil 90 cerca de 275 m). No lleva buffer
tamano_bloque <- 400

# Cantidad de veces que se reparten los bloques entre los folds (semillas distintas)
n_semillas <- 3

# Texto que se agrega al nombre de los archivos de salida, para no sobrescribir
# los de corridas anteriores
sufijo <- "_coord"

n_arboles <- 500
vars_modelo <- c("surface_covered", "bedrooms", "bathrooms", "property_type",
                 "month", "year", "estrato", "valor_ref", "uso", "incidentes",
                 "dist_est5", "dist_est6", "prop_comercio", "dist_parque",
                 "dist_restaurante", "dist_universidad", "dist_financiero",
                 "dist_avenida", "dist_tm", "coord_x", "coord_y")

dir.create(dir_submit, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------------
# (2) Importar las bases y el shapefile
# ------------------------------------------------------------------------------

train_planas <- read_csv(file.path(dir_clean, "train_planas.csv"), show_col_types = FALSE)
test_planas <- read_csv(file.path(dir_clean, "test_planas.csv"), show_col_types = FALSE)

# El shapefile viene en grados (MAGNA). Todo se trabaja en el sistema MAGNA
# Colombia Bogota (EPSG 3116, metros), que es el que usa el buffer
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


# ------------------------------------------------------------------------------
# (4) Localidad de cada anuncio y validacion espacial con buffer
# ------------------------------------------------------------------------------

puntos_train <- fun_puntos(train_planas)
puntos_test <- fun_puntos(test_planas)

# Coordenadas en metros (EPSG 3116) como predictores
train_planas$coord_x <- st_coordinates(puntos_train)[, 1]
train_planas$coord_y <- st_coordinates(puntos_train)[, 2]
test_planas$coord_x <- st_coordinates(puntos_test)[, 1]
test_planas$coord_y <- st_coordinates(puntos_test)[, 2]

# Localidad mas cercana a cada anuncio (los puntos con coordenadas en la calle
# quedan asignados igual)
puntos_train$localidad <- localidades$LocNombre[st_nearest_feature(puntos_train, localidades)]
train_planas$localidad <- puntos_train$localidad
print(count(train_planas, localidad, sort = TRUE))

# Cada fold valida una localidad. El entrenamiento deja por fuera esa localidad y
# todo lo que este a menos de distancia_buffer metros de ella
set.seed(2026)
folds_loc_buffer <- spatial_leave_location_out_cv(
  puntos_train, group = localidad, buffer = distancia_buffer
)
folds_loc_buffer

# Los folds se pasan a caret
folds_caret <- rsample2caret(folds_loc_buffer)

ctrl_mae <- trainControl(
  method = "cv",
  index = folds_caret$index,
  indexOut = folds_caret$indexOut,
  summaryFunction = fun_mae,
  savePredictions = "final"
)


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


# ------------------------------------------------------------------------------
# (6) Random forest
# ------------------------------------------------------------------------------

# Los mejores modelos anteriores quedaron en el borde de la grilla (mtry 8 y
# nodo 40), asi que se extiende hacia ese lado. Los nodos grandes tambien usan
# menos memoria
rf_grid <- expand.grid(
  mtry = c(5, 8, 11, 14),
  splitrule = "variance",
  min.node.size = c(40, 80)
)

set.seed(2026)
rf_mae <- train(
  x = db_train[, vars_modelo],
  y = db_train$precio,
  method = "ranger",
  metric = "MAE",
  maximize = FALSE,
  trControl = ctrl_mae,
  tuneGrid = rf_grid,
  num.trees = n_arboles,
  importance = "impurity"
)

rf_mae
rf_mae$bestTune

# Importancia de las variables
print(varImp(rf_mae))


# ------------------------------------------------------------------------------
# (7) Resultados de la validacion
# ------------------------------------------------------------------------------

# Predicciones fuera de muestra con los mejores hiperparametros
pred_cv_rf <- rf_mae$pred |>
  mutate(localidad = db_train$localidad[rowIndex])

# MAE total (millones de pesos) y por localidad
mae_rf <- mean(abs(pred_cv_rf$obs - pred_cv_rf$pred))
mae_rf

mae_localidad_rf <- pred_cv_rf |>
  group_by(localidad) |>
  summarise(n = n(), MAE = mean(abs(obs - pred)), .groups = "drop") |>
  arrange(desc(n))
print(mae_localidad_rf)

# Chapinero es el fold mas parecido al test
filter(mae_localidad_rf, localidad == "CHAPINERO")


# ------------------------------------------------------------------------------
# (8) Prediccion del test y exportar
# ------------------------------------------------------------------------------

predict_rf <- test_planas |>
  mutate(price = predict(rf_mae, newdata = db_test[, vars_modelo]) * 1e6) |>
  select(property_id, price)

name_rf <- paste0("RF_mtry_", rf_mae$bestTune$mtry,
                  "_nodo_", rf_mae$bestTune$min.node.size, sufijo, ".csv")
write_csv(predict_rf, file.path(dir_submit, name_rf))

# Tabla de resultados para comparar con los demas modelos
resultados <- data.frame(modelo = "RF", archivo = name_rf, mtry = rf_mae$bestTune$mtry,
                         min.node.size = rf_mae$bestTune$min.node.size,
                         MAE_cv_millones = mae_rf)
write_csv(resultados, file.path(dir_submit, paste0("resultados_rf", sufijo, ".csv")))
print(resultados)


# ------------------------------------------------------------------------------
# (9) XGBoost
# ------------------------------------------------------------------------------

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

# El mejor modelo anterior tuvo eta 0.03, profundidad 6 y 300 rondas, los valores
# extremos de la grilla. Se prueban eta menores, mas profundidad y mas rondas
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
xgb_mae <- train(
  x = x_train,
  y = db_train$precio,
  method = "xgbTree",
  metric = "MAE",
  maximize = FALSE,
  trControl = ctrl_mae,
  tuneGrid = xgb_grid,
  verbosity = 0
)

xgb_mae
xgb_mae$bestTune

# Importancia de las variables
print(varImp(xgb_mae))


# ------------------------------------------------------------------------------
# (10) Resultados de la validacion de XGBoost
# ------------------------------------------------------------------------------

# Predicciones fuera de muestra con los mejores hiperparametros
pred_cv_xgb <- xgb_mae$pred |>
  mutate(localidad = db_train$localidad[rowIndex])

# MAE total (millones de pesos) y por localidad
mae_xgb <- mean(abs(pred_cv_xgb$obs - pred_cv_xgb$pred))
mae_xgb

mae_localidad_xgb <- pred_cv_xgb |>
  group_by(localidad) |>
  summarise(n = n(), MAE = mean(abs(obs - pred)), .groups = "drop") |>
  arrange(desc(n))
print(mae_localidad_xgb)

# Chapinero es el fold mas parecido al test
filter(mae_localidad_xgb, localidad == "CHAPINERO")


# ------------------------------------------------------------------------------
# (11) Prediccion del test y exportar XGBoost
# ------------------------------------------------------------------------------

predict_xgb <- test_planas |>
  mutate(price = predict(xgb_mae, newdata = x_test) * 1e6) |>
  select(property_id, price)

name_xgb <- paste0("XGB_nrounds_", xgb_mae$bestTune$nrounds,
                   "_depth_", xgb_mae$bestTune$max_depth,
                   "_eta_", xgb_mae$bestTune$eta, sufijo, ".csv")
write_csv(predict_xgb, file.path(dir_submit, name_xgb))

# Tabla de resultados para comparar con los demas modelos
resultados_xgb <- data.frame(modelo = "XGB", archivo = name_xgb,
                             nrounds = xgb_mae$bestTune$nrounds,
                             max_depth = xgb_mae$bestTune$max_depth,
                             eta = xgb_mae$bestTune$eta,
                             min_child_weight = xgb_mae$bestTune$min_child_weight,
                             MAE_cv_millones = mae_xgb)
write_csv(resultados_xgb, file.path(dir_submit, paste0("resultados_xgb", sufijo, ".csv")))
print(resultados_xgb)


# ------------------------------------------------------------------------------
# (12) Validacion espacial por bloques sin buffer
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
# (13) Random forest con validacion por bloques
# ------------------------------------------------------------------------------

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
                                    MAE_cv_millones = mae_rf_bloques)
write_csv(resultados_rf_bloques, file.path(dir_submit, paste0("resultados_rf_bloques", sufijo, ".csv")))
print(resultados_rf_bloques)


# ------------------------------------------------------------------------------
# (14) XGBoost con validacion por bloques
# ------------------------------------------------------------------------------

set.seed(2026)
xgb_bloques <- train(
  x = x_train,
  y = db_train$precio,
  method = "xgbTree",
  metric = "MAE",
  maximize = FALSE,
  trControl = ctrl_bloques,
  tuneGrid = xgb_grid,
  verbosity = 0
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
                                     MAE_cv_millones = mae_xgb_bloques)
write_csv(resultados_xgb_bloques, file.path(dir_submit, paste0("resultados_xgb_bloques", sufijo, ".csv")))
print(resultados_xgb_bloques)