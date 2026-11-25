# ------------------------------------------------------------------------------
# Archivo: 04_modelos_imagen.R
# Proyecto: Taller 3 - Big Data y Machine Learning
#
# Descripcion:
#   Agrega como predictores los 32 rasgos de imagen que produce la red
#   convolucional en Colab (una fila por celda de unos 50 m). Los rasgos cambian
#   en cada fold, porque la red de cada fold se entreno sin las celdas que se
#   validan (y, dentro del entrenamiento, con validacion interna). Por eso la
#   validacion se hace con un ciclo propio y no con caret. Se compara XGBoost sin
#   y con los rasgos en los mismos 5 folds por bloques (semilla 1) y se exporta
#   la prediccion del test con la red final.
# Input:
#   Se corre despues de 03_modelos.R, en la misma sesion de main.R, que deja los
#   objetos x_train, x_test, db_train, train_planas, test_planas, indices_bloques
#   e indices_fuera_bloques (con la prediccion de texto ya incluida en x_train).
#   Los CSV de Colab (features_s1_fold1_n32.csv a features_s1_fold5_n32.csv y
#   features_final_n32.csv) van en 01_data/imagenes.
# Output:
#   CSV de predicciones y tabla de resultados en 03_outputs/submits.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (15) Rutas, hiperparametros y celda de cada anuncio
# ------------------------------------------------------------------------------

# dir_imagenes, dir_submit y sufijo vienen de main.R y 03_modelos.R

# Hiperparametros del mejor XGBoost con validacion por bloques
res_xgb <- read_csv(file.path(dir_submit, paste0("resultados_xgb_bloques", sufijo, ".csv")),
                    show_col_types = FALSE)
xgb_params <- list(
  objective = "reg:squarederror",
  eta = res_xgb$eta,
  max_depth = res_xgb$max_depth,
  min_child_weight = res_xgb$min_child_weight,
  gamma = 0,
  colsample_bytree = 0.8,
  subsample = 0.8
)
xgb_nrounds <- res_xgb$nrounds

# Misma regla de celda que usa el cuaderno de Colab (unos 50 m)
fun_celda <- function(.lon, .lat) {
  sprintf("%d_%d", as.integer(floor(.lon / 0.00045)), as.integer(floor(.lat / 0.00045)))
}

train_planas$celda_id <- fun_celda(train_planas$lon, train_planas$lat)
test_planas$celda_id <- fun_celda(test_planas$lon, test_planas$lat)


# ------------------------------------------------------------------------------
# (16) Funciones
# ------------------------------------------------------------------------------

# Rasgos de imagen de cada anuncio, segun su celda
fun_rasgos <- function(.archivo, .celdas) {
  rasgos <- read_csv(file.path(dir_imagenes, .archivo), show_col_types = FALSE)
  posicion <- match(.celdas, rasgos$celda_id)
  stopifnot(!anyNA(posicion))
  matriz <- as.matrix(rasgos[posicion, -1])
  colnames(matriz) <- paste0("img_", colnames(matriz))
  matriz
}

# Ajusta un XGBoost y predice
fun_xgb <- function(.x_ent, .y_ent, .x_pred) {
  modelo <- xgb.train(params = xgb_params,
                      data = xgb.DMatrix(.x_ent, label = .y_ent),
                      nrounds = xgb_nrounds, verbose = 0)
  predict(modelo, xgb.DMatrix(.x_pred))
}

# Validacion por bloques (semilla 1, folds 1 a 5). Con imagen, cada fold usa los
# rasgos de su propia red
fun_cv_imagen <- function(.con_imagen) {
  pred <- rep(NA_real_, nrow(x_train))
  for (k in 1:5) {
    x_k <- x_train
    if (.con_imagen) {
      x_k <- cbind(x_train, fun_rasgos(paste0("features_s1_fold", k, "_n32.csv"),
                                       train_planas$celda_id))
    }
    dentro <- indices_bloques[[k]]
    fuera <- indices_fuera_bloques[[k]]
    pred[fuera] <- fun_xgb(x_k[dentro, ], db_train$precio[dentro], x_k[fuera, ])
  }
  stopifnot(!anyNA(pred))
  pred
}


# ------------------------------------------------------------------------------
# (17) Resultados de la validacion sin y con imagen
# ------------------------------------------------------------------------------

pred_sin <- fun_cv_imagen(FALSE)
pred_con <- fun_cv_imagen(TRUE)

es_chapinero <- db_train$localidad == "CHAPINERO"
resultados_imagen <- data.frame(
  modelo = c("XGB sin imagen", "XGB con imagen"),
  MAE_total = c(mean(abs(pred_sin - db_train$precio)),
                mean(abs(pred_con - db_train$precio))),
  MAE_chapinero = c(mean(abs(pred_sin - db_train$precio)[es_chapinero]),
                    mean(abs(pred_con - db_train$precio)[es_chapinero]))
)
print(resultados_imagen)
write_csv(resultados_imagen, file.path(dir_submit, paste0("resultados_imagen", sufijo, ".csv")))


# ------------------------------------------------------------------------------
# (18) Prediccion del test con la red final y exportar
# ------------------------------------------------------------------------------

x_train_img <- cbind(x_train, fun_rasgos("features_final_n32.csv", train_planas$celda_id))
x_test_img <- cbind(x_test, fun_rasgos("features_final_n32.csv", test_planas$celda_id))
stopifnot(identical(colnames(x_train_img), colnames(x_test_img)))

set.seed(2026)
predict_imagen <- test_planas |>
  mutate(price = fun_xgb(x_train_img, db_train$precio, x_test_img) * 1e6) |>
  select(property_id, price)

name_imagen <- paste0("XGB_imagen_nrounds_", xgb_nrounds, "_depth_", xgb_params$max_depth,
                      "_eta_", xgb_params$eta, sufijo, ".csv")
write_csv(predict_imagen, file.path(dir_submit, name_imagen))
print(name_imagen)