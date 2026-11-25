# ------------------------------------------------------------------------------
# Archivo: 10_models_regresion_basica.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion: Estimar regresion lineal, Elastic Net y CART con datos enriquecidos.
#   Ajusta por cuadrados en niveles o en logaritmos al llamarlo desde el bloque 11.
# Input: RDS de 01_cleaning.R, plantilla y rutas de main.R. El bloque 11 indica
#   el uso de logaritmos y su carpeta de salida.
# Output: Tres modelos y tres CSV de entrega, y resumen.xlsx en
#   03_outputs/05_models_regresion_basica/.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir la validacion y los hiperparametros
# ------------------------------------------------------------------------------

semilla <- 2026L
n_folds <- 5L
lado_bloque_m <- 400L
escala_precio <- 1e6
if (!exists("usar_log", inherits = FALSE)) usar_log <- FALSE
escala_objetivo <- if (usar_log) "log" else "niveles"
criterio_cv <- if (usar_log) "mae_cop" else "perdida_asim_cop"

# Una trayectoria de lambda por alpha. Las fracciones se aplican a la mayor
# correlacion absoluta con el precio en el conjunto de ajuste estandarizado.
alphas <- c(0, 0.5, 1)
fracciones_lambda <- c(0.3, 0.1, 0.03, 0.01)
grilla <- expand.grid(fraccion_lambda = fracciones_lambda, alpha = alphas,
                     KEEP.OUT.ATTRS = FALSE)

# Un solo CART. Se usa la validacion externa por bloques para compararlo.
cp_cart <- 0.01
profundidad_cart <- 6L
min_split_cart <- 30L
min_hoja_cart <- 10L
control_cart <- rpart.control(cp = cp_cart, maxdepth = profundidad_cart,
                             minsplit = min_split_cart, minbucket = min_hoja_cart,
                             xval = 0L, maxsurrogate = 0L, maxcompete = 0L)

# ------------------------------------------------------------------------------
# (2) Leer datos y formar la matriz completa de predictores
# ------------------------------------------------------------------------------

train <- readRDS(archivo_train_clean)
test <- readRDS(archivo_test_clean)
plantilla <- read_csv(archivo_plantilla, col_types = cols(property_id = col_character()),
                       col_select = property_id, show_col_types = FALSE)
stopifnot(all(is.finite(train$price)), !anyNA(test$property_id),
          !anyNA(plantilla$property_id), !anyDuplicated(test$property_id),
          !anyDuplicated(plantilla$property_id), setequal(test$property_id, plantilla$property_id))
orden_test <- match(plantilla$property_id, test$property_id)

# Misma matriz que la del bloque de datos limpios. Las predicciones de texto
# supervisadas se produjeron con otra particion y quedan fuera de esta CV.
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
    valor[!valor %in% niveles[[variable]]] <- "sin_dato"
    categorias <- c(niveles[[variable]], "sin_dato")
    indicadores <- 1 * outer(valor, categorias, "==")
    colnames(indicadores) <- paste(variable, make.names(categorias, unique = TRUE), sep = "_")
    x <- cbind(x, indicadores)
  }
  x
}

x_train <- armar_matriz(train)
x_test <- armar_matriz(test)
precio <- train$price
y <- precio / escala_precio
if (usar_log) {
  stopifnot(all(y > 0))
  y <- log(y)
}

# ------------------------------------------------------------------------------
# (3) Preparar folds y funciones de estimacion
# ------------------------------------------------------------------------------

coords <- as.matrix(train[c("coord_x", "coord_y")])
stopifnot(all(is.finite(coords)))
origen <- apply(coords, 2L, min)
celdas <- floor(sweep(coords, 2L, origen) / lado_bloque_m)
id_bloque <- paste(celdas[, 1L], celdas[, 2L], sep = "_")
bloques <- sort(unique(id_bloque), method = "radix")
stopifnot(length(bloques) >= n_folds)
set.seed(semilla)
fold_bloque <- setNames(sample(rep(seq_len(n_folds), length.out = length(bloques))), bloques)
fold <- unname(fold_bloque[id_bloque])
validaciones <- split(seq_len(nrow(train)), fold)

imputar <- function(x, mediana) {
  faltantes <- which(is.na(x), arr.ind = TRUE)
  x[faltantes] <- mediana[faltantes[, 2L]]
  x
}

preparar <- function(ajuste, nuevo) {
  # Aprender la imputacion y la escala solo con los avisos de ajuste.
  mediana <- apply(ajuste, 2L, median, na.rm = TRUE)
  mediana[is.na(mediana)] <- 0
  ajuste <- scale(imputar(ajuste, mediana))
  centro <- attr(ajuste, "scaled:center")
  escala <- attr(ajuste, "scaled:scale")
  constantes <- escala == 0
  escala[constantes] <- 1
  ajuste[, constantes] <- 0
  nuevo <- scale(imputar(nuevo, mediana), center = centro, scale = escala)
  list(ajuste = ajuste, nuevo = nuevo, mediana = mediana, centro = centro, escala = escala)
}

ajustar_lineal <- function(x, y) {
  fit <- lm.fit(cbind("(Intercept)" = 1, x), y)
  # Conservar todos los predictores en la entrada y anular coeficientes redundantes.
  beta <- fit$coefficients
  beta[is.na(beta)] <- 0
  list(coeficientes = beta, rango = fit$rank)
}

ajustar_cart <- function(x, y) {
  # Una formula simple no necesita guardar el entorno con las matrices de ajuste.
  formula_cart <- as.formula("precio ~ .", env = baseenv())
  rpart(formula_cart, data = data.frame(precio = y, x), method = "anova",
        control = control_cart, y = FALSE)
}

lambda_referencia <- function(x, y) {
  max(abs(crossprod(x, y - mean(y)))) / length(y)
}

a_pesos <- function(estimado) {
  escala_precio * (if (usar_log) exp(estimado) else estimado)
}

medir <- function(estimado, real) {
  error <- a_pesos(estimado) - real
  stopifnot(all(is.finite(error)))
  sobre <- error > 0
  sub <- error < 0
  metricas <- data.frame(
    mae_cop = mean(abs(error)),
    mae_sobre_cop = if (any(sobre)) mean(error[sobre]) else NA_real_,
    mae_sub_cop = if (any(sub)) mean(-error[sub]) else NA_real_,
    pct_sobre = 100 * mean(sobre),
    sesgo_cop = mean(error)
  )
  if (usar_log) return(metricas)
  cbind(perdida_asim_cop = mean((2 / 3) * pmax(error, 0) + (1 / 3) * pmax(-error, 0)),
        metricas)
}

# ------------------------------------------------------------------------------
# (4) Validar y seleccionar la penalizacion de Elastic Net
# ------------------------------------------------------------------------------

pred_rl <- pred_cart <- rep(NA_real_, length(y))
pred_en <- matrix(NA_real_, nrow = length(y), ncol = nrow(grilla))
for (k in seq_along(validaciones)) {
  b <- validaciones[[k]]
  a <- seq_along(y)[-b]
  base <- preparar(x_train[a, , drop = FALSE], x_train[b, , drop = FALSE])

  rl <- ajustar_lineal(base$ajuste, y[a])
  pred_rl[b] <- cbind(1, base$nuevo) %*% rl$coeficientes
  cart <- ajustar_cart(base$ajuste, y[a])
  pred_cart[b] <- predict(cart, newdata = as.data.frame(base$nuevo))

  lambda <- lambda_referencia(base$ajuste, y[a]) * fracciones_lambda
  for (alpha in alphas) {
    columnas <- which(grilla$alpha == alpha)
    en <- glmnet(base$ajuste, y[a], alpha = alpha, lambda = lambda,
                 standardize = FALSE)
    pred_en[b, columnas] <- predict(en, newx = base$nuevo, s = lambda)
  }
  message(sprintf("Regresiones | fold %d/%d | %.0f%%", k, n_folds, 100 * k / n_folds))
}

ranking <- cbind(grilla, bind_rows(lapply(seq_len(ncol(pred_en)), function(i) {
  medir(pred_en[, i], precio)
})))
ranking <- ranking[order(ranking[[criterio_cv]], ranking$mae_cop), ]
ranking$puesto <- seq_len(nrow(ranking))
mejor <- ranking[1L, ]

# ------------------------------------------------------------------------------
# (5) Reajustar y exportar modelos, predicciones y resumen
# ------------------------------------------------------------------------------

base <- preparar(x_train, x_test)
rl <- ajustar_lineal(base$ajuste, y)
cart <- ajustar_cart(base$ajuste, y)
lambda_base <- lambda_referencia(base$ajuste, y)
lambda_final <- lambda_base * mejor$fraccion_lambda
en <- glmnet(base$ajuste, y, alpha = mejor$alpha,
             lambda = lambda_base * fracciones_lambda,
             standardize = FALSE)
modelos <- list(rl, en, cart)
predicciones <- list(
  drop(cbind(1, base$nuevo) %*% rl$coeficientes),
  drop(predict(en, newx = base$nuevo, s = lambda_final)),
  predict(cart, newdata = as.data.frame(base$nuevo))
)

carpetas <- c("01_linear_regression", "02_elastic_net", "03_cart")
nombres <- c(
  sprintf("linear_regression_ols_full_%d", ncol(x_train)),
  sprintf("elastic_net_alpha_%g_lambda_rel_%g_lambda_%g",
          mejor$alpha, mejor$fraccion_lambda, lambda_final),
  sprintf("cart_cp_%g_maxdepth_%d_minsplit_%d_minhoja_%d",
          cp_cart, profundidad_cart, min_split_cart, min_hoja_cart)
)
if (usar_log) nombres <- paste0(nombres, "_log")
preparacion <- list(mediana = base$mediana, centro = base$centro, escala = base$escala,
                    niveles = niveles, predictores = colnames(x_train), escala_precio = escala_precio,
                    escala_objetivo = escala_objetivo)

for (i in seq_along(modelos)) {
  salida <- file.path(dir_modelos_regresion, carpetas[i])
  dir.create(salida, recursive = TRUE, showWarnings = FALSE)
  pronostico <- a_pesos(predicciones[[i]])
  stopifnot(length(pronostico) == nrow(test), all(is.finite(pronostico)))
  entrega <- data.frame(property_id = plantilla$property_id, price = pronostico[orden_test])
  write_csv(entrega, file.path(salida, paste0(nombres[i], ".csv")))
  saveRDS(list(modelo = modelos[[i]], preparacion = preparacion, familia = carpetas[i],
               lambda = if (i == 2L) lambda_final else NULL),
          file.path(salida, paste0(nombres[i], ".rds")))
}

metricas_rl <- medir(pred_rl, precio)
resumen <- cbind(
  modelo = c("linear_regression", "elastic_net", "cart"),
  archivo = paste0(nombres, ".csv"),
  escala_objetivo = escala_objetivo, criterio_cv = criterio_cv,
  n_predictores = ncol(x_train), n_folds = n_folds,
  lado_bloque_m = lado_bloque_m, semilla = semilla,
  bind_rows(metricas_rl, mejor[names(metricas_rl)],
            medir(pred_cart, precio))
)
resumen <- resumen[order(resumen[[criterio_cv]], resumen$mae_cop), ]
write.xlsx(list(resumen = resumen, elastic_net = ranking,
                predictores = data.frame(predictor = colnames(x_train)),
                folds = data.frame(property_id = train$property_id, bloque = id_bloque, fold = fold)),
           file.path(dir_modelos_regresion, "resumen.xlsx"), overwrite = TRUE)
print(resumen[unique(c("modelo", criterio_cv, "mae_cop"))], row.names = FALSE)
