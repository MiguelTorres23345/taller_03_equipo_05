# ==============================================================================
# Archivo: 15_ablacion_fuentes.R
# Proyecto: Problem Set 3 - Prediccion de precios de vivienda en Chapinero
# Descripcion:
#   Ablacion por fuentes de informacion con el MEJOR modelo (XGBoost con
#   nrounds 1000, max_depth 8, eta 0.01 y min_child_weight 20; ver
#   03_outputs/04_modelos_texto). Los hiperparametros quedan fijos para que la
#   unica diferencia entre pasos sea el conjunto de variables.
#   Se empieza solo con las variables de la limpieza y se agrega una fuente por
#   paso, de forma acumulada:
#     0. Limpieza        (area, habitaciones, banos, tipo, mes, anio)
#     1. + Coordenadas   (coord_x, coord_y)
#     2. + Estratificacion (estrato, dist_est5, dist_est6)
#     3. + Valor de referencia por m2
#     4. + Uso del suelo (uso, prop_comercio)
#     5. + Seguridad     (incidentes)
#     6. + TransMilenio  (dist_tm)
#     7. + OpenStreetMap (parque, restaurante, universidad, financiero, avenida)
#     8. + Texto         (TF-IDF y MiniLM)
#     9. + Texto fine-tuning (txt_ft_minilm_lnp)
#   Opcionalmente (individual <- TRUE) tambien mide cada fuente sola sobre la
#   limpieza, para ver su aporte sin depender del orden.
#   Todos los pasos usan los mismos folds espaciales (folds.csv de 13_folds.R).
# Input:
#   train_planas.csv, folds.csv, variables de texto y Loca.shp.
# Output:
#   03_outputs/06_ablacion/ablacion_fuentes.csv y ablacion_fuentes.png.
# ==============================================================================


# ------------------------------------------------------------------------------
# (1) Paquetes, rutas y opciones
# ------------------------------------------------------------------------------

library(dplyr)
library(readr)
library(sf)
library(xgboost)
library(ggplot2)

if (!exists("root")) root <- "."
dir_clean <- file.path(root, "01_data", "02_clean")
dir_text <- file.path(root, "01_data", "02_text")
dir_salida <- file.path(root, "03_outputs", "06_ablacion")
archivo_localidades <- file.path(root, "01_data", "01_raw", "fuentes_externas", "localidades", "Loca.shp")
dir.create(dir_salida, recursive = TRUE, showWarnings = FALSE)

if (!exists("muestra")) muestra <- Inf          # anuncios de train (Inf = todos)
if (!exists("individual")) individual <- TRUE   # TRUE: tambien cada fuente sola
if (!exists("n_hilos")) n_hilos <- max(1, min(8, parallel::detectCores() - 2))

# Hiperparametros del mejor modelo (no se ajustan aqui). Con muestra pequena
# se reducen las rondas solo para probar que el codigo corre
parametros <- list(objective = "reg:squarederror", eta = 0.01, max_depth = 8,
                   min_child_weight = 20, subsample = 0.8, colsample_bytree = 0.8,
                   nthread = n_hilos)
n_rondas <- if (is.finite(muestra)) 50 else 1000

# Variables de cada fuente
fuentes <- list(
  "Coordenadas" = c("coord_x", "coord_y"),
  "Estratificacion" = c("estrato", "dist_est5", "dist_est6"),
  "Valor de referencia" = "valor_ref",
  "Uso del suelo" = c("uso", "prop_comercio"),
  "Seguridad" = "incidentes",
  "TransMilenio" = "dist_tm",
  "OpenStreetMap" = c("dist_parque", "dist_restaurante", "dist_universidad",
                      "dist_financiero", "dist_avenida")
)
vars_limpieza <- c("surface_covered", "bedrooms", "bathrooms", "property_type",
                   "month", "year")


# ------------------------------------------------------------------------------
# (2) Importar y preparar las bases
# ------------------------------------------------------------------------------

train_planas <- read_csv(file.path(dir_clean, "train_planas.csv"), show_col_types = FALSE)
folds <- read_csv(file.path(dir_clean, "folds.csv"), show_col_types = FALSE)

texto <- read_csv(file.path(dir_text, "text_features_train.csv"), show_col_types = FALSE) |>
  left_join(read_csv(file.path(dir_text, "text_finetune_minilm_train.csv"),
                     show_col_types = FALSE), by = "property_id")
vars_texto_ft <- "txt_ft_minilm_lnp"
vars_texto <- setdiff(names(texto), c("property_id", vars_texto_ft))

train_planas <- train_planas |>
  inner_join(folds, by = "property_id") |>
  left_join(texto, by = "property_id")

# Coordenadas en metros (EPSG 3116) y localidad (solo para reportar Chapinero)
puntos <- train_planas |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
  st_transform(3116)
train_planas$coord_x <- st_coordinates(puntos)[, 1]
train_planas$coord_y <- st_coordinates(puntos)[, 2]
localidades <- st_read(archivo_localidades, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(3116)
train_planas$localidad <- localidades$LocNombre[st_nearest_feature(puntos, localidades)]

if (is.finite(muestra)) {
  set.seed(2026)
  train_planas <- slice_sample(train_planas, n = muestra)
}

# Vacios: mediana (numericas) o moda (categoricas)
vars_todas <- c(vars_limpieza, unlist(fuentes), vars_texto, vars_texto_ft)
for (v in vars_todas) {
  if (is.numeric(train_planas[[v]])) {
    train_planas[[v]][is.na(train_planas[[v]])] <- median(train_planas[[v]], na.rm = TRUE)
  } else {
    moda <- names(which.max(table(train_planas[[v]])))
    train_planas[[v]][is.na(train_planas[[v]])] <- moda
    train_planas[[v]] <- factor(train_planas[[v]])
  }
}
precio <- train_planas$price / 1e6   # millones de pesos


# ------------------------------------------------------------------------------
# (3) Funcion: MAE con validacion espacial para un conjunto de variables
# ------------------------------------------------------------------------------

fun_mae_cv <- function(.vars) {
  x <- model.matrix(~ . - 1, data = train_planas[, .vars, drop = FALSE])
  pred <- rep(NA_real_, nrow(x))
  for (k in sort(unique(train_planas$fold_espacial))) {
    dentro <- which(train_planas$fold_espacial != k)
    fuera <- which(train_planas$fold_espacial == k)
    modelo <- xgb.train(params = parametros,
                        data = xgb.DMatrix(x[dentro, , drop = FALSE], label = precio[dentro]),
                        nrounds = n_rondas, verbose = 0)
    pred[fuera] <- predict(modelo, x[fuera, , drop = FALSE])
  }
  error <- pred - precio
  chap <- train_planas$localidad == "CHAPINERO"
  tibble(n_variables = length(.vars),
         MAE = mean(abs(error)),
         MAE_chapinero = mean(abs(error[chap])),
         error_medio = mean(error),
         pct_sobreestima = 100 * mean(error > 0))
}


# ------------------------------------------------------------------------------
# (4) Ablacion acumulada: se agrega una fuente por paso
# ------------------------------------------------------------------------------

pasos <- c(fuentes, list("Texto (TF-IDF y MiniLM)" = vars_texto,
                         "Texto fine-tuning" = vars_texto_ft))

resultados <- list()
vars_actuales <- vars_limpieza
message("\n>>> 0. Solo limpieza")
resultados[[1]] <- fun_mae_cv(vars_actuales) |>
  mutate(tipo = "acumulado", paso = 0, fuente = "Solo limpieza")
print(resultados[[1]])

for (i in seq_along(pasos)) {
  vars_actuales <- c(vars_actuales, pasos[[i]])
  message("\n>>> ", i, ". + ", names(pasos)[i])
  resultados[[length(resultados) + 1]] <- fun_mae_cv(vars_actuales) |>
    mutate(tipo = "acumulado", paso = i, fuente = paste("+", names(pasos)[i]))
  print(resultados[[length(resultados)]])
}


# ------------------------------------------------------------------------------
# (5) Cada fuente sola sobre la limpieza (opcional)
# ------------------------------------------------------------------------------

if (individual) {
  for (i in seq_along(pasos)) {
    message("\n>>> Individual: limpieza + ", names(pasos)[i])
    resultados[[length(resultados) + 1]] <- fun_mae_cv(c(vars_limpieza, pasos[[i]])) |>
      mutate(tipo = "individual", paso = i, fuente = names(pasos)[i])
  }
}


# ------------------------------------------------------------------------------
# (6) Tabla y grafico
# ------------------------------------------------------------------------------

tabla <- bind_rows(resultados)
# Mejora (%) del MAE respecto a usar solo la limpieza
ref <- tabla$MAE[tabla$fuente == "Solo limpieza"]
tabla <- tabla |>
  mutate(mejora_vs_limpieza = 100 * (ref - MAE) / ref) |>
  select(tipo, paso, fuente, n_variables, MAE, MAE_chapinero, error_medio,
         pct_sobreestima, mejora_vs_limpieza)
write_csv(tabla, file.path(dir_salida, "ablacion_fuentes.csv"))
print(tabla, width = Inf)

acum <- filter(tabla, tipo == "acumulado") |>
  mutate(fuente = factor(fuente, levels = rev(fuente)))
g <- ggplot(acum, aes(MAE, fuente)) +
  geom_col(fill = "#2b6c8f") +
  geom_text(aes(label = round(MAE, 1)), hjust = 1.1, colour = "white", size = 3.5) +
  labs(x = "MAE de la validacion espacial (millones de pesos)", y = NULL,
       title = "Aporte de cada fuente de informacion (acumulado)") +
  theme_minimal()
ggsave(file.path(dir_salida, "ablacion_fuentes.png"), g, width = 8, height = 5, dpi = 150)
