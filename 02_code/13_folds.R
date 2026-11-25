# ------------------------------------------------------------------------------
# Archivo: 13_folds.R
# Proyecto: Taller 3 - Big Data y Machine Learning
#
# Descripcion:
#   Crear UN solo archivo de folds que usan todos los modelos (R y Python) y los
#   scripts de texto (04 y 05). Asi la validacion es identica en todos los pasos,
#   las comparaciones son justas y se evita la filtracion entre el modelo de texto
#   y el modelo final.
#   Hay dos esquemas, sobre los mismos anuncios unicos de train_planas.csv:
#     - fold_espacial: bloques cuadrados de tamano_bloque metros asignados al
#       azar a 5 folds, sin buffer (el esquema de 05_modelos.R, seccion 12).
#     - fold_estandar: 5 folds al azar. Solo sirve para mostrar cuanto se
#       equivoca la validacion que ignora la geografia.
# Input:
#   01_data/02_clean/train_planas.csv (salida de 02_variables_planas.R).
# Output:
#   01_data/02_clean/folds.csv con property_id, fold_espacial y fold_estandar.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Paquetes y rutas
# ------------------------------------------------------------------------------

library(dplyr)
library(readr)
library(sf)
library(rsample)
library(spatialsample)

if (!exists("root")) root <- "."
dir_clean <- file.path(root, "01_data", "02_clean")

# Mismo tamano de bloque de 05_modelos.R (la mediana de la distancia de lo que se
# valida al entrenamiento se parece a la del test)
tamano_bloque <- 400
n_folds <- 5

# ------------------------------------------------------------------------------
# (2) Folds espaciales por bloques
# ------------------------------------------------------------------------------

train_planas <- read_csv(file.path(dir_clean, "train_planas.csv"), show_col_types = FALSE)

puntos <- train_planas |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
  st_transform(3116)

set.seed(2026)
folds_bloques <- spatial_block_cv(
  puntos, v = n_folds, method = "random", cellsize = tamano_bloque,
  radius = NULL, buffer = NULL
)

# complement() devuelve las filas que se validan en cada split
fold_espacial <- integer(nrow(train_planas))
for (k in seq_along(folds_bloques$splits)) {
  fold_espacial[complement(folds_bloques$splits[[k]])] <- k
}
stopifnot(all(fold_espacial > 0))

# ------------------------------------------------------------------------------
# (3) Folds estandar (al azar)
# ------------------------------------------------------------------------------

set.seed(2026)
fold_estandar <- sample(rep(seq_len(n_folds), length.out = nrow(train_planas)))

# ------------------------------------------------------------------------------
# (4) Exportar
# ------------------------------------------------------------------------------

folds <- tibble(property_id = train_planas$property_id,
                fold_espacial = fold_espacial,
                fold_estandar = fold_estandar)
write_csv(folds, file.path(dir_clean, "folds.csv"))

print(count(folds, fold_espacial))
