# ------------------------------------------------------------------------------
# Proyecto: Problem Set 3 - Prediccion de precios de vivienda en Chapinero
# Script:   18_figura_bloques.R
# Objetivo: Figura de una sola imagen con la validacion espacial por bloques de
#           400 m, la que se uso para el XGBoost final. Muestra un fold: los
#           anuncios que se validan, los que entrenan y los cuadros del fold.
# Entradas: 01_data/02_clean/train_planas.csv y test_planas.csv (coordenadas),
#           01_data/01_raw/fuentes_externas/localidades/Loca.shp (localidades de Bogota, IDECA).
# Salida:   03_outputs/mapas/validacion_bloques.png
# Nota:     Los bloques se arman igual que en 14_models_text.R: spatial_block_cv
#           con cellsize 400, sin buffer, tres repartos (semillas 2026, 2027 y
#           2028) en 5 folds. Se dibuja el primer fold de la primera semilla y la
#           distancia mediana usa los 15 folds. No calcula los folds de localidad
#           con buffer, asi que corre en pocos minutos (17_figura_validacion.R
#           tiene los tres paneles).
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Paquetes, rutas y opciones
# ------------------------------------------------------------------------------

library(dplyr)
library(readr)
library(sf)
library(ggplot2)
library(rsample)
library(spatialsample)

if (!exists("root")) root <- "."
dir_clean <- file.path(root, "01_data", "02_clean")
dir_salida <- file.path(root, "03_outputs", "mapas")
archivo_localidades <- file.path(root, "01_data", "01_raw", "fuentes_externas", "localidades", "Loca.shp")
dir.create(dir_salida, recursive = TRUE, showWarnings = FALSE)

tamano_bloque <- 400   # metros, como en 14_models_text.R
n_semillas <- 3
fold_bloques <- 1      # fold que se dibuja (de la primera semilla)
margen <- 2500         # metros alrededor de la caja de puntos_test

# Colores del Problem Set 2
col_navy <- "#1F3864"
col_azul <- "#4F81BD"  # se valida
col_gris <- "#C8CACE"  # entrena
niveles <- c("se valida", "entrena")


# ------------------------------------------------------------------------------
# (2) Importar las bases y el shapefile
# ------------------------------------------------------------------------------

train_planas <- read_csv(file.path(dir_clean, "train_planas.csv"), show_col_types = FALSE,
                         col_select = c(property_id, lat, lon))
test_planas <- read_csv(file.path(dir_clean, "test_planas.csv"), show_col_types = FALSE,
                        col_select = c(property_id, lat, lon))

# El shapefile viene en grados (MAGNA). Todo se trabaja en EPSG 3116 (metros)
localidades <- st_read(archivo_localidades, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(3116)

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

fun_m <- function(.x) format(round(.x), big.mark = ".", decimal.mark = ",", trim = TRUE)

puntos_train <- fun_puntos(train_planas)
puntos_test <- fun_puntos(test_planas)
n_train <- nrow(puntos_train)
chapinero <- st_sf(geometry = st_union(localidades[localidades$LocNombre == "CHAPINERO", ]))


# ------------------------------------------------------------------------------
# (3) Folds de bloques de 400 m y distancia al entrenamiento
# ------------------------------------------------------------------------------

# Tres repartos de bloques con semillas 2026, 2027 y 2028, cada uno en 5 folds
bloques <- lapply(seq_len(n_semillas), function(.s) {
  set.seed(2025 + .s)
  folds_s <- spatial_block_cv(
    puntos_train, v = 5, method = "random", cellsize = tamano_bloque,
    radius = NULL, buffer = NULL
  )
  fold <- integer(n_train)
  for (k in seq_along(folds_s$splits)) fold[complement(folds_s$splits[[k]])] <- k
  stopifnot(all(fold > 0))
  fold
})
puntos_train$fold <- bloques[[1]]

# Distancia mediana de lo que se valida al anuncio de entrenamiento mas cercano,
# sobre los 15 folds, y la de la prueba contra todo el entrenamiento
dist_bloques <- unlist(lapply(bloques, function(.f) {
  unlist(lapply(1:5, function(.k) {
    fun_dist_cercana(puntos_train[.f == .k, ], puntos_train[.f != .k, ])
  }))
}))
dist_test <- fun_dist_cercana(puntos_test, puntos_train)

prob <- c(0.10, 0.25, 0.50, 0.75, 0.90)
cat("\nDistancia al anuncio de entrenamiento mas cercano (m):\n")
print(round(rbind(`Bloques de 400 m (15 folds)` = quantile(dist_bloques, prob),
                  `Prueba (test contra train)` = quantile(dist_test, prob)), 1))


# ------------------------------------------------------------------------------
# (4) Cuadros del fold que se dibuja
# ------------------------------------------------------------------------------

# Grilla de spatial_block_cv: caja de los anuncios ampliada en 1e-5 (relativo)
# por cada lado. Solo se usan los cuadros con anuncios
caja_grilla <- st_bbox(puntos_train)
caja_grilla[1:2] <- caja_grilla[1:2] - abs(caja_grilla[1:2] * 1e-05)
caja_grilla[3:4] <- caja_grilla[3:4] + abs(caja_grilla[3:4] * 1e-05)
grilla <- st_sf(geometry = st_make_grid(caja_grilla, cellsize = tamano_bloque))
grilla$celda <- seq_len(nrow(grilla))
puntos_train$celda <- vapply(st_intersects(puntos_train, grilla),
                             function(.i) .i[1], numeric(1))
cuadros <- puntos_train |>
  st_drop_geometry() |>
  group_by(celda) |>
  summarise(n_folds = n_distinct(fold), fold = first(fold), .groups = "drop")
# Comprobacion: la grilla reproduce los folds (cada cuadro, un solo fold)
stopifnot(all(cuadros$n_folds == 1))
cuadros_con_anuncios <- grilla[grilla$celda %in% cuadros$celda, ]
cuadros_validados <- grilla[grilla$celda %in% cuadros$celda[cuadros$fold == fold_bloques], ]


# ------------------------------------------------------------------------------
# (5) Figura
# ------------------------------------------------------------------------------

caja_test <- st_bbox(puntos_test)
xlim <- c(caja_test[["xmin"]] - margen, caja_test[["xmax"]] + margen)
ylim <- c(caja_test[["ymin"]] - margen, caja_test[["ymax"]] + margen)

fun_rol <- function(.idx, .rol) {
  x <- puntos_train[.idx, "property_id"]
  x$rol <- factor(.rol, levels = niveles)
  x
}
val <- which(puntos_train$fold == fold_bloques)
ent <- setdiff(seq_len(n_train), val)

# Barra de escala de 1 km abajo a la derecha
x_barra <- xlim[2] - diff(xlim) * 0.08 - 1000
y_barra <- ylim[1] + diff(ylim) * 0.04

g <- ggplot() +
  geom_sf(data = cuadros_con_anuncios, fill = NA, color = "#E6E7EA", linewidth = 0.15) +
  geom_sf(data = cuadros_validados, fill = alpha(col_azul, 0.2), color = col_azul,
          linewidth = 0.3) +
  geom_sf(data = fun_rol(ent, "entrena"), aes(color = rol), size = 0.3, alpha = 0.8) +
  geom_sf(data = fun_rol(val, "se valida"), aes(color = rol), size = 0.6, alpha = 0.9) +
  geom_sf(data = chapinero, aes(linetype = "límite de Chapinero"), fill = NA,
          color = col_navy, linewidth = 0.6, show.legend = "line") +
  scale_linetype_manual(values = "dashed", name = NULL) +
  scale_color_manual(values = c(col_azul, col_gris), limits = niveles, name = NULL,
                     guide = guide_legend(
                       override.aes = list(size = 3.5, alpha = 1, linetype = 0))) +
  annotate("segment", x = x_barra, xend = x_barra + 1000, y = y_barra, yend = y_barra,
           linewidth = 1.2, color = "#333333") +
  annotate("text", x = x_barra + 500, y = y_barra + diff(ylim) * 0.025, label = "1 km",
           size = 4, color = "#333333") +
  coord_sf(xlim = xlim, ylim = ylim, expand = FALSE, datum = NA) +
  labs(title = "Bloques de 400 m",
       caption = paste0("Distancia mediana al entrenamiento: ", fun_m(median(dist_bloques)),
                        " m (prueba: ", fun_m(median(dist_test)), " m)")) +
  theme_void(base_size = 14) +
  theme(plot.title = element_text(color = col_navy, face = "bold", hjust = 0.5,
                                  size = 16, margin = margin(b = 6)),
        plot.caption = element_text(color = "#333333", hjust = 0.5, size = 11,
                                    margin = margin(t = 6)),
        plot.caption.position = "panel",
        panel.border = element_rect(color = col_gris, fill = NA, linewidth = 0.4),
        legend.position = "bottom",
        legend.text = element_text(size = 13),
        plot.margin = margin(6, 8, 6, 8),
        plot.background = element_rect(fill = "white", color = NA))

# El alto sale de la forma de la ventana, mas espacio para titulo, texto y leyenda
ancho <- 5
alto <- ancho * diff(ylim) / diff(xlim) + 1.5
ggsave(file.path(dir_salida, "validacion_bloques.png"), g,
       width = ancho, height = alto, dpi = 300, bg = "white")

cat("Figura guardada en", file.path(dir_salida, "validacion_bloques.png"), "\n")
