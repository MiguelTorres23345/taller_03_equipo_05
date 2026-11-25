# ------------------------------------------------------------------------------
# Proyecto: Problem Set 3 - Prediccion de precios de vivienda en Chapinero
# Script:   17_figura_validacion.R
# Objetivo: Figura de una diapositiva que explica por que se usa validacion
#           espacial por bloques de 400 m. Tres paneles con el mismo encuadre:
#           folds al azar, una localidad con buffer de 300 m y bloques de 400 m.
#           Cada panel dibuja UNA ronda y reporta, sobre todos los folds, la
#           mediana de la distancia de lo que se valida al anuncio de
#           entrenamiento mas cercano, junto a la de la prueba.
# Entradas: 01_data/02_clean/train_planas.csv y test_planas.csv (coordenadas),
#           01_data/02_clean/folds.csv (folds al azar y por bloques de 13_folds.R),
#           01_data/01_raw/fuentes_externas/localidades/Loca.shp (localidades de Bogota, IDECA).
# Salida:   03_outputs/mapas/validacion_espacial.png
# Nota:     No cambia nada del modelo. Los objetos (puntos_train, puntos_test y
#           los folds de localidad con buffer) se reconstruyen con el mismo
#           codigo de 05_modelos.R. Los folds de localidad con buffer tardan
#           unos 4 minutos en generarse.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Paquetes, rutas y opciones
# ------------------------------------------------------------------------------

library(dplyr)
library(readr)
library(sf)
library(ggplot2)
library(patchwork)
library(rsample)
library(spatialsample)

if (!exists("root")) root <- "."
dir_clean <- file.path(root, "01_data", "02_clean")
dir_salida <- file.path(root, "03_outputs", "mapas")
archivo_localidades <- file.path(root, "01_data", "01_raw", "fuentes_externas", "localidades", "Loca.shp")
dir.create(dir_salida, recursive = TRUE, showWarnings = FALSE)

# Mismos valores de 05_modelos.R y 13_folds.R
distancia_buffer <- 300
tamano_bloque <- 400

# Margen (metros) alrededor de la caja de puntos_test
margen <- 2500

# Fold que se dibuja en los paneles de folds al azar y de bloques
fold_azar <- 1
fold_bloques <- 1

# Colores del Problem Set 2
col_navy <- "#1F3864"
col_azul <- "#4F81BD"             # RGB(79,129,189): se valida
col_gris <- "#C8CACE"             # RGB(200,202,206): entrena
col_rojo <- "#B5473A"             # RGB(181,71,58): se excluye por el buffer
niveles <- c("se valida", "entrena", "se excluye (buffer)")


# ------------------------------------------------------------------------------
# (2) Importar las bases, el shapefile y los folds (como en 05_modelos.R)
# ------------------------------------------------------------------------------

train_planas <- read_csv(file.path(dir_clean, "train_planas.csv"), show_col_types = FALSE,
                         col_select = c(property_id, lat, lon))
test_planas <- read_csv(file.path(dir_clean, "test_planas.csv"), show_col_types = FALSE,
                        col_select = c(property_id, lat, lon))
folds <- read_csv(file.path(dir_clean, "folds.csv"), show_col_types = FALSE)

# El shapefile viene en grados (MAGNA). Todo se trabaja en EPSG 3116 (metros)
localidades <- st_read(archivo_localidades, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(3116)

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

puntos_train <- fun_puntos(train_planas)
puntos_test <- fun_puntos(test_planas)
n_train <- nrow(puntos_train)

# Folds al azar (13_folds.R: semilla 2026, mismo orden de filas)
puntos_train <- puntos_train |> left_join(folds, by = "property_id")
stopifnot(!anyNA(puntos_train$fold_estandar), !anyNA(puntos_train$fold_espacial))

# Bloques de 400 m como en 14_models_text.R (el modelo final de XGBoost): tres
# repartos con semillas 2026, 2027 y 2028, cada uno en 5 folds (15 folds)
n_semillas <- 3
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
# La primera semilla (2026) es el reparto que guarda 13_folds.R en folds.csv
cat("Primera semilla igual a folds.csv:",
    all(bloques[[1]] == puntos_train$fold_espacial), "\n")
# El fold que se dibuja sale de la primera semilla
puntos_train$fold_espacial <- bloques[[1]]

# Localidad mas cercana a cada anuncio y folds de localidad con buffer
puntos_train$localidad <- localidades$LocNombre[st_nearest_feature(puntos_train, localidades)]
set.seed(2026)
folds_loc_buffer <- spatial_leave_location_out_cv(
  puntos_train, group = localidad, buffer = distancia_buffer
)


# ------------------------------------------------------------------------------
# (3) Distancia al anuncio de entrenamiento mas cercano, sobre todos los folds
# ------------------------------------------------------------------------------

# Folds al azar y por bloques: lo que se valida contra lo que entrena
fun_dist_folds <- function(.columna) {
  unlist(lapply(sort(unique(puntos_train[[.columna]])), function(.k) {
    fun_dist_cercana(puntos_train[puntos_train[[.columna]] == .k, ],
                     puntos_train[puntos_train[[.columna]] != .k, ])
  }))
}
dist_azar <- fun_dist_folds("fold_estandar")

# Bloques: los 15 folds del modelo final (3 semillas). Tambien se guarda la
# primera semilla sola, que es la que imprime el log de 14_models_text.R
dist_bloques_semilla <- lapply(bloques, function(.f) {
  unlist(lapply(1:5, function(.k) {
    fun_dist_cercana(puntos_train[.f == .k, ], puntos_train[.f != .k, ])
  }))
})
dist_bloques <- unlist(dist_bloques_semilla)

# Localidad con buffer: out_id es lo que se valida e in_id lo que entrena
dist_localidad <- unlist(lapply(folds_loc_buffer$splits, function(.s) {
  fun_dist_cercana(puntos_train[.s$out_id, ], puntos_train[.s$in_id, ])
}))

# Referencia: cada anuncio de test contra todo train
dist_test <- fun_dist_cercana(puntos_test, puntos_train)

prob <- c(0.10, 0.25, 0.50, 0.75, 0.90)
resumen <- rbind(
  `Folds al azar` = quantile(dist_azar, prob),
  `Una localidad + 300 m` = quantile(dist_localidad, prob),
  `Bloques de 400 m (15 folds)` = quantile(dist_bloques, prob),
  `Bloques, solo semilla 2026` = quantile(dist_bloques_semilla[[1]], prob),
  `Prueba (test contra train)` = quantile(dist_test, prob))
cat("\nDistancia al anuncio de entrenamiento mas cercano (m):\n")
print(round(resumen, 1))

fun_m <- function(.x) format(round(.x), big.mark = ".", decimal.mark = ",", trim = TRUE)
med_test <- median(dist_test)
fun_texto <- function(.d) {
  paste0("Distancia mediana al entrenamiento: ", fun_m(median(.d)), " m")
}


# ------------------------------------------------------------------------------
# (4) Encuadre, limite de Chapinero y rotulo
# ------------------------------------------------------------------------------

caja_test <- st_bbox(puntos_test)
xlim <- c(caja_test[["xmin"]] - margen, caja_test[["xmax"]] + margen)
ylim <- c(caja_test[["ymin"]] - margen, caja_test[["ymax"]] + margen)

chapinero <- st_sf(geometry = st_union(localidades[localidades$LocNombre == "CHAPINERO", ]))

# Parte de los anuncios de test que cae dentro de Chapinero, para la referencia
pct_test_chapinero <- 100 * mean(lengths(st_intersects(puntos_test, chapinero)) > 0)
texto_referencia <- paste0("Referencia: anuncios de test (", fun_m(pct_test_chapinero),
                           " % en Chapinero) a ", fun_m(med_test),
                           " m del entrenamiento más cercano")

# El rotulo va en la parte de cerros de Chapinero (al este de la prueba, casi sin
# anuncios), para no tapar puntos. Si no hay, va en un punto interior
fun_caja <- function(.x, .y) {
  .x <- unname(.x)
  .y <- unname(.y)
  st_as_sfc(st_bbox(c(xmin = .x[1], xmax = .x[2], ymin = .y[1], ymax = .y[2]),
                    crs = st_crs(3116)))
}
x_cerros <- quantile(st_coordinates(puntos_test)[, 1], 0.99) + 300
zona_cerros <- suppressWarnings(st_intersection(
  st_intersection(st_geometry(chapinero), fun_caja(xlim, ylim)),
  fun_caja(c(x_cerros, xlim[2]), ylim)))
if (length(zona_cerros) == 0 || all(st_is_empty(zona_cerros))) {
  zona_cerros <- suppressWarnings(
    st_intersection(st_geometry(chapinero), fun_caja(xlim, ylim)))
}
rotulo <- st_sf(geometry = st_point_on_surface(st_union(zona_cerros)))
# Se corre 1,5 km al este para que no roce los anuncios del borde
st_geometry(rotulo) <- st_geometry(rotulo) + c(1500, 0)
st_crs(rotulo) <- st_crs(3116)


# ------------------------------------------------------------------------------
# (5) Una ronda de cada validacion
# ------------------------------------------------------------------------------

todos <- seq_len(n_train)

# Panel 1: un fold al azar
val_azar <- which(puntos_train$fold_estandar == fold_azar)
ent_azar <- setdiff(todos, val_azar)

# Panel 2: el fold donde se valida Chapinero. Se busca el fold cuyo out_id es
# toda la localidad de Chapinero
es_chapinero <- vapply(folds_loc_buffer$splits, function(.s) {
  length(.s$out_id) > 0 && all(puntos_train$localidad[.s$out_id] == "CHAPINERO")
}, logical(1))
stopifnot(sum(es_chapinero) == 1)
split_chap <- folds_loc_buffer$splits[[which(es_chapinero)]]
val_loc <- split_chap$out_id
ent_loc <- split_chap$in_id
exc_loc <- setdiff(todos, c(val_loc, ent_loc))
cat("\nFold de Chapinero: valida", length(val_loc), "| entrena", length(ent_loc),
    "| excluidos por el buffer", length(exc_loc), "\n")

# Panel 3: un fold de bloques de 400 m
val_bloques <- which(puntos_train$fold_espacial == fold_bloques)
ent_bloques <- setdiff(todos, val_bloques)

# Cuadros de 400 m (grilla de spatial_block_cv: caja de los anuncios ampliada en
# 1e-5 relativo por cada lado). Solo se usan los cuadros con anuncios
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
  summarise(n_folds = n_distinct(fold_espacial), fold = first(fold_espacial),
            .groups = "drop")
# Comprobacion: la grilla reproduce los folds (cada cuadro, un solo fold)
stopifnot(all(cuadros$n_folds == 1))
cuadros_con_anuncios <- grilla[grilla$celda %in% cuadros$celda, ]
cuadros_validados <- grilla[grilla$celda %in% cuadros$celda[cuadros$fold == fold_bloques], ]


# ------------------------------------------------------------------------------
# (6) Funcion de panel y figura
# ------------------------------------------------------------------------------

# Puntos con su rol como factor (todos los niveles, para una leyenda unica)
fun_rol <- function(.idx, .rol) {
  x <- puntos_train[.idx, "property_id"]
  x$rol <- factor(.rol, levels = niveles)
  x
}

# .capas_previas: cuadros u otras capas que van debajo de los puntos
fun_panel <- function(.val, .ent, .exc = integer(0), .titulo, .dist,
                      .capas_previas = list(), .rotular = FALSE) {
  g <- ggplot()
  for (capa in .capas_previas) g <- g + capa
  g <- g +
    geom_sf(data = fun_rol(.ent, "entrena"), aes(color = rol), size = 0.3, alpha = 0.8)
  if (length(.exc) > 0) {
    g <- g + geom_sf(data = fun_rol(.exc, "se excluye (buffer)"), aes(color = rol),
                     size = 0.6, alpha = 0.9)
  }
  g <- g +
    geom_sf(data = fun_rol(.val, "se valida"), aes(color = rol), size = 0.6, alpha = 0.9) +
    # Limite de Chapinero con entrada propia en la leyenda
    geom_sf(data = chapinero, aes(linetype = "límite de Chapinero"), fill = NA,
            color = col_navy, linewidth = 0.6, show.legend = "line") +
    scale_linetype_manual(values = "dashed", name = NULL) +
    scale_color_manual(values = c(col_azul, col_gris, col_rojo), limits = niveles,
                       drop = FALSE, name = NULL,
                       guide = guide_legend(
                         # linetype = 0 evita que la raya del limite de Chapinero
                         # aparezca dentro de las llaves de los puntos
                         override.aes = list(size = 3.5, alpha = 1, linetype = 0))) +
    coord_sf(xlim = xlim, ylim = ylim, expand = FALSE, datum = NA) +
    labs(title = .titulo, caption = fun_texto(.dist)) +
    theme_void(base_size = 14) +
    theme(plot.title = element_text(color = col_navy, face = "bold", hjust = 0.5,
                                    size = 15, margin = margin(b = 6)),
          plot.caption = element_text(color = "#333333", hjust = 0.5, size = 10.5,
                                      margin = margin(t = 6)),
          plot.caption.position = "panel",
          panel.border = element_rect(color = col_gris, fill = NA, linewidth = 0.4),
          legend.text = element_text(size = 13),
          plot.background = element_rect(fill = "white", color = NA))
  # El rotulo "Chapinero" solo va en el primer panel
  if (.rotular) {
    g <- g + geom_sf_text(data = rotulo, label = "Chapinero", color = col_navy,
                          fontface = "bold", size = 4.2)
  }
  g
}

capas_bloques <- list(
  geom_sf(data = cuadros_con_anuncios, fill = NA, color = "#E6E7EA", linewidth = 0.15),
  geom_sf(data = cuadros_validados, fill = alpha(col_azul, 0.2), color = col_azul,
          linewidth = 0.3))

# La leyenda sale solo del panel 2, que es el unico con los tres roles (los
# paneles 1 y 3 no tienen anuncios excluidos y su leyenda no coincidiria)
p1 <- fun_panel(val_azar, ent_azar, .titulo = "Folds al azar", .dist = dist_azar,
                .rotular = TRUE) +
  guides(color = "none")
p2 <- fun_panel(val_loc, ent_loc, exc_loc, .titulo = "Una localidad + 300 m",
                .dist = dist_localidad)

# Barra de escala de 1 km abajo a la derecha del panel 3
x_barra <- xlim[2] - diff(xlim) * 0.08 - 1000
y_barra <- ylim[1] + diff(ylim) * 0.04
p3 <- fun_panel(val_bloques, ent_bloques, .titulo = "Bloques de 400 m",
                .dist = dist_bloques, .capas_previas = capas_bloques) +
  annotate("segment", x = x_barra, xend = x_barra + 1000, y = y_barra, yend = y_barra,
           linewidth = 1.2, color = "#333333") +
  annotate("text", x = x_barra + 500, y = y_barra + diff(ylim) * 0.025, label = "1 km",
           size = 4, color = "#333333") +
  guides(color = "none")

# Referencia (distancia de la prueba) y leyenda unica debajo de los tres paneles
fila_referencia <- wrap_elements(
  grid::textGrob(texto_referencia,
                 gp = grid::gpar(col = col_navy, fontface = "bold", fontsize = 13)))

figura <- p1 + p2 + p3 + fila_referencia + guide_area() +
  plot_layout(design = "ABC\nDDD\nEEE", heights = c(1, 0.07, 0.09), guides = "collect") &
  theme(legend.position = "bottom", legend.box = "horizontal")

ggsave(file.path(dir_salida, "validacion_espacial.png"), figura,
       width = 12, height = 6.4, dpi = 300, bg = "white")

cat("Figura guardada en", file.path(dir_salida, "validacion_espacial.png"), "\n")
