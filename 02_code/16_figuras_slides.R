# ------------------------------------------------------------------------------
# Proyecto: Problem Set 3 - Prediccion de precios de vivienda en Chapinero
# Script:   16_figuras_slides.R
# Objetivo: Generar las figuras de las presentaciones (best y compare) con el
#           mismo estilo del Problem Set 2 (ggplot, azul marino y azul).
# Entradas: 01_data/02_clean/train_clean.csv y test_clean.csv (coordenadas),
#           01_data/01_raw/fuentes_externas/localidades/Loca.shp (localidades de Bogota, IDECA).
# Salidas:  03_outputs/figuras_slides/*.png
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Paquetes, rutas y opciones
# ------------------------------------------------------------------------------

library(dplyr)
library(readr)
library(sf)
library(ggplot2)

if (!exists("root")) root <- "."
dir_clean <- file.path(root, "01_data", "02_clean")
dir_salida <- file.path(root, "03_outputs", "figuras_slides")
archivo_localidades <- file.path(root, "01_data", "01_raw", "fuentes_externas", "localidades", "Loca.shp")
dir.create(dir_salida, recursive = TRUE, showWarnings = FALSE)

# Colores (azul marino y azul del Problem Set 2, mas un naranja para la prueba)
col_navy <- "#1F3864"
col_azul <- "#4F81BD"
col_prueba <- "#E8833A"
col_vecinas <- "#D9D9D9"   # relleno gris de las localidades
col_borde <- "white"       # borde de las localidades
col_chapinero <- "#BFD0E8" # relleno de Chapinero

# Tema limpio para lamina: sin ejes ni cuadricula, leyenda abajo
tema_slides <- theme_void(base_size = 16) +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(color = "#333333", size = 15),
        legend.margin = margin(t = 6),
        plot.margin = margin(4, 4, 4, 4),
        plot.background = element_rect(fill = "white", color = NA))

fun_miles <- function(.n) format(.n, big.mark = ".", decimal.mark = ",", trim = TRUE)


# ------------------------------------------------------------------------------
# (2) Importar las bases y el shapefile (todo en metros, MAGNA-SIRGAS Bogota)
# ------------------------------------------------------------------------------

fun_puntos <- function(.archivo, .base) {
  read_csv(file.path(dir_clean, .archivo), show_col_types = FALSE,
           col_select = c(property_id, lat, lon)) |>
    mutate(base = .base)
}

anuncios <- bind_rows(fun_puntos("train_clean.csv", "Entrenamiento"),
                      fun_puntos("test_clean.csv", "Prueba")) |>
  filter(!is.na(lat), !is.na(lon)) |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326) |>
  st_transform(3116)

n_base <- table(anuncios$base)
anuncios <- anuncios |>
  mutate(base = factor(base, levels = c("Entrenamiento", "Prueba"),
                       labels = c(paste0("Entrenamiento (", fun_miles(n_base[["Entrenamiento"]]), ")"),
                                  paste0("Prueba (", fun_miles(n_base[["Prueba"]]), ")")))) |>
  arrange(desc(base))   # la prueba se dibuja encima

localidades <- st_read(archivo_localidades, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(3116) |>
  mutate(es_chapinero = LocNombre == "CHAPINERO")

cat("Anuncios por base:\n")
print(n_base)


# ------------------------------------------------------------------------------
# (3) Funcion del mapa
# ------------------------------------------------------------------------------

# .xlim, .ylim: ventana en metros. .tamano: tamano de los puntos.
fun_mapa <- function(.xlim, .ylim, .tamano) {
  .xlim <- unname(.xlim)
  .ylim <- unname(.ylim)
  ventana <- st_as_sfc(st_bbox(c(xmin = .xlim[1], xmax = .xlim[2],
                                 ymin = .ylim[1], ymax = .ylim[2]), crs = st_crs(3116)))

  # Barra de escala de 1 km abajo a la derecha (zona vacia)
  x0 <- .xlim[2] - diff(.xlim) * 0.05 - 1000
  y0 <- .ylim[1] + diff(.ylim) * 0.05

  g <- ggplot() +
    geom_sf(data = filter(localidades, !es_chapinero),
            fill = col_vecinas, color = col_borde, linewidth = 0.5) +
    geom_sf(data = filter(localidades, es_chapinero),
            fill = col_chapinero, color = col_azul, linewidth = 0.5) +
    geom_sf(data = anuncios, aes(color = base), size = .tamano, alpha = 0.6) +
    scale_color_manual(values = setNames(c(col_navy, col_prueba), levels(anuncios$base)),
                       guide = guide_legend(override.aes = list(size = 4.5, alpha = 1))) +
    annotate("segment", x = x0, xend = x0 + 1000, y = y0, yend = y0,
             linewidth = 1.2, color = "#333333") +
    annotate("text", x = x0 + 500, y = y0 + diff(.ylim) * 0.025, label = "1 km",
             size = 4.2, color = "#333333") +
    coord_sf(xlim = .xlim, ylim = .ylim, expand = FALSE, datum = NA) +
    tema_slides
  g
}

fun_guardar <- function(.g, .nombre, .xlim, .ylim, .ancho) {
  # El alto sale de la forma de la ventana, mas espacio para la leyenda
  alto <- .ancho * diff(.ylim) / diff(.xlim) + 0.9
  ggsave(file.path(dir_salida, .nombre), .g,
         width = .ancho, height = alto, dpi = 300, bg = "white")
}


# ------------------------------------------------------------------------------
# (4) Mapas: entrenamiento y prueba
# ------------------------------------------------------------------------------

# (a) Bogota completa: donde hay anuncios, con un margen
caja <- st_bbox(anuncios)
xlim_a <- c(caja["xmin"] - diff(c(caja["xmin"], caja["xmax"])) * 0.03,
            caja["xmax"] + diff(c(caja["xmin"], caja["xmax"])) * 0.03)
ylim_a <- c(caja["ymin"] - diff(c(caja["ymin"], caja["ymax"])) * 0.03,
            caja["ymax"] + diff(c(caja["ymin"], caja["ymax"])) * 0.03)
mapa <- fun_mapa(xlim_a, ylim_a, 0.35)
fun_guardar(mapa, "mapa_train_test.png", xlim_a, ylim_a, 7)

# (b) Zoom en Chapinero y sus alrededores: la ventana sale de la zona de prueba
#     (percentiles 1 y 99 para que unos pocos anuncios lejanos no la estiren)
coord_test <- st_coordinates(filter(anuncios, startsWith(as.character(base), "Prueba")))
lim_x <- quantile(coord_test[, 1], c(0.01, 0.99))
lim_y <- quantile(coord_test[, 2], c(0.01, 0.99))
xlim_b <- c(lim_x[[1]] - diff(lim_x) * 0.6, lim_x[[2]] + diff(lim_x) * 0.35)
ylim_b <- c(lim_y[[1]] - diff(lim_y) * 0.08, lim_y[[2]] + diff(lim_y) * 0.2)
mapa_zoom <- fun_mapa(xlim_b, ylim_b, 0.6)
fun_guardar(mapa_zoom, "mapa_train_test_zoom.png", xlim_b, ylim_b, 6)


# ------------------------------------------------------------------------------
# (5) Validacion: bloques espaciales, validacion estandar y buffer
# ------------------------------------------------------------------------------

# Bases de entrenamiento con sus folds (los mismos de 13_folds.R)
folds <- read_csv(file.path(dir_clean, "folds.csv"), show_col_types = FALSE)
puntos_val <- fun_puntos("train_clean.csv", "Entrenamiento") |>
  inner_join(folds, by = "property_id") |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326) |>
  st_transform(3116)
puntos_val$localidad <- localidades$LocNombre[st_nearest_feature(puntos_val, localidades)]
puntos_prueba <- fun_puntos("test_clean.csv", "Prueba") |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326) |>
  st_transform(3116)

# Mismo tamano de bloque de 13_folds.R
tamano_bloque <- 400
distancia_buffer <- 300

# (5.1) Mapa de los bloques: cada cuadro de 400 m va completo a un solo fold.
# La grilla es la de spatial_block_cv(): st_make_grid() sobre la caja de los
# anuncios, ampliada en 1e-5 (relativo) por cada lado, como hace spatialsample.
caja_grilla <- st_bbox(puntos_val)
caja_grilla[1:2] <- caja_grilla[1:2] - abs(caja_grilla[1:2] * 1e-05)
caja_grilla[3:4] <- caja_grilla[3:4] + abs(caja_grilla[3:4] * 1e-05)
grilla <- st_sf(geometry = st_make_grid(caja_grilla, cellsize = tamano_bloque))
grilla$celda <- seq_len(nrow(grilla))
celda_de_punto <- vapply(st_intersects(puntos_val, grilla), function(.i) .i[1], numeric(1))
puntos_val$celda <- celda_de_punto

tabla_celdas <- puntos_val |>
  st_drop_geometry() |>
  group_by(celda) |>
  summarise(n_folds = n_distinct(fold_espacial), fold = first(fold_espacial),
            n_anuncios = n(), .groups = "drop")
# Comprobacion: la grilla reproduce los folds (cada cuadro, un solo fold)
stopifnot(all(tabla_celdas$n_folds == 1))
cat("Cuadros con anuncios:", nrow(tabla_celdas), "\n")

celdas <- grilla |>
  inner_join(tabla_celdas, by = "celda") |>
  mutate(fold = factor(fold, labels = paste("Fold", sort(unique(fold)))))

col_folds <- c("#1F3864", "#4F81BD", "#7FB77E", "#B565A7", "#C9A227")

mapa_bloques <- ggplot() +
  geom_sf(data = filter(localidades, !es_chapinero),
          fill = col_vecinas, color = col_borde, linewidth = 0.5) +
  geom_sf(data = filter(localidades, es_chapinero),
          fill = col_chapinero, color = col_azul, linewidth = 0.5) +
  geom_sf(data = celdas, aes(fill = fold), color = "white", linewidth = 0.15, alpha = 0.9) +
  scale_fill_manual(values = setNames(col_folds, levels(celdas$fold)),
                    guide = guide_legend(nrow = 1, override.aes = list(alpha = 1))) +
  geom_sf(data = puntos_prueba, color = col_prueba, size = 0.3, alpha = 0.5) +
  coord_sf(xlim = xlim_b, ylim = ylim_b, expand = FALSE, datum = NA) +
  tema_slides
fun_guardar(mapa_bloques, "mapa_bloques_cv.png", xlim_b, ylim_b, 6)

# (5.2) Distancia de lo que se valida al entrenamiento, segun el tipo de validacion.
# Es lo que decide si la validacion se parece a la prueba.
fun_dist_cercana <- function(.a, .b) {
  idx <- st_nearest_feature(.a, .b)
  as.numeric(st_distance(.a, .b[idx, ], by_element = TRUE))
}

# Prueba contra todo el entrenamiento
dist_prueba <- fun_dist_cercana(puntos_prueba, puntos_val)

# Bloques de 400 m y folds al azar: lo validado contra lo entrenado de cada fold
fun_dist_folds <- function(.columna) {
  unlist(lapply(sort(unique(puntos_val[[.columna]])), function(.k) {
    fun_dist_cercana(puntos_val[puntos_val[[.columna]] == .k, ],
                     puntos_val[puntos_val[[.columna]] != .k, ])
  }))
}
dist_bloques <- fun_dist_folds("fold_espacial")
dist_estandar <- fun_dist_folds("fold_estandar")

# Una localidad por fold, sin entrenar con lo que esta a menos de 300 m de ella
dist_buffer <- unlist(lapply(unique(puntos_val$localidad), function(.l) {
  validacion <- puntos_val[puntos_val$localidad == .l, ]
  otros <- puntos_val[puntos_val$localidad != .l, ]
  cerca <- lengths(st_is_within_distance(otros, validacion, dist = distancia_buffer)) > 0
  entrenamiento <- otros[!cerca, ]
  if (nrow(validacion) < 10 || nrow(entrenamiento) < 10) return(numeric(0))
  fun_dist_cercana(validacion, entrenamiento)
}))

# Resumen para citar en la lamina (metros)
prob <- c(0.10, 0.25, 0.50, 0.75, 0.90)
resumen_dist <- rbind(
  `Prueba` = quantile(dist_prueba, prob),
  `Bloques de 400 m` = quantile(dist_bloques, prob),
  `Validacion estandar` = quantile(dist_estandar, prob),
  `Localidad con buffer` = quantile(dist_buffer, prob))
cat("\nDistancia al anuncio de entrenamiento mas cercano (m):\n")
print(round(resumen_dist, 1))

niveles <- c("Prueba", "Bloques de 400 m", "Validación estándar", "Localidad con buffer de 300 m")
distancias <- bind_rows(
  tibble(tipo = niveles[1], d = dist_prueba),
  tibble(tipo = niveles[2], d = dist_bloques),
  tibble(tipo = niveles[3], d = dist_estandar),
  tibble(tipo = niveles[4], d = dist_buffer)) |>
  mutate(tipo = factor(tipo, levels = niveles), d = pmax(d, 1))   # 1 m para el eje log

g_dist <- ggplot(distancias, aes(d, color = tipo, linewidth = tipo, linetype = tipo)) +
  stat_ecdf(geom = "step") +
  scale_x_log10(breaks = c(1, 10, 100, 1000, 10000),
                labels = c("1", "10", "100", "1.000", "10.000")) +
  scale_y_continuous(labels = function(.x) paste0(.x * 100, " %")) +
  scale_color_manual(values = c(col_prueba, col_navy, "#9AA5B1", col_azul)) +
  scale_linewidth_manual(values = c(1.8, 1.3, 1.1, 1.1)) +
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed")) +
  labs(x = "Distancia al anuncio de entrenamiento más cercano (metros)",
       y = "Anuncios validados") +
  theme_minimal(base_size = 16) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.direction = "vertical",
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", color = NA))
ggsave(file.path(dir_salida, "distancia_validacion.png"), g_dist,
       width = 8, height = 6.2, dpi = 300, bg = "white")


# ------------------------------------------------------------------------------
# (6) Error al sumar las fuentes una tras otra (ablacion acumulada)
# ------------------------------------------------------------------------------

ablacion <- read_csv(file.path(root, "03_outputs", "06_ablacion", "ablacion_fuentes.csv"),
                     show_col_types = FALSE) |>
  filter(tipo == "acumulado") |>
  arrange(paso)

etiquetas_fuentes <- c("Solo limpieza", "+ Coordenadas", "+ Estrato",
                       "+ Valor de referencia", "+ Uso del suelo", "+ Seguridad",
                       "+ TransMilenio", "+ OpenStreetMap",
                       "+ Texto (TF-IDF y MiniLM)", "+ Texto con fine-tuning")
stopifnot(nrow(ablacion) == length(etiquetas_fuentes))

ablacion <- ablacion |>
  mutate(etiqueta = factor(etiquetas_fuentes, levels = etiquetas_fuentes),
         grupo = case_when(paso == 0 ~ "base", paso == 1 ~ "coord",
                           paso >= 8 ~ "texto", TRUE ~ "otras"),
         texto_mae = format(round(MAE, 1), nsmall = 1, decimal.mark = ","),
         # La etiqueta del paso 1 se corre a la derecha: la linea que baja de
         # "Solo limpieza" pasaria por encima
         hjust_et = if_else(paso == 1, -0.35, 0.5),
         vjust_et = if_else(paso == 1, -0.7, -1.4))

col_grupos <- c(base = "#6E737D", coord = col_navy, otras = "#9DB4D8", texto = col_azul)

g_ablacion <- ggplot(ablacion, aes(etiqueta, MAE, group = 1)) +
  geom_line(color = col_navy, linewidth = 0.9) +
  geom_point(aes(color = grupo), size = 4.5) +
  geom_text(aes(label = texto_mae, hjust = hjust_et, vjust = vjust_et), size = 4.2,
            fontface = "bold", color = "#333333") +
  scale_color_manual(values = col_grupos, guide = "none") +
  scale_y_continuous(limits = c(105, 165), breaks = seq(110, 160, 10)) +
  labs(x = NULL, y = "MAE, validación espacial (millones)") +
  theme_minimal(base_size = 15) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 1, color = "#333333"),
        axis.title.y = element_text(margin = margin(r = 8)),
        plot.background = element_rect(fill = "white", color = NA))
ggsave(file.path(dir_salida, "ablacion_fuentes.png"), g_ablacion,
       width = 9, height = 5.2, dpi = 300, bg = "white")

cat("Figuras guardadas en", dir_salida, "\n")
