# ------------------------------------------------------------------------------
# Archivo: 02_variables_planas.R
# Proyecto: Taller 3 - Big Data y Machine Learning
#
# Descripcion:
#   Segundo paso del proyecto. Agregar a train y test variables que vienen de
#   fuentes externas y estan definidas por manzana: el estrato, el valor de
#   referencia por m2, el uso predominante, la seguridad
#   y variables de entorno calculadas con las capas anteriores. Cada anuncio se
#   ubica con su latitud y longitud en una manzana. Tambien se calculan las
#   distancias a parques, restaurantes, universidades, bancos y avenidas
#   (OpenStreetMap, extracto de Bogota de BBBike) y a estaciones de TransMilenio. Al final se hace un mapa para
#   revisar que los anuncios caen donde deben.
# Input:
#   train_clean.csv y test_clean.csv (salida de 01_cleaning.R), el shapefile
#   ManzanaEstratificacion de la Secretaria Distrital de Planeacion y el
#   shapefile Valor_Ref_M_2020 de Catastro Distrital y el shapefile
#   GUsoMZ2020 (uso predominante por manzana), IRSCAT (incidentes
#   reportados por sector catastral, IDECA) y el shapefile de estaciones
#   troncales de TransMilenio, todos en 01_data. El extracto de
#   OpenStreetMap se descarga con osmextract la primera vez y se guarda en
#   01_data/osm. Se corre desde main.R, que carga los paquetes y
#   define las rutas.
# Output:
#   train_planas y test_planas (CSV en 01_data/clean) con las columnas
#   estrato, valor_ref, uso (factor), incidentes, indice_seg, dist_est5,
#   dist_est6, prop_comercio, dist_parque, dist_restaurante, dist_universidad,
#   dist_financiero, dist_avenida y dist_tm, y los mapas en 03_outputs/mapas.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Rutas y parametros
# ------------------------------------------------------------------------------

# dir_clean y dir_mapas vienen de main.R. Las capas externas estan en dir_externos
archivo_manzanas <- file.path(dir_externos, "estratificacion", "ManzanaEstratificacion.shp")
archivo_valor <- file.path(dir_externos, "valor_referencia", "Valor_Ref_M_2020.shp")
archivo_uso <- file.path(dir_externos, "uso_predominante", "GUsoMZ2020.shp")
archivo_seg <- file.path(dir_externos, "seguridad", "IRSCAT.shp")
archivo_tm <- file.path(dir_externos, "transmilenio", "Estación_troncal.shp")
dir_osm <- file.path(dir_externos, "osm")

# Distancia maxima (metros) para asignar la manzana mas cercana a un anuncio
dist_manzana <- 50
# Radio (metros) para buscar manzanas vecinas cuando la manzana no tiene dato
dist_vecinas <- 100
# Radio (metros) para medir la proporcion de manzanas comerciales alrededor
dist_comercio <- 300


# ------------------------------------------------------------------------------
# (2) Importar las bases y el shapefile
# ------------------------------------------------------------------------------

train_clean <- read_csv(file.path(dir_clean, "train_clean.csv"), show_col_types = FALSE)
test_clean <- read_csv(file.path(dir_clean, "test_clean.csv"), show_col_types = FALSE)

# Las manzanas de estrato vienen en una proyeccion local de Bogota (metros)
manzanas <- st_read(archivo_manzanas, quiet = TRUE) |> st_make_valid()

# El valor de referencia viene en grados (WGS84), se pasa al mismo sistema de
# coordenadas de las manzanas de estrato para medir distancias en metros
valor_ref <- st_read(archivo_valor, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(st_crs(manzanas))

# El uso predominante tambien viene en grados
uso <- st_read(archivo_uso, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(st_crs(manzanas))

# Clases del uso predominante, de la mas a la menos frecuente. Se toman de la
# capa completa para que train y test tengan los mismos niveles aunque alguna
# clase no aparezca en una de las bases
niveles_uso <- names(sort(table(uso$GRUPOUSOEC), decreasing = TRUE))
print(niveles_uso)

# Incidentes reportados por sector catastral (llamadas a la linea 123). Cada
# anio es el acumulado de enero a septiembre, entonces los anios son comparables.
# Familias: R, N, AOP, MM, M, D, PIA, H y HC
seg <- st_read(archivo_seg, quiet = TRUE) |>
  st_make_valid() |>
  st_transform(st_crs(manzanas))

# Total de incidentes del sector en cada anio de publicacion
anios <- sort(unique(c(train_clean$year, test_clean$year)))
incidentes_sector <- sapply(anios, function(a) {
  columnas <- grep(paste0("^CM(R|N|AOP|MM|M|D|PIA|H|HC)", substr(a, 3, 4), "CON"), names(seg))
  rowSums(st_drop_geometry(seg)[, columnas])
})
colnames(incidentes_sector) <- anios

# Mediana de los sectores en cada anio, para el indice
mediana_anio <- apply(incidentes_sector, 2, median)


# ------------------------------------------------------------------------------
# (3) Funciones
# ------------------------------------------------------------------------------

# Convertir la base a puntos con el mismo sistema de coordenadas del shapefile.
# Latitud y longitud estan en grados (WGS84).
fun_puntos <- function(.db) {
  .db |>
    st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) |>
    st_transform(st_crs(manzanas))
}

# Moda de un vector
fun_moda <- function(.x) {
  as.numeric(names(which.max(table(.x))))
}

# Dato de una capa de manzanas para cada punto. Las coordenadas de los anuncios
# son aproximadas y muchos puntos caen en la calle, entonces se toma la manzana
# mas cercana si esta a menos de dist_manzana metros. Si esa manzana no tiene
# dato (valor 0) o no hay ninguna cerca, se resume el dato de las manzanas a
# menos de dist_vecinas metros (.resumen es la moda o la mediana).
fun_asignar <- function(.puntos, .capa, .col, .resumen) {
  valores <- .capa[[.col]]
  cercana <- st_nearest_feature(.puntos, .capa)
  distancia <- st_distance(.puntos, .capa[cercana, ], by_element = TRUE)
  x <- valores[cercana]
  x[as.numeric(distancia) > dist_manzana | x == 0] <- NA
  
  falta <- is.na(x)
  con_dato <- .capa[valores > 0, ]
  vecinas <- st_is_within_distance(.puntos[falta, ], con_dato, dist = dist_vecinas)
  x[falta] <- sapply(vecinas, function(i) {
    if (length(i) == 0) NA_real_ else .resumen(con_dato[[.col]][i])
  })
  x
}

# Uso predominante de cada punto. Misma logica de fun_asignar, pero con una
# categoria de texto: manzana mas cercana a menos de dist_manzana metros y, si
# no hay, la clase mas frecuente entre las manzanas a menos de dist_vecinas
fun_asignar_uso <- function(.puntos) {
  cercana <- st_nearest_feature(.puntos, uso)
  distancia <- st_distance(.puntos, uso[cercana, ], by_element = TRUE)
  x <- uso$GRUPOUSOEC[cercana]
  x[as.numeric(distancia) > dist_manzana] <- NA
  
  falta <- is.na(x)
  vecinas <- st_is_within_distance(.puntos[falta, ], uso, dist = dist_vecinas)
  x[falta] <- sapply(vecinas, function(i) {
    if (length(i) == 0) NA_character_ else names(which.max(table(uso$GRUPOUSOEC[i])))
  })
  factor(x, levels = niveles_uso)
}

# Incidentes del sector donde cae cada punto, en el anio de publicacion. Se toma
# el sector mas cercano a menos de dist_manzana metros y, si no hay, la mediana
# de los sectores a menos de dist_vecinas metros
fun_incidentes <- function(.puntos, .anio) {
  columna <- match(.anio, anios)
  cercana <- st_nearest_feature(.puntos, seg)
  distancia <- st_distance(.puntos, seg[cercana, ], by_element = TRUE)
  x <- incidentes_sector[cbind(cercana, columna)]
  x[as.numeric(distancia) > dist_manzana] <- NA
  
  falta <- which(is.na(x))
  vecinas <- st_is_within_distance(.puntos[falta, ], seg, dist = dist_vecinas)
  x[falta] <- vapply(seq_along(falta), function(k) {
    i <- vecinas[[k]]
    if (length(i) == 0) NA_real_ else median(incidentes_sector[i, columna[falta[k]]])
  }, numeric(1))
  x
}

# Distancia (metros) de cada punto a la manzana mas cercana de un estrato. Vale 0
# si el punto esta dentro de una manzana de ese estrato. Es un proxy de la
# preferencia por vivir cerca de zonas de estratos altos
fun_distancia <- function(.puntos, .estrato) {
  capa <- manzanas[manzanas$ESTRATO == .estrato, ]
  cercana <- st_nearest_feature(.puntos, capa)
  as.numeric(st_distance(.puntos, capa[cercana, ], by_element = TRUE))
}

# Proporcion de manzanas de comercio y oficinas dentro de un circulo de radio
# dist_comercio alrededor del punto, con la capa de uso predominante
fun_prop_comercio <- function(.puntos) {
  vecinas <- st_intersects(st_buffer(.puntos, dist_comercio), uso)
  vapply(vecinas, function(i) {
    if (length(i) == 0) NA_real_ else mean(uso$GRUPOUSOEC[i] == "COMERCIO Y OFICINAS")
  }, numeric(1))
}


# Lee una capa del extracto de OpenStreetMap de Bogota (BBBike). La primera vez
# descarga el archivo a la carpeta osm y despues lo reutiliza. .capa es la capa
# del extracto (points, lines o multipolygons), .consulta es el filtro en SQL y
# .extra son etiquetas adicionales que hay que pedir al crear el archivo
fun_osm <- function(.capa, .consulta, .extra = NULL) {
  oe_get("Bogota", provider = "bbbike", layer = .capa, query = .consulta,
         extra_tags = .extra, download_directory = dir_osm, quiet = TRUE) |>
    fun_geometrias()
}

# Juntar geometrias de varias capas (las que no tienen datos vienen
# como NULL) en el sistema de coordenadas de las manzanas
fun_geometrias <- function(...) {
  capas <- Filter(Negate(is.null), list(...))
  do.call(c, lapply(capas, st_geometry)) |>
    st_make_valid() |>
    st_transform(st_crs(manzanas))
}

# Distancia en metros de cada punto al elemento mas cercano de una capa
# (0 si el punto esta dentro de un poligono)
fun_dist_osm <- function(.puntos, .capa) {
  cercano <- st_nearest_feature(.puntos, .capa)
  as.numeric(st_distance(.puntos, .capa[cercano], by_element = TRUE))
}


# ------------------------------------------------------------------------------
# (4) Capas de OpenStreetMap y TransMilenio
# ------------------------------------------------------------------------------

puntos_train <- fun_puntos(train_clean)
puntos_test <- fun_puntos(test_clean)

dir.create(dir_osm, recursive = TRUE, showWarnings = FALSE)

# En la capa de puntos las etiquetas amenity y office no vienen por defecto
etiquetas_puntos <- c("amenity", "office")

# Restaurantes y bancos u oficinas financieras (puntos, como en clase)
capa_restaurantes <- fun_osm(
  "points", "SELECT * FROM points WHERE amenity = 'restaurant'", etiquetas_puntos)
capa_financiero <- fun_osm(
  "points", "SELECT * FROM points WHERE amenity = 'bank' OR office = 'financial'",
  etiquetas_puntos)

# Universidades: los campus suelen estar mapeados como poligonos, por eso se
# juntan los puntos y los poligonos
capa_universidades <- c(
  fun_osm("points", "SELECT * FROM points WHERE amenity = 'university'", etiquetas_puntos),
  fun_osm("multipolygons", "SELECT * FROM multipolygons WHERE amenity = 'university'"))

# Parques (poligonos)
capa_parques <- fun_osm("multipolygons", "SELECT * FROM multipolygons WHERE leisure = 'park'")
capa_parques <- capa_parques[!st_is_empty(capa_parques)]

# Avenidas principales (lineas): vias troncales y primarias
capa_avenidas <- fun_osm("lines", "SELECT * FROM lines WHERE highway IN ('trunk', 'primary')")

# Estaciones troncales de TransMilenio (datos abiertos de TransMilenio)
capa_tm <- st_read(archivo_tm, quiet = TRUE) |>
  st_zm() |>
  fun_geometrias()

# Cantidad de elementos de cada capa
capas_osm <- list(parques = capa_parques, restaurantes = capa_restaurantes,
                  universidades = capa_universidades, financiero = capa_financiero,
                  avenidas = capa_avenidas, estaciones_tm = capa_tm)
print(sapply(capas_osm, length))
stopifnot(all(sapply(capas_osm, length) > 0))

# El extracto de Bogota es un rectangulo. Se verifica que todos los anuncios
# queden dentro del area que cubren las avenidas
cobertura <- st_within(c(st_geometry(puntos_train), st_geometry(puntos_test)),
                       st_as_sfc(st_bbox(capa_avenidas)))
stopifnot(all(lengths(cobertura) > 0))


# ------------------------------------------------------------------------------
# (5) Agregar estrato, valor de referencia, uso, seguridad y entorno
# ------------------------------------------------------------------------------

# El estrato es una categoria (moda) y el valor de referencia es un monto (mediana)
train_planas <- train_clean |>
  mutate(estrato = fun_asignar(puntos_train, manzanas, "ESTRATO", fun_moda),
         valor_ref = fun_asignar(puntos_train, valor_ref, "V_REF", median),
         uso = fun_asignar_uso(puntos_train),
         incidentes = fun_incidentes(puntos_train, year),
         indice_seg = incidentes / mediana_anio[as.character(year)],
         dist_est5 = fun_distancia(puntos_train, 5),
         dist_est6 = fun_distancia(puntos_train, 6),
         prop_comercio = fun_prop_comercio(puntos_train),
         dist_parque = fun_dist_osm(puntos_train, capa_parques),
         dist_restaurante = fun_dist_osm(puntos_train, capa_restaurantes),
         dist_universidad = fun_dist_osm(puntos_train, capa_universidades),
         dist_financiero = fun_dist_osm(puntos_train, capa_financiero),
         dist_avenida = fun_dist_osm(puntos_train, capa_avenidas),
         dist_tm = fun_dist_osm(puntos_train, capa_tm))
test_planas <- test_clean |>
  mutate(estrato = fun_asignar(puntos_test, manzanas, "ESTRATO", fun_moda),
         valor_ref = fun_asignar(puntos_test, valor_ref, "V_REF", median),
         uso = fun_asignar_uso(puntos_test),
         incidentes = fun_incidentes(puntos_test, year),
         indice_seg = incidentes / mediana_anio[as.character(year)],
         dist_est5 = fun_distancia(puntos_test, 5),
         dist_est6 = fun_distancia(puntos_test, 6),
         prop_comercio = fun_prop_comercio(puntos_test),
         dist_parque = fun_dist_osm(puntos_test, capa_parques),
         dist_restaurante = fun_dist_osm(puntos_test, capa_restaurantes),
         dist_universidad = fun_dist_osm(puntos_test, capa_universidades),
         dist_financiero = fun_dist_osm(puntos_test, capa_financiero),
         dist_avenida = fun_dist_osm(puntos_test, capa_avenidas),
         dist_tm = fun_dist_osm(puntos_test, capa_tm))

print(count(train_planas, estrato))
print(count(test_planas, estrato))
print(summary(train_planas$valor_ref))
print(summary(test_planas$valor_ref))
print(count(train_planas, uso))
print(count(test_planas, uso))
print(summary(train_planas[, c("incidentes", "indice_seg")]))
print(summary(test_planas[, c("incidentes", "indice_seg")]))
print(summary(train_planas[, c("dist_est5", "dist_est6", "prop_comercio")]))
print(summary(test_planas[, c("dist_est5", "dist_est6", "prop_comercio")]))
print(summary(train_planas[, c("dist_parque", "dist_restaurante", "dist_universidad", "dist_financiero", "dist_avenida", "dist_tm")]))
print(summary(test_planas[, c("dist_parque", "dist_restaurante", "dist_universidad", "dist_financiero", "dist_avenida", "dist_tm")]))

# Verificacion: mismas filas que las bases de entrada
stopifnot(nrow(train_planas) == nrow(train_clean),
          nrow(test_planas) == nrow(test_clean))


# ------------------------------------------------------------------------------
# (6) Mapas
# ------------------------------------------------------------------------------

dir.create(dir_mapas, recursive = TRUE, showWarnings = FALSE)

# Anuncios de train y test en un solo objeto
puntos_train$base <- "train"
puntos_test$base <- "test"
puntos <- rbind(puntos_train[, "base"], puntos_test[, "base"])

colores_estrato <- c("0" = "grey85", "1" = "#d73027", "2" = "#fc8d59", "3" = "#fee08b",
                     "4" = "#d9ef8b", "5" = "#91cf60", "6" = "#1a9850")
colores_base <- c("train" = "black", "test" = "blue")

mapa <- ggplot() +
  geom_sf(data = manzanas, aes(fill = factor(ESTRATO)), color = NA) +
  geom_sf(data = puntos, aes(color = base), size = 0.2, alpha = 0.4) +
  scale_fill_manual(values = colores_estrato, name = "Estrato") +
  scale_color_manual(values = colores_base, name = "Base") +
  guides(color = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  labs(title = "Anuncios sobre las manzanas de Bogota") +
  theme_minimal()

ggsave(file.path(dir_mapas, "mapa_bogota.png"), mapa, width = 8, height = 10, dpi = 200, bg = "white")

# Zona de test (Chapinero) con las manzanas que estan alrededor
zona <- st_bbox(puntos_test)
mapa_zona <- mapa +
  coord_sf(xlim = zona[c("xmin", "xmax")], ylim = zona[c("ymin", "ymax")]) +
  labs(title = "Zona de test")

ggsave(file.path(dir_mapas, "mapa_zona_test.png"), mapa_zona, width = 8, height = 10, dpi = 200, bg = "white")


# ------------------------------------------------------------------------------
# (7) Exportar las bases
# ------------------------------------------------------------------------------

write_csv(train_planas, file.path(dir_clean, "train_planas.csv"))
write_csv(test_planas, file.path(dir_clean, "test_planas.csv"))