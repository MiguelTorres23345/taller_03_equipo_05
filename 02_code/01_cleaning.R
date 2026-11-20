# ------------------------------------------------------------------------------
# Archivo: 01_cleaning.R
# Proyecto: Taller 3 - Big Data y Machine Learning
#
# Descripcion:
#   Primer paso del proyecto. Importar train y test, eliminar los anuncios
#   duplicados de train y completar el area (surface_covered) y los banos
#   (bathrooms) con la informacion del titulo y la descripcion de cada anuncio.
#   El titulo y la descripcion no se modifican: el texto se limpia solo en una
#   copia temporal para buscar los patrones.
# Input:
#   train.csv y test.csv en la carpeta Taller 3.
# Output:
#   train_clean y test_clean (objetos y archivos CSV en la carpeta clean) con
#   surface_covered y bathrooms sin NA. Cada valor completado queda marcado en
#   surface_source y bathrooms_source (base, texto o relleno), y area_text y
#   bathrooms_text conservan lo encontrado en el texto.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Paquetes y rutas
# ------------------------------------------------------------------------------

library(dplyr)
library(readr)

dir_datos <- "C:/Users/aeste/OneDrive/Documentos/10 Semestre/Big Data ML/Taller 3"
dir_clean <- file.path(dir_datos, "clean")

# Tamanio minimo de un grupo para usar su moda o mediana en el relleno
n_minimo <- 10


# ------------------------------------------------------------------------------
# (2) Importar las bases originales
# ------------------------------------------------------------------------------

db_train <- read_csv(file.path(dir_datos, "train.csv"), show_col_types = FALSE)
db_test <- read_csv(file.path(dir_datos, "test.csv"), show_col_types = FALSE)


# ------------------------------------------------------------------------------
# (3) Definir las funciones de procesamiento
# ------------------------------------------------------------------------------

# (3.1) Eliminar duplicados
# Un mismo inmueble puede estar publicado en varios meses con la misma
# descripcion, precio, ubicacion, habitaciones, banos y tipo. Se deja una sola
# fila: la que tiene menos NA y, si empatan, la mas reciente. Los NA de esa fila
# se completan con la copia mas reciente que tenga el dato. Las copias de un
# mismo mes no se tocan, pueden ser inmuebles distintos del mismo edificio.
fun_eliminar_duplicados <- function(.db) {
  llaves <- c("description", "price", "lat", "lon", "bedrooms", "bathrooms",
              "property_type")
  
  .db <- .db |>
    mutate(
      fila = row_number(),
      periodo = year * 12 + month,
      n_na = is.na(surface_total) + is.na(surface_covered) + is.na(rooms) +
        is.na(bathrooms)
    ) |>
    group_by(across(all_of(llaves))) |>
    mutate(grupo = cur_group_id(), n_copias = n(), n_meses = n_distinct(periodo)) |>
    ungroup()
  
  repetidos <- .db |> filter(n_copias > 1, n_meses > 1)
  unicos <- .db |> filter(!(n_copias > 1 & n_meses > 1))
  
  # Primer dato disponible, de la copia mas reciente a la mas antigua
  recientes <- repetidos |>
    arrange(grupo, desc(periodo)) |>
    group_by(grupo) |>
    summarise(
      across(c(surface_total, surface_covered, rooms),
             ~ .x[!is.na(.x)][1], .names = "{.col}_rec"),
      .groups = "drop"
    )
  
  conservados <- repetidos |>
    arrange(grupo, n_na, desc(periodo)) |>
    group_by(grupo) |>
    slice(1) |>
    ungroup() |>
    left_join(recientes, by = "grupo") |>
    mutate(
      surface_total = coalesce(surface_total, surface_total_rec),
      surface_covered = coalesce(surface_covered, surface_covered_rec),
      rooms = coalesce(rooms, rooms_rec)
    )
  
  bind_rows(unicos, conservados) |>
    arrange(fila) |>
    select(-c(fila, periodo, n_na, grupo, n_copias, n_meses),
           -ends_with("_rec"))
}


# (3.2) Texto temporal para buscar patrones
# El titulo y la descripcion vienen sin tildes ni enes y con restos de
# entidades HTML ("ba ntilde o", "baos", "rea"). Esta funcion devuelve una
# copia en minuscula con esos restos unificados; no cambia la base.
fun_texto <- function(.titulo, .descripcion) {
  texto <- tolower(paste(coalesce(.titulo, ""), coalesce(.descripcion, ""),
                         sep = " . "))
  texto <- gsub("\\s*\\b([aeiou])acute\\b\\s*", "\\1", texto, perl = TRUE)
  texto <- gsub("\\s*\\bntilde\\b\\s*", "n", texto, perl = TRUE)
  texto <- gsub("\\bbaa o(s?)\\b", "bano\\1", texto, perl = TRUE)
  texto <- gsub("\\bbao(s?)\\b", "bano\\1", texto, perl = TRUE)
  texto <- gsub("\\bhabitacin(es)?\\b", "habitacion\\1", texto, perl = TRUE)
  gsub("\u00b2", "2", texto, fixed = TRUE)
}

# Palabras (letras y numeros) de un texto
fun_palabras <- function(.texto) {
  if (!nzchar(.texto)) return(character(0))
  regmatches(.texto, gregexpr("[a-z0-9]+", .texto, perl = TRUE))[[1]]
}


# (3.3) Area construida desde el texto
# Se busca una cifra seguida de una unidad ("85 m2", "85mts", "ochenta metros")
# y se decide si es el area construida segun la palabra que la rodea.

# Cifra: entera, con separador de miles ("1.200") o con decimales ("85,5").
# Unidad: m2, mts, mt2, mtrs, metros, mc, etc. Puede venir pegada al numero
# siguiente ("m23" es "m2" y luego "3 habitaciones").
patron_cifra <- "(?<![a-z0-9.,])(\\d{1,3}(?:\\.\\d{3})+(?:,\\d{1,2})?|\\d{2,6}(?:[.,]\\d{1,2})?)"
patron_unidad <- "(msup2|m\\s?sup2|(?:metros|metro|mtros|mtrs|mtr|mts|mt|m)(?:2\\d*)?|mc)"
patron_area <- paste0(patron_cifra, "\\s{0,2}", patron_unidad, "(?![a-z])")
# Cifra con unidad seguida de "de area construida", justo antes de otra cifra
patron_consumida <- paste0("\\d\\s{0,2}(?:metros|metro|mtros|mtrs|mtr|mts|mt|m)\\s?2?\\d*",
                           "\\s*(?:de|del|en)?\\s*(?:area\\s*)?",
                           "(?:construida|construido|construidos|total)\\s*$")

# Palabras que se ignoran al buscar la palabra clave a la izquierda
palabras_relleno <- c("de", "del", "la", "el", "los", "las", "un", "una", "y",
                      "en", "es", "son", "con", "aprox", "aproximadamente",
                      "aproximado", "mide", "tiene", "cuenta", "solo", "sola",
                      "apenas", "casi", "unos", "unas", "que", "se", "su", "sus",
                      "al", "lo", "ubicado", "ubicada", "desde", "hasta", "por", "a")
# Cifras que describen otro espacio y no el area construida
otros_espacios <- c("terraza", "terrazas", "balcon", "balcones", "patio", "patios",
                    "garaje", "garajes", "parqueadero", "parqueaderos", "jardin",
                    "deposito", "depositos", "altura", "frente", "fondo", "ancho",
                    "largo", "libre")
otros_espacios_izq <- c(otros_espacios, "x", "por", "mas", "ms")
palabras_lote <- c("lote", "terreno")
# Palabras que se saltan entre la unidad y la palabra que describe la cifra
palabras_salto <- c("de", "del", "para", "en", "la", "el", "una", "un", "los", "las",
                    "cuadrados", "cuadrado")
# Palabras a las que se les perdio la tilde ("balcn" es balcon)
tildes_mal <- c("rea", "reas", "balcn", "jardn", "depsito", "depsitos", "construccin")
tildes_bien <- c("area", "areas", "balcon", "jardin", "deposito", "depositos", "construccion")
# A la derecha de la cifra, "85 m2 balcon" o "85 m2 deposito" son listas de
# caracteristicas y la cifra si es el area
otros_espacios_der <- setdiff(otros_espacios, c("balcon", "deposito"))
palabras_area <- c("construida", "area", "apartamento", "apto", "casa", "duplex",
                   "piso", "consta", "total", "interior", "exterior")

# Numeros escritos en letras ("cuarenta y cuatro metros")
decenas <- c(veinte = 20, treinta = 30, cuarenta = 40, cincuenta = 50,
             sesenta = 60, setenta = 70, ochenta = 80, noventa = 90)
centenas <- c(cien = 100, ciento = 100, doscientos = 200, trescientos = 300,
              cuatrocientos = 400, quinientos = 500)
unidades <- c(un = 1, uno = 1, dos = 2, tres = 3, cuatro = 4, cinco = 5,
              seis = 6, siete = 7, ocho = 8, nueve = 9)
patron_letras <- paste0(
  "\\b(?:(ciento|cien|doscientos|trescientos|cuatrocientos|quinientos)\\s+)?",
  "(?:(veinte|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa)",
  "(?:\\s+y\\s+(un|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve))?)",
  "\\s+(metros|mts|m2|mt2|mts2)\\b"
)

fun_letras_a_digitos <- function(.texto) {
  m <- gregexpr(patron_letras, .texto, perl = TRUE)[[1]]
  if (m[1] == -1) return(.texto)
  ini <- attr(m, "capture.start")
  largo <- attr(m, "capture.length")
  grupo <- function(i, k) substr(.texto, ini[i, k], ini[i, k] + largo[i, k] - 1)
  
  nuevos <- character(length(m))
  for (i in seq_along(m)) {
    valor <- decenas[[grupo(i, 2)]]
    if (largo[i, 1] > 0) valor <- valor + centenas[[grupo(i, 1)]]
    if (largo[i, 3] > 0) valor <- valor + unidades[[grupo(i, 3)]]
    nuevos[i] <- paste(valor, grupo(i, 4))
  }
  regmatches(.texto, gregexpr(patron_letras, .texto, perl = TRUE)) <- list(nuevos)
  .texto
}

fun_numero <- function(.cifra) {
  if (grepl("^\\d{1,3}(\\.\\d{3})+(,\\d{1,2})?$", .cifra, perl = TRUE)) {
    .cifra <- gsub(".", "", .cifra, fixed = TRUE)
  }
  as.numeric(sub(",", ".", .cifra, fixed = TRUE))
}

# Devuelve el area construida del anuncio o NA si el texto no la indica
fun_area_texto <- function(.texto, .tipo) {
  # Palabras a las que se les perdio la tilde al bajar los datos
  texto <- .texto
  for (j in seq_along(tildes_mal)) {
    texto <- gsub(paste0("\\b", tildes_mal[j], "\\b"), tildes_bien[j], texto, perl = TRUE)
  }
  texto <- fun_letras_a_digitos(texto)
  m <- gregexpr(patron_area, texto, perl = TRUE)[[1]]
  if (m[1] == -1) return(NA_real_)
  ini <- attr(m, "capture.start")
  largo <- attr(m, "capture.length")
  
  prioridad <- c()
  posicion <- c()
  valor <- c()
  for (i in seq_along(m)) {
    inicio <- as.integer(m[i])
    fin <- inicio + attr(m, "match.length")[i] - 1L
    cifra <- substr(texto, ini[i, 1], ini[i, 1] + largo[i, 1] - 1)
    unidad <- substr(texto, ini[i, 2], ini[i, 2] + largo[i, 2] - 1)
    v <- fun_numero(cifra)
    
    # "m23" se lee como "m2"
    unidad <- sub("^(metros|metro|mtros|mtrs|mtr|mts|mt|m)2\\d+$", "\\12", unidad,
                  perl = TRUE)
    
    # Palabras antes de la cifra (se ignoran las de relleno y los numeros)
    antes <- tail(fun_palabras(substr(texto, 1, inicio - 1)), 6)
    antes[antes == "rea"] <- "area"
    previa <- if (length(antes) > 0) tail(antes, 1) else "<inicio>"
    antes <- antes[!(antes %in% palabras_relleno) & !grepl("^\\d+$", antes)]
    izq1 <- if (length(antes) > 0) tail(antes, 1) else "<inicio>"
    izq2 <- if (length(antes) > 1) tail(antes, 2)[1] else "<inicio>"
    
    # Palabra despues de la unidad (se saltan articulos, "de", "en", "cuadrados")
    despues <- head(fun_palabras(substr(texto, fin + 1, fin + 40)), 4)
    while (length(despues) > 0 && despues[1] %in% palabras_salto) {
      despues <- despues[-1]
    }
    der1 <- if (length(despues) > 0) despues[1] else "<fin>"
    
    # "85 m" solo cuenta si va despues de una palabra tipica de area
    if (unidad == "m" && !(izq1 %in% palabras_area)) next
    # "40 m2 de area construida 20 m2 de terraza": la palabra "construida" ya
    # describe la cifra anterior y no la 20
    if (izq1 %in% c("area", "total", "construida", "construido", "construidos") &&
        grepl(patron_consumida, substr(texto, 1, inicio - 1), perl = TRUE)) {
      izq1 <- "<consumida>"
    }
    # La cifra viene precedida por una palabra que dice que es el area
    dice_area <- izq1 %in% c("area", "total", "construida", "construido", "construidos")
    # La coma decimal se pierde al bajar los datos: "6169 m2" es 61,69 m2
    if (grepl("^\\d{4,6}$", cifra, perl = TRUE)) {
      v <- v / 100
      # "1695 m2" es 169,5 y no 16,95. Un apartamento menor a 25 m2 y una casa
      # menor a 50 m2 son poco probables, entonces solo se pierde un decimal
      minimo <- ifelse(.tipo == "Casa", 50, 25)
      if (nchar(cifra) == 4 && v < minimo) v <- as.numeric(cifra) / 10
    }
    if (v < 15 || v > 1500) next
    if (.tipo == "Apartamento" && v > 600) next
    # Terrazas, balcones, alturas, dimensiones y sumas no son el area. Si la
    # cifra viene despues de "area", "total" o "construida" se acepta aunque
    # la siguiente palabra sea una terraza ("area 95 m2 terraza cubierta").
    if ((izq1 %in% otros_espacios_izq || previa %in% otros_espacios_izq) &&
        izq1 != "construida") next
    # "terraza cubierta de 15 m2": la terraza esta dos palabras antes
    if (!dice_area && izq2 %in% otros_espacios &&
        !(izq1 %in% c("construida", "construido", "construidos"))) next
    if (der1 %in% otros_espacios_der && !dice_area) next
    # El lote o terreno no es area construida
    if (izq1 %in% palabras_lote || der1 %in% palabras_lote) next
    # Area privada: solo si dice "area privada"
    if (izq1 == "privada" || der1 %in% c("privada", "privados")) {
      if (izq2 == "area") {
        prioridad <- c(prioridad, 1)
        posicion <- c(posicion, inicio)
        valor <- c(valor, v)
      }
      next
    }
    # "85 m2 mas 12 m2 de terraza": la segunda cifra es un adicional
    if (previa %in% c("ms", "mas")) next
    
    # Prioridad: dice construida (0), area o total (1), resto (2)
    if (izq1 %in% c("construida", "construccion", "construido", "construidos") ||
        izq2 %in% c("construida", "construccion") ||
        der1 %in% c("construidos", "construida", "construido", "construccion") ||
        (der1 == "area" && length(despues) > 1 &&
         despues[2] %in% c("construida", "construido", "construidos"))) {
      p <- 0
    } else if (izq1 %in% c("area", "total")) {
      p <- 1
    } else {
      p <- 2
    }
    prioridad <- c(prioridad, p)
    posicion <- c(posicion, inicio)
    valor <- c(valor, v)
  }
  
  if (length(valor) == 0) return(NA_real_)
  valor[order(prioridad, posicion)][1]
}


# (3.4) Banos desde el texto
# "N banos" con N de 2 o mas y sin calificador (social, privado, servicio...)
# es el total. Con calificador o con N = 1 es solo una parte, asi que no se usa
# como total. Ademas se calculan tres datos del texto para el relleno: cuantas
# veces aparece "bano", cuantas veces aparece "1 bano ..." y si hay bano de
# servicio.
numeros_banos <- c(un = 1, una = 1, uno = 1, dos = 2, tres = 3, cuatro = 4,
                   cinco = 5, seis = 6, siete = 7, ocho = 8, nueve = 9, diez = 10)
calificadores <- c("social", "sociales", "privado", "privados", "auxiliar",
                   "auxiliares", "servicio", "completo", "de", "y", "mas",
                   "principal", "independiente", "independientes", "compartido",
                   "compartidos", "visitas", "adicional")
patron_servicio <- paste0(
  "bano\\s+(?:de[l]?\\s+)?servicio|servicio\\s+con\\s+(?:su\\s+)?bano|",
  "cuarto\\s+(?:y|con)\\s+(?:su\\s+)?bano|bano\\s+para\\s+(?:el\\s+)?servicio"
)

# Numero escrito antes de "bano" (cifra de 1 a 12 o palabra), NA si no hay
fun_numero_bano <- function(.palabra) {
  if (grepl("^\\d{1,2}$", .palabra) && as.integer(.palabra) %in% 1:12) {
    return(as.integer(.palabra))
  }
  if (.palabra %in% names(numeros_banos)) return(unname(numeros_banos[.palabra]))
  NA_integer_
}

fun_banos_texto <- function(.texto) {
  palabras <- fun_palabras(.texto)
  posiciones <- which(palabras %in% c("bano", "banos"))
  
  n_explicito <- NA_real_
  suma_uno <- 0
  for (j in posiciones) {
    if (j == 1) next
    n <- fun_numero_bano(palabras[j - 1])
    if (is.na(n)) next
    siguiente <- if (j < length(palabras)) palabras[j + 1] else "<fin>"
    if (n == 1) suma_uno <- suma_uno + 1
    if (is.na(n_explicito) && n >= 2 && !(siguiente %in% calificadores)) {
      n_explicito <- n
    }
  }
  
  data.frame(
    n_explicito = n_explicito,
    n_menciones = length(posiciones),
    suma_uno = suma_uno,
    servicio = as.numeric(grepl(patron_servicio, .texto, perl = TRUE))
  )
}


# (3.5) Extraer del texto: junta area y banos en la base
fun_extraer_texto <- function(.db) {
  texto <- fun_texto(.db$title, .db$description)
  banos <- bind_rows(lapply(texto, fun_banos_texto))
  
  .db |>
    mutate(
      area_text = mapply(fun_area_texto, texto, property_type, USE.NAMES = FALSE),
      n_explicito = banos$n_explicito,
      n_menciones = pmin(banos$n_menciones, 5),
      suma_uno = pmin(banos$suma_uno, 3),
      servicio = banos$servicio,
      hab = pmin(bedrooms, 6),
      # Si el numero del texto queda 3 o mas por debajo de las habitaciones,
      # probablemente describe solo una parte del inmueble y se descarta.
      bathrooms_text = if_else(n_explicito - bedrooms <= -3, NA_real_, n_explicito)
    ) |>
    select(-n_explicito)
}


# (3.6) Relleno segun tipo de inmueble
# Moda (si hay empate, el menor valor)
fun_moda <- function(.x) {
  .x <- .x[!is.na(.x)]
  valores <- sort(unique(.x))
  valores[which.max(tabulate(match(.x, valores)))]
}

# Tabla con el valor tipico de .y por cada combinacion de .llaves, usando solo
# los grupos con al menos n_minimo anuncios
fun_tabla_relleno <- function(.db, .y, .llaves, .resumen) {
  .db |>
    filter(!is.na(.data[[.y]])) |>
    group_by(across(all_of(.llaves))) |>
    summarise(relleno = .resumen(.data[[.y]]), n = n(), .groups = "drop") |>
    filter(n >= n_minimo) |>
    select(-n)
}


# ------------------------------------------------------------------------------
# (4) Aplicar el procesamiento a train y test
# ------------------------------------------------------------------------------

# (4.1) Duplicados: solo en train
n_train_original <- nrow(db_train)
db_train <- fun_eliminar_duplicados(db_train)
print(tibble(filas_antes = n_train_original, filas_despues = nrow(db_train),
             eliminadas = n_train_original - nrow(db_train)))

# (4.2) Area y banos que indica el texto
db_train <- fun_extraer_texto(db_train)
db_test <- fun_extraer_texto(db_test)

# (4.3) Traslape entre lo que trae la base y lo que dice el texto
# Area: surface_covered o, si falta, surface_total
fun_traslape <- function(.base, .texto, .coincide) {
  tibble(
    en_base = sum(!is.na(.base)),
    en_texto = sum(!is.na(.texto)),
    en_ambos = sum(!is.na(.base) & !is.na(.texto)),
    solo_texto = sum(is.na(.base) & !is.na(.texto)),
    en_ninguno = sum(is.na(.base) & is.na(.texto)),
    coincidencia = mean(.coincide[!is.na(.base) & !is.na(.texto)])
  )
}
fun_traslape_area <- function(.db) {
  base <- coalesce(.db$surface_covered, .db$surface_total)
  fun_traslape(base, .db$area_text, abs(.db$area_text / base - 1) <= 0.10)
}
fun_traslape_banos <- function(.db) {
  fun_traslape(.db$bathrooms, .db$bathrooms_text, .db$bathrooms_text == .db$bathrooms)
}

print(bind_rows(train = fun_traslape_area(db_train), test = fun_traslape_area(db_test),
                .id = "area"))
print(bind_rows(train = fun_traslape_banos(db_train), test = fun_traslape_banos(db_test),
                .id = "banos"))

# (4.4) Banos: base -> texto -> relleno con la moda segun tipo
# Las tablas se arman solo con train y se usan en train y test.
llaves_banos_1 <- c("property_type", "hab", "n_menciones", "suma_uno", "servicio")
llaves_banos_2 <- c("property_type", "hab", "n_menciones")
llaves_banos_3 <- c("property_type", "hab")
llaves_banos_4 <- c("property_type")

tab_banos_1 <- fun_tabla_relleno(db_train, "bathrooms", llaves_banos_1, fun_moda) |>
  rename(relleno_1 = relleno)
tab_banos_2 <- fun_tabla_relleno(db_train, "bathrooms", llaves_banos_2, fun_moda) |>
  rename(relleno_2 = relleno)
tab_banos_3 <- fun_tabla_relleno(db_train, "bathrooms", llaves_banos_3, fun_moda) |>
  rename(relleno_3 = relleno)
tab_banos_4 <- fun_tabla_relleno(db_train, "bathrooms", llaves_banos_4, fun_moda) |>
  rename(relleno_4 = relleno)

fun_imputar_banos <- function(.db) {
  .db |>
    left_join(tab_banos_1, by = llaves_banos_1) |>
    left_join(tab_banos_2, by = llaves_banos_2) |>
    left_join(tab_banos_3, by = llaves_banos_3) |>
    left_join(tab_banos_4, by = llaves_banos_4) |>
    mutate(
      bathrooms_source = case_when(
        !is.na(bathrooms) ~ "base",
        !is.na(bathrooms_text) ~ "texto",
        TRUE ~ "relleno"
      ),
      bathrooms = coalesce(bathrooms, bathrooms_text, relleno_1, relleno_2,
                           relleno_3, relleno_4),
      banos_cat = pmin(bathrooms, 5)
    ) |>
    select(-starts_with("relleno_"))
}
db_train <- fun_imputar_banos(db_train)
db_test <- fun_imputar_banos(db_test)

# (4.5) Area: base -> texto -> relleno con la mediana segun tipo y banos
# La base es surface_covered o, si falta, surface_total.
db_train <- db_train |>
  mutate(area_base = coalesce(surface_covered, surface_total, area_text))
db_test <- db_test |>
  mutate(area_base = coalesce(surface_covered, surface_total, area_text))

llaves_area_1 <- c("property_type", "hab", "banos_cat")
llaves_area_2 <- c("property_type", "hab")
llaves_area_3 <- c("property_type")

tab_area_1 <- fun_tabla_relleno(db_train, "area_base", llaves_area_1, median) |>
  rename(relleno_1 = relleno)
tab_area_2 <- fun_tabla_relleno(db_train, "area_base", llaves_area_2, median) |>
  rename(relleno_2 = relleno)
tab_area_3 <- fun_tabla_relleno(db_train, "area_base", llaves_area_3, median) |>
  rename(relleno_3 = relleno)

fun_imputar_area <- function(.db) {
  .db |>
    left_join(tab_area_1, by = llaves_area_1) |>
    left_join(tab_area_2, by = llaves_area_2) |>
    left_join(tab_area_3, by = llaves_area_3) |>
    mutate(
      surface_source = case_when(
        !is.na(surface_covered) ~ "surface_covered",
        !is.na(surface_total) ~ "surface_total",
        !is.na(area_text) ~ "texto",
        TRUE ~ "relleno"
      ),
      surface_covered = coalesce(area_base, relleno_1, relleno_2, relleno_3)
    ) |>
    select(-starts_with("relleno_"))
}
db_train <- fun_imputar_area(db_train)
db_test <- fun_imputar_area(db_test)


# ------------------------------------------------------------------------------
# (5) Preparar las bases finales
# ------------------------------------------------------------------------------

# Quitar las columnas auxiliares del relleno
train_clean <- db_train |> select(-c(suma_uno, servicio, n_menciones, hab, banos_cat, area_base, surface_total, rooms))
test_clean <- db_test |> select(-c(suma_uno, servicio, n_menciones, hab, banos_cat, area_base, surface_total, rooms))
# Fuente de cada valor
print(count(train_clean, surface_source))
print(count(train_clean, bathrooms_source))
print(count(test_clean, surface_source))
print(count(test_clean, bathrooms_source))

# Verificacion: sin NA en area y banos
stopifnot(sum(is.na(train_clean$surface_covered)) == 0,
          sum(is.na(train_clean$bathrooms)) == 0,
          sum(is.na(test_clean$surface_covered)) == 0,
          sum(is.na(test_clean$bathrooms)) == 0)


# ------------------------------------------------------------------------------
# (6) Exportar las bases finales
# ------------------------------------------------------------------------------

dir.create(dir_clean, recursive = TRUE, showWarnings = FALSE)
write_csv(train_clean, file.path(dir_clean, "train_clean.csv"))
write_csv(test_clean, file.path(dir_clean, "test_clean.csv"))