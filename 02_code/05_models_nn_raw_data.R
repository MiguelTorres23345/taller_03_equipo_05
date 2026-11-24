# ------------------------------------------------------------------------------
# Archivo: 05_models_nn_raw_data.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion:
#   Definir y ejecutar la red de datos crudos para validacion aleatoria o espacial.
#   Al ejecutar este archivo desde la raiz, correr la validacion aleatoria.
# Input:
#   CSV originales y rutas de resultados y dependencias definidos en main.R.
#   Particiones espaciales de 06_models_nn_bloques.R o 07_models_nn_localidades.R.
#   Paquetes de R y binarios de Torch y ONNX preparados en 00_setup.R.
#   Pesos de E5 verificados y guardados en el cache local del taller.
# Output:
#   Un modelo RDS y un CSV de test por arquitectura, y resumen.xlsx con
#   metricas fuera de fold en la carpeta de cada experimento.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir predictores, arquitecturas e hiperparametros
# ------------------------------------------------------------------------------

predictores <- c(
  "month", "year", "surface_total", "surface_covered", "rooms",
  "bedrooms", "bathrooms", "lat", "lon", "property_type"
)
columnas_texto <- c("title", "description")

# Cada nombre coincide con una subcarpeta. Cada vector enumera las capas ocultas;
# integer(0) indica salida directa. Ejemplo de dos capas: c(32L, 64L).
arquitecturas <- list(
  "01_transfer_in_layer_out_layer" = integer(0),
  "02_transfer_1_hidden_layer_32n" = 32L,
  "03_transfer_1_hidden_layer_64n" = 64L
)
semilla <- 2026L
n_folds <- 5L
epocas <- 25L
tamano_lote <- 256L
tasa <- 0.001
dropout <- 0.20                      # Se aplica despues de cada ReLU.
criterio <- "asimetrica"             # Alternativas: "mae","asimetrica".
peso_sobre <- 2 / 3                  # Para el criterio y reporte asimetricos.
peso_sub <- 1 - peso_sobre
escala_precio <- 1e9                  # Entrenar en miles de millones de COP.Pero devuelve la prediccion en niveles

modelo_e5 <- "multilingual-e5-small" # Revision fija en el registro de sbert.
prefijo_e5 <- "query: "
separador_texto <- " . "
lote_e5 <- 32L
hilos <- 4L

# ------------------------------------------------------------------------------
# (2) Leer los datos y definir la generacion de embeddings
# ------------------------------------------------------------------------------

if (is.null(getOption("taller3.modo_nn"))) {
  # Obtener las rutas y cargar los paquetes de main.R cuando se ejecuta 05 solo.
  opcion_nn <- options(taller3.modo_nn = "preparar")
  tryCatch(
    source("main.R", local = TRUE, encoding = "UTF-8"),
    finally = options(opcion_nn)
  )
}

train <- read_csv(archivo_train, show_col_types = FALSE)
test <- read_csv(archivo_test, show_col_types = FALSE)
plantilla <- read_csv(archivo_plantilla, show_col_types = FALSE)
orden_test <- match(plantilla$property_id, test$property_id)
stopifnot(!anyNA(train$price), !anyNA(orden_test))

armar_texto <- function(base) {
  titulo <- coalesce(base[[columnas_texto[1]]], "")
  descripcion <- coalesce(base[[columnas_texto[2]]], "")
  paste0(titulo, separador_texto, descripcion)
}

generar_embeddings_nn <- function() {
  torch_set_num_threads(hilos)
  stopifnot(criterio %in% c("mae", "asimetrica"), peso_sobre > 0, peso_sobre < 1)

  # Codificar una vez cada combinacion exacta de titulo y descripcion.
  textos <- c(armar_texto(train), armar_texto(test))
  textos_unicos <- unique(textos)
  indice_texto <- match(textos, textos_unicos)
  message(sprintf("E5: %d textos distintos. La primera corrida descarga los pesos.",
                  length(textos_unicos)))

  # sbert comprueba los archivos locales o descarga su revision fija de E5.
  cache_e5 <- file.path(cache_nn, "sbert")
  sbert::model_download(modelo_e5, cache_dir = cache_e5)
  encoder <- sbert::load_model(modelo_e5, cache_dir = cache_e5,
                              backend = "cpu", threads = hilos)
  codificar <- function(filas) {
    sbert::encode(textos_unicos[filas], model = encoder, batch_size = lote_e5,
                  normalize = TRUE, prefix = prefijo_e5, sort_by_length = TRUE)
  }

  # Medir una muestra dispersa y luego codificar en grupos para mostrar el avance.
  inicio_e5 <- Sys.time()
  n_textos <- length(textos_unicos)
  muestra <- unique(as.integer(round(seq(1, n_textos, length.out = min(256L, n_textos)))))
  pendientes <- setdiff(seq_len(n_textos), muestra)
  primera <- codificar(muestra)
  segundos_muestra <- as.numeric(difftime(Sys.time(), inicio_e5, units = "secs"))
  message(sprintf("E5: estimacion %.1f min restantes de codificacion.",
                  segundos_muestra * length(pendientes) / length(muestra) / 60))

  embeddings_unicos <- matrix(NA_real_, nrow = n_textos, ncol = ncol(primera))
  embeddings_unicos[muestra, ] <- primera
  grupos <- split(pendientes, (seq_along(pendientes) - 1L) %/% (32L * lote_e5))
  procesados <- length(muestra)
  for (grupo in grupos) {
    embeddings_unicos[grupo, ] <- codificar(grupo)
    procesados <- procesados + length(grupo)
    message(sprintf("E5: %d/%d (%.1f%%)", procesados, n_textos, 100 * procesados / n_textos))
  }
  segundos_e5 <- as.numeric(difftime(Sys.time(), inicio_e5, units = "secs"))
  stopifnot(ncol(embeddings_unicos) == 384L, all(is.finite(embeddings_unicos)))

  # Conservar el orden de ambas bases al recuperar los textos repetidos.
  list(
    train = embeddings_unicos[indice_texto[seq_len(nrow(train))], , drop = FALSE],
    test = embeddings_unicos[indice_texto[nrow(train) + seq_len(nrow(test))], , drop = FALSE],
    segundos = segundos_e5,
    encoder = list(modelo = encoder$id, alias = modelo_e5, revision = encoder$revision,
                   columnas = columnas_texto, prefijo = prefijo_e5,
                   separador = separador_texto, normalizar = TRUE,
                   backend = "onnx", version_sbert = as.character(getNamespaceVersion("sbert")))
  )
}

# ------------------------------------------------------------------------------
# (3) Preparar entradas y definir la red y las metricas
# ------------------------------------------------------------------------------

# Al ajustar se aprenden medianas y escala. Test reutiliza la misma referencia.
preparar_entrada <- function(base, embedding, ajuste = seq_len(nrow(base)),
                             referencia = NULL) {
  tabla <- as.data.frame(base[predictores])
  tabla$property_type <- as.integer(tabla$property_type == "Apartamento")
  tabla <- as.matrix(tabla)
  medianas <- if (is.null(referencia)) {
    apply(tabla[ajuste, , drop = FALSE], 2, median, na.rm = TRUE)
  } else {
    referencia$medianas
  }

  for (variable in colnames(tabla)) {
    tabla[is.na(tabla[, variable]), variable] <- medianas[[variable]]
  }
  if (is.null(referencia)) {
    grupo <- tabla[ajuste, , drop = FALSE]
    escala <- apply(grupo, 2, sd)
    escala[escala == 0] <- 1
    referencia <- list(medianas = medianas, centro = colMeans(grupo), escala = escala)
  }
  tabla <- scale(tabla, center = referencia$centro, scale = referencia$escala)
  x <- cbind(tabla, embedding)
  list(x = torch_tensor(x, dtype = torch_float()), referencia = referencia)
}

crear_red <- function(entradas, ocultas, dropout) {
  capas <- list()
  for (neuronas in ocultas) {
    capas <- c(capas, list(nn_linear(entradas, neuronas), nn_relu(), nn_dropout(dropout)))
    entradas <- neuronas
  }
  do.call(nn_sequential, c(capas, list(nn_linear(entradas, 1L))))
}

ajustar_red <- function(x, y, filas, ocultas, semilla_ajuste) {
  set.seed(semilla_ajuste)
  torch_manual_seed(semilla_ajuste)
  red <- crear_red(x$shape[2], ocultas, dropout)
  optimizador <- optim_adam(red$parameters, lr = tasa)

  red$train()
  for (epoca in seq_len(epocas)) {
    orden <- filas[sample.int(length(filas))]
    for (inicio in seq(1L, length(orden), by = tamano_lote)) {
      fin <- min(length(orden), inicio + tamano_lote - 1L)
      lote <- orden[inicio:fin]
      optimizador$zero_grad()
      error <- red(x[lote, , drop = FALSE])$squeeze(2) - y[lote]
      perdida <- if (criterio == "mae") {
        torch_abs(error)$mean()
      } else {
        torch_where(error > 0, peso_sobre * error, -peso_sub * error)$mean()
      }
      perdida$backward()
      optimizador$step()
    }
  }
  red
}

predecir_precio <- function(red, x) {
  red$eval()
  prediccion <- with_no_grad({ red(x)$squeeze(2) })
  as.numeric(as_array(prediccion)) * escala_precio
}

medir_error <- function(pred, real) {
  error <- pred - real
  sobre <- error > 0
  sub <- error < 0
  mae <- mean(abs(error))
  # Los aportes usan todos los avisos. Los MAE por lado solo los de ese signo.
  aporte_sobre <- peso_sobre * mean(pmax(error, 0))
  aporte_sub <- peso_sub * mean(pmax(-error, 0))
  asimetrica <- aporte_sobre + aporte_sub
  data.frame(
    perdida_cv_cop = if (criterio == "mae") mae else asimetrica,
    mae_cop = mae,
    mae_sobre_cop = if (any(sobre)) mean(error[sobre]) else NA_real_,
    mae_sub_cop = if (any(sub)) mean(-error[sub]) else NA_real_,
    perdida_asim_cop = asimetrica,
    perdida_sobre_cop = aporte_sobre,
    perdida_sub_cop = aporte_sub,
    pct_sobre = 100 * mean(sobre),
    sesgo_cop = mean(error)
  )
}

# ------------------------------------------------------------------------------
# (4) Definir particiones y validar la red
# ------------------------------------------------------------------------------

armar_esquema_nn <- function(fold, buffer_m = 0L, excluir = list()) {
  stopifnot(!anyNA(fold), length(unique(fold)) >= 2L)
  validaciones <- split(seq_along(fold), fold)
  particiones <- lapply(names(validaciones), function(nombre) {
    validacion <- validaciones[[nombre]]
    fuera <- excluir[[nombre]]
    ajuste <- setdiff(seq_along(fold), c(validacion, fuera))
    stopifnot(length(ajuste) > 1L)
    list(ajuste = ajuste, validacion = validacion, n_buffer = length(fuera))
  })
  names(particiones) <- names(validaciones)
  list(buffer_m = buffer_m, particiones = particiones)
}

experimento_aleatorio_nn <- function() {
  set.seed(semilla)
  fold <- sample(rep(seq_len(n_folds), length.out = nrow(train)))
  list(salida = dir_modelos_nn, esquemas = list(aleatoria = armar_esquema_nn(fold)))
}

validar_nn <- function(esquema, nombre, capas, ocultas, embeddings, y) {
  pred_cv <- rep(NA_real_, nrow(train))
  detalle <- vector("list", length(esquema$particiones))
  inicio_cv <- Sys.time()

  for (k in seq_along(esquema$particiones)) {
    particion <- esquema$particiones[[k]]
    ajuste <- particion$ajuste
    validacion <- particion$validacion
    entrada <- preparar_entrada(train, embeddings, ajuste = ajuste)
    red <- ajustar_red(entrada$x, y, ajuste, ocultas, semilla + k)
    pred_cv[validacion] <- predecir_precio(red, entrada$x[validacion, , drop = FALSE])
    detalle[[k]] <- cbind(
      validacion = nombre, buffer_m = esquema$buffer_m,
      fold = names(esquema$particiones)[k], n_ajuste = length(ajuste),
      n_validacion = length(validacion), n_excluidos_buffer = particion$n_buffer,
      medir_error(pred_cv[validacion], train$price[validacion])
    )
    message(sprintf("%s | %s | fold %d/%d (%.1f%%) | ajusta %d, valida %d, buffer %d",
                    capas, nombre, k, length(detalle), 100 * k / length(detalle),
                    length(ajuste), length(validacion), particion$n_buffer))
  }

  stopifnot(all(is.finite(pred_cv)))
  list(
    resumen = cbind(validacion = nombre, buffer_m = esquema$buffer_m,
                    medir_error(pred_cv, train$price),
                    k_folds = length(detalle), n_cv = nrow(train),
                    segundos_cv = as.numeric(difftime(Sys.time(), inicio_cv, units = "secs"))),
    detalle = bind_rows(detalle)
  )
}

# ------------------------------------------------------------------------------
# (5) Reajustar y exportar las arquitecturas y sus metricas
# ------------------------------------------------------------------------------

reajustar_nn <- function(ocultas, embeddings, y) {
  inicio_final <- Sys.time()
  entrada <- preparar_entrada(train, embeddings$train)
  entrada_test <- preparar_entrada(test, embeddings$test, referencia = entrada$referencia)
  red <- ajustar_red(entrada$x, y, seq_len(nrow(train)), ocultas, semilla)
  pred_test <- predecir_precio(red, entrada_test$x)
  stopifnot(all(is.finite(pred_test)))

  # Los pesos se guardan como arrays de R para reconstruir la red.
  arquitectura <- list(entradas = length(predictores) + ncol(embeddings$train),
                       ocultas = ocultas, salidas = 1L, activacion = "relu",
                       dropout = if (length(ocultas)) dropout else 0)
  modelo <- list(
    arquitectura = arquitectura, pesos = lapply(red$state_dict(), as_array),
    predictores = predictores, referencia_tabular = entrada$referencia,
    escala_precio = escala_precio,
    version_r = as.character(getRversion()),
    encoder = embeddings$encoder
  )
  predicciones <- plantilla
  predicciones$price <- pred_test[orden_test]
  list(modelo = modelo, predicciones = predicciones,
       segundos = as.numeric(difftime(Sys.time(), inicio_final, units = "secs")))
}

ejecutar_experimentos_nn <- function(experimentos) {
  embeddings <- generar_embeddings_nn()
  y <- torch_tensor(train$price / escala_precio, dtype = torch_float())
  reajustes <- vector("list", length(arquitecturas))

  for (experimento in experimentos) {
    esquemas <- experimento$esquemas
    filas_resumen <- vector("list", length(arquitecturas))
    hojas <- list(resumen = NULL)

    for (i in seq_along(arquitecturas)) {
      nombre <- names(arquitecturas)[i]
      ocultas <- arquitecturas[[i]]
      capas <- paste(c(length(predictores) + ncol(embeddings$train), ocultas, 1L),
                     collapse = "-")
      evaluaciones <- lapply(names(esquemas), function(esquema) {
        validar_nn(esquemas[[esquema]], esquema, capas, ocultas, embeddings$train, y)
      })

      # El reajuste usa todo train, independientemente del esquema de validacion.
      if (is.null(reajustes[[i]])) reajustes[[i]] <- reajustar_nn(ocultas, embeddings, y)
      final <- reajustes[[i]]
      salida <- file.path(experimento$salida, nombre)
      dir.create(salida, recursive = TRUE, showWarnings = FALSE)
      saveRDS(final$modelo, file.path(salida, "modelo.rds"))
      write_csv(final$predicciones, file.path(salida, "predicciones_test.csv"))
      message(sprintf("%s | modelo y CSV guardados", capas))

      filas_resumen[[i]] <- cbind(
        modelo = nombre, arquitectura = capas, criterio = criterio,
        bind_rows(lapply(evaluaciones, "[[", "resumen")),
        parametros = sum(lengths(final$modelo$pesos)), semilla = semilla,
        epocas = epocas, lote = tamano_lote, tasa = tasa,
        dropout = final$modelo$arquitectura$dropout, peso_sobre_asim = peso_sobre,
        peso_sub_asim = peso_sub, segundos_e5_compartidos = embeddings$segundos,
        segundos_reajuste_compartidos = final$segundos
      )
      hojas[[sprintf("nn_%02d", i)]] <- bind_rows(lapply(evaluaciones, "[[", "detalle"))
    }

    resumen <- bind_rows(filas_resumen)
    resumen <- resumen[order(resumen$validacion, resumen$perdida_cv_cop, resumen$mae_cop), ]
    hojas$resumen <- resumen
    if (!is.null(experimento$unidades)) hojas$unidades_espaciales <- experimento$unidades
    libro <- buildWorkbook(hojas, asTable = TRUE, firstRow = TRUE, colWidths = "auto")
    formato_cop <- createStyle(numFmt = "#,##0")
    for (hoja in names(hojas)) {
      cols_cop <- grep("_cop$", names(hojas[[hoja]]))
      if (length(cols_cop)) {
        addStyle(libro, hoja, formato_cop, rows = seq_len(nrow(hojas[[hoja]])) + 1L,
                 cols = cols_cop, gridExpand = TRUE)
      }
    }
    saveWorkbook(libro, file.path(experimento$salida, "resumen.xlsx"), overwrite = TRUE)
    print(resumen[c("validacion", "arquitectura", "perdida_cv_cop", "mae_cop",
                    "perdida_asim_cop")], row.names = FALSE)
  }
  message(sprintf("E5 compartido: %.1f min", embeddings$segundos / 60))
}

# ------------------------------------------------------------------------------
# (6) Ejecutar la validacion aleatoria
# ------------------------------------------------------------------------------

# Los bloques espaciales llaman al motor con sus propias particiones.
if (!identical(getOption("taller3.modo_nn"), "espacial")) {
  ejecutar_experimentos_nn(list(experimento_aleatorio_nn()))
}

