# ------------------------------------------------------------------------------
# Archivo: 08_models_nn_raw_data_base_r.R
# Proyecto: Taller 3 - Big Data y Machine Learning
# Descripcion:
#   Definir y ejecutar la red de datos crudos con cinco folds aleatorios.
# Input:
#   CSV originales y rutas de resultados y dependencias definidos en main.R.
#   Paquetes de R y binarios de ONNX preparados en 00_setup.R.
#   Pesos de E5 en el cache local del taller o descargados en la primera corrida.
# Output:
#   Base de embeddings E5 con property_id en 01_data/02_clean/.
#   Un modelo RDS y un CSV de test por arquitectura, y resumen.xlsx con
#   metricas fuera de fold en 03_outputs/02_models_nn_raw_data_base_r/.
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

tipo_id <- cols(property_id = col_character())
train <- read_csv(archivo_train, col_types = tipo_id, show_col_types = FALSE)
test <- read_csv(archivo_test, col_types = tipo_id, show_col_types = FALSE)
plantilla <- read_csv(archivo_plantilla, col_types = tipo_id, col_select = property_id,
                       show_col_types = FALSE)
orden_test <- match(plantilla$property_id, test$property_id)
stopifnot(!anyNA(train$price), !anyNA(test$property_id), !anyNA(plantilla$property_id),
          !anyDuplicated(test$property_id), !anyDuplicated(plantilla$property_id),
          setequal(test$property_id, plantilla$property_id), !anyNA(orden_test))

armar_texto <- function(base) {
  titulo <- coalesce(base[[columnas_texto[1]]], "")
  descripcion <- coalesce(base[[columnas_texto[2]]], "")
  paste0(titulo, separador_texto, descripcion)
}

generar_embeddings_nn <- function() {
  # La huella depende del texto y del encoder. Ignora el orden y los folds.
  textos <- c(armar_texto(train), armar_texto(test))
  id_train <- as.character(train$property_id)
  id_test <- as.character(test$property_id)
  ids <- c(id_train, id_test)
  muestras <- rep(c("train", "test"), c(length(id_train), length(id_test)))
  registro <- sbert::models(detail = TRUE)
  modelo <- registro[match(modelo_e5, registro$model), ]
  stopifnot(nrow(modelo) == 1L, !is.na(modelo$revision), !anyNA(ids),
            !anyDuplicated(id_train), !anyDuplicated(id_test))
  ficha <- list(modelo = modelo$id, alias = modelo_e5, revision = modelo$revision,
                columnas = columnas_texto, prefijo = prefijo_e5,
                separador = separador_texto, normalizar = TRUE, lote = lote_e5,
                backend = "onnx", version_sbert = as.character(getNamespaceVersion("sbert")))
  orden <- order(muestras, ids)
  huella <- digest::digest(list(ficha, muestras[orden], ids[orden], textos[orden]), algo = "sha256")
  archivo <- file.path(dir_datos_clean, paste0("embeddings_e5_raw_data_", substr(huella, 1, 12), ".rds"))
  if (file.exists(archivo)) {
    base_e5 <- readRDS(archivo)
    stopifnot(identical(base_e5$huella, huella))
    base_e5$archivo <- archivo
    base_e5$train <- base_e5$train[id_train, , drop = FALSE]
    base_e5$test <- base_e5$test[id_test, , drop = FALSE]
    base_e5$segundos <- 0
    message("Embeddings E5 leidos de ", archivo)
    return(base_e5)
  }

  # Codificar una vez cada combinacion exacta de titulo y descripcion.
  textos_unicos <- unique(textos)
  indice_texto <- match(textos, textos_unicos)
  message(sprintf("E5: %d textos distintos. La primera corrida descarga los pesos.",
                  length(textos_unicos)))

  # sbert comprueba los archivos locales o descarga su revision fija de E5.
  preparar_onnx()
  cache_e5 <- file.path(dir_dependencias, "sbert")
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

  # La fila se identifica por property_id. Guardar solo despues de codificar todo.
  base_e5 <- list(
    train = embeddings_unicos[indice_texto[seq_len(nrow(train))], , drop = FALSE],
    test = embeddings_unicos[indice_texto[nrow(train) + seq_len(nrow(test))], , drop = FALSE],
    segundos = segundos_e5, encoder = ficha, huella = huella
  )
  rownames(base_e5$train) <- id_train
  rownames(base_e5$test) <- id_test
  colnames(base_e5$train) <- colnames(base_e5$test) <- sprintf("e5_%03d", seq_len(ncol(embeddings_unicos)))
  dir.create(dir_datos_clean, recursive = TRUE, showWarnings = FALSE)
  temporal <- tempfile("embeddings-e5-", tmpdir = dir_datos_clean)
  on.exit(unlink(temporal), add = TRUE)
  saveRDS(base_e5, temporal, compress = FALSE)
  if (!file.rename(temporal, archivo)) {
    stop("No se pudo mover la base de embeddings a ", archivo, call. = FALSE)
  }
  message("Base de embeddings guardada en ", archivo)
  base_e5$archivo <- archivo
  base_e5
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
  list(x = x, referencia = referencia)
}

crear_red <- function(entradas, ocultas) {
  dimensiones <- c(entradas, ocultas, 1L)
  lapply(seq_len(length(dimensiones) - 1L), function(k) {
    anteriores <- dimensiones[k]
    neuronas <- dimensiones[k + 1L]
    limite <- 1 / sqrt(anteriores)
    list(pesos = matrix(runif(anteriores * neuronas, -limite, limite), anteriores),
         sesgo = runif(neuronas, -limite, limite))
  })
}

propagar_red <- function(red, x, tasa_dropout = 0) {
  entradas <- vector("list", length(red))
  derivadas <- vector("list", length(red) - 1L)
  for (k in seq_along(red)) {
    entradas[[k]] <- x
    salida <- sweep(x %*% red[[k]]$pesos, 2L, red[[k]]$sesgo, "+")
    if (k < length(red)) {
      # Dropout invertido: escalar las activaciones que siguen encendidas.
      mascara <- if (tasa_dropout == 0) 1 else {
        matrix(runif(length(salida)) >= tasa_dropout, nrow(salida)) / (1 - tasa_dropout)
      }
      derivadas[[k]] <- (salida > 0) * mascara
      x <- pmax(salida, 0) * mascara
    }
  }
  list(prediccion = as.numeric(salida), entradas = entradas, derivadas = derivadas)
}

ajustar_red <- function(x, y, filas, ocultas, semilla_ajuste) {
  set.seed(semilla_ajuste)
  red <- crear_red(ncol(x), ocultas)
  media_grad <- lapply(red, function(capa) lapply(capa, function(p) 0 * p))
  media_cuadrado <- media_grad
  paso <- 0L

  for (epoca in seq_len(epocas)) {
    orden <- filas[sample.int(length(filas))]
    for (inicio in seq(1L, length(orden), by = tamano_lote)) {
      fin <- min(length(orden), inicio + tamano_lote - 1L)
      lote <- orden[inicio:fin]
      paso <- paso + 1L
      avance <- propagar_red(red, x[lote, , drop = FALSE], dropout)
      error <- avance$prediccion - y[lote]
      pendiente <- if (criterio == "mae") sign(error) else ifelse(error > 0, peso_sobre, -peso_sub)
      grad_salida <- matrix(pendiente / length(lote), ncol = 1L)

      # Derivar desde la salida hacia la entrada y actualizar con Adam.
      for (k in rev(seq_along(red))) {
        grad <- list(pesos = crossprod(avance$entradas[[k]], grad_salida),
                     sesgo = colSums(grad_salida))
        if (k > 1L) {
          grad_salida <- tcrossprod(grad_salida, red[[k]]$pesos) * avance$derivadas[[k - 1L]]
        }
        for (nombre in names(grad)) {
          g <- grad[[nombre]]
          m <- 0.9 * media_grad[[k]][[nombre]] + 0.1 * g
          v <- 0.999 * media_cuadrado[[k]][[nombre]] + 0.001 * g^2
          red[[k]][[nombre]] <- red[[k]][[nombre]] -
            tasa * (m / (1 - 0.9^paso)) / (sqrt(v / (1 - 0.999^paso)) + 1e-8)
          media_grad[[k]][[nombre]] <- m
          media_cuadrado[[k]][[nombre]] <- v
        }
      }
    }
  }
  red
}

predecir_precio <- function(red, x) {
  propagar_red(red, x)$prediccion * escala_precio
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

set.seed(semilla)
fold <- sample(rep(seq_len(n_folds), length.out = nrow(train)))
validaciones <- split(seq_along(fold), fold)

validar_nn <- function(capas, ocultas, embeddings, y) {
  pred_cv <- rep(NA_real_, nrow(train))
  inicio_cv <- Sys.time()

  for (k in seq_along(validaciones)) {
    validacion <- validaciones[[k]]
    ajuste <- setdiff(seq_len(nrow(train)), validacion)
    entrada <- preparar_entrada(train, embeddings, ajuste = ajuste)
    red <- ajustar_red(entrada$x, y, ajuste, ocultas, semilla + k)
    pred_cv[validacion] <- predecir_precio(red, entrada$x[validacion, , drop = FALSE])
    message(sprintf("%s | fold %d/%d (%.1f%%)", capas, k, length(validaciones),
                    100 * k / length(validaciones)))
  }

  stopifnot(all(is.finite(pred_cv)))
  cbind(validacion = "aleatoria", medir_error(pred_cv, train$price),
        k_folds = length(validaciones), n_cv = nrow(train),
        segundos_cv = as.numeric(difftime(Sys.time(), inicio_cv, units = "secs")))
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
  stopifnot(length(pred_test) == nrow(test), all(is.finite(pred_test)))

  # Los pesos se guardan como arrays de R para reconstruir la red.
  arquitectura <- list(entradas = length(predictores) + ncol(embeddings$train),
                       ocultas = ocultas, salidas = 1L, activacion = "relu",
                       dropout = if (length(ocultas)) dropout else 0)
  modelo <- list(
    arquitectura = arquitectura, pesos = red,
    predictores = predictores, referencia_tabular = entrada$referencia,
    escala_precio = escala_precio,
    version_r = as.character(getRversion()),
    encoder = embeddings$encoder
  )
  predicciones <- data.frame(property_id = plantilla$property_id, price = pred_test[orden_test])
  list(modelo = modelo, predicciones = predicciones,
       segundos = as.numeric(difftime(Sys.time(), inicio_final, units = "secs")))
}

ejecutar_nn <- function(embeddings) {
  stopifnot(criterio %in% c("mae", "asimetrica"), peso_sobre > 0, peso_sobre < 1)
  y <- train$price / escala_precio
  filas_resumen <- vector("list", length(arquitecturas))

  for (i in seq_along(arquitecturas)) {
    ocultas <- arquitecturas[[i]]
    capas <- paste(c(length(predictores) + ncol(embeddings$train), ocultas, 1L),
                   collapse = "-")
    cv <- validar_nn(capas, ocultas, embeddings$train, y)
    final <- reajustar_nn(ocultas, embeddings, y)
    salida <- file.path(dir_modelos_nn, names(arquitecturas)[i])
    dir.create(salida, recursive = TRUE, showWarnings = FALSE)
    saveRDS(final$modelo, file.path(salida, "modelo.rds"))
    nombre <- sprintf("nn_e5_capas_%s_dropout_%.2f_epocas_%d_tasa_%g_lote_%d.csv",
                      capas, final$modelo$arquitectura$dropout, epocas, tasa, tamano_lote)
    write_csv(final$predicciones, file.path(salida, nombre))

    filas_resumen[[i]] <- cbind(
      modelo = nombre, arquitectura = capas, criterio = criterio, cv,
      semilla = semilla, epocas = epocas, lote = tamano_lote, tasa = tasa,
      dropout = final$modelo$arquitectura$dropout,
      segundos_e5 = embeddings$segundos, segundos_reajuste = final$segundos
    )
  }
  resumen <- bind_rows(filas_resumen)
  resumen <- resumen[order(resumen$perdida_cv_cop, resumen$mae_cop), ]
  write.xlsx(list(resumen = resumen), file.path(dir_modelos_nn, "resumen.xlsx"),
             overwrite = TRUE)
  print(resumen[c("arquitectura", "perdida_cv_cop", "mae_cop")], row.names = FALSE)
}
