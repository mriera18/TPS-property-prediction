# ==============================================================================
# PROYECTO DE PROPIEDADES DE BIOPLÁSTICOS (SHINY APP)
# ==============================================================================

library(shiny)
library(bslib)
library(dplyr)
library(randomForest)
library(xgboost)
library(nnet)
library(caret)

# 1. CARGA DE METADATOS (Se ejecuta una vez al encender la app)
# Leemos el "mapa" que creaste con tu script principal
tryCatch({
  metadata <- readRDS("modelos/metadata_shiny.rds")
  targets_disponibles <- metadata$Propiedad
}, error = function(e) {
  stop("Error: No se encontró metadata_shiny.rds. Ejecuta el script de entrenamiento primero.")
})

# 2. INTERFAZ VISUAL (UI)
ui <- page_sidebar(
  theme = bs_theme(bootswatch = "flatly", primary = "#2C3E50"),
  title = "Predictor de Bioplásticos Asistido por Machine Learning",
  
  # BARRA LATERAL (Inputs)
  sidebar = sidebar(
    width = 350,
    h5("1. Propiedad a predecir", style = "color: #2C3E50; font-weight: bold;"),
    selectInput("target_sel", label = NULL, choices = targets_disponibles),
    
    hr(),
    
    h5("2. Formulación y Proceso", style = "color: #2C3E50; font-weight: bold;"),
    p("Ingrese los valores para las variables seleccionadas por el algoritmo:"),
    
    # Aquí aparecerán los cuadros de texto dinámicamente
    uiOutput("controles_dinamicos"),
    
    hr(),
    actionButton("btn_predict", "Proyectar Resultado", 
                 class = "btn-success btn-lg w-100", 
                 icon = icon("calculator"))
  ),
  
  # PANEL PRINCIPAL (Resultados)
  layout_columns(
    col_widths = 12,
    
    # Tarjeta de Resultado Principal
    card(
      full_screen = TRUE,
      card_header(
        class = "bg-primary text-white",
        span(icon("chart-line"), " Resultado de la Predicción")
      ),
      card_body(
        class = "text-center",
        h3("Valor Estimado:", style = "color: #7F8C8D; margin-top: 20px;"),
        h1(textOutput("pred_valor"), style = "color: #E74C3C; font-size: 4rem; font-weight: bold;"),
        hr(),
        h5("Detalles del Modelo:", style = "color: #7F8C8D;"),
        h4(textOutput("info_modelo"), style = "color: #2980B9;")
      )
    ),
    
    # Tarjeta de Metodología (Para dar peso científico)
    card(
      card_header("Información Metodológica"),
      card_body(
        p("Esta proyección es generada por un modelo de Machine Learning entrenado mediante un metaanálisis de la literatura científica. Los hiperparámetros y el proceso de feature selection fueron optimizados utilizando la metodología CRISP-ML."),
        p(strong("Nota para investigadores:"), " Las variables requeridas en el panel izquierdo cambian dinámicamente dependiendo de la propiedad seleccionada, ya que los algoritmos (Random Forest, XGBoost o Redes Neuronales) filtran los predictores más relevantes para evitar la maldición de la dimensionalidad.")
      )
    )
  )
)

# 3. LÓGICA DEL SERVIDOR (El cerebro matemático)
server <- function(input, output, session) {
  
  # Reactivo: Obtener qué modelo ganó para el target actual y sus variables
  info_target <- reactive({
    req(input$target_sel)
    target <- input$target_sel
    algo <- metadata$Mejor_Modelo[metadata$Propiedad == target]
    
    # Leer el archivo con los nombres de las columnas correctas
    ruta_features <- paste0("features/features_", algo, "_", target, ".rds")
    vars <- readRDS(ruta_features)
    
    list(target = target, algo = algo, vars = vars)
  })
  
  # Renderizar los inputs dinámicamente según la propiedad elegida
  output$controles_dinamicos <- renderUI({
    vars <- info_target()$vars
    
    # Crea un cuadro de número para cada variable que el modelo necesita
    lapply(vars, function(v) {
      numericInput(inputId = v, 
                   label = paste(v, ":"), 
                   value = 0, # Empiezan en 0 por defecto
                   step = 0.1)
    })
  })
  
  # Evento: Cuando el usuario presiona el botón "Proyectar"
  observeEvent(input$btn_predict, {
    info <- info_target()
    
    # 1. Recolectar lo que el usuario escribió
    valores_usuario <- lapply(info$vars, function(v) {
      # Si por algún motivo está vacío, asigna 0
      val <- input[[v]]
      if (is.null(val) || is.na(val)) return(0) else return(val)
    })
    
    # 2. Armar el dataframe (Tiene que tener el mismo orden y nombres)
    df_input <- as.data.frame(valores_usuario)
    colnames(df_input) <- info$vars
    
    # 3. Hacer la predicción dependiendo del algoritmo ganador
    resultado <- NA
    
    tryCatch({
      if (info$algo == "rf") {
        mod <- readRDS(paste0("modelos/modelo_rf_", info$target, ".rds"))
        resultado <- predict(mod, df_input)
        
      } else if (info$algo == "xgb") {
        mod <- readRDS(paste0("modelos/modelo_xgb_", info$target, ".rds"))
        # XGBoost requiere matriz
        resultado <- predict(mod, as.matrix(df_input))
        
      } else if (info$algo == "nn") {
        mod <- readRDS(paste0("modelos/modelo_nn_", info$target, ".rds"))
        preproc <- readRDS(paste0("preprocess/preproc_", info$target, ".rds"))
        # Red neuronal requiere escalar los datos del usuario primero
        df_escalado <- predict(preproc, df_input)
        resultado <- predict(mod, df_escalado)
      }
      
      # 4. Mostrar Resultados
      output$pred_valor <- renderText({ 
        paste(round(resultado, 3)) 
      })
      
      # Traducir el nombre del algoritmo para la pantalla
      nombre_algo_bonito <- switch(info$algo,
                                   "rf" = "Random Forest",
                                   "xgb" = "Extreme Gradient Boosting (XGBoost)",
                                   "nn" = "Red Neuronal Artificial (ANN)")
      
      output$info_modelo <- renderText({ 
        paste("Algoritmo Óptimo Utilizado:", nombre_algo_bonito) 
      })
      
    }, error = function(e) {
      output$pred_valor <- renderText({ "Error en cálculo" })
      output$info_modelo <- renderText({ paste("Revise los datos ingresados.", e$message) })
    })
  })
}

# 4. LANZAR APP
shinyApp(ui, server)