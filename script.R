# =========================================================
# 1. LIBRERÍAS
# =========================================================
library(readxl)
library(tidyverse)
library(VIM)
library(corrplot)
library(randomForest)
library(caret)
library(Metrics)
library(xgboost)
library(nnet)
library(ggplot2)
library(dplyr)
library(tidytext)
library(reshape2)
library(tidyr)

# Creamos las carpetas una sola vez al inicio
dir.create("modelos", showWarnings = FALSE)
dir.create("features", showWarnings = FALSE)
dir.create("preprocess", showWarnings = FALSE)

set.seed(123)

# =========================================================
# 2. CARGA DE DATOS
# =========================================================

df <- read_excel("C:/Users/UTM/Documents/Maestría Cs Datos/TESIS/DATOS Y PAPER/BD_version9.xlsx")

# =========================================================
# 3. PREPROCESAMIENTO
# =========================================================

# --- Almidones
nombres_almidones <- names(df)[1:51]

df_starch <- df %>%
  select(all_of(nombres_almidones)) %>%
  mutate(across(everything(), as.numeric)) %>%
  mutate(across(everything(), ~replace_na(., 0)))

df_starch_clean <- df_starch %>% select(where(~ var(.) > 0))

df$Dominant_Starch <- colnames(df_starch_clean)[max.col(df_starch_clean, ties.method = "first")]
df$Dominant_Starch <- as.factor(df$Dominant_Starch)

# --- Limpieza lógica
df <- df %>%
  mutate(
    screw_speed = ifelse(extrusion == 0, 0, screw_speed),
    rpm = ifelse(magnetic_stirring == 0 & mixed == 0, 0, rpm),
    Pressure = ifelse(extrusion == 0 & compression == 0, 0, Pressure),
    Tdrying2 = replace_na(Tdrying2, 0),
    tdrying2 = replace_na(tdrying2, 0)
  )

# --- SOLUCIÓN REVISOR 2: SE ELIMINA LA IMPUTACIÓN GLOBAL PARA EVITAR DATA LEAKAGE ---
variables_a_imputar <- c("process_temperature", "process_time", "rpm", 
                         "screw_speed", "Pressure", "Tdrying1", "tdrying1",
                         "Tdrying2", "tdrying2")

# Generamos la variable calculada (los NA se propagan temporalmente y se imputarán dentro del loop)
df <- df %>%
  mutate(
    ratio_temp_agua = process_temperature / (water + 1e-5)
  )

# =========================================================
# 4. VARIABLES NUMÉRICAS
# =========================================================
df_num <- df %>% select(where(is.numeric))

# =========================================================
# 🔍 DIAGNÓSTICO DEL DATASET (PARA DISCUSIÓN DEL PAPER)
# =========================================================

# 1. % NA por variable
na_summary <- data.frame(
  Variable = colnames(df_num),
  NA_pct = colMeans(is.na(df_num)) * 100
) %>%
  arrange(desc(NA_pct))

print("Top variables con más NA:")
print(head(na_summary, 15))

top_na <- na_summary %>% head(15)

#####
#FIGURA 2: Top variables con más NA###

ggplot(top_na, aes(x = reorder(Variable, NA_pct), y = NA_pct)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    title = "Variables con mayor porcentaje de datos faltantes",
    x = "Variable",
    y = "% NA"
  ) +
  theme_minimal()


# 2. % NA global
total_na_pct <- sum(is.na(df_num)) / 
  (nrow(df_num) * ncol(df_num)) * 100

cat("Porcentaje total de NA en el dataset:", total_na_pct, "%\n")


# 3. Variables de baja varianza
varianza <- sapply(df_num, var, na.rm = TRUE)
low_var <- names(varianza[varianza < 1e-5])

cat("Número de variables con baja varianza:", length(low_var), "\n")
print(head(low_var, 20))


# 4. Variables binarias
binarias <- sapply(df_num, function(x) length(unique(na.omit(x))) == 2)

cat("Número de variables binarias:", sum(binarias), "\n")
print(head(names(df_num)[binarias], 20))


# 5. Número de datos por target
lista_targets <- c("tensile_strength", "Young", "elongation_p", "T50", "Tm", "Td", 
                   "water_abs", "WVP", "water_solubility", "degradation_weight")

n_por_target <- sapply(lista_targets, function(t) sum(!is.na(df_num[[t]])))

cat("Número de datos por target:\n")
print(n_por_target)

#######
##FIGURA 1: Número de datos por target

df_targets <- data.frame(
  Target = names(n_por_target),
  N = as.numeric(n_por_target)
)


ggplot(df_targets, aes(x = reorder(Target, N), y = N, fill = N)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "lightblue", high = "#004b96") + 
  coord_flip() +
  scale_x_discrete(labels = c(
    "tensile_strength" = "Tensile strength",
    "elongation_p"     = "Elongation %",
    "water_abs"        = "Water absorption",
    "Young" = "Young's modulus",
    "degradation_weight" = "Weight degradation",
    "water_solubility" = "Water solubility",
    "WVP" = "WVP",
    "T50" = "T50",
    "Td" = "Td",
    "Tm" = "Tm"
  )) +
  labs(
    x = "Target variable",
    y = "Number of data points"
  ) +
  theme_minimal() +
  theme(
    text = element_text(family = "serif"), 
    legend.position = "none",
    panel.grid.major.y = element_blank()
  )

# =========================================================
# 5. TARGETS
# =========================================================
categorias <- list(
  Mecanicas   = c("tensile_strength", "Young", "elongation_p"),
  Termicas    = c("T50", "Tm", "Td"),
  Barrera     = c("water_abs", "WVP", "water_solubility"),
  Degradacion = c("degradation_weight")
)

todos_targets <- unlist(categorias)

# =========================================================
# 6. FUNCIÓN LIMPIEZA POR TARGET (CORREGIDA)
# =========================================================
# MODIFICACIÓN: Ya no eliminamos columnas con NA, ya que se imputarán legítimamente dentro del loop.
limpiar_por_target <- function(df_num, target, todos_targets) {
  predictores <- setdiff(colnames(df_num), todos_targets)
  df_num %>%
    select(all_of(target), all_of(predictores)) %>%
    drop_na(all_of(target)) %>%
    select(where(~ sd(., na.rm = TRUE) > 0))
}

# =========================================================
# OBSERVACIONES DESPUÉS DEL PREPROCESAMIENTO
# =========================================================

tabla_observaciones <- data.frame(
  Target = todos_targets,
  Observaciones = sapply(todos_targets, function(target) {
    nrow(limpiar_por_target(df_num, target, todos_targets))
  })
)

print(tabla_observaciones)

# =========================================================
# 7. INICIALIZACIÓN DE VARIABLES
# =========================================================
resultados <- NULL
lista_importancias <- list()
lista_correlaciones <- list()
predicciones <- list()

# =========================================================
# 8. LOOP PRINCIPAL (CORREGIDO CON FILTRO DE VARIANZA CERO)
# =========================================================

###Fijar semilla
set.seed(123)

r2 <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

for (categoria in names(categorias)) {
  
  for (target in categorias[[categoria]]) {
    
    if (!target %in% colnames(df_num)) next
    
    cat("\nProcesando:", target, "\n")
    
    # Limpieza específica aislando el target actual
    df_modelo <- limpiar_por_target(df_num, target, todos_targets)
    
    if (nrow(df_modelo) < 30) next
    
    # -----------------------------
    # División 70:20:10
    # -----------------------------
    idx_train <- createDataPartition(df_modelo[[target]], p = 0.7, list = FALSE)
    train <- df_modelo[idx_train, ]
    temp  <- df_modelo[-idx_train, ]
    
    idx_val <- createDataPartition(temp[[target]], p = 2/3, list = FALSE)
    val  <- temp[idx_val, ]
    test <- temp[-idx_val, ]
    
    if(target == "tensile_strength"){
      
      cat("\n===============================\n")
      cat("TENSILE STRENGTH\n")
      cat("===============================\n")
      
      cat("\nTRAIN\n")
      print(summary(train[[target]]))
      
      cat("\nVALIDATION\n")
      print(summary(val[[target]]))
      
      cat("\nTEST\n")
      print(summary(test[[target]]))
      
      cat("\nNúmero de datos\n")
      cat("Train:", nrow(train), "\n")
      cat("Validation:", nrow(val), "\n")
      cat("Test:", nrow(test), "\n")
    }
    
    if(target == "tensile_strength"){
      
      cat("\nValores mayores de 40 MPa\n")
      
      cat("Train :", sum(train[[target]] > 40), "\n")
      cat("Validation :", sum(val[[target]] > 40), "\n")
      cat("Test :", sum(test[[target]] > 40), "\n")
      
    }
    
    
    # -------------------------------------------------------------
    # IMPUTACIÓN INTERNA LEGÍTMA
    # -------------------------------------------------------------
    imputador_sano <- preProcess(train, method = "medianImpute")
    
    train <- predict(imputador_sano, train)
    val   <- predict(imputador_sano, val)
    test  <- predict(imputador_sano, test)
    
    # -------------------------------------------------------------
    # ELIMINAR VARIABLES CONSTANTES EN ESTE SPLIT
    # -------------------------------------------------------------
    pred_vars_check <- train %>% select(-all_of(target))
    vars_constantes <- names(pred_vars_check)[sapply(pred_vars_check, function(x) sd(x, na.rm = TRUE) == 0)]
    
    if (length(vars_constantes) > 0) {
      cat("  Variables con varianza cero detectadas en este split y removidas:", vars_constantes, "\n")
      train <- train %>% select(-all_of(vars_constantes))
      val   <- val   %>% select(-all_of(vars_constantes))
      test  <- test  %>% select(-all_of(vars_constantes))
    }
    
    # -------------------------------------------------------------
    # ELIMINAR COLINEALIDAD (Umbral > 0.85) - Ahora seguro y sin NAs
    # -------------------------------------------------------------
    pred_vars <- train %>% select(-all_of(target))
    cor_matrix <- cor(pred_vars, use = "pairwise.complete.obs")
    
    altamente_correlacionadas <- findCorrelation(cor_matrix, cutoff = 0.85)
    
    if (length(altamente_correlacionadas) > 0) {
      vars_a_quitar <- names(pred_vars)[altamente_correlacionadas]
      cat("  Variables colineales removidas:", vars_a_quitar, "\n")
      train <- train %>% select(-all_of(vars_a_quitar))
      val   <- val   %>% select(-all_of(vars_a_quitar))
      test  <- test  %>% select(-all_of(vars_a_quitar))
    }
    
    cat("Número de variables tras filtro:", ncol(train)-1, "\n")
    
    # -----------------------------
    # FEATURE SELECTION (XGBoost)
    # -----------------------------
    train_x <- as.matrix(train %>% select(-all_of(target)))
    train_y <- train[[target]]
    
    modelo_fs <- xgboost(
      x = train_x,
      y = train_y,
      nrounds = 100,
      objective = "reg:squarederror",
      verbosity = 0
    )
    
    imp <- xgb.importance(model = modelo_fs)
    if (nrow(imp) == 0) next
    
    imp_top <- imp[1:min(15, nrow(imp)), ]
    imp_top$Target <- target
    
    lista_importancias[[target]] <- imp_top
    
    top_vars <- imp_top$Feature
    top_vars <- top_vars[top_vars %in% colnames(train)]
    
    # Guardar features para cada modelo (para Shiny)
    saveRDS(top_vars, paste0("features/features_rf_", target, ".rds"))
    saveRDS(top_vars, paste0("features/features_xgb_", target, ".rds"))
    saveRDS(top_vars, paste0("features/features_nn_", target, ".rds"))
    
    if (length(top_vars) == 0) next
    
    cat("Top variables:\n")
    print(top_vars)
    
    # -----------------------------
    # Reducir datasets
    # -----------------------------
    train_reduced <- train %>% select(all_of(top_vars), all_of(target))
    val_reduced   <- val   %>% select(all_of(top_vars), all_of(target))
    test_reduced  <- test  %>% select(all_of(top_vars), all_of(target))
    
    # =========================================================
    # DIAGNÓSTICO DEL TARGET ACTUAL
    # =========================================================
    
    cat("\n===============================\n")
    cat("TARGET:", target, "\n")
    cat("===============================\n")
    
    cat("Observaciones:\n")
    cat("Train:", nrow(train_reduced), "\n")
    cat("Validation:", nrow(val_reduced), "\n")
    cat("Test:", nrow(test_reduced), "\n\n")
    
    cat("Resumen del target (TRAIN)\n")
    print(summary(train_reduced[[target]]))
    
    cat("\nResumen del target (TEST)\n")
    print(summary(test_reduced[[target]]))
    
    cat("\nDesviación estándar del target\n")
    cat("Train:", sd(train_reduced[[target]]), "\n")
    cat("Test :", sd(test_reduced[[target]]), "\n")
    
    cat("\nNúmero de valores únicos\n")
    cat("Train:", length(unique(train_reduced[[target]])), "\n")
    cat("Test :", length(unique(test_reduced[[target]])), "\n")
    
    cat("\nNA en Train:", sum(is.na(train_reduced[[target]])), "\n")
    cat("NA en Test :", sum(is.na(test_reduced[[target]])), "\n")
    
    cat("===============================\n")
    
    # -----------------------------
    # MATRIZ DE CORRELACIÓN
    # -----------------------------
    top_vars_cor <- top_vars[1:min(10, length(top_vars))]
    
    df_cor <- train_reduced %>%
      select(all_of(top_vars_cor)) %>%
      select(where(~ sd(., na.rm = TRUE) > 0))
    
    if (ncol(df_cor) > 2) {
      matriz_cor <- cor(df_cor, use = "pairwise.complete.obs")
      matriz_cor[upper.tri(matriz_cor)] <- NA
      diag(matriz_cor) <- NA  
      
      df_cor_long <- reshape2::melt(matriz_cor)
      df_cor_long$Target <- target
      
      lista_correlaciones[[target]] <- df_cor_long
    }
    
    # =========================================================
    # MODELOS
    # =========================================================
    
    # Si el test tiene varianza casi nula se avisa
    if (sd(test_reduced[[target]]) < 1e-10) {
      
      cat("\n---------------------------------\n")
      cat("ATENCIÓN\n")
      cat("El conjunto TEST tiene varianza prácticamente cero\n")
      cat("R² no será confiable para:", target, "\n")
      cat("---------------------------------\n")
      
    }
    
    # --- RANDOM FOREST
    modelo_rf <- randomForest(as.formula(paste(target, "~ .")), data = train_reduced)
    
    pred_rf_train <- predict(modelo_rf, train_reduced)
    pred_rf_val   <- predict(modelo_rf, val_reduced)
    pred_rf_test  <- predict(modelo_rf, test_reduced)
    
    # --- XGBOOST
    modelo_xgb <- xgboost(
      x = as.matrix(train_reduced %>% select(-all_of(target))),
      y = train_reduced[[target]],
      nrounds = 100,
      objective = "reg:squarederror",
      verbosity = 0
    )
    
    pred_xgb_train <- predict(modelo_xgb, as.matrix(train_reduced %>% select(-all_of(target))))
    pred_xgb_val   <- predict(modelo_xgb, as.matrix(val_reduced %>% select(-all_of(target))))
    pred_xgb_test  <- predict(modelo_xgb, as.matrix(test_reduced %>% select(-all_of(target))))
    
    # --- ANN
    pre <- preProcess(train_reduced %>% select(-all_of(target)), method = c("center", "scale"))
    
    train_nn <- predict(pre, train_reduced %>% select(-all_of(target)))
    val_nn   <- predict(pre, val_reduced %>% select(-all_of(target)))
    test_nn  <- predict(pre, test_reduced %>% select(-all_of(target)))
    
    train_nn$y <- train_reduced[[target]]
    
    modelo_nn <- nnet(y ~ ., data = train_nn, size = 5, decay = 0.1, linout = TRUE, trace = FALSE, maxit = 500)
    
    pred_nn_train <- predict(modelo_nn, train_nn)
    pred_nn_val   <- predict(modelo_nn, val_nn)
    pred_nn_test  <- predict(modelo_nn, test_nn)
    
    saveRDS(modelo_rf, paste0("modelos/modelo_rf_", target, ".rds"))
    saveRDS(modelo_xgb, paste0("modelos/modelo_xgb_", target, ".rds"))
    saveRDS(modelo_nn, paste0("modelos/modelo_nn_", target, ".rds"))
    saveRDS(pre, paste0("preprocess/preproc_", target, ".rds"))
    
    # =========================================================
    # GUARDAR PREDICCIONES 
    # =========================================================
    predicciones[[target]] <- data.frame(
      Target = target,
      Real   = test_reduced[[target]],
      RF     = pred_rf_test,
      XGB    = pred_xgb_test,
      ANN     = pred_nn_test
    )
    
    # =========================================================
    # MÉTRICAS COMPLETAS (TRAIN / VAL / TEST)
    # =========================================================
    
    # -------- R2 --------
    R2_RF_train  <- r2(train_reduced[[target]], pred_rf_train)
    R2_RF_val    <- r2(val_reduced[[target]], pred_rf_val)
    R2_RF_test   <- r2(test_reduced[[target]], pred_rf_test)
    
    R2_XGB_train <- r2(train_reduced[[target]], pred_xgb_train)
    R2_XGB_val   <- r2(val_reduced[[target]], pred_xgb_val)
    R2_XGB_test  <- r2(test_reduced[[target]], pred_xgb_test)
    
    R2_NN_train  <- r2(train_reduced[[target]], pred_nn_train)
    R2_NN_val    <- r2(val_reduced[[target]], pred_nn_val)
    R2_NN_test   <- r2(test_reduced[[target]], pred_nn_test)
    
    resultados <- rbind(resultados, data.frame(
      Categoria = categoria,
      Target = target,
      
      # -------- R2 --------
      R2_RF_train = R2_RF_train,
      R2_RF_val   = R2_RF_val,
      R2_RF_test  = R2_RF_test,
      
      R2_XGB_train = R2_XGB_train,
      R2_XGB_val   = R2_XGB_val,
      R2_XGB_test  = R2_XGB_test,
      
      R2_NN_train = R2_NN_train,
      R2_NN_val   = R2_NN_val,
      R2_NN_test  = R2_NN_test,
      
      # -------- OVERFITTING --------
      Overfit_RF  = R2_RF_train - R2_RF_test,
      Overfit_XGB = R2_XGB_train - R2_XGB_test,
      Overfit_NN  = R2_NN_train - R2_NN_test,
      
      # -------- RMSE --------
      RMSE_RF_train = rmse(train_reduced[[target]], pred_rf_train),
      RMSE_RF_val   = rmse(val_reduced[[target]], pred_rf_val),
      RMSE_RF_test  = rmse(test_reduced[[target]], pred_rf_test),
      
      RMSE_XGB_train = rmse(train_reduced[[target]], pred_xgb_train),
      RMSE_XGB_val   = rmse(val_reduced[[target]], pred_xgb_val),
      RMSE_XGB_test  = rmse(test_reduced[[target]], pred_xgb_test),
      
      RMSE_NN_train = rmse(train_reduced[[target]], pred_nn_train),
      RMSE_NN_val   = rmse(val_reduced[[target]], pred_nn_val),
      RMSE_NN_test  = rmse(test_reduced[[target]], pred_nn_test),
      
      # -------- MAE --------
      MAE_RF_train = mae(train_reduced[[target]], pred_rf_train),
      MAE_RF_val   = mae(val_reduced[[target]], pred_rf_val),
      MAE_RF_test  = mae(test_reduced[[target]], pred_rf_test),
      
      MAE_XGB_train = mae(train_reduced[[target]], pred_xgb_train),
      MAE_XGB_val   = mae(val_reduced[[target]], pred_xgb_val),
      MAE_XGB_test  = mae(test_reduced[[target]], pred_xgb_test),
      
      MAE_NN_train = mae(train_reduced[[target]], pred_nn_train),
      MAE_NN_val   = mae(val_reduced[[target]], pred_nn_val),
      MAE_NN_test  = mae(test_reduced[[target]], pred_nn_test)
    ))
  }}

# =========================================================
# GUARDAR PREDICCIONES POR TARGET
# =========================================================
str(predicciones)
length(predicciones)

# =========================================================
# RESULTADOS
# =========================================================
print(resultados)

#Promedios por modelo
# =========================
# PROMEDIOS R2
# =========================
colMeans(resultados[,c(
  "R2_RF_train","R2_RF_val","R2_RF_test",
  "R2_XGB_train","R2_XGB_val","R2_XGB_test",
  "R2_NN_train","R2_NN_val","R2_NN_test"
)])

# =========================
# PROMEDIOS RMSE
# =========================
colMeans(resultados[,c(
  "RMSE_RF_train","RMSE_RF_val","RMSE_RF_test",
  "RMSE_XGB_train","RMSE_XGB_val","RMSE_XGB_test",
  "RMSE_NN_train","RMSE_NN_val","RMSE_NN_test"
)])

# =========================
# PROMEDIOS MAE
# =========================
colMeans(resultados[,c(
  "MAE_RF_train","MAE_RF_val","MAE_RF_test",
  "MAE_XGB_train","MAE_XGB_val","MAE_XGB_test",
  "MAE_NN_train","MAE_NN_val","MAE_NN_test"
)])

df_long <- resultados %>%
  select(Target,
         R2_RF_train, R2_RF_val, R2_RF_test,
         R2_XGB_train, R2_XGB_val, R2_XGB_test,
         R2_NN_train, R2_NN_val, R2_NN_test) %>%
  pivot_longer(-Target)

ggplot(df_long, aes(x=name, y=value)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title="Distribución de R² por modelo y dataset")


#Best Model
resultados$Best_Model <- apply(
  resultados[,c("R2_RF_test","R2_XGB_test","R2_NN_test")],
  1,
  function(x) c("RF","XGB","ANN")[which.max(x)]
)

table(resultados$Best_Model)

ggplot(resultados, aes(x=Best_Model)) +
  geom_bar() +
  theme_minimal() +
  labs(
    title="Frecuencia del mejor modelo por variable objetivo",
    x="Modelo",
    y="Número de veces que fue el mejor"
  )

# =========================================================
# FIGURAS: Heatmaps individuales por target
# =========================================================

df_cor_total <- bind_rows(lista_correlaciones) %>%
  drop_na(value)

# Identificar targets válidos que realmente se procesaron y guardaron correlación
targets_cor <- intersect(todos_targets, unique(df_cor_total$Target))

for (t in targets_cor) {
  
  df_plot <- df_cor_total %>% filter(Target == t)
  
  # ordenar variables
  df_plot$Var1 <- factor(df_plot$Var1, levels = unique(df_plot$Var1))
  df_plot$Var2 <- factor(df_plot$Var2, levels = unique(df_plot$Var2))
  
  p <- ggplot(df_plot, aes(Var1, Var2, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(
      low = "#2166ac",
      mid = "white",
      high = "#b2182b",
      midpoint = 0,
      limits = c(-1, 1)
    ) +
    coord_fixed() +
    theme_minimal(base_size = 12) +
    labs(
      title = paste("Correlation matrix -", t),
      x = "",
      y = "",
      fill = "r"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
      axis.text.y = element_text(size = 7),
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  print(p)
  
  ggsave(
    filename = paste0("Correlacion_", t, ".png"),
    plot = p,
    width = 7,
    height = 6,
    dpi = 300
  )
}

# =========================================================
# FIGURA 3: Importancia de Variables
# =========================================================

df_imp_total <- bind_rows(lista_importancias) %>%
  mutate(
    Feature = str_replace_all(Feature, "_", " "),
    Feature = str_to_sentence(Feature),
    Target = factor(Target, levels = sort(unique(Target)))
  )

targets_alfabetico <- levels(df_imp_total$Target)
label_map_imp <- setNames(
  paste0("(", letters[seq_along(targets_alfabetico)], ")"), 
  targets_alfabetico
)

p_imp <- ggplot(df_imp_total, aes(x = tidytext::reorder_within(Feature, Gain, Target), 
                                  y = Gain, 
                                  fill = Gain)) +
  geom_bar(stat = "identity") +
  tidytext::scale_x_reordered() +
  scale_fill_gradient(low = "#9ecae1", high = "#084594") +
  coord_flip() +
  facet_wrap(~Target, scales = "free", ncol = 2, 
             labeller = labeller(Target = label_map_imp)) +
  labs(
    x = "Predictor variables",
    y = "Ganancia (Gain)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    text = element_text(family = "serif"), # Times New Roman
    legend.position = "none",
    panel.grid.major.y = element_blank(),
    strip.text = element_text(face = "bold", size = 12), 
    axis.text.y = element_text(size = 10)
  )

print(p_imp)
ggsave("Figura_Importancia_Literales.png", p_imp, width = 12, height = 16, dpi = 300)

# =========================================================
# FIGURA: Heatmaps de correlación general facetados
# =========================================================

df_cor_total <- bind_rows(lista_correlaciones) %>%
  drop_na(value) %>%
  distinct(Target, Var1, Var2, .keep_all = TRUE) %>% 
  mutate(
    Var1 = str_to_sentence(str_replace_all(Var1, "_", " ")),
    Var2 = str_to_sentence(str_replace_all(Var2, "_", " ")),
    Target = factor(Target, levels = targets_cor)
  )

label_map <- setNames(
  paste0("(", letters[seq_along(targets_cor)], ")"),
  targets_cor
)

p_cor <- ggplot(df_cor_total, aes(Var1, Var2, fill = value)) +
  
  geom_tile(color = "white") +
  
  # Números dentro de cada celda
  geom_text(
    aes(label = sprintf("%.2f", value)),
    family = "serif",
    size = 5,
    check_overlap = TRUE
  ) + 
  
  scale_fill_gradient2(
    low = "#2166ac",
    mid = "white",
    high = "#b2182b",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  
  facet_wrap(
    ~Target,
    scales = "free",
    ncol = 5,
    labeller = labeller(Target = label_map)
  ) +
  
  labs(
    x = "Predictor variables",
    y = "Predictor variables",
    fill = "r"
  ) +
  
  theme_minimal(base_size = 16) +
  
  theme(
    legend.position = "bottom",
    text = element_text(family = "serif", size = 10),
    axis.text.x = element_text(angle = 60, hjust = 1, size = 10, color = "black"),
    axis.text.y = element_text(size = 16, color = "black"),
    axis.title = element_text(size = 16, face = "bold"),
    strip.text = element_text(face = "bold", size = 16),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 16),
    panel.grid = element_blank(),
    panel.spacing = unit(0.4, "lines"),
    plot.title = element_text(face = "bold", size = 10, hjust = 0.5)
  )

ggsave(
  "Figura_correlacion_final_revisada.png",
  plot = p_cor,
  width = 60,
  height = 28,
  units = "cm",
  dpi = 300
)


# =========================================================
# PREDICCIÓN VS. REAL
# =========================================================

df_all <- bind_rows(lapply(names(predicciones), function(target) {
  
  data.frame(
    Target = target,
    Real   = predicciones[[target]]$Real,
    RF     = predicciones[[target]]$RF,
    XGB    = predicciones[[target]]$XGB,
    ANN     = predicciones[[target]]$ANN
  )
})) %>%
  pivot_longer(
    cols = c(RF, XGB, ANN),
    names_to = "Modelo",
    values_to = "Predicho"
  )

valid_targets <- names(predicciones)

# =========================================================
# GRAFICO FACET_WRAP
# =========================================================

df_all <- df_all %>%
  mutate(
    Target = factor(Target, levels = targets_cor)
  )

label_map_pp2 <- setNames(
  paste0("(", letters[seq_along(targets_cor)], ")"),
  targets_cor
)

pp2 <- ggplot(df_all, aes(x = Real, y = Predicho, color = Modelo)) +
  geom_point(alpha = 0.7, size = 3) +
  geom_abline(slope = 1, intercept = 0, linewidth = 1.2, linetype = "dashed", color = "black") +
  scale_color_manual(values = c("RF"="#1b9e77","XGB"="#d95f02","ANN"="#7570b3")) +
  facet_wrap(
    ~Target, 
    scales = "free", 
    ncol = 2,
    labeller = labeller(Target = label_map_pp2)
  ) +
  theme_minimal() +
  labs(
    x = "Observed value",
    y = "Predicted value",
    color = "Model"
  ) +
  theme(
    text = element_text(family = "serif"),
    
    plot.title = element_text(
      face = "bold",
      size = 22,
      hjust = 0.5
    ),
    
    strip.text = element_text(
      face = "bold",
      size = 22
    ), 
    
    axis.title = element_text(
      size = 18,
      face = "bold"
    ),
    
    axis.text = element_text(
      size = 16,
      color = "black",
      face = "bold"
    ),
    
    legend.position = "bottom",
    
    # MEJORA DE DISTRIBUCIÓN
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.key.width = unit(2.2, "cm"),
    legend.spacing.x = unit(0.7, "cm"),
    
    legend.text = element_text(size = 16),
    
    legend.title = element_text(
      size = 16,
      face = "bold"
    ),
    
    panel.spacing = unit(1.2, "lines")
  )

ggsave(
  "pp2.tiff",
  plot = pp2,
  width = 30,
  height = 40,
  units = "cm",
  dpi = 600,
  compression = "lzw"
)

# PREDICCIÓN CON LINEA DE TENDENCIA
targets_plot <- valid_targets

label_map <- setNames(
  paste0("(", letters[seq_along(targets_plot)], ")"),
  targets_plot
)

df_all$Target <- factor(df_all$Target, levels = targets_plot)

p2 <- ggplot(df_all, aes(x = Real, y = Predicho, color = Modelo)) +
  geom_point(alpha = 0.4, size = 1.2) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "solid", color = "black") +
  scale_color_manual(values = c(
    "RF" = "#1b9e77",
    "XGB" = "#d95f02",
    "ANN" = "#7570b3"
  )) +
  facet_wrap(
    ~Target,
    scales = "free",
    ncol = 3,
    labeller = labeller(Target = label_map)
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(size = 12, hjust = 1),
    axis.text.y = element_text(size = 12),
    axis.title = element_text(size = 14),
    strip.text = element_text(size = 13, face = "bold"),
    plot.title = element_text(size = 16, face = "bold")
  ) +
  labs(
    title = "Predicted vs Observed with model trends",
    x = "Valor real",
    y = "Valor predicho",
    color = "Modelo"
  )

ggsave("Fig4.png", p2, width = 14, height = 16, dpi = 300)

print(p2)

#==============================================================
# DESEMPEÑO GLOBAL DEL MODELO
#==============================================================
df_r2_test <- resultados %>%
  select(Target, R2_RF_test, R2_XGB_test, R2_NN_test) %>%
  pivot_longer(-Target)

ggplot(df_r2_test, aes(x = Target, y = value, fill = name)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Comparación de R² en test por modelo",
    x = "Target",
    y = "R²"
  )

# =========================================================
# CREAR METADATA PARA SHINY
# =========================================================

metadata <- resultados %>%
  select(Target, Best_Model) %>%
  distinct() %>%
  mutate(Mejor_Modelo = tolower(Best_Model)) %>%
  select(
    Propiedad = Target,
    Mejor_Modelo
  )

saveRDS(metadata, "modelos/metadata_shiny.rds")
