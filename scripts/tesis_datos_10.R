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

# --- Imputación
variables_a_imputar <- c("process_temperature", "process_time", "rpm", 
                         "screw_speed", "Pressure", "Tdrying1", "tdrying1",
                         "Tdrying2", "tdrying2")

df <- kNN(df, variable = variables_a_imputar, k = 5, imp_var = FALSE)

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

ggplot(df_targets, aes(x = reorder(Target, N), y = N)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    title = "Número de observaciones por variable objetivo",
    x = "Variable objetivo",
    y = "Número de datos"
  ) +
  theme_minimal()

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
# 6. FUNCIÓN LIMPIEZA POR TARGET
# =========================================================
limpiar_por_target <- function(df, target) {
  df %>%
    drop_na(all_of(target)) %>%
    select(where(~ sd(., na.rm = TRUE) > 0)) %>%
    select(where(~ sum(is.na(.)) == 0))
}

#
# =========================================================
resultados <- NULL

# =========================================================
# LISTA PARA GUARDAR IMPORTANCIAS
# =========================================================
lista_importancias <- list()
lista_correlaciones <- list()
predicciones <- list()


# =========================================================
# 8. LOOP PRINCIPAL
# =========================================================

r2 <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

for (categoria in names(categorias)) {
  
  for (target in categorias[[categoria]]) {
    
    if (!target %in% colnames(df_num)) next
    
    cat("\nProcesando:", target, "\n")
    
    # -----------------------------
    # Limpieza específica
    # -----------------------------
    df_modelo <- limpiar_por_target(df_num, target)
    
    if (nrow(df_modelo) < 30) next
    
    cat("Número de variables:", ncol(df_modelo)-1, "\n")
    
    # -----------------------------
    # División 70:20:10
    # -----------------------------
    idx_train <- createDataPartition(df_modelo[[target]], p = 0.7, list = FALSE)
    train <- df_modelo[idx_train, ]
    temp  <- df_modelo[-idx_train, ]
    
    idx_val <- createDataPartition(temp[[target]], p = 2/3, list = FALSE)
    val  <- temp[idx_val, ]
    test <- temp[-idx_val, ]
    
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
    
    if (length(top_vars) == 0) next
    
    cat("Top variables:\n")
    print(top_vars)
    
    # -----------------------------
    # Reducir datasets
    # -----------------------------
    train <- train %>% select(all_of(top_vars), all_of(target))
    val   <- val   %>% select(all_of(top_vars), all_of(target))
    test  <- test  %>% select(all_of(top_vars), all_of(target))
    
    # -----------------------------
    # MATRIZ DE CORRELACIÓN
    # -----------------------------
    top_vars_cor <- top_vars[1:min(10, length(top_vars))]
    
    df_cor <- train %>%
      select(all_of(top_vars_cor)) %>%
      select(where(~ sd(., na.rm = TRUE) > 0))
    
    if (ncol(df_cor) > 2) {
      
      matriz_cor <- cor(df_cor, use = "pairwise.complete.obs")
      matriz_cor[upper.tri(matriz_cor)] <- NA
      
      df_cor_long <- reshape2::melt(matriz_cor)
      df_cor_long$Target <- target
      
      lista_correlaciones[[target]] <- df_cor_long
    }
  
    # =========================================================
    # MODELOS
    # =========================================================
    
    # --- RANDOM FOREST
    modelo_rf <- randomForest(as.formula(paste(target, "~ .")), data = train)
    
    pred_rf_train <- predict(modelo_rf, train)
    pred_rf_val   <- predict(modelo_rf, val)
    pred_rf_test  <- predict(modelo_rf, test)
    
    # --- XGBOOST
    modelo_xgb <- xgboost(
      x = as.matrix(train %>% select(-all_of(target))),
      y = train[[target]],
      nrounds = 100,
      objective = "reg:squarederror",
      verbosity = 0
    )
    
    pred_xgb_train <- predict(modelo_xgb, as.matrix(train %>% select(-all_of(target))))
    pred_xgb_val   <- predict(modelo_xgb, as.matrix(val %>% select(-all_of(target))))
    pred_xgb_test  <- predict(modelo_xgb, as.matrix(test %>% select(-all_of(target))))
    
    # --- ANN
    pre <- preProcess(train %>% select(-all_of(target)), method = c("center", "scale"))
    
    train_nn <- predict(pre, train %>% select(-all_of(target)))
    val_nn   <- predict(pre, val %>% select(-all_of(target)))
    test_nn  <- predict(pre, test %>% select(-all_of(target)))
    
    train_nn$y <- train[[target]]
    
    modelo_nn <- nnet(y ~ ., data = train_nn, size = 5, linout = TRUE, trace = FALSE)
    
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
      Real   = test[[target]],
      RF     = pred_rf_test,
      XGB    = pred_xgb_test,
      NN     = pred_nn_test
    )
    
    # =========================================================
    # MÉTRICAS COMPLETAS (TRAIN / VAL / TEST)
    # =========================================================
    
    # -------- R2 --------
    R2_RF_train  <- r2(train[[target]], pred_rf_train)
    R2_RF_val    <- r2(val[[target]], pred_rf_val)
    R2_RF_test   <- r2(test[[target]], pred_rf_test)
    
    R2_XGB_train <- r2(train[[target]], pred_xgb_train)
    R2_XGB_val   <- r2(val[[target]], pred_xgb_val)
    R2_XGB_test  <- r2(test[[target]], pred_xgb_test)
    
    R2_NN_train  <- r2(train[[target]], pred_nn_train)
    R2_NN_val    <- r2(val[[target]], pred_nn_val)
    R2_NN_test   <- r2(test[[target]], pred_nn_test)
    
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
      RMSE_RF_train = rmse(train[[target]], pred_rf_train),
      RMSE_RF_val   = rmse(val[[target]], pred_rf_val),
      RMSE_RF_test  = rmse(test[[target]], pred_rf_test),
      
      RMSE_XGB_train = rmse(train[[target]], pred_xgb_train),
      RMSE_XGB_val   = rmse(val[[target]], pred_xgb_val),
      RMSE_XGB_test  = rmse(test[[target]], pred_xgb_test),
      
      RMSE_NN_train = rmse(train[[target]], pred_nn_train),
      RMSE_NN_val   = rmse(val[[target]], pred_nn_val),
      RMSE_NN_test  = rmse(test[[target]], pred_nn_test),
      
      # -------- MAE --------
      MAE_RF_train = mae(train[[target]], pred_rf_train),
      MAE_RF_val   = mae(val[[target]], pred_rf_val),
      MAE_RF_test  = mae(test[[target]], pred_rf_test),
      
      MAE_XGB_train = mae(train[[target]], pred_xgb_train),
      MAE_XGB_val   = mae(val[[target]], pred_xgb_val),
      MAE_XGB_test  = mae(test[[target]], pred_xgb_test),
      
      MAE_NN_train = mae(train[[target]], pred_nn_train),
      MAE_NN_val   = mae(val[[target]], pred_nn_val),
      MAE_NN_test  = mae(test[[target]], pred_nn_test)
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
  function(x) c("RF","XGB","NN")[which.max(x)]
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
# FIGURA 3: Heatmaps de correlación (Top variables)
# =========================================================

df_cor_total <- bind_rows(lista_correlaciones) %>%
  drop_na(value)

targets_cor <- todos_targets[todos_targets %in% unique(df_cor_total$Target)]

df_cor_total$Target <- factor(df_cor_total$Target, levels = targets_cor)

levels(df_cor_total$Target)

label_map <- setNames(
  paste0("(", letters[seq_along(targets_cor)], ") ", targets_cor),
  targets_cor
)

# =========================================================
# HEATMAP LIMPIO
# =========================================================

p_cor <- ggplot(df_cor_total, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
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
    ncol = 2,
    labeller = labeller(Target = label_map)
  ) +
  theme_minimal(base_size = 11) +
  labs(
    title = "Correlation matrices of the most relevant variables",
    x = "",
    y = "",
    fill = "r"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8.5),
    axis.text.y = element_text(size = 8.5),
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold")
  )

print(p_cor)

data.frame(
  letra = paste0("(", letters[seq_along(targets_cor)], ")"),
  Target = targets_cor
)

ggsave("Figura_correlacion_global.png", p_cor,
       width = 10, height = 14, dpi = 300)



# =========================================================
# PREDICCIÓN VS. REAL
# =========================================================


# CONSTRUIR DATASET GLOBAL
#
df_all <- bind_rows(lapply(names(predicciones), function(target) {
  
  data.frame(
    Target = target,
    Real   = predicciones[[target]]$Real,
    RF     = predicciones[[target]]$RF,
    XGB    = predicciones[[target]]$XGB,
    NN     = predicciones[[target]]$NN
  )
})) %>%
  pivot_longer(
    cols = c(RF, XGB, NN),
    names_to = "Modelo",
    values_to = "Predicho"
  )

valid_targets <- names(predicciones)

# =========================================================
# GRAFICO FACET_WRAP
# =========================================================
pp2 <- ggplot(df_all, aes(x = Real, y = Predicho, color = Modelo)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = c("RF"="#1b9e77","XGB"="#d95f02","NN"="#7570b3")) +
  facet_wrap(~Target, scales = "free") +
  theme_minimal(base_size = 10) +
  labs(
    title = "Predicted vs Observed (comparison across models)",
    x = "Valor real",
    y = "Valor predicho"
  )

print(pp2)

#PREDICCIÓN CON LINEA

targets_plot <- valid_targets

label_map <- setNames(
  paste0("(", letters[seq_along(targets_plot)], ")"),
  targets_plot
)

df_all$Target <- factor(df_all$Target, levels = targets_plot)

p2 <- ggplot(df_all, aes(x = Real, y = Predicho, color = Modelo)) +
  
  # puntos (datos reales)
  geom_point(alpha = 0.4, size = 1.2) +
  
  # línea de tendencia por modelo
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", size = 0.8) +
  
  # línea ideal
  geom_abline(slope = 1, intercept = 0, linetype = "solid", color = "black") +
  
  scale_color_manual(values = c(
    "RF" = "#1b9e77",
    "XGB" = "#d95f02",
    "NN" = "#7570b3"
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
#  DESEMPEÑO GLOBAL DEL MODELO
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
