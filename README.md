# TPS Property Prediction using Machine Learning

This project presents a machine learning-based framework for predicting the physicochemical and mechanical properties of thermoplastic starch (TPS) materials as a function of formulation and processing conditions.

The study integrates data extracted from scientific literature and applies a structured workflow following the CRISP-ML methodology, including data preprocessing, feature selection, model training, and evaluation.

Three predictive models were developed and compared:

- Random Forest (RF)
- Extreme Gradient Boosting (XGBoost)
- Artificial Neural Networks (ANN)

Feature selection was performed using XGBoost importance metrics to reduce dimensionality and improve model generalization. Model performance was evaluated using R², RMSE, and MAE across training, validation, and test sets.

## Interactive Application

An interactive web application was developed using the :contentReference[oaicite:0]{index=0} framework to allow non-specialized users to estimate TPS properties in real time based on user-defined input variables.

The application dynamically selects the optimal model for each target variable and adjusts input parameters according to the most relevant predictors identified during training.

## Repository Structure

- `scripts/`: main training pipeline
- `modelos/`: trained machine learning models (.rds)
- `preprocess/`: preprocessing objects
- `features/`: selected features per target
- `app.R`: interactive Shiny application

## Methodology

The workflow follows a CRISP-ML approach:

1. Data cleaning and preprocessing
2. Missing value imputation (kNN)
3. Feature selection using XGBoost importance
4. Model training (RF, XGB, ANN)
5. Model evaluation (R², RMSE, MAE)
6. Selection of best-performing model per target
7. Deployment via Shiny application

## Objective

To provide a predictive tool that supports the design and optimization of biodegradable materials based on thermoplastic starch.

## Reproducibility

All trained models, preprocessing objects, and selected features are stored as `.rds` files to ensure full reproducibility of results and seamless deployment of the application.

## Author

Maria Riera
