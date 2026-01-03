# California Housing Price Prediction Pipeline

## Project Overview
This project implements an **end-to-end Machine Learning pipeline** to predict California housing prices using the **Random Forest Regressor**. The pipeline covers all stages of the ML workflow including data preprocessing, model training, evaluation, model persistence, and inference on new data.

This project demonstrates the full lifecycle of an ML project and is a great example of **practical application of Python, scikit-learn, and data engineering principles**.

---

## Features

1. **Data Preprocessing**
   - Handles missing values using median imputation for numerical features.
   - Scales numerical features using StandardScaler.
   - Encodes categorical features using One-Hot Encoding.
   - Fully integrated preprocessing pipeline using `ColumnTransformer`.

2. **Stratified Train-Test Split**
   - Ensures representative samples across the median income categories.
   - Prevents biased model training.

3. **Model Training**
   - Uses **Random Forest Regressor** for robust regression performance.
   - Performs cross-validation for accurate performance assessment.

4. **Model Evaluation**
   - Calculates **Root Mean Squared Error (RMSE)** as the evaluation metric.
   - Provides cross-validated RMSE to check model stability.

5. **Model Persistence**
   - Saves the **entire preprocessing + model pipeline** using `joblib`.
   - Ensures reproducible inference without retraining or manual preprocessing.

6. **Inference**
   - Predicts median house values for new input data using the saved pipeline.
   - Saves predictions to a CSV file without modifying original input data.

---

## Project Structure

## 2. Train the Model

If the model does not exist (model.pkl), running the script will:

Load the dataset

Split into training and test sets

Build the preprocessing and Random Forest pipeline

Train the model

Save the trained pipeline

python scripts/housing_pipeline.py

## 3. Run Inference

Once the model is trained:

Place new data in data/input.csv (same feature columns, excluding target)

Run the script to generate predictions:

python scripts/housing_pipeline.py


Predictions will be saved in predictions.csv in the same folder.

Evaluation Metric

Root Mean Squared Error (RMSE) is used as the main metric for performance evaluation:

Lower RMSE indicates better predictive performance.

## Result
we can see through the files properly but a short view here also
<img width="712" height="537" alt="Screenshot 2026-01-03 171445" src="https://github.com/user-attachments/assets/feec80ef-24d2-4c30-87d8-74cdab95f823" />
<img width="833" height="695" alt="Screenshot 2026-01-03 171534" src="https://github.com/user-attachments/assets/1aa1ad76-a472-4244-ab0f-9e00af0b4c36" />
<img width="405" height="514" alt="Screenshot 2026-01-03 163924" src="https://github.com/user-attachments/assets/1ddaffbe-a60e-4a8e-83c4-8b205e966861" />

## Key Highlights

Entire preprocessing + model saved as a single pipeline for production-ready inference

Handles missing values, scaling, and categorical encoding automatically

Cross-validation ensures robust and unbiased performance assessment

Fully modular and reusable for other datasets

## Future Enhancements

Hyperparameter tuning with GridSearchCV or RandomizedSearchCV

Feature importance analysis and visualization

Deployment as a REST API (Flask / FastAPI)

Versioning and automatic retraining pipeline

Technologies Used

Python 3.x

Pandas, NumPy

Scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor)

Joblib for model persistence

## Author

## Parmeet – Student & ML Enthusiast

## First end-to-end ML project demonstrating training, evaluation, persistence, and inference pipeline.


