# 🏡 House Price Prediction using Machine Learning

Predicting residential house prices using the Ames Housing dataset with advanced regression techniques. This project was developed as part of the **Housing Prices Competition for Kaggle Learn Users**.

---

## 📌 Project Overview

House prices are influenced by many factors beyond the number of bedrooms and bathrooms. This project aims to build a machine learning model capable of accurately predicting the **SalePrice** of residential homes using **79 explanatory features** from the Ames Housing dataset.

The project covers the complete machine learning workflow, including data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## 📊 Dataset

- **Dataset:** Ames Housing Dataset
- **Training Samples:** 1,460
- **Testing Samples:** 1,459
- **Features:** 79 explanatory variables
- **Target Variable:** `SalePrice`

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Kaggle Notebook

---

## 📂 Project Workflow

### 1. Data Exploration
- Dataset inspection
- Missing value analysis
- Feature type identification
- Statistical summary

### 2. Data Cleaning
- Removed columns with excessive missing values
- Imputed missing numerical values using Median
- Imputed missing categorical values using Mode

### 3. Feature Engineering
- One-Hot Encoding using `OneHotEncoder`
- Feature preprocessing using `ColumnTransformer`

### 4. Model Training
The following regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### 5. Hyperparameter Tuning
- RandomizedSearchCV
- 5-Fold Cross Validation
- Best model selected based on validation performance

### 6. Final Prediction
- Trained the best-performing model on the complete training dataset
- Generated predictions for the Kaggle test dataset
- Submitted predictions to Kaggle

---

## 📈 Model Performance

| Model | RMSE | R² Score | Log RMSE |
|--------|------:|---------:|---------:|
| Linear Regression | 29638.43 | 0.8855 | 0.1791 |
| Decision Tree | 44801.58 | 0.7383 | 0.2374 |
| Random Forest | 29151.92 | 0.8892 | 0.1537 |
| **Gradient Boosting** | **27226.59** | **0.9034** | **0.1364** |

> **Best Model:** Gradient Boosting Regressor

---

## 🔍 Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV** with 5-fold cross-validation.

Although tuning improved the cross-validation score, the tuned model did not outperform the original Gradient Boosting model on the validation dataset. Therefore, the original model was selected for the final submission.

---

## 🏆 Kaggle Result

- Competition: **Housing Prices Competition for Kaggle Learn Users**
- Successfully generated and submitted predictions.
- **Leaderboard Rank:** **459**

---

## 📁 Project Structure

```
├── HousePricePrediction.ipynb
├── submission.csv
├── README.md
```

---

## 🚀 Future Improvements

- XGBoost Regressor
- LightGBM
- CatBoost
- Advanced Feature Engineering
- Stacking & Ensemble Learning
- Feature Importance Analysis

---

## 🎯 Key Learning Outcomes

- Data Cleaning
- Missing Value Imputation
- Feature Encoding
- Regression Modeling
- Model Evaluation
- Hyperparameter Tuning
- Machine Learning Pipeline
- Kaggle Competition Workflow

---

## 🤝 Connect With Me

**Parmeet Singh**

- LinkedIn: *(Add your LinkedIn profile)*
- GitHub: *(Add your GitHub profile)*

---

## ⭐ If you found this project useful, consider giving it a star!
