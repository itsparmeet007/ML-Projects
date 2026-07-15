# 📊 Student Performance Prediction using Multiple Linear Regression

## 📌 Overview

This project implements **Multiple Linear Regression** to predict students' academic performance based on various study and lifestyle factors.

The notebook covers the complete machine learning workflow from data preprocessing and exploratory data analysis (EDA) to model evaluation. It also includes a custom implementation of Multiple Linear Regression using the **Normal Equation** without relying on Scikit-Learn's `LinearRegression`.

---

## 📂 Dataset Features

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

**Target Variable**

- Performance Index

---

## 📊 Exploratory Data Analysis

The notebook includes:

- Data Inspection
- Missing Value Analysis
- Descriptive Statistics
- Distribution Plots
- Boxplots
- Correlation Heatmap
- Scatter Plots
- Regression Plots

---

## ⚙️ Data Preprocessing

- Train-Test Split
- Feature Scaling using StandardScaler
- Categorical Encoding using OneHotEncoder
- ColumnTransformer Pipeline

---

## 🤖 Models Used

### Scikit-Learn Multiple Linear Regression

Built using:

- `LinearRegression`

### Multiple Linear Regression from Scratch

Implemented using the **Normal Equation**

- `fit()`
- `predict()`

---

## 📈 Model Performance

| Metric | Value |
|---------|-------|
| R² Score | **0.9889** |
| Mean Absolute Error (MAE) | **1.61** |

The custom implementation produced results nearly identical to Scikit-Learn's implementation.

---

## 📦 Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📁 Project Structure

```
Student_Performance_Multiple_Linear_Regression/
│
├── Student_Performance_Multiple_Linear_Regression.ipynb
├── README.md
└── dataset.csv
```

---

## 🎯 Key Learnings

- Multiple Linear Regression
- Data Preprocessing
- Feature Scaling
- One-Hot Encoding
- ColumnTransformer
- Model Evaluation
- Residual Analysis
- Multiple Linear Regression from Scratch
- Normal Equation

---

## 🚀 Future Improvements

- Ridge Regression
- Lasso Regression
- Elastic Net
- Polynomial Regression
- Model Deployment

---

## 👨‍💻 Author

**Parmeet Singh**

B.Tech CSE (Honors in Data Science)

⭐ If you found this project helpful, feel free to star the repository.
