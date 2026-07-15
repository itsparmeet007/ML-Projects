# 🎓 Student Performance Prediction using Multiple Linear Regression

## 📌 Project Overview

This project predicts a student's **Performance Index** using **Multiple Linear Regression** based on various academic and lifestyle factors.

The notebook demonstrates the complete Machine Learning workflow, including data preprocessing, exploratory data analysis (EDA), model building, evaluation, and implementing Multiple Linear Regression from scratch using the **Normal Equation**.

---

## 📂 Dataset

The dataset contains the following features:

| Feature | Description |
|---------|-------------|
| Hours Studied | Number of study hours |
| Previous Scores | Student's previous academic scores |
| Extracurricular Activities | Participation in extracurricular activities (Yes/No) |
| Sleep Hours | Average daily sleep hours |
| Sample Question Papers Practiced | Number of sample papers practiced |
| Performance Index | Target variable representing student performance |

---

## 🎯 Problem Statement

Build a Multiple Linear Regression model capable of predicting the **Performance Index** of students based on their academic and personal habits.

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset Overview
- Missing Value Analysis
- Descriptive Statistics
- Distribution of Numerical Features
- Count Plot for Categorical Feature
- Boxplots for Outlier Detection
- Correlation Heatmap
- Scatter Plots
- Regression Plots
- Actual vs Predicted Visualization
- Residual Analysis

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Train-Test Split
- Feature Scaling using **StandardScaler**
- Categorical Encoding using **OneHotEncoder**
- Pipeline implementation using **ColumnTransformer**

---

## 🤖 Machine Learning Model

### Scikit-Learn Implementation

- Linear Regression (`sklearn.linear_model.LinearRegression`)

### Custom Implementation

Implemented Multiple Linear Regression from scratch using the **Normal Equation**:

\[
\beta = (X^TX)^{-1}X^Ty
\]

The custom implementation includes:

- `fit()`
- `predict()`

without using Scikit-Learn's Linear Regression.

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | **0.9889** |
| Mean Absolute Error (MAE) | **1.61** |

The custom implementation produced results nearly identical to Scikit-Learn's implementation, validating the correctness of the mathematical implementation.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📁 Project Structure

```
Student-Performance-Multiple-Regression/
│
├── Student_Performance.ipynb
├── README.md
└── dataset.csv
```

---

## 🚀 Key Learning Outcomes

- Understanding Multiple Linear Regression
- Feature Scaling
- One-Hot Encoding
- Building preprocessing pipelines
- Exploratory Data Analysis (EDA)
- Model Evaluation
- Residual Analysis
- Actual vs Predicted Analysis
- Implementing Multiple Linear Regression using the Normal Equation
- Comparing custom implementation with Scikit-Learn

---

## 📷 Results

The model achieved excellent predictive performance with an **R² Score of approximately 98.9%**, indicating that it explains almost all of the variance in the target variable.

The Actual vs Predicted plot demonstrates that the predicted values closely align with the true values, confirming the effectiveness of the model.

---

## 🔮 Future Improvements

- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Cross Validation
- Hyperparameter Tuning
- Feature Selection
- Model Deployment using Streamlit or Flask

---

## 👨‍💻 Author

**Parmeet Singh**

B.Tech CSE (Honors in Data Science)

GitHub: *Add your GitHub profile link here*

LinkedIn: *Add your LinkedIn profile link here*

---

⭐ If you found this project useful, consider giving it a star!
