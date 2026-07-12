# 🏥 Student Health Risk Prediction

An end-to-end Machine Learning classification project that predicts a student's **health condition** based on lifestyle, physical activity, dietary habits, and health-related attributes.

This project was developed as part of the **Kaggle Playground Series 2026** and follows a complete data science workflow from data exploration to model comparison and prediction.

---

## 📌 Problem Statement

The objective is to predict the **health condition** of a student into one of the following classes:

- 🟢 Fit
- 🟡 Unhealthy
- 🔴 At-Risk

The competition is evaluated using **Balanced Accuracy**, making it suitable for handling the highly imbalanced target classes.

---

## 📂 Dataset Information

### Training Dataset
- **Rows:** 690,088
- **Columns:** 15

### Test Dataset
- **Rows:** 295,753
- **Columns:** 14

### Features

| Feature | Description |
|----------|-------------|
| sleep_duration | Average sleeping hours |
| heart_rate | Heart rate |
| bmi | Body Mass Index |
| calorie_expenditure | Calories burned |
| step_count | Daily steps |
| exercise_duration | Exercise duration |
| water_intake | Daily water intake |
| diet_type | Dietary preference |
| stress_level | Stress level |
| sleep_quality | Quality of sleep |
| physical_activity_level | Activity level |
| smoking_alcohol | Smoking/Alcohol habits |
| gender | Gender |

Target Variable:

- **health_condition**

---

# 🛠️ Project Workflow

- Data Loading
- Data Inspection
- Missing Value Analysis
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Train-Test Split
- Baseline Model Comparison
- Best Model Selection
- Final Model Training
- Kaggle Submission

---

# 📊 Exploratory Data Analysis

Performed detailed EDA including:

- Target class distribution
- Missing value analysis
- Numerical feature distributions
- Boxplots for outlier detection
- Correlation heatmap
- Categorical feature analysis
- Relationship between features and target variable

---

# ⚙️ Data Preprocessing

### Numerical Features

- Median Imputation
- Standard Scaling (for linear/distance-based models)

### Categorical Features

- Most Frequent Imputation
- One-Hot Encoding

Implemented using:

- Pipeline
- ColumnTransformer

to avoid data leakage and ensure reproducibility.

---

# 🤖 Machine Learning Models

The following classification models were trained and compared using **Balanced Accuracy**.

| Model | Balanced Accuracy |
|--------|------------------:|
| HistGradientBoosting | **0.8574** |
| Gradient Boosting | 0.8561 |
| Decision Tree | 0.8529 |
| Random Forest | 0.8525 |
| Extra Trees | 0.8416 |
| Logistic Regression | 0.8138 |
| K-Nearest Neighbors | 0.7833 |
| AdaBoost | 0.5847 |

---

# 🏆 Best Performing Model

**HistGradientBoostingClassifier**

Balanced Accuracy:

```text
0.8574
```

---

# 📈 Evaluation Metric

The competition uses:

- **Balanced Accuracy**

Balanced Accuracy computes the average recall across all classes, making it appropriate for imbalanced datasets.

---

# 🧰 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

---

# 📁 Repository Structure

```
Student-Health-Risk-Prediction/
│
├── Student_Health_Risk_Prediction.ipynb
├── README.md
├── submission.csv
└── dataset/
```

---

# 🚀 Future Improvements

- Hyperparameter Optimization
- Feature Engineering
- CatBoost Classifier
- XGBoost Classifier
- LightGBM Classifier
- Ensemble Learning
- SHAP Explainability
- Cross Validation

---

# 📌 Results

Successfully built a complete end-to-end machine learning pipeline for predicting student health risk and achieved a **Balanced Accuracy of 0.8574** using **HistGradientBoostingClassifier**.

---

# 👨‍💻 Author

**Parmeet Singh**

B.Tech CSE (Data Science)

Aspiring Machine Learning Engineer & Data Scientist

GitHub: *(Add your GitHub Profile Link)*

LinkedIn: *(Add your LinkedIn Profile Link)*

---

## ⭐ If you found this project helpful, consider giving it a star!
