# 📉 Lasso Regression

A complete implementation and explanation of **Lasso Regression (Least Absolute Shrinkage and Selection Operator)** using **Python** and **Scikit-learn**. This project demonstrates how Lasso Regression performs linear regression while automatically selecting important features by shrinking less important coefficients to zero.

---

## 📌 Table of Contents

- Introduction
- What is Lasso Regression?
- How Lasso Regression Works
- Mathematical Formula
- Advantages
- Disadvantages
- Applications
- Dataset
- Installation
- Project Structure
- Implementation
- Model Evaluation
- Results
- Hyperparameter Tuning
- Future Improvements
- Technologies Used
- Conclusion
- References

---

# 📖 Introduction

Lasso Regression is a supervised machine learning algorithm used for predicting continuous numerical values. It is an extension of Linear Regression that includes **L1 Regularization**, which penalizes the absolute values of regression coefficients.

Unlike Ordinary Linear Regression, Lasso Regression can reduce some coefficients exactly to zero, effectively performing **feature selection**.

---

# 🎯 What is Lasso Regression?

Lasso Regression is a regularized version of Linear Regression that minimizes prediction error while preventing overfitting.

It is especially useful when:

- Dataset has many features
- Some features are irrelevant
- Feature selection is required
- Multicollinearity exists

Lasso automatically removes unnecessary variables by shrinking their coefficients to zero.

---

# ⚙️ How Lasso Regression Works

Lasso Regression minimizes the following cost function:

\[
Loss = RSS + \alpha \sum_{i=1}^{n}|w_i|
\]

Where:

- RSS = Residual Sum of Squares
- α = Regularization parameter
- \(w_i\) = Model coefficients

The L1 penalty forces small coefficients to become exactly zero.

This results in:

- Reduced model complexity
- Better generalization
- Automatic feature selection

---

# 🧮 Mathematical Formula

Linear Regression:

\[
y=\beta_0+\beta_1x_1+\beta_2x_2+...+\beta_nx_n
\]

Lasso Cost Function:

\[
J(\beta)=\sum_{i=1}^{m}(y_i-\hat y_i)^2+\alpha\sum_{j=1}^{n}|\beta_j|
\]

where

- \(y_i\) = Actual value
- \(\hat y_i\) = Predicted value
- \(\beta_j\) = Coefficients
- α = Regularization strength

---

# 🚀 Advantages

✔ Prevents overfitting

✔ Performs automatic feature selection

✔ Reduces model complexity

✔ Handles multicollinearity

✔ Improves generalization

✔ Produces simpler and more interpretable models

---

# ❌ Disadvantages

- Can underfit if α is too large
- Sensitive to feature scaling
- Requires standardized features
- Not suitable when all features are important
- Coefficients become biased because of regularization

---

# 💼 Applications

Lasso Regression is widely used in:

- House Price Prediction
- Medical Diagnosis
- Stock Price Forecasting
- Sales Prediction
- Marketing Analytics
- Credit Risk Analysis
- Feature Selection
- Bioinformatics
- Financial Modeling

---

# 📊 Dataset

You can use any regression dataset such as:

- Boston Housing Dataset
- California Housing Dataset
- House Prices Dataset (Kaggle)
- Custom CSV Dataset

Example features:

- Area
- Bedrooms
- Bathrooms
- Age
- Distance from City
- Parking

Target:

- House Price

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Lasso-Regression.git
```

Move into project directory

```bash
cd Lasso-Regression
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Project Structure

```
Lasso-Regression/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── LassoRegression.ipynb
│
├── models/
│
├── images/
│
├── requirements.txt
│
├── README.md
│
└── lasso_regression.py
```

---

# 🧠 Implementation

### Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```

---

### Load Dataset

```python
df = pd.read_csv("dataset.csv")
```

---

### Split Dataset

```python
X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### Train Model

```python
model = Lasso(alpha=0.1)

model.fit(X_train, y_train)
```

---

### Prediction

```python
y_pred = model.predict(X_test)
```

---

### Evaluation

```python
print("R2 Score :", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("Coefficients :", model.coef_)
```

---

# 📈 Model Evaluation

Common evaluation metrics include:

- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Example:

```
R² Score : 0.91

MSE : 1.82

RMSE : 1.35
```

---

# 🔍 Hyperparameter Tuning

The most important hyperparameter is:

```
alpha
```

Example:

```python
alphas = [0.001,0.01,0.1,1,10]

for a in alphas:
    model = Lasso(alpha=a)
    model.fit(X_train,y_train)
    print(a, model.score(X_test,y_test))
```

Choosing the optimal alpha improves model performance.

---

# 📊 Difference Between Linear, Ridge and Lasso Regression

| Feature | Linear | Ridge | Lasso |
|----------|--------|--------|--------|
| Regularization | ❌ | L2 | L1 |
| Feature Selection | ❌ | ❌ | ✅ |
| Prevents Overfitting | ❌ | ✅ | ✅ |
| Shrinks Coefficients | ❌ | Yes | Yes |
| Sets Coefficients to Zero | ❌ | ❌ | ✅ |

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# 📌 Future Improvements

- Cross Validation
- GridSearchCV for Alpha Selection
- Polynomial Features
- Pipeline Implementation
- Real-world Dataset
- Model Deployment using Streamlit
- Hyperparameter Optimization

---

# 🎯 Conclusion

Lasso Regression is one of the most useful regularized regression algorithms. It not only improves prediction performance by reducing overfitting but also performs automatic feature selection by shrinking unimportant coefficients to zero.

It is an excellent choice for datasets containing many features, especially when only a subset contributes significantly to the prediction.

---

# 📚 References

- Scikit-learn Documentation
- ISLR (Introduction to Statistical Learning)
- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow
- Pattern Recognition and Machine Learning – Christopher Bishop

---

## 👨‍💻 Author

**Parmeet Singh**

B.Tech CSE (Data Science)

GitHub: https://github.com/itsparmeet007

LinkedIn: https://www.linkedin.com/in/parmeet-singh-9414a8349/

---

⭐ If you found this project helpful, don't forget to **star the repository**.
