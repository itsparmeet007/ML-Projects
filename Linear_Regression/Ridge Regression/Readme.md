# 📈 Ridge Regression

A complete implementation and explanation of **Ridge Regression** using **Python** and **Scikit-learn**. This project demonstrates how Ridge Regression improves Linear Regression by adding **L2 Regularization**, reducing overfitting while keeping all features in the model.

---

# 📌 Table of Contents

- Introduction
- What is Ridge Regression?
- How Ridge Regression Works
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
- Difference Between Linear, Ridge & Lasso Regression
- Technologies Used
- Conclusion
- References

---

# 📖 Introduction

Ridge Regression is a supervised machine learning algorithm used for predicting continuous numerical values. It is an extension of Linear Regression that includes **L2 Regularization**, which penalizes the square of the regression coefficients.

Unlike Lasso Regression, Ridge Regression **does not eliminate features**. Instead, it shrinks their coefficients toward zero, making the model less sensitive to noise and reducing overfitting.

---

# 🎯 What is Ridge Regression?

Ridge Regression is a regularized linear regression technique that adds a penalty term to the cost function to reduce model complexity.

It is particularly useful when:

- The dataset has many features
- Features are highly correlated (multicollinearity)
- The model is overfitting
- Better generalization is required

---

# ⚙️ How Ridge Regression Works

Ridge Regression minimizes the following cost function:

\[
Loss = RSS + \alpha \sum_{i=1}^{n}w_i^2
\]

Where:

- RSS = Residual Sum of Squares
- α = Regularization parameter
- \(w_i\) = Regression coefficients

The L2 penalty shrinks coefficients toward zero but rarely makes them exactly zero.

This results in:

- Reduced overfitting
- Lower variance
- Improved prediction on unseen data
- Better handling of multicollinearity

---

# 🧮 Mathematical Formula

Linear Regression:

\[
y=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n
\]

Ridge Cost Function:

\[
J(\beta)=\sum_{i=1}^{m}(y_i-\hat y_i)^2+\alpha\sum_{j=1}^{n}\beta_j^2
\]

Where:

- \(y_i\) = Actual value
- \(\hat y_i\) = Predicted value
- \(\beta_j\) = Regression coefficients
- α = Regularization strength

---

# 🚀 Advantages

✔ Reduces overfitting

✔ Handles multicollinearity effectively

✔ Improves model generalization

✔ Keeps all features in the model

✔ Stable coefficient estimates

✔ Works well for high-dimensional datasets

---

# ❌ Disadvantages

- Does not perform feature selection
- Sensitive to feature scaling
- Requires tuning of α
- Can underfit if α is too large
- Coefficients become biased due to regularization

---

# 💼 Applications

Ridge Regression is widely used in:

- House Price Prediction
- Sales Forecasting
- Stock Market Prediction
- Medical Data Analysis
- Financial Modeling
- Customer Demand Forecasting
- Weather Prediction
- Economic Forecasting

---

# 📊 Dataset

You can use any regression dataset such as:

- California Housing Dataset
- Boston Housing Dataset
- House Prices Dataset (Kaggle)
- Custom CSV Dataset

Example Features:

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
git clone https://github.com/yourusername/Ridge-Regression.git
```

Move into the project directory

```bash
cd Ridge-Regression
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Project Structure

```
Ridge-Regression/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── RidgeRegression.ipynb
│
├── models/
│
├── images/
│
├── requirements.txt
│
├── README.md
│
└── ridge_regression.py
```

---

# 🧠 Implementation

### Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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
model = Ridge(alpha=1.0)

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
print("R² Score :", r2_score(y_test, y_pred))
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

Example Output:

```
R² Score : 0.92

MSE : 1.56

RMSE : 1.25
```

---

# 🔍 Hyperparameter Tuning

The primary hyperparameter is:

```
alpha
```

Example:

```python
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    print(a, model.score(X_test, y_test))
```

Choose the alpha value that provides the best balance between bias and variance.

---

# 📊 Difference Between Linear, Ridge and Lasso Regression

| Feature | Linear | Ridge | Lasso |
|----------|--------|--------|--------|
| Regularization | ❌ | L2 | L1 |
| Feature Selection | ❌ | ❌ | ✅ |
| Prevents Overfitting | ❌ | ✅ | ✅ |
| Shrinks Coefficients | ❌ | ✅ | ✅ |
| Sets Coefficients to Zero | ❌ | ❌ | ✅ |
| Handles Multicollinearity | ❌ | ✅ | Moderate |

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
- Polynomial Ridge Regression
- Pipeline Implementation
- Model Deployment using Streamlit
- Hyperparameter Optimization

---

# 🎯 Conclusion

Ridge Regression is a powerful extension of Linear Regression that uses **L2 Regularization** to reduce overfitting and improve generalization. Unlike Lasso Regression, it retains all features by shrinking coefficients instead of removing them, making it especially useful when all variables contribute to the prediction.

---

# 📚 References

- Scikit-learn Documentation
- ISLR (Introduction to Statistical Learning)
- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow
- Pattern Recognition and Machine Learning – Christopher Bishop

---

## 👨‍💻 Author

**Parmeet Singh**

**B.Tech CSE (Data Science)**

GitHub: https://github.com/itsparmeet007

LinkedIn: https://www.linkedin.com/in/parmeet-singh-9414a8349/

---

⭐ If you found this project helpful, don't forget to **star the repository**.
