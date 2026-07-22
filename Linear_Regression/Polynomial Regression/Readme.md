# 📈 Polynomial Regression

> A Machine Learning algorithm that models **non-linear relationships** between the independent and dependent variables by fitting a polynomial equation to the data.

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [What is Polynomial Regression?](#-what-is-polynomial-regression)
- [Why Use Polynomial Regression?](#-why-use-polynomial-regression)
- [Real-World Applications](#-real-world-applications)
- [Mathematical Equation](#-mathematical-equation)
- [Working Process](#-working-process)
- [Advantages](#-advantages)
- [Disadvantages](#-disadvantages)
- [Implementation in Python](#-implementation-in-python)
- [Model Evaluation](#-model-evaluation)
- [Visualization](#-visualization)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

# 📌 Introduction

Linear Regression works well when the relationship between variables is linear.

However, many real-world datasets contain **curved or non-linear patterns**.

Polynomial Regression extends Linear Regression by adding polynomial terms (such as x², x³, etc.) to better capture these complex relationships.

Despite its name, Polynomial Regression is still considered a **Linear Regression model** because it is linear with respect to its coefficients.

---

# 📚 What is Polynomial Regression?

Polynomial Regression is a supervised machine learning algorithm used to predict continuous values when the relationship between the input feature and target variable is **non-linear**.

Instead of fitting a straight line, it fits a **curve**.

Example:

Instead of

\[
y = b_0 + b_1x
\]

Polynomial Regression fits

\[
y = b_0 + b_1x + b_2x^2 + b_3x^3 + ... + b_nx^n
\]

where:

- **x** = Independent Variable
- **y** = Dependent Variable
- **n** = Degree of Polynomial

---

# ❓ Why Use Polynomial Regression?

Use Polynomial Regression when:

- Data follows a curved trend
- Linear Regression underfits the data
- Higher-order relationships exist
- Better prediction accuracy is needed for non-linear datasets

Example:

- Salary vs Experience
- Temperature vs Electricity Consumption
- Population Growth
- Stock Trend Approximation
- Manufacturing Cost Analysis

---

# 🌍 Real-World Applications

## Finance

- Stock market trend approximation
- Investment forecasting

## Healthcare

- Disease progression prediction
- Drug dosage estimation

## Agriculture

- Crop yield prediction
- Rainfall estimation

## Economics

- Demand forecasting
- Sales prediction

## Engineering

- Stress-Strain analysis
- Material deformation prediction

---

# 🧮 Mathematical Equation

For degree 2:

\[
y = b_0 + b_1x + b_2x^2
\]

Degree 3:

\[
y = b_0 + b_1x + b_2x^2 + b_3x^3
\]

General Formula:

\[
y = b_0 + b_1x + b_2x^2 + ... + b_nx^n
\]

---

# ⚙️ Working Process

### Step 1

Collect Dataset

↓

### Step 2

Choose Polynomial Degree

↓

### Step 3

Transform Features using PolynomialFeatures

↓

### Step 4

Train Linear Regression Model

↓

### Step 5

Predict Values

↓

### Step 6

Evaluate Model

---

# 🛠 Implementation in Python

## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
```

---

## Load Dataset

```python
df = pd.read_csv("data.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

---

## Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## Polynomial Feature Transformation

```python
poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

---

## Train Model

```python
model = LinearRegression()

model.fit(X_train_poly, y_train)
```

---

## Prediction

```python
y_pred = model.predict(X_test_poly)
```

---

## Accuracy

```python
print("R² Score:", r2_score(y_test, y_pred))
```

---

# 📊 Visualization

```python
plt.scatter(X, y, color="red")

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.plot(
    X_grid,
    model.predict(poly.transform(X_grid)),
    color="blue"
)

plt.title("Polynomial Regression")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")

plt.show()
```

---

# 📈 Model Evaluation

Common evaluation metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Example:

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print(mae)
print(mse)
print(rmse)
```

---

# ✅ Advantages

- Handles non-linear data effectively
- Easy to implement
- Better fit than Linear Regression for curved data
- High prediction accuracy on polynomial relationships
- Uses the same Linear Regression algorithm after feature transformation

---

# ❌ Disadvantages

- Can overfit with high-degree polynomials
- Sensitive to outliers
- Computationally expensive for large degrees
- Choosing the optimal degree can be difficult
- Poor extrapolation outside the training range

---

# 📂 Project Structure

```
Polynomial-Regression/
│
├── data.csv
├── Polynomial_Regression.ipynb
├── polynomial_regression.py
├── requirements.txt
├── README.md
└── images/
```

---

# 📦 Requirements

Install dependencies using:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
```

Or

```bash
pip install -r requirements.txt
```

---

# 🚀 Future Improvements

- Hyperparameter tuning
- Cross Validation
- Pipeline implementation
- Regularization (Ridge/Lasso)
- Degree optimization using GridSearchCV
- Deploy using Flask/FastAPI/Streamlit

---

# 📚 References

- Scikit-Learn Documentation
- NumPy Documentation
- Pandas Documentation
- Matplotlib Documentation

---

# 👨‍💻 Author

**Parmeet Singh**

**B.Tech CSE (Data Science)**

Passionate about:

- Machine Learning
- Data Science
- Artificial Intelligence
- Deep Learning
- MLOps

---

## ⭐ If you found this project helpful, consider giving it a Star!
