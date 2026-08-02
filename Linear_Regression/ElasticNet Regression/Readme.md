# ElasticNet Regression

This notebook demonstrates **ElasticNet Regression**, a regularized linear regression technique that combines the advantages of **Ridge Regression (L2 regularization)** and **Lasso Regression (L1 regularization)**.

ElasticNet is especially useful when a dataset contains many features, correlated features, or when we want both regularization and feature selection.

---

## 📌 Topics Covered

* Linear Regression
* Regularization
* L1 Regularization
* L2 Regularization
* Lasso Regression
* Ridge Regression
* ElasticNet Regression
* `alpha`
* `l1_ratio`
* Model Training
* Model Prediction
* Model Evaluation

---

# 🧠 What is ElasticNet Regression?

ElasticNet Regression is a linear regression algorithm that combines **L1 and L2 regularization**.

It can be considered a combination of:

* **Lasso Regression** → L1 regularization
* **Ridge Regression** → L2 regularization

The objective is to minimize the prediction error while also penalizing large coefficients.

The regularization helps reduce **overfitting** and can improve model generalization.

---

# 📐 ElasticNet Equation

The ElasticNet objective function combines the L1 and L2 penalties.

[
\text{Loss} =
\text{MSE}
+
\alpha
\left(
l1_ratio \sum |w_j|
+
(1-l1_ratio)\sum w_j^2
\right)
]

Where:

* `MSE` = Mean Squared Error
* (\alpha) = Overall regularization strength
* `l1_ratio` = Balance between L1 and L2 regularization
* (w_j) = Model coefficients

---

# 🔥 L1 vs L2 vs ElasticNet

| Model             | Regularization | Main Effect                                         |
| ----------------- | -------------- | --------------------------------------------------- |
| Linear Regression | None           | No regularization                                   |
| Ridge Regression  | L2             | Shrinks coefficients                                |
| Lasso Regression  | L1             | Can make coefficients exactly zero                  |
| ElasticNet        | L1 + L2        | Shrinks coefficients and performs feature selection |

---

# ⚙️ ElasticNet Parameters

Scikit-learn provides ElasticNet through:

```python
from sklearn.linear_model import ElasticNet
```

A basic model can be created using:

```python
model = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5
)
```

### `alpha`

`alpha` controls the overall strength of regularization.

* Small `alpha` → weaker regularization
* Large `alpha` → stronger regularization

Example:

```python
ElasticNet(alpha=0.01)
```

has weaker regularization than:

```python
ElasticNet(alpha=1.0)
```

---

## `l1_ratio`

`l1_ratio` determines how much L1 and L2 regularization contribute.

| `l1_ratio` | Behavior                       |
| ---------: | ------------------------------ |
|        `0` | Ridge-like / L2 regularization |
|      `0.5` | Equal combination of L1 and L2 |
|        `1` | Lasso-like / L1 regularization |

For example:

```python
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

uses an equal mixture of L1 and L2 regularization.

---

# 🚀 Implementation

Import ElasticNet:

```python
from sklearn.linear_model import ElasticNet
```

Create the model:

```python
reg = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

Train the model:

```python
reg.fit(X_train, y_train)
```

Make predictions:

```python
y_pred = reg.predict(X_test)
```

---

# 📊 Model Evaluation

ElasticNet can be evaluated using common regression metrics.

### Mean Squared Error

[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
]

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

### R² Score

[
R^2 =
1 -
\frac{\sum(y_i-\hat{y_i})^2}
{\sum(y_i-\bar{y})^2}
]

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

A higher (R^2) generally indicates that the model explains more of the variance in the target variable.

---

# 🎯 Why Use ElasticNet?

ElasticNet is particularly useful when:

* There are many input features.
* Features are highly correlated.
* You want regularization.
* You want some form of feature selection.
* Lasso alone is unstable because of correlated features.
* Ordinary Linear Regression is overfitting.

---

# 🔍 ElasticNet vs Ridge vs Lasso

| Feature                      | Ridge   | Lasso           | ElasticNet |
| ---------------------------- | ------- | --------------- | ---------- |
| L1 Regularization            | ❌       | ✅               | ✅          |
| L2 Regularization            | ✅       | ❌               | ✅          |
| Feature Selection            | ❌       | ✅               | ✅          |
| Handles Correlated Features  | ✅       | Can be unstable | ✅          |
| Coefficients Can Become Zero | ❌       | ✅               | ✅          |
| Controls Regularization      | `alpha` | `alpha`         | `alpha`    |
| L1/L2 Balance                | ❌       | Fixed L1        | `l1_ratio` |

---

# 🧪 Effect of `alpha`

The regularization strength can significantly affect the learned coefficients.

For example:

```python
alphas = [0.01, 0.1, 1, 10]
```

As `alpha` increases, the model applies stronger regularization, which generally causes coefficients to become smaller.

However, excessive regularization can cause **underfitting**.

---

# 🧪 Effect of `l1_ratio`

Different values of `l1_ratio` change the balance between L1 and L2 regularization.

```python
l1_ratio = 0
```

→ Ridge-like behavior

```python
l1_ratio = 0.5
```

→ Combination of L1 and L2

```python
l1_ratio = 1
```

→ Lasso-like behavior

This parameter can therefore be tuned to find a suitable balance for the dataset.

---

# 🔄 General Workflow

```text
Load Dataset
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Create ElasticNet Model
      ↓
Set alpha and l1_ratio
      ↓
Train Model
      ↓
Make Predictions
      ↓
Evaluate Model
      ↓
Tune Hyperparameters
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# 🎯 Learning Objectives

After completing this notebook, you should understand:

* What ElasticNet Regression is.
* Why regularization is required.
* The difference between L1 and L2 regularization.
* How ElasticNet combines Ridge and Lasso.
* The purpose of `alpha`.
* The purpose of `l1_ratio`.
* How to train an ElasticNet model using Scikit-learn.
* How to evaluate a regression model.
* How regularization affects model coefficients.

---

# 🏁 Conclusion

ElasticNet Regression combines **L1 and L2 regularization** to provide a flexible regularized linear regression model.

It is particularly useful for datasets with **many features and correlated predictors**, where using only Ridge or Lasso may not provide the desired behavior.

By adjusting `alpha` and `l1_ratio`, ElasticNet can be tuned to control the strength and type of regularization applied to the model.

