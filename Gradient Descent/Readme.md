# 📉 Gradient Descent for Linear Regression (Optimizing Intercept)

## 📌 Project Overview

This notebook demonstrates how **Gradient Descent** works by manually optimizing the **intercept (b)** of a simple linear regression model while keeping the slope **(m)** fixed.

Instead of relying on Scikit-learn's optimization, we calculate gradients ourselves and observe how the regression line gradually converges toward the optimal solution.

This project is ideal for beginners who want to understand the mathematics behind Gradient Descent.

---

## 🎯 Objectives

- Generate a synthetic regression dataset.
- Train a Linear Regression model using Ordinary Least Squares (OLS).
- Keep the slope fixed.
- Optimize only the intercept using Gradient Descent.
- Visualize the movement of the regression line after every iteration.
- Understand the effect of learning rate and epochs.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 📂 Workflow

### 1. Generate Dataset

A synthetic regression dataset is created using:

- 10 samples
- 1 feature
- Random noise

---

### 2. Train Linear Regression

The dataset is first fitted using Scikit-learn's `LinearRegression()` model to obtain the optimal slope and intercept using OLS.

---

### 3. Fix the Slope

Instead of updating both parameters,

- Slope (m) is kept constant.
- Intercept (b) is initialized with different values.

---

### 4. Calculate Gradient

The gradient of the Mean Squared Error with respect to the intercept is calculated manually.

Gradient Formula:

\[
\frac{\partial J}{\partial b}
=
-2 \times \text{mean}(y-\hat y)
\]

---

### 5. Update Intercept

The intercept is updated using the Gradient Descent update rule:

\[
b = b - \alpha \times \frac{\partial J}{\partial b}
\]

where

- **b** → intercept
- **α** → learning rate

---

### 6. Repeat for Multiple Epochs

The process continues until the intercept converges toward its optimal value.

---

### 7. Visualize Results

The notebook plots:

- Dataset
- OLS Regression Line
- Initial Guess
- Updated Regression Line after Gradient Descent

This helps visualize how Gradient Descent moves closer to the optimal solution.

---

## 📊 Concepts Covered

- Linear Regression
- Ordinary Least Squares (OLS)
- Cost Function
- Gradient
- Learning Rate
- Epochs
- Gradient Descent Optimization
- Model Convergence

---

## 📈 Key Learning

Through this notebook you will learn:

- Why Gradient Descent works.
- How gradients are calculated manually.
- How learning rate affects convergence.
- Difference between OLS and Gradient Descent.
- Why multiple epochs are required.
- How regression parameters are optimized iteratively.

---

## 🚀 Future Improvements

- Optimize both **Slope (m)** and **Intercept (b)** simultaneously.
- Animate Gradient Descent.
- Plot Cost vs Epoch.
- Visualize the loss surface.
- Extend to Multiple Linear Regression.

---

## 📚 Conclusion

This project provides an intuitive understanding of Gradient Descent by manually updating the intercept parameter of a linear regression model. It bridges the gap between the mathematical equations and their practical implementation, making it easier to understand how machine learning models learn from data.

---

## 👨‍💻 Author

**Parmeet Singh**

Aspiring Data Scientist | Machine Learning Enthusiast
