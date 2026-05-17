# Power Transformer (Box-Cox & Yeo-Johnson)

## Introduction

Power Transformation is a feature transformation technique used to make data:

- More normally distributed
- Less skewed
- More suitable for machine learning algorithms

It is commonly used in:

- Linear Regression
- Logistic Regression
- SVM
- KNN
- Neural Networks

because many machine learning algorithms perform better when data follows a normal distribution.

---

# Why Power Transformation?

Real-world datasets often contain:

- Right-skewed data
- Left-skewed data
- Outliers
- Non-normal distributions

Power transformation helps in:

✅ Reducing skewness  
✅ Stabilizing variance  
✅ Improving model performance  
✅ Making data more Gaussian-like

---

# Types of Power Transformation

There are two major types:

1. Box-Cox Transformation  
2. Yeo-Johnson Transformation

---

# 1. Box-Cox Transformation

## Definition

Box-Cox transformation is a mathematical power transformation used to normalize positively skewed data.

---

# Important Condition

Box-Cox only works on:

```python
Positive values (> 0)
