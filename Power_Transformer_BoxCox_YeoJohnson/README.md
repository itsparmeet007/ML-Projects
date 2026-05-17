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

# Power Transformer Techniques in Machine Learning

This project demonstrates the implementation of **Power Transformation techniques** used in Machine Learning and Data Preprocessing.

The notebook covers:

- Box-Cox Transformation
- Yeo-Johnson Transformation
- Feature Distribution Transformation
- Data Normalization
- Skewness Reduction
- Handling Non-Normal Data

---

# Project Overview

Power Transformations are statistical techniques used to transform data into a more Gaussian-like distribution.

These transformations help improve:

- Model performance
- Feature scaling
- Statistical assumptions
- Data symmetry
- Variance stabilization

This notebook provides practical implementation examples using Python and Scikit-learn.

---

# Techniques Covered

## 1. Box-Cox Transformation

Box-Cox transformation is applied to strictly positive numerical data.

### Benefits

- Reduces skewness
- Stabilizes variance
- Improves normality

---

## 2. Yeo-Johnson Transformation

Yeo-Johnson transformation works with both:

- Positive values
- Negative values

Unlike Box-Cox, it does not require strictly positive data.

### Benefits

- Handles zero and negative values
- Reduces skewness
- Improves feature distribution

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

# Project Structure

```bash
Power_Transformer_BoxCox_YeoJohnson/
│
├── Power Transformer.ipynb
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/itsparmeet007/ML-Projects.git
```

Move into the project directory:

```bash
cd ML-Projects/Power_Transformer_BoxCox_YeoJohnson
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Notebook

Open Jupyter Notebook or VS Code and run:

```bash
Power Transformer.ipynb
```

---

# Learning Outcomes

After completing this project, you will understand:

- Why feature transformation is important
- When to use Box-Cox transformation
- When to use Yeo-Johnson transformation
- How to reduce skewness in datasets
- How transformed data improves ML performance

---

# Author

Parmeet Singh

GitHub:
https://github.com/itsparmeet007

