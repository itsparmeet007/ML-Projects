# Categorical Imputation in Machine Learning

## Overview

This project demonstrates different techniques for handling missing values in **categorical features** using Python and Scikit-learn.

Missing data is a common problem in real-world datasets. In this notebook, we explore and compare two popular categorical imputation methods:

1. **Frequent Value Imputation (Mode Imputation)**
2. **Missing Category Imputation**

The analysis is performed using features from the House Prices dataset.

---

## Dataset Features Used

* `GarageQual`
* `FireplaceQu`
* `SalePrice`

`SalePrice` is used to analyze how different imputation techniques affect the distribution of the target variable.

---

## Techniques Covered

### 1. Frequent Value Imputation

Missing values are replaced with the most frequently occurring category (mode).

Example:

| Original | After Imputation |
| -------- | ---------------- |
| TA       | TA               |
| NaN      | TA               |
| TA       | TA               |

#### Advantages

* Simple and fast
* Preserves dataset size
* Easy to implement

#### Limitations

* Can distort category distributions
* May introduce bias if missing values are not random
* Works best when one category is clearly dominant

---

### 2. Missing Category Imputation

Missing values are replaced with a new category such as `"Missing"`.

Example:

| Original | After Imputation |
| -------- | ---------------- |
| Gd       | Gd               |
| NaN      | Missing          |
| TA       | TA               |

#### Advantages

* Preserves information about missingness
* Useful when missing values may carry meaning
* Does not artificially increase existing category frequencies

#### Limitations

* Adds an extra category
* May increase dimensionality after encoding

---

## Exploratory Analysis

The notebook includes:

* Missing value analysis
* Category frequency visualization
* Distribution comparison using KDE plots
* Impact analysis on `SalePrice`
* Before and after imputation comparisons

---

## Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Scikit-Learn Implementation

### Frequent Value Imputation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
```

### Missing Category Imputation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy='constant',
    fill_value='Missing'
)
```

---

## Project Structure

```text
Categorical-Imputation/
│
├── Categorical_Imputation.ipynb
├── README.md
└── train.csv
```

---

## Key Learning Outcomes

* Understanding categorical missing data
* Choosing appropriate imputation strategies
* Evaluating the impact of imputation on feature distributions
* Implementing imputation using Scikit-learn
* Visualizing the effects of missing value handling

---

## Future Improvements

* Compare with Random Sample Imputation
* Compare model performance before and after imputation
* Automate preprocessing using Scikit-learn Pipelines
* Apply techniques to additional categorical features

---

## Author

**Parmeet Singh**

Machine Learning & Data Science Enthusiast

Currently exploring data preprocessing, feature engineering, machine learning, and real-world data analysis projects.

