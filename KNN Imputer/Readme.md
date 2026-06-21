# 🧠 K-Nearest Neighbors (KNN) Imputation

## 📌 Overview

Missing values are one of the most common challenges encountered during data preprocessing in Machine Learning. Many machine learning algorithms cannot handle missing data directly, making imputation an essential step before model training.

This project demonstrates the implementation of **KNN Imputer (K-Nearest Neighbors Imputation)**, a powerful technique that fills missing values by utilizing information from similar data points in the dataset.

---

## 🎯 Objective

The primary objectives of this notebook are:

* Understand the concept of missing data.
* Learn why missing value treatment is important.
* Explore the KNN Imputation technique.
* Implement KNN Imputer using Scikit-Learn.
* Compare original and imputed datasets.
* Understand the advantages and limitations of KNN Imputation.

---

## 📚 What is KNN Imputation?

KNN Imputation is a missing value treatment technique that uses the **K-Nearest Neighbors algorithm** to estimate missing values.

Instead of replacing missing values with simple statistics such as mean, median, or mode, KNN Imputer identifies observations that are most similar to the incomplete observation and uses their values to fill in the missing data.

The underlying assumption is:

> Similar data points tend to have similar feature values.

---

## 🔍 How KNN Imputer Works

Suppose a dataset contains missing values in a feature.

### Step 1: Select the Missing Observation

Identify the row containing the missing value.

### Step 2: Compute Distances

Calculate the distance between the incomplete row and all other rows using available features.

Commonly used distance metric:

* Euclidean Distance

Formula:

[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
]

### Step 3: Find K Nearest Neighbors

Select the K most similar observations based on the calculated distances.

Example:

* K = 3
* Choose the three closest rows.

### Step 4: Aggregate Neighbor Values

For numerical features:

* Average of neighbors is used.

For categorical features:

* Majority voting may be used.

### Step 5: Replace Missing Value

The computed value is assigned to the missing entry.

---

## 🧮 Example

### Original Dataset

| Age | Salary |
| --- | ------ |
| 25  | 30000  |
| 30  | 40000  |
| NaN | 35000  |
| 35  | 45000  |

For the missing Age value:

1. Calculate similarity with other rows.
2. Select nearest neighbors.
3. Compute average age.

Imputed Age:

[
(25 + 30)/2 = 27.5
]

Final Dataset:

| Age  | Salary |
| ---- | ------ |
| 25   | 30000  |
| 30   | 40000  |
| 27.5 | 35000  |
| 35   | 45000  |

---

## ⚙️ Implementation Using Scikit-Learn

### Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
```

### Create Sample Dataset

```python
data = {
    'Age': [25, 30, np.nan, 35],
    'Salary': [30000, 40000, 35000, 45000]
}

df = pd.DataFrame(data)
```

### Apply KNN Imputer

```python
imputer = KNNImputer(n_neighbors=2)

df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

print(df_imputed)
```

---

## 🔧 Important Parameters

### `n_neighbors`

Number of neighboring samples used for imputation.

```python
KNNImputer(n_neighbors=5)
```

Default:

```python
n_neighbors = 5
```

---

### `weights`

Determines how neighbors contribute.

Options:

```python
weights='uniform'
```

All neighbors contribute equally.

```python
weights='distance'
```

Closer neighbors have more influence.

---

### `metric`

Distance metric used to find neighbors.

Default:

```python
metric='nan_euclidean'
```

Specially designed for datasets containing missing values.

---

## 📊 Advantages of KNN Imputation

### 1. Preserves Relationships

Unlike mean or median imputation, KNN considers relationships among features.

### 2. More Accurate

Often produces more realistic estimates.

### 3. Non-Parametric

No assumptions about data distribution.

### 4. Handles Complex Patterns

Useful when data contains non-linear relationships.

---

## ⚠️ Limitations

### 1. Computationally Expensive

Distance calculations become costly for large datasets.

### 2. Sensitive to Scaling

Features should be standardized before applying KNN.

Example:

```python
from sklearn.preprocessing import StandardScaler
```

### 3. Memory Intensive

Requires storing and comparing multiple observations.

### 4. Not Ideal for Extremely Sparse Data

Performance decreases when too many values are missing.

---

## 🚀 Best Practices

### Normalize Data

Before applying KNN:

```python
from sklearn.preprocessing import StandardScaler
```

### Experiment with K Values

Try:

```python
K = 3
K = 5
K = 7
```

to find the most suitable value.

### Evaluate Imputation Quality

Compare different techniques:

* Mean Imputation
* Median Imputation
* Mode Imputation
* KNN Imputation

---

## 📈 When to Use KNN Imputation

✅ Dataset contains moderate missing values.

✅ Features have meaningful relationships.

✅ Dataset size is manageable.

✅ Higher imputation accuracy is desired.

---

## ❌ When Not to Use

❌ Very large datasets.

❌ Extremely sparse datasets.

❌ Real-time systems requiring fast preprocessing.

---

## 🏆 Conclusion

KNN Imputation is an advanced missing value handling technique that estimates missing values using neighboring observations rather than relying on simple statistical measures. It often provides better and more realistic imputations by preserving relationships within the data. However, it comes with increased computational cost and requires proper feature scaling for optimal performance.

By mastering KNN Imputation, data scientists can significantly improve data quality and model performance during the preprocessing stage of Machine Learning workflows.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Jupyter Notebook

---

## 📖 References

* Scikit-Learn Documentation
* Machine Learning Data Preprocessing Techniques
* K-Nearest Neighbors Algorithm Fundamentals

---

### ⭐ If you found this project useful, consider giving it a star!
