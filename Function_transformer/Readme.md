# Feature Transformation and Skewness

## What is Feature Transformation?

Feature Transformation is a preprocessing technique used to change the scale or distribution of data so that machine learning models can perform better.

Transformations help in:

- Reducing skewness
- Handling outliers
- Improving normality
- Improving model performance

---

# What is Skewness?

Skewness describes the asymmetry of data distribution.

There are 3 types:

| Type | Meaning |
|---|---|
| Normal Distribution | Symmetrical data |
| Right Skewed | Tail extends toward right |
| Left Skewed | Tail extends toward left |

---

# 1. Right Skewed Distribution

## Characteristics

- Tail on the right side
- Large outliers present
- Mean > Median

Example:

```text
1, 2, 3, 5, 100
```

## Common Examples

- Salary
- House prices
- Population

---

# 2. Left Skewed Distribution

## Characteristics

- Tail on the left side
- Small outliers present
- Mean < Median

Example:

```text
-100, 1, 2, 3, 4
```

## Common Examples

- Easy exam marks
- Retirement age

---

# How to Detect Skewness

## Method 1 — Histogram

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['column'], kde=True)
plt.show()
```

---

## Method 2 — Skewness Value

```python
df['column'].skew()
```

### Interpretation

| Skewness Value | Meaning |
|---|---|
| = 0 | Normal Distribution |
| > 0 | Right Skewed |
| < 0 | Left Skewed |

---

# Transformations

---

# 1. Log Transformation

## Formula

```text
y = log(x)
```

## Purpose

- Reduces right skewness
- Compresses large values
- Reduces outlier effect

## Best For

- Highly right-skewed data

## Python Example

```python
import numpy as np

x = np.array([1,10,100,1000])

log_x = np.log(x)

print(log_x)
```

---

# log1p()

```python
np.log1p(x)
```

Means:

```text
log(1 + x)
```

Used because:

```text
log(0) is undefined
```

---

# Advantages

- Strong skew reduction
- Improves normality

## Disadvantages

- Cannot directly handle negative values

---

# 2. Square Root Transformation

## Formula

```text
y = √x
```

## Purpose

- Reduces moderate skewness
- Compresses values gently

## Best For

- Moderately right-skewed data

## Python Example

```python
import numpy as np

x = np.array([1,4,9,16])

sqrt_x = np.sqrt(x)

print(sqrt_x)
```

---

# Advantages

- Easy interpretation
- Less aggressive than log

## Disadvantages

- Cannot directly handle negative values

---

# 3. Reciprocal Transformation

## Formula

```text
y = 1/x
```

## Purpose

- Strongly reduces large values
- Handles extreme skewness

## Best For

- Highly skewed data with large outliers

## Python Example

```python
import numpy as np

x = np.array([1,2,5,10])

reciprocal_x = np.reciprocal(x.astype(float))

print(reciprocal_x)
```

---

# Advantages

- Strong skewness reduction

## Disadvantages

- Cannot handle zero
- Difficult interpretation

---

# 4. Square Transformation

## Formula

```text
y = x²
```

## Purpose

- Expands large values
- Useful for left-skewed data

## Best For

- Left-skewed distributions

## Python Example

```python
import numpy as np

x = np.array([1,2,3,4])

square_x = np.square(x)

print(square_x)
```

---

# Advantages

- Captures non-linear relationships

## Disadvantages

- Can increase skewness

---

# Comparison Table

| Transformation | Formula | Use Case |
|---|---|---|
| Log | log(x) | Strong right skew |
| Square Root | √x | Moderate right skew |
| Reciprocal | 1/x | Extreme right skew |
| Square | x² | Left skew |

---

# Function Transformer

Function Transformer is a preprocessing tool in scikit-learn used to apply custom mathematical functions to features.

---

# Import

```python
from sklearn.preprocessing import FunctionTransformer
```

---

# Examples

## Log Transform

```python
transformer = FunctionTransformer(np.log1p)
```

## Square Root Transform

```python
transformer = FunctionTransformer(np.sqrt)
```

## Reciprocal Transform

```python
transformer = FunctionTransformer(np.reciprocal)
```

## Square Transform

```python
transformer = FunctionTransformer(np.square)
```

---

# When to Use Which Transformation

| Situation | Recommended Transformation |
|---|---|
| Highly right skewed | Log / Reciprocal |
| Moderate right skew | Square Root |
| Left skewed | Square |
| Extreme outliers | Reciprocal |

---

# Interview Definitions

## Skewness

> Skewness measures the asymmetry of data distribution.

## Log Transformation

> Log transformation reduces skewness by compressing large values using logarithms.

## Function Transformer

> Function Transformer applies mathematical functions to features during preprocessing.

---
