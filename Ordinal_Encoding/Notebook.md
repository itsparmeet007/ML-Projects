# Ordinal Encoding vs Label Encoding

## 1. Ordinal Encoding

### Definition
Ordinal Encoding is a technique used to convert categorical data into numerical values while preserving the order or ranking of categories.

It is used when categories have a meaningful relationship.

---

## Example

### Input

| Size |
|------|
| Small |
| Medium |
| Large |

### Encoded Output

| Size | Encoded |
|------|----------|
| Small | 0 |
| Medium | 1 |
| Large | 2 |

---

## Why Use Ordinal Encoding?

Machine Learning algorithms cannot understand text values directly.

Ordinal Encoding converts categories into numbers while maintaining their ranking.

---

## When to Use

Use Ordinal Encoding when data has a natural order, such as:

- Education Level:
  - School < College < PhD

- Ratings:
  - Poor < Average < Good < Excellent

- Shirt Sizes:
  - S < M < L < XL

---

## When NOT to Use

Do not use Ordinal Encoding when categories do not have any order.

### Example

| Color |
|------|
| Red |
| Blue |
| Green |

There is no ranking among colors.

Use One Hot Encoding instead.

---

## Python Example

```python
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

df = pd.DataFrame({
    'Size': ['Small', 'Medium', 'Large']
})

encoder = OrdinalEncoder(
    categories=[['Small', 'Medium', 'Large']]
)

df['Size_encoded'] = encoder.fit_transform(df[['Size']])

print(df)
