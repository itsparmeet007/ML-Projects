# 🏷️ Categorical Missing Value Imputation
 main

## 📌 Overview

**Categorical Missing Value Imputation** is the process of replacing missing values in categorical (non-numeric) features with appropriate values so that machine learning models can effectively process the data.

Categorical features represent labels or groups (e.g., Gender, City, Department, Product Category), and missing values in these features can negatively impact data analysis and model performance if left untreated.

Choosing the right imputation technique depends on the amount of missing data, the importance of the feature, and the machine learning algorithm being used.

---

# 🎯 Why Do We Need Categorical Imputation?

Missing values in categorical features may occur due to:

- Incomplete surveys
- Data entry errors
- System failures
- User privacy concerns
- Data integration issues

Most machine learning algorithms cannot directly handle missing categorical values. Therefore, imputing these values is an essential preprocessing step.

---

# 🧠 Common Categorical Imputation Techniques

## 1. Frequent Category (Mode) Imputation

The missing values are replaced with the **most frequently occurring category** in the feature.

### Example

Original Data

| City |
|------|
| Delhi |
| Mumbai |
| Delhi |
| NaN |
| Chennai |

After Mode Imputation

| City |
|------|
| Delhi |
| Mumbai |
| Delhi |
| Delhi |
| Chennai |

### Python Example

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")

df["City"] = imputer.fit_transform(df[["City"]])
```

### Advantages

- Simple and fast
- Preserves dataset size
- Works well when missing values are few

### Disadvantages

- May increase the frequency of the dominant category
- Can introduce bias if many values are missing

---

## 2. Missing Category Imputation

Instead of replacing missing values with an existing category, a **new category** such as `"Missing"` or `"Unknown"` is created.

### Example

Original Data

| Department |
|------------|
| HR |
| Sales |
| NaN |
| Finance |

After Imputation

| Department |
|------------|
| HR |
| Sales |
| Missing |
| Finance |

### Python Example

```python
df["Department"] = df["Department"].fillna("Missing")
```

or

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="constant",
    fill_value="Missing"
)

df["Department"] = imputer.fit_transform(df[["Department"]])
```

### Advantages

- Preserves information that values were missing
- Easy to implement
- Useful when missing values are meaningful

### Disadvantages

- Introduces a new category
- May slightly increase feature cardinality

---

## 3. Random Sample Imputation

Missing values are replaced with randomly selected existing categories from the same feature.

### Example

Original

```
Red
Blue
Green
NaN
Blue
Red
```

Possible Output

```
Red
Blue
Green
Blue
Blue
Red
```

### Advantages

- Preserves category distribution
- Maintains dataset variability

### Disadvantages

- Randomness may produce slightly different results each time
- More complex than mode imputation

---

## 4. Predictive Imputation

A machine learning model predicts the missing category using other available features.

### Example

Predict the missing **Education Level** using:

- Age
- Salary
- Occupation
- Experience

### Advantages

- Often more accurate
- Utilizes relationships between features

### Disadvantages

- Computationally expensive
- More difficult to implement
- Requires sufficient training data

---

# 📊 Example Dataset

Before Imputation

| Gender |
|---------|
| Male |
| Female |
| NaN |
| Female |
| Male |

---

### Mode Imputation

| Gender |
|---------|
| Male |
| Female |
| Female |
| Female |
| Male |

---

### Missing Category Imputation

| Gender |
|---------|
| Male |
| Female |
| Missing |
| Female |
| Male |

---

# ⚙️ Implementation Using Pandas

## Mode Imputation

```python
mode = df["Gender"].mode()[0]

df["Gender"] = df["Gender"].fillna(mode)
```

---

## Missing Category Imputation

```python
df["Gender"] = df["Gender"].fillna("Missing")
```

---

# ⚙️ Implementation Using Scikit-Learn

## Most Frequent Strategy

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")

df["Gender"] = imputer.fit_transform(df[["Gender"]])
```

---

## Constant Strategy

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value="Missing"
)

df["Gender"] = imputer.fit_transform(df[["Gender"]])
```

---

# ✅ Advantages of Categorical Imputation

- Removes missing values efficiently
- Enables machine learning algorithms to process categorical features
- Prevents loss of valuable data
- Improves model training
- Easy to implement using Pandas and Scikit-learn

---

# ❌ Disadvantages

- Incorrect imputation may introduce bias
- Mode imputation can overrepresent the most common category
- Missing category imputation increases the number of unique categories
- Predictive methods require additional computation

---

# 📌 Which Technique Should You Choose?

| Situation | Recommended Technique |
|-----------|-----------------------|
| Few missing values | Frequent Category (Mode) |
| Missing values are informative | Missing Category |
| Preserve category distribution | Random Sample |
| High accuracy required | Predictive Imputation |

---

# 🚫 Common Mistakes

- Using the mode when a large percentage of values are missing
- Replacing missing values without understanding why they are missing
- Using different imputation methods for training and testing data
- Fitting the imputer separately on training and test datasets (causes data leakage)

---

# 💡 Best Practices

- Analyze the percentage of missing values before selecting an imputation technique.
- Fit the imputer **only on the training data**, then transform both training and testing datasets.
- If missing values carry useful information, consider creating a separate `"Missing"` category.
- Compare multiple imputation methods and evaluate their impact on model performance.
- Document the chosen imputation strategy for reproducibility.

---

# 📝 Key Takeaways

- Categorical imputation handles missing values in non-numeric features.
- **Mode (Most Frequent) Imputation** is the simplest and most commonly used technique.
- **Missing Category Imputation** creates a new category to preserve missingness information.
- **Random Sample Imputation** maintains the original category distribution.
- **Predictive Imputation** uses machine learning models to estimate missing categories.
- The choice of imputation method should depend on the dataset, the extent of missing values, and the machine learning task.

---

# 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/impute.html
- Pandas Documentation: https://pandas.pydata.org/docs/
- Feature Engineering for Machine Learning by Alice Zheng & Amanda Casari
- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurélien Géron

---

## 👨‍💻 Author

**Parmeet Singh**

Aspiring AI & Machine Learning Engineer passionate about Data Science, Machine Learning, Deep Learning, Computer Vision, NLP, and MLOps.

- 🌐 GitHub: https://github.com/your-github-username
- 💼 LinkedIn: https://linkedin.com/in/your-linkedin-profile

> ⭐ If you found this repository helpful, consider giving it a star!
