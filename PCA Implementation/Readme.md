# Principal Component Analysis (PCA)

## 📌 Overview

Principal Component Analysis (PCA) is a **dimensionality reduction** technique used in Machine Learning and Data Science. It transforms a dataset with many features into a smaller set of uncorrelated variables called **Principal Components**, while preserving as much information (variance) as possible.

PCA helps simplify datasets, reduce computational cost, remove redundancy, and improve model performance.

---

## 🎯 Why Use PCA?

- Reduce the number of features
- Remove multicollinearity
- Speed up model training
- Reduce storage requirements
- Improve visualization of high-dimensional data
- Reduce overfitting in some machine learning models

---

## ⚙️ How PCA Works

1. Standardize the dataset.
2. Compute the covariance matrix.
3. Calculate eigenvalues and eigenvectors.
4. Sort eigenvalues in descending order.
5. Select the top **k** principal components.
6. Transform the original data into the new feature space.

---

## 📐 Mathematical Intuition

PCA finds new axes that maximize the variance in the data.

- **Eigenvectors** → Direction of maximum variance
- **Eigenvalues** → Amount of variance explained by each principal component

The first principal component captures the highest variance, the second captures the next highest variance, and so on.

---

## 📚 Types of PCA

### 1. Standard PCA
Used for most machine learning tasks.

### 2. Incremental PCA
Processes data in batches for large datasets.

### 3. Kernel PCA
Used for non-linear dimensionality reduction.

### 4. Sparse PCA
Produces sparse principal components for better interpretability.

---

## 📦 Python Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(X_pca)
```

---

## 📊 Explained Variance Ratio

The explained variance ratio indicates how much information each principal component retains.

```python
print(pca.explained_variance_ratio_)
```

Example Output:

```
[0.72 0.18]
```

This means:

- PC1 explains **72%** of the variance.
- PC2 explains **18%** of the variance.
- Total variance retained = **90%**.

---

## 📈 Choosing the Number of Components

You can retain a desired percentage of variance instead of specifying the number of components.

```python
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
```

This keeps enough principal components to preserve **95%** of the variance.

---

## 📉 Visualizing Explained Variance

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_scaled)

plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(),
         marker='o')

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA")
plt.grid(True)
plt.show()
```

---

## ✅ Advantages

- Reduces dimensionality
- Removes redundant features
- Eliminates multicollinearity
- Faster model training
- Improves visualization
- Can reduce overfitting
- Preserves most important information

---

## ❌ Disadvantages

- Loss of interpretability
- Information loss is inevitable
- Sensitive to feature scaling
- Assumes linear relationships
- Not suitable for every dataset

---

## 📌 Applications

- Image Compression
- Face Recognition
- Data Visualization
- Feature Engineering
- Noise Reduction
- Bioinformatics
- Finance
- Recommendation Systems
- Machine Learning Preprocessing

---

## 📁 Scikit-Learn PCA Parameters

| Parameter | Description |
|-----------|-------------|
| `n_components` | Number of principal components to keep |
| `svd_solver` | Solver used for decomposition |
| `whiten` | Scales principal components to unit variance |
| `random_state` | Ensures reproducibility |

---

## 📊 Example Workflow

```
Raw Dataset
      │
      ▼
StandardScaler
      │
      ▼
Covariance Matrix
      │
      ▼
Eigenvalues & Eigenvectors
      │
      ▼
Select Top Components
      │
      ▼
Transform Dataset
      │
      ▼
Reduced Dataset
      │
      ▼
Machine Learning Model
```

---

## 🛠️ Libraries Used

- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install them using:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 📖 Key Terms

| Term | Meaning |
|------|---------|
| Feature | Original variable |
| Principal Component | New transformed feature |
| Variance | Amount of information |
| Eigenvalue | Variance explained |
| Eigenvector | Direction of maximum variance |
| Explained Variance Ratio | Percentage of variance retained |
| Dimensionality Reduction | Reducing the number of features |

---

## 🚀 Real-World Example

Suppose a dataset contains **100 features**.

After applying PCA:

- Original Features → **100**
- Principal Components → **20**
- Variance Retained → **95%**

The model trains faster while maintaining nearly the same predictive performance.

---

## 📝 Conclusion

Principal Component Analysis (PCA) is one of the most widely used dimensionality reduction techniques in machine learning. By transforming correlated features into a smaller set of orthogonal principal components, PCA simplifies datasets, reduces computational complexity, and often improves model performance while retaining most of the original information.

---

### ⭐ If you found this project helpful, consider giving it a star on GitHub!
