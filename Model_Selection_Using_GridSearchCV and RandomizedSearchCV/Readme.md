# 🔢 Handwritten Digit Classification — Model Selection & Hyperparameter Tuning

Comparing five ML classifiers on the sklearn Digits dataset using GridSearchCV and RandomizedSearchCV to find the best model and optimal hyperparameters.

---

## 📌 Problem Statement

Given 8×8 grayscale images of handwritten digits (0–9), the goal is to:
1. Train and evaluate multiple classifiers
2. Tune each model's hyperparameters systematically
3. Identify the best-performing model using cross-validated accuracy

---

## 📂 Dataset

| Property | Details |
|---|---|
| Source | `sklearn.datasets.load_digits` |
| Samples | 1,797 |
| Features | 64 (8×8 pixel values, flattened) |
| Classes | 10 (digits 0–9) |
| Train / Test Split | 80% / 20% |

---

## 🤖 Models Compared

| Model | Hyperparameters Tuned |
|---|---|
| Random Forest | `n_estimators` |
| Logistic Regression | `C` (regularization strength) |
| Gaussian Naive Bayes | `var_smoothing` |
| Multinomial Naive Bayes | `alpha`, `fit_prior` |
| Decision Tree | `criterion`, `splitter`, `max_depth` |

---

## ⚙️ Methodology

### 1. GridSearchCV
- Exhaustive search over all hyperparameter combinations
- 5-fold cross-validation for each combination
- Best score and best params recorded per model

### 2. RandomizedSearchCV
- Randomly samples hyperparameter combinations
- Same 5-fold CV setup
- Useful comparison: similar accuracy at lower compute cost

### 3. Pipeline (Logistic Regression)
- Built a `StandardScaler → LogisticRegression` pipeline
- Tuned `C` values: `[0.01, 0.1, 1, 5, 10, 50]`
- Prevents data leakage during cross-validation

---

## 📊 Results (GridSearchCV)

| Model | Best CV Accuracy |
|---|---|
| Logistic Regression | **92.2%** ✅ |
| Random Forest | 90.3% |
| Multinomial Naive Bayes | 87.1% |
| Gaussian Naive Bayes | 83.2% |
| Decision Tree | 81.5% |

> **Winner:** Logistic Regression with `C=5`, further boosted to ~97%+ with `StandardScaler` in a Pipeline.

---

<img width="656" height="497" alt="image" src="https://github.com/user-attachments/assets/a3ebbb4e-4332-42ad-bdd5-56563a3e3ca0" />

<img width="627" height="502" alt="image" src="https://github.com/user-attachments/assets/5d40f4e3-2f1e-4f4d-990d-0785012cbf3e" />

<img width="665" height="517" alt="image" src="https://github.com/user-attachments/assets/6601b048-bf2e-47ee-be75-fa29c86c729f" />

<img width="602" height="495" alt="image" src="https://github.com/user-attachments/assets/ba91285f-dec7-4e95-b3b4-19960acfb84c" />

## 🛠️ Tech Stack

- Python 3
- scikit-learn
- pandas
- matplotlib
- seaborn

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/digit-classification-model-selection.git
cd digit-classification-model-selection

# Install dependencies
pip install scikit-learn pandas matplotlib seaborn jupyter

# Launch notebook
jupyter notebook ex16.ipynb
```

---

## 💡 Key Takeaways

- **GridSearchCV vs RandomizedSearchCV**: Both gave nearly identical results here, but RandomizedSearchCV is preferred when the hyperparameter space is large
- **Pipelines matter**: Applying `StandardScaler` inside a Pipeline (instead of before splitting) prevents data leakage and gives a more honest accuracy estimate
- **Logistic Regression wins**: Despite being a "simple" model, LR outperformed tree-based methods on this structured, low-noise dataset

---

## 📁 Project Structure

```
├── ex16.ipynb        # Main notebook with all experiments
└── README.md         # Project documentation
```
