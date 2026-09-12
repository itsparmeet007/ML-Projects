# Gradient Boosting

A hands-on project to understand and implement **Gradient Boosting Regression** using Python and Scikit-learn.

## 📌 Overview

This project focuses on understanding how Gradient Boosting works step by step rather than simply using the built-in model.

The notebook covers:

- Initial prediction using the mean of the target
- Calculating residuals
- Training weak Decision Tree learners
- Updating predictions using a learning rate
- Repeating the process to reduce residual errors
- Using Scikit-learn's `GradientBoostingRegressor`
- Evaluating the trained model
- Saving and loading the trained model
- Making predictions on new data

## 🧠 Gradient Boosting

Gradient Boosting is an ensemble learning technique in which weak learners are built **sequentially**.

Each new learner tries to correct the errors made by the previous learners.

```text
Initial Prediction
       ↓
Calculate Residuals
       ↓
Train Weak Learner
       ↓
Update Prediction
       ↓
Calculate New Residuals
       ↓
Repeat
```

## 📊 Dataset

A small dataset containing:

- R&D Spend
- Administration
- Marketing Spend
- Profit

was used for learning. The dataset is intentionally small so that the Gradient Boosting calculations can be understood manually.

## 🔍 Manual Gradient Boosting

The project first explores the algorithm step by step.

The initial prediction is calculated as the mean of `Profit`.

Residuals are calculated as:

```text
Residual = Actual Value - Predicted Value
```

A Decision Tree with `max_depth=1` is used as a weak learner to predict the residuals.

The prediction is updated using:

```text
New Prediction = Previous Prediction + Learning Rate × Weak Learner Prediction
```

With repeated iterations, the residuals become progressively smaller.

## 🤖 Scikit-learn Implementation

After understanding the manual process, `GradientBoostingRegressor` is used:

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=1,
    random_state=42
)

gbr.fit(X, y)

y_pred = gbr.predict(X)
```

## 📈 Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Training results:

```text
MAE  : 0.6641
MSE  : 0.6294
RMSE : 0.7934
R²   : 0.9996
```

> **Note:** These metrics are training results from a very small dataset and should not be interpreted as real-world model performance.

## 💾 Saving the Model

The trained model is saved using `joblib`:

```python
import joblib

joblib.dump(gbr, "gradient_boosting_model.pkl")
```

This creates:

```text
gradient_boosting_model.pkl
```

## 🔄 Loading the Model

The saved model can be loaded without retraining:

```python
loaded_model = joblib.load("gradient_boosting_model.pkl")
```

## 🧪 Testing on New Data

The saved model was tested using new input samples:

```python
X_new = [
    [120, 100, 300],
    [150, 80, 400],
    [50, 110, 180]
]

predictions = loaded_model.predict(X_new)

for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}: Predicted Profit = {prediction:.2f}")
```

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook / Kaggle

## 📁 Project Structure

```text
Gradient-Boosting/
│
├── Gradient_Boosting.ipynb
├── gradient_boosting_model.pkl
└── README.md
```

## 🎯 Learning Outcomes

Through this project, I learned:

1. How Gradient Boosting works conceptually.
2. How the initial prediction is calculated.
3. How residuals are generated.
4. How weak Decision Trees learn residuals.
5. How the learning rate controls each correction.
6. How predictions are updated sequentially.
7. How `n_estimators` affects the model.
8. How to train a Gradient Boosting Regressor using Scikit-learn.
9. How to evaluate a regression model.
10. How to save, load, and reuse a trained ML model.

## 🚀 Future Improvements

- Use a larger real-world dataset
- Perform proper train/test splitting
- Tune `n_estimators`
- Tune `learning_rate`
- Experiment with different tree depths
- Compare Gradient Boosting with Random Forest
- Explore XGBoost, LightGBM, and CatBoost

## 📚 Conclusion

This project helped me understand Gradient Boosting from both the **algorithmic** and **practical Scikit-learn** perspectives.

The key idea is:

> **Gradient Boosting builds weak learners sequentially, with each new learner trying to correct the errors made by the previous learners.**
