# Logistic Regression [Perceptron Trick]

This notebook demonstrates the **Perceptron Trick** for binary classification and compares its decision boundary with the decision boundary produced by **Logistic Regression**.

The implementation starts from scratch using NumPy and then compares the result with Scikit-learn's `LogisticRegression`.

---

## 📌 Topics Covered

* Binary Classification
* Perceptron Algorithm
* Perceptron Trick
* Step Function
* Dot Product
* Weight Updates
* Decision Boundary
* Logistic Regression
* Perceptron vs Logistic Regression
* Effect of `class_sep`

---

## 📊 Dataset

The dataset is generated using Scikit-learn's `make_classification()` function.

```python
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=10
)
```

### Dataset Configuration

| Parameter            | Value |
| -------------------- | ----: |
| Samples              |   100 |
| Features             |     2 |
| Informative Features |     1 |
| Classes              |     2 |
| Clusters per Class   |     1 |
| Class Separation     |    10 |

A second experiment is also performed with:

```python
class_sep = 20
```

This increases the separation between the two classes.

---

# 🧠 Perceptron Trick

The Perceptron is a simple binary classification algorithm.

The basic idea is:

1. Calculate the weighted sum of the input features.
2. Apply a step function.
3. Compare the predicted output with the actual output.
4. Update the weights if the prediction is incorrect.
5. Repeat the process.

---

## Step Function

The notebook uses the following step function:

```python
def step(z):
    return 1 if z > 0 else 0
```

The logic is:

* If the dot product is greater than `0` → prediction = `1`
* Otherwise → prediction = `0`

Mathematically:

[
\hat{y} =
\begin{cases}
1 & \text{if } z > 0 \
0 & \text{otherwise}
\end{cases}
]

---

# ⚙️ Perceptron Implementation

The Perceptron is implemented from scratch using NumPy.

```python
def perceptron(X, y):

    X = np.insert(X, 0, 1, axis=1)

    weights = np.ones(X.shape[1])

    lr = 0.1

    for i in range(1000):

        j = np.random.randint(0, 100)

        y_hat = step(np.dot(X[j], weights))

        weights = weights + lr * (y[j] - y_hat) * X[j]

    return weights[0], weights[1:]
```

### Important Components

#### Bias Term

A column containing `1` is inserted into the dataset:

```python
X = np.insert(X, 0, 1, axis=1)
```

This allows the model to learn an intercept/bias.

#### Initial Weights

```python
weights = np.ones(X.shape[1])
```

The weights are initialized to `1`.

#### Learning Rate

```python
lr = 0.1
```

The learning rate controls how much the weights are changed during each update.

#### Prediction

```python
y_hat = step(np.dot(X[j], weights))
```

The weighted sum is calculated using the dot product and passed through the step function.

---

# 🔄 Weight Update Rule

The Perceptron updates its weights using:

```python
weights = weights + lr * (y[j] - y_hat) * X[j]
```

The update can be understood as:

[
w = w + \eta(y-\hat{y})x
]

where:

* (w) = weights
* (\eta) = learning rate
* (y) = actual output
* (\hat{y}) = predicted output
* (x) = input features

If the prediction is correct, the error becomes zero and the weights remain unchanged.

---

# 📐 Decision Boundary

For two features, the decision boundary can be represented as:

[
w_0+w_1x_1+w_2x_2=0
]

Rearranging:

[
x_2=-\frac{w_1}{w_2}x_1-\frac{w_0}{w_2}
]

In the notebook:

```python
m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])
```

Then the decision boundary is generated using:

```python
y_input = m * x_input + b
```

---

# 🤖 Logistic Regression

After implementing the Perceptron from scratch, Scikit-learn's Logistic Regression is trained on the same dataset.

```python
from sklearn.linear_model import LogisticRegression

lor = LogisticRegression()

lor.fit(X, y)
```

The Logistic Regression decision boundary is calculated from the learned coefficients:

```python
m = -(lor.coef_[0][0] / lor.coef_[0][1])

b = -(lor.intercept_[0] / lor.coef_[0][1])
```

---

# ⚔️ Logistic Regression vs Perceptron

The notebook plots both decision boundaries on the same dataset.

```python
plt.plot(
    x_input,
    y_input,
    color='red',
    linewidth=3,
    label='Perceptron'
)

plt.plot(
    x_input1,
    y_input1,
    color='blue',
    linewidth=3,
    label='Logistic Regression'
)
```

The final visualization compares:

* **Red line** → Perceptron
* **Blue line** → Logistic Regression
* Scatter points → Dataset observations

### Key Difference

| Perceptron                                            | Logistic Regression                             |
| ----------------------------------------------------- | ----------------------------------------------- |
| Uses a step function                                  | Uses the sigmoid function                       |
| Produces hard binary predictions                      | Produces probabilities                          |
| Simple classification algorithm                       | Probabilistic classification algorithm          |
| Updates weights based on misclassification            | Optimizes a loss function                       |
| Can be implemented using the Perceptron learning rule | Commonly optimized using gradient-based methods |

---

# 📈 Effect of `class_sep`

The notebook also performs a second experiment by changing:

```python
class_sep = 20
```

The first experiment uses:

```python
class_sep = 10
```

Increasing `class_sep` makes the classes more separated in the generated dataset.

This experiment helps visualize how the decision boundaries behave when the classes become more clearly separated.

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# 📂 Notebook Structure

The notebook follows this general workflow:

```text
Generate Dataset
      ↓
Visualize Dataset
      ↓
Create Step Function
      ↓
Implement Perceptron from Scratch
      ↓
Calculate Perceptron Decision Boundary
      ↓
Train Logistic Regression
      ↓
Calculate Logistic Regression Boundary
      ↓
Compare Both Boundaries
      ↓
Experiment with class_sep = 20
```

---

# 🎯 Learning Objectives

By completing this notebook, you can understand:

* How a Perceptron performs binary classification.
* How the Perceptron updates its weights.
* How a step function converts a score into a class.
* How a decision boundary is derived from model coefficients.
* How Logistic Regression can be used for binary classification.
* How Perceptron and Logistic Regression decision boundaries can be compared visually.
* How changing `class_sep` affects a classification dataset.

---

# 🚀 Conclusion

This project provides an intuitive introduction to **binary classification** by implementing the Perceptron learning rule from scratch and comparing it with **Logistic Regression**.

The Perceptron demonstrates the fundamental idea of learning a separating hyperplane through weight updates, while Logistic Regression provides a probabilistic approach to classification.

The visual comparison makes it easier to understand how different classification algorithms can produce decision boundaries for the same dataset.

