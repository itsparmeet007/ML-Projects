# ⚡ Mini Batch Gradient Descent (MBGD)

> Mini Batch Gradient Descent is an optimization algorithm used to train Machine Learning and Deep Learning models. It updates the model parameters using **small batches of training data**, combining the speed of Stochastic Gradient Descent (SGD) with the stability of Batch Gradient Descent.

---

# 📑 Table of Contents

- [Introduction](#-introduction)
- [What is Mini Batch Gradient Descent?](#-what-is-mini-batch-gradient-descent)
- [Why Use Mini Batch Gradient Descent?](#-why-use-mini-batch-gradient-descent)
- [How It Works](#-how-it-works)
- [Working Process](#-working-process)
- [Algorithm](#-algorithm)
- [Implementation in Python](#-implementation-in-python)
- [Visualization](#-visualization)
- [Advantages](#-advantages)
- [Disadvantages](#-disadvantages)
- [Comparison with Other Gradient Descent Methods](#-comparison-with-other-gradient-descent-methods)
- [Hyperparameters](#-hyperparameters)
- [Applications](#-applications)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

# 📖 Introduction

Gradient Descent is one of the most widely used optimization algorithms for minimizing a model's loss function.

There are three major types of Gradient Descent:

- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)
- Mini Batch Gradient Descent (MBGD)

Mini Batch Gradient Descent strikes a balance between computational efficiency and convergence stability by updating the model using **small subsets (mini-batches)** of the dataset instead of the entire dataset or a single sample.

---

# 📚 What is Mini Batch Gradient Descent?

Mini Batch Gradient Descent divides the training dataset into **small batches** (typically 16, 32, 64, 128, or 256 samples) and updates the model parameters after processing each batch.

Instead of computing gradients on:

- Entire dataset (Batch GD)
- One sample (SGD)

MBGD computes gradients on a **small batch of samples**.

This approach reduces computation time while maintaining smoother convergence.

---

# ❓ Why Use Mini Batch Gradient Descent?

Mini Batch Gradient Descent is preferred because it:

- Trains faster than Batch Gradient Descent
- Produces more stable updates than SGD
- Efficiently utilizes GPU acceleration
- Handles large datasets efficiently
- Reduces memory usage
- Improves convergence speed

---

# ⚙️ How It Works

Suppose we have:

- Dataset Size = **10,000 samples**
- Batch Size = **100**

Then:

- Total Mini Batches = 10,000 / 100 = **100 batches**

The model performs:

1. Forward Pass
2. Loss Calculation
3. Gradient Computation
4. Parameter Update

for each mini-batch.

After all mini-batches are processed, **one epoch** is completed.

---

# 🔄 Working Process

```
Training Dataset
        │
        ▼
Split into Mini Batches
        │
        ▼
Forward Propagation
        │
        ▼
Calculate Loss
        │
        ▼
Backpropagation
        │
        ▼
Update Parameters
        │
        ▼
Next Mini Batch
        │
        ▼
Repeat Until All Batches Complete
        │
        ▼
One Epoch Completed
```

---

# 📝 Algorithm

```
Initialize Parameters

Repeat until convergence:

    Shuffle dataset

    Divide dataset into mini batches

    For each mini batch:

        Compute Predictions

        Calculate Loss

        Compute Gradients

        Update Parameters

Repeat for multiple epochs
```

---

# 🧮 Mathematical Update Rule

For each mini batch:

\[
\theta = \theta - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}\nabla J_i(\theta)
\]

Where:

- **θ** = Model parameters (weights)
- **α** = Learning Rate
- **m** = Batch Size
- **J(θ)** = Cost Function
- **∇J(θ)** = Gradient of the loss function

---

# 💻 Implementation in Python

## Import Libraries

```python
import numpy as np
```

---

## Sample Dataset

```python
X = np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
```

---

## Hyperparameters

```python
learning_rate = 0.01
epochs = 100
batch_size = 32
```

---

## Initialize Parameters

```python
weight = np.random.randn(1)
bias = np.random.randn()
```

---

## Mini Batch Gradient Descent

```python
for epoch in range(epochs):

    indices = np.random.permutation(len(X))

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(0, len(X), batch_size):

        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        y_pred = weight * X_batch + bias

        dw = (-2/len(X_batch)) * np.sum(X_batch * (y_batch - y_pred))
        db = (-2/len(X_batch)) * np.sum(y_batch - y_pred)

        weight -= learning_rate * dw
        bias -= learning_rate * db

print("Weight:", weight)
print("Bias:", bias)
```

---

# 📊 Visualization

```python
import matplotlib.pyplot as plt

plt.scatter(X, y, alpha=0.5)

plt.plot(
    X,
    weight * X + bias,
    color='red',
    linewidth=2
)

plt.title("Mini Batch Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")

plt.show()
```

---

# ⚖️ Comparison with Other Gradient Descent Methods

| Feature | Batch GD | Stochastic GD | Mini Batch GD |
|----------|----------|---------------|----------------|
| Dataset Used | Entire Dataset | One Sample | Small Batch |
| Speed | Slow | Fast | Moderate to Fast |
| Memory Usage | High | Low | Moderate |
| Update Frequency | Once per Epoch | Every Sample | Every Batch |
| Stability | Very Stable | Noisy | Stable |
| GPU Friendly | Limited | Poor | Excellent |
| Large Dataset Support | Poor | Good | Excellent |

---

# ⚙️ Hyperparameters

### Learning Rate

Controls how large each update step is.

Example:

```python
learning_rate = 0.01
```

---

### Batch Size

Common values:

- 16
- 32
- 64
- 128
- 256

Smaller batches:

- Faster updates
- More noisy gradients

Larger batches:

- Stable gradients
- Higher memory consumption

---

### Epochs

Number of complete passes through the training dataset.

Example:

```python
epochs = 100
```

---

# 🌍 Applications

Mini Batch Gradient Descent is widely used in:

- Deep Learning
- Artificial Neural Networks
- Computer Vision
- Natural Language Processing (NLP)
- Recommendation Systems
- Time Series Forecasting
- Image Classification
- Speech Recognition
- Reinforcement Learning

---

# ✅ Advantages

- Faster than Batch Gradient Descent
- More stable than SGD
- Efficient memory utilization
- Works well with large datasets
- Supports GPU acceleration
- Better convergence in practice
- Standard optimization approach for deep learning

---

# ❌ Disadvantages

- Requires selecting an appropriate batch size
- Performance depends on learning rate
- Still introduces some gradient noise
- Can converge to local minima or saddle points
- More hyperparameters to tune

---

# 📂 Project Structure

```
Mini-Batch-Gradient-Descent/
│
├── Mini_Batch_GD.ipynb
├── mini_batch_gd.py
├── requirements.txt
├── README.md
└── images/
```

---

# 📦 Requirements

Install the required libraries:

```bash
pip install numpy
pip install matplotlib
```

Or install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 🚀 Future Improvements

- Learning Rate Scheduling
- Momentum Optimization
- RMSProp
- Adam Optimizer
- Early Stopping
- Mini Batch Training with PyTorch
- TensorFlow/Keras Implementation

---

# 📚 References

- Scikit-Learn Documentation
- TensorFlow Documentation
- PyTorch Documentation
- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurélien Géron

---

# 👨‍💻 Author

**Parmeet Singh**

**B.Tech CSE (Data Science)**

Passionate about:

- Machine Learning
- Data Science
- Artificial Intelligence
- Deep Learning
- MLOps

---

## ⭐ If you found this project helpful, consider giving it a Star!
