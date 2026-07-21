# 📉 Stochastic Gradient Descent (SGD)

> A step-by-step implementation and understanding of **Stochastic Gradient Descent (SGD)** from scratch using Python. This project demonstrates how machine learning models learn by updating parameters after every training example instead of using the complete dataset.

---

## 📌 Overview

**Stochastic Gradient Descent (SGD)** is an optimization algorithm used to minimize the cost (loss) function in Machine Learning and Deep Learning.

Unlike **Batch Gradient Descent**, which updates model parameters after processing the entire dataset, **SGD updates the parameters after every single training example**, making it significantly faster for large datasets.

This notebook covers:

- Understanding the intuition behind SGD
- Mathematical formulation
- Algorithm implementation from scratch
- Model training using SGD
- Visualization of parameter updates
- Comparison with Batch Gradient Descent

---

## 📂 Project Structure

```
Stochastic-Gradient-Descent/
│
├── Stochastic_Gradient_Descent.ipynb
├── README.md
└── images/
```

---

## 🚀 What is Stochastic Gradient Descent?

Stochastic Gradient Descent is an iterative optimization algorithm that updates the model parameters using **one randomly selected training sample at a time**.

Instead of computing gradients using the entire dataset,

- Select one random data point
- Compute prediction
- Calculate loss
- Compute gradient
- Update weights immediately

This process repeats until convergence.

---

## ⚙️ Algorithm

1. Initialize weights randomly.
2. Shuffle the training data.
3. Pick one random training example.
4. Compute prediction.
5. Calculate error.
6. Compute gradients.
7. Update weights.
8. Repeat for all samples (one epoch).
9. Continue for multiple epochs until convergence.

---

## 📐 Mathematical Formulation

For Linear Regression,

Prediction:

\[
\hat{y}=mx+b
\]

Loss Function (Mean Squared Error):

\[
J(m,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
\]

Weight Update:

\[
m = m - \alpha \frac{\partial J}{\partial m}
\]

\[
b = b - \alpha \frac{\partial J}{\partial b}
\]

Where:

- **α** = Learning Rate
- **m** = Slope (Weight)
- **b** = Bias

---

## 💻 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn (for comparison)

---

## 📊 Workflow

```
Dataset
   │
   ▼
Initialize Parameters
   │
   ▼
Randomly Select One Sample
   │
   ▼
Forward Pass
   │
   ▼
Calculate Error
   │
   ▼
Compute Gradient
   │
   ▼
Update Parameters
   │
   ▼
Repeat Until Convergence
```

---

## 📈 Advantages

✅ Much faster than Batch Gradient Descent

✅ Suitable for very large datasets

✅ Requires less memory

✅ Can escape local minima due to noisy updates

✅ Works well for online learning

---

## ❌ Disadvantages

- Noisy parameter updates
- Loss function fluctuates
- May take longer to converge
- Sensitive to learning rate
- Requires data shuffling

---

## 🔄 Batch GD vs Stochastic GD

| Feature | Batch Gradient Descent | Stochastic Gradient Descent |
|----------|-----------------------|-----------------------------|
| Gradient Calculation | Entire Dataset | One Sample |
| Speed | Slower | Faster |
| Memory Usage | High | Low |
| Updates | One per Epoch | One per Sample |
| Convergence | Smooth | Noisy |
| Large Dataset | Less Efficient | Highly Efficient |

---

## 📉 Learning Behavior

During training:

- The loss decreases over time.
- Parameters move toward the optimal solution.
- Updates fluctuate because only one sample is used each step.
- Eventually, the model converges near the global minimum.

---

## 📚 Applications

Stochastic Gradient Descent is widely used in:

- Linear Regression
- Logistic Regression
- Neural Networks
- Deep Learning
- Recommendation Systems
- Natural Language Processing (NLP)
- Computer Vision

---

## 🎯 Key Hyperparameters

| Hyperparameter | Description |
|---------------|-------------|
| Learning Rate | Controls update size |
| Epochs | Number of complete passes through the dataset |
| Batch Size | Equals 1 in SGD |
| Shuffle | Randomizes training data each epoch |

---

## 📸 Sample Output

- Model learns optimal parameters iteratively.
- Loss decreases across epochs.
- Parameter updates show stochastic behavior.
- Visualization illustrates convergence despite noisy updates.

---

## 🧠 Key Learning Outcomes

By completing this project, you will understand:

- Gradient Descent optimization
- Why SGD is faster than Batch GD
- Parameter update mechanism
- Effect of learning rate
- Convergence behavior
- Trade-offs between speed and stability

---

## 🔮 Future Improvements

- Mini-Batch Gradient Descent
- Momentum SGD
- Nesterov Accelerated Gradient (NAG)
- AdaGrad
- RMSProp
- Adam Optimizer
- Learning Rate Scheduling

---

## 👨‍💻 Author

**Parmeet Singh**

Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ If you found this project helpful, consider giving it a star!
