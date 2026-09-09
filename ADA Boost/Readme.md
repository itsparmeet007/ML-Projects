# AdaBoost (Adaptive Boosting)

## 📌 Introduction

**AdaBoost**, short for **Adaptive Boosting**, is an ensemble machine learning algorithm that combines multiple **weak learners** to create a powerful **strong learner**.

The main idea behind AdaBoost is:

> Learn from previous mistakes by giving more importance to incorrectly classified data points.

---

# 🧠 What is a Weak Learner?

A **weak learner** is a machine learning model that performs only slightly better than random guessing.

For a binary classification problem:

* Random guessing accuracy ≈ **50%**
* A weak learner should perform **better than 50%**

Example:

```text
Accuracy = 60%
```

This model may not be very powerful individually, but AdaBoost combines multiple weak learners to create a **strong learner**.

---

# 🌳 Decision Stump

A **Decision Stump** is a Decision Tree with only **one split**.

Example:

```text
          Age < 30?
         /         \
       Yes          No
       +1           -1
```

The decision stump makes only one decision.

For example:

```text
If Age < 30  → +1
Otherwise    → -1
```

Because Decision Stumps are simple models, they are commonly used as **weak learners in AdaBoost**.

---

# ➕ What does +1 and -1 mean?

In binary classification, AdaBoost traditionally represents classes using:

| Class          | Value |
| -------------- | ----- |
| Positive Class | +1    |
| Negative Class | -1    |

Example:

```text
Spam      → +1
Not Spam  → -1
```

Another example:

```text
Disease     → +1
No Disease  → -1
```

A weak learner predicts:

```text
h(x) = +1
```

or

```text
h(x) = -1
```

---

# ⚙️ How AdaBoost Works

### Step 1: Assign Equal Weights

Initially, all training samples are given equal importance.

```text
Sample 1 → Weight = Equal
Sample 2 → Weight = Equal
Sample 3 → Weight = Equal
Sample 4 → Weight = Equal
```

---

### Step 2: Train a Weak Learner

A simple model, such as a Decision Stump, is trained on the dataset.

```text
Weak Learner 1
```

---

### Step 3: Find Incorrect Predictions

AdaBoost checks which samples were classified incorrectly.

```text
Correct Prediction   → Lower Importance
Wrong Prediction     → Higher Importance
```

---

### Step 4: Update Sample Weights

Misclassified samples receive more weight.

This means that the next weak learner focuses more on the difficult samples.

---

### Step 5: Train Another Weak Learner

The next Decision Stump is trained using the updated weights.

```text
Weak Learner 1 → Makes mistakes

Weak Learner 2 → Focuses more on those mistakes

Weak Learner 3 → Focuses on remaining mistakes
```

---

### Step 6: Combine All Weak Learners

AdaBoost combines all weak learners using weighted voting.

```text
Weak Learner 1  ──┐
Weak Learner 2  ──┤
Weak Learner 3  ──┤──> Strong Learner
Weak Learner 4  ──┘
```

More accurate weak learners receive more importance.

---

# 📐 AdaBoost Formula

The importance of a weak learner is calculated using:

$$
\alpha = \frac{1}{2}\ln\left(\frac{1-\epsilon}{\epsilon}\right)
$$

Where:

* `ε` = Error rate of the weak learner
* `α` = Importance or weight of the weak learner

A lower error generally gives a higher importance to the weak learner.

---

# 🎯 Final Prediction

AdaBoost combines predictions using:

$$
F(x)=\alpha_1h_1(x)+\alpha_2h_2(x)+\alpha_3h_3(x)+\cdots
$$

Where:

* `h(x)` = Prediction of a weak learner (`+1` or `-1`)
* `α` = Importance of that weak learner

The final prediction is:

$$
Prediction = sign(F(x))
$$

### If:

```text
F(x) > 0  → +1
F(x) < 0  → -1
```

---

# 🔍 Example

Suppose three weak learners make predictions:

```text
h₁(x) = +1
h₂(x) = +1
h₃(x) = -1
```

Their weights are:

```text
α₁ = 2
α₂ = 3
α₃ = 1
```

The final score is:

$$
F(x)=2(+1)+3(+1)+1(-1)
$$

$$
F(x)=2+3-1=4
$$

Since:

$$
F(x)>0
$$

The final prediction is:

```text
+1
```

---

# 📊 Advantages

* Combines weak learners to create a strong model.
* Can achieve high classification accuracy.
* Focuses on difficult and misclassified examples.
* Simple Decision Trees can be used as base models.
* Reduces bias.

---

# ⚠️ Disadvantages

* Sensitive to noisy data.
* Sensitive to outliers.
* Sequential training can be slower.
* Incorrectly labelled data may receive too much importance.

---

# 🔑 Key Concepts

```text
Weak Learner
      ↓
Decision Stump
      ↓
Predict +1 or -1
      ↓
Identify Mistakes
      ↓
Increase Weight of Misclassified Samples
      ↓
Train Next Weak Learner
      ↓
Weighted Combination
      ↓
Strong Learner
```

---

# 📝 Summary

> **AdaBoost builds a strong classifier by sequentially combining multiple weak learners. Each new learner focuses more on the mistakes made by previous learners.**

### Important Terms

* **Weak Learner** → A model slightly better than random guessing.
* **Decision Stump** → A Decision Tree with only one split.
* **+1** → Positive class.
* **-1** → Negative class.
* **Sample Weight** → Importance assigned to each training example.
* **Alpha (α)** → Importance of each weak learner.
* **Final Model** → Weighted combination of all weak learners.

