# One Hot Encoding in Machine Learning

This project demonstrates the implementation of **One Hot Encoding**, a commonly used feature engineering technique in Machine Learning for handling categorical data.

The notebook explains how categorical variables are transformed into numerical format using One Hot Encoding techniques with Python and Scikit-learn.

---

# Project Overview

Machine Learning algorithms generally work with numerical data.  
Categorical features such as:

- Gender
- City
- Country
- Color
- Education

must be converted into numerical representations before training ML models.

This project demonstrates:

- What One Hot Encoding is
- Why it is important
- How to implement it using Pandas
- How to implement it using Scikit-learn
- Avoiding Dummy Variable Trap
- Real dataset preprocessing workflow

---

# What is One Hot Encoding?

One Hot Encoding converts categorical values into binary vectors.

Example:

| Color | Red | Blue | Green |
|------|------|------|------|
| Red | 1 | 0 | 0 |
| Blue | 0 | 1 | 0 |
| Green | 0 | 0 | 1 |

Each category becomes a separate feature column.

---

# Why Use One Hot Encoding?

One Hot Encoding helps:

- Convert categorical data into numerical format
- Improve ML model compatibility
- Prevent incorrect ordinal relationships
- Prepare datasets for training

---

# Techniques Covered

## 1. Pandas One Hot Encoding

Using:

```python
pd.get_dummies()
```

---

## 2. Scikit-learn One Hot Encoder

Using:

```python
OneHotEncoder()
```

from Scikit-learn preprocessing module.

---

## 3. Dummy Variable Trap

The notebook also explains:

- Multicollinearity
- Redundant features
- Dropping one encoded column

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- VS Code

---

# Project Structure

```bash
One_Hot_Encoding/
│
├── OneHotEncoder.ipynb
├── README.md
├── requirements.txt
└── dataset.csv
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/itsparmeet007/ML-Projects.git
```

Move into the project directory:

```bash
cd ML-Projects/One_Hot_Encoding
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Notebook

Open Jupyter Notebook or VS Code and run:

```bash
OneHotEncoder.ipynb
```

---

# Learning Outcomes

After completing this project, you will understand:

- Handling categorical variables
- Feature engineering basics
- One Hot Encoding implementation
- Dummy variable trap
- Preprocessing workflow in ML pipelines

---

# Author

Parmeet Singh

GitHub:
https://github.com/itsparmeet007
