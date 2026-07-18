# Random Forest Algorithm

This repository demonstrates the use of the **Random Forest** algorithm for two common supervised machine learning tasks:

- **Classification:** Predict passenger survival using the **Titanic Dataset**.
- **Regression:** Predict medical insurance expenses using the **Medical Insurance Dataset**.

Each project follows a complete machine learning workflow, including:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model implementation
- Model evaluation
- Feature importance analysis

## Repository Structure

```
Random Forest
│
├── Random Forest Classification
│   ├── Mini_Project_Classification.ipynb
│   ├── Titanic-Dataset.csv
│   └── README.md
│
└── Random Forest Regression
    ├── regression_mini_project.ipynb
    ├── insurance.csv
    └── README.md
```
---

# What is Random Forest?

**Random Forest** is a **Supervised Machine Learning** algorithm that belongs to the **Ensemble Learning** family.

Instead of relying on a single Decision Tree, Random Forest builds many decision trees and combines their predictions to produce a more accurate and stable result.

- **Classification:** predicts a category (e.g., Yes/No, Survived/Not Survived).
- **Regression:** predicts a numerical value (e.g., House Price, Insurance Cost).

---

# What is Ensemble Learning?

**Ensemble Learning** is a machine learning technique that combines multiple models to produce a better prediction than a single model.

The main idea is often called the **"Wisdom of the Crowd"**:
> Many weak learners working together usually perform better than a single learner.
> In Random Forest, the weak learners are individual Decision Trees that work together to produce a stronger final model.

Random Forest is an example of **Bagging (Bootstrap Aggregating)**, one of the most popular Ensemble Learning methods.

---

# How Random Forest Works

### 1. Bootstrap Sampling (Bagging)

Random Forest creates multiple random training datasets by sampling the original dataset **with replacement**.

Each dataset is then used to train a separate Decision Tree.

---

### 2. Random Feature Selection

When building each tree, Random Forest does **not** consider every feature at each split.

Instead, it randomly selects a subset of features, making every tree different and reducing overfitting.

---

### 3. Build Multiple Decision Trees

Each tree learns independently using its own random data and random feature selection.

Because every tree is slightly different, they make different predictions.

---

### 4. Final Prediction

The predictions from all trees are combined:

- **Classification:** Majority Vote
- **Regression:** Average Prediction

This makes the final prediction more reliable than relying on a single Decision Tree.

---

### 5. Better Generalization

Since every tree is trained on different data and different feature subsets, Random Forest reduces overfitting and usually generalizes better to unseen data than a single Decision Tree.

---

# Advantages
- High prediction accuracy.
- Reduces overfitting compared to a single Decision Tree.
- Works for both classification and regression tasks.
- Handles noisy data and outliers effectively.
- Provides built-in feature importance.
- Requires little data preprocessing.
- Does not require feature scaling.

---

# Limitations
- Slower to train than a single Decision Tree.
- Requires more memory because it stores many trees.
- Less interpretable than an individual Decision Tree.
- Prediction can be slower when using a large number of trees.

---

# Classification vs Regression

| Criteria | Random Forest Classifier | Random Forest Regressor |
|-----------|-------------------------|-------------------------|
| Target | Categorical | Continuous |
| Output | Class Label | Numerical Value |
| Final Prediction | Majority Vote | Average Prediction |
| Common Metrics | Accuracy, Precision, Recall, F1-score | R², MAE, RMSE |

