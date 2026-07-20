# Titanic Survival Prediction using Random Forest

## Project Overview

This project applies the **Random Forest Classification** algorithm to predict whether a passenger survived the Titanic disaster based on demographic and travel information.

# Project Workflow

```
      Dataset
         ↓
        EDA
         ↓
    Data Cleaning
         ↓
  Feature Encoding
         ↓
  Train/Test Split
         ↓
Random Forest Classifier
         ↓
   Model Evaluation
```
---

# Dataset

- **Dataset:** `Titanic-Dataset.csv`
- **Instances:** 891 passengers
- **Features:** 12
- **Target Variable:** `Survived`
  - `1` → Survived
  - `0` → Did Not Survive

### Main Features

- Pclass
- Sex
- Age
- Fare
- SibSp
- Parch
- Embarked

---
# Exploratory Data Analysis

Some observations from the dataset:
- Most passengers did not survive.
- Female passengers had a noticeably higher survival rate than males.
- First-class passengers were more likely to survive.
- Younger passengers generally had higher survival rates.
---

# Data Preprocessing

The following preprocessing steps were performed before training the model:
- Removed the **Cabin** column due to a large number of missing values (687).
- Filled missing values in **Age** using the median.
- Filled missing values in **Embarked** using the most frequent value (mode).
- Removed unnecessary columns: (PassengerId, Name, Ticket)
- Applied **One-Hot Encoding** to categorical features.

---

# Model

The classification model was built using **RandomForestClassifier** from Scikit-learn.

```python
RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
```

---

# Model Performance

Since the dataset is slightly imbalanced, model performance is evaluated using **Precision**, **Recall**, and **F1-score** in addition to Accuracy.

| Metric | Score |
|---------|------:|
| Accuracy | **0.83** |
| Precision | **0.80** |
| Recall | **0.77** |
| F1-Score | **0.79** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|---------:|------:|--------:|--------:|
| 0 (Did Not Survive) | 0.84 | 0.87 | 0.85 | 105 |
| 1 (Survived) | 0.80 | 0.77 | 0.79 | 74 |
| **Macro Avg** | **0.82** | **0.82** | **0.82** | **179** |
| **Weighted Avg** | **0.83** | **0.83** | **0.83** | **179** |

---

# Feature Importance

The Random Forest model identified the following features as the most influential for predicting passenger survival.

| Feature | Importance |
|---------|-----------:|
| Fare | 27.47% |
| Sex | 26.93% |
| Age | 25.33% |
| Pclass | 8.35% |
| SibSp | 5.02% |
| Parch | 3.74% |
| Embarked (Southampton) | 2.18% |
| Embarked (Queenstown) | 0.98% |

# Key Insights
- The model achieved balanced classification performance with an **F1-score of 0.79**.
- **Fare**, **Sex**, and **Age** were the strongest predictors of survival.
- Data preprocessing, including handling missing values and encoding categorical variables, played an important role in the model's performance.
The results indicate that **Fare**, **Sex**, and **Age** contributed the most to the model's predictions, while the embarkation port had relatively little influence.
