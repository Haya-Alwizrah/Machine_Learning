# Medical Insurance Cost Prediction using Random Forest

## Project Overview

This project applies the **Random Forest Regression** algorithm to predict individual medical insurance expenses based on demographic, lifestyle, and health-related information.

---

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
Random Forest Regressor
         ↓
   Model Evaluation
```

---

# Dataset

- **Dataset:** `insurance.csv`
- **Instances:** 1,338
- **Features:** 7
- **Target Variable:** `expenses`

### Main Features

- Age
- Sex
- BMI
- Children
- Smoker
- Region

---

# Exploratory Data Analysis

Some observations from the dataset:

- Medical expenses were highly right-skewed.
- Smokers generally had much higher medical expenses than non-smokers.
- Higher BMI was associated with increased insurance charges.
- Medical expenses tended to increase with age.

---

# Data Preprocessing

The following preprocessing steps were performed before training the model:

- Verified that the dataset contained no missing values.
- Applied **One-Hot Encoding** to categorical variables (`sex`, `smoker`, and `region`).
- Split the dataset into training and testing sets.

---

# Model

The regression model was built using **RandomForestRegressor** from Scikit-learn.

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

---

# Model Performance

| Metric | Score |
|---------|------:|
| R² Score | **0.8813** |
| MAE | **2627.26** |
| RMSE | **4670.16** |

---

# Feature Importance

The Random Forest model identified the most influential features for predicting medical insurance expenses.


---

# Key Insights

- The Random Forest regressor explained approximately **88%** of the variance in medical insurance expenses.
- Smoking status, BMI, and age were expected to be the most influential factors affecting medical costs.
- The model effectively captured the complex, non-linear relationships between patient characteristics and insurance expenses.
