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

The Random Forest model identified the following features as the most influential for predicting medical insurance expenses.

| Feature | Importance |
|---------|-----------:|
| Smoker | 59.93% |
| BMI | 21.43% |
| Age | 13.77% |
| Children | 2.33% |
| Sex | 0.73% |
| Region (Southeast) | 0.67% |
| Region (Northwest) | 0.66% |
| Region (Southwest) | 0.48% |

The results indicate that **Smoking Status**, **BMI**, and **Age** contributed the most to the model's predictions, while **Sex** and **Region** had relatively little influence.


---

# Key Insights

- The Random Forest regressor explained approximately **88%** of the variance in medical insurance expenses (**R² = 0.8813**).
- **Smoking status** was by far the strongest predictor, accounting for nearly **60%** of the model's total feature importance.
- **BMI** and **Age** also had a significant impact on insurance costs, while **Sex** and **Region** contributed relatively little to the predictions.
- The model effectively captured the complex relationships between patient characteristics and medical expenses, resulting in low prediction errors (**MAE = 2627.26**, **RMSE = 4670.16**).
