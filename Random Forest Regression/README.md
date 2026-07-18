# Regression Random Forest:

## Project Overview:
This project applies **Random Forest Regression** to predict continuous individual medical insurance expenses based on patient demographics, lifestyle choices, and physiological indicators. By leveraging an ensemble of decision trees, the model effectively captures complex, non-linear relationships—most notably the compounding financial impact of smoking combined with a high Body Mass Index (BMI)—without requiring manual interaction terms.

---
## Dataset Specifications
* **Dataset File:** `insurance.csv`
* **Total Instances (Rows):** 1,338 beneficiaries
* **Total Features (Columns):** 7 variables
* **Target Variable:** `expenses` (Continuous individual medical costs billed by insurance in USD)
* **Primary Predictors:**
  * `age`: Age of the primary beneficiary (years)
  * `sex`: Gender of contractor (`male` / `female`)
  * `bmi`: Body Mass Index, providing an understanding of body weight relative to height (kg/m²)
  * `children`: Number of children/dependents covered by the health insurance plan
  * `smoker`: Smoking status (`yes` / `no`)
  * `region`: Beneficiary's residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`)

---

## Data Preprocessing:
During the Exploratory Data Analysis (EDA) and data preparation phase, the dataset underwent strict verification and transformation:
* **No missing Values found.**
* **Categorical Encoding:**
  * Binary variables (`sex`, `smoker`) were transformed using One-Hot Encoding (`smoker_yes` = 1, `no` = 0).
  * The multi-class nominal variable (`region`) was One-Hot Encoded into dummy variables to prevent the regression trees from assuming an artificial numerical hierarchy.
* **Target Distribution Handling:** The `expenses` column displays a right-skewed distribution driven by high-cost medical treatments. Random Forest handles this non-parametric skewness robustly without necessitating a log-transformation of the target variable.

---

## Model Training & Hyperparameter Tuning
The regression engine is implemented using `scikit-learn`'s `RandomForestRegressor`. Key hyperparameters optimized during model development include:
* `n_estimators`: Set to `100` decision trees to stabilize variance and ensure smooth prediction averages across feature sub-spaces.
* `max_depth`: Tuned to prevent individual trees from memorizing localized noise or extreme expense outliers.
* `min_samples_leaf`: Set to maintain sufficient sample depth at leaf nodes for reliable cost estimation.
* `random_state`: Fixed at `42` to guarantee exact reproducibility across identical training splits.

---

## Model Evaluation & Results

### Regression Performance Metrics
| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **R² Score (Coefficient of Determination)** | `0.8655` | The Random Forest model explains **86.6%** of the variance in individual medical expenses. |
| **RMSE (Root Mean Squared Error)** | `4,569.89` | The average prediction deviation from actual billed insurance charges, penalizing larger errors. |
| **MAE (Mean Absolute Error)** | `2,545.64` | The median absolute error across all patient cost predictions. |
| **MAPE (Mean Absolute Percentage Error)** | `32.20%` | Relative percentage error across varied cost brackets. |

### Key Takeaways & Feature Importance
1. **Dominant Cost Drivers:** Feature importance analysis revealed that **Smoking Status (`smoker`)** is by far the single most influential driver of medical costs (~60.9% importance), followed by **BMI** (~21.5%) and **Age** (~13.5%). Together, these three attributes account for over **95%** of the model's predictive capacity.
2. **Non-Linear Interactions:** The ensemble naturally captured non-linear risk factors; for example, smokers with a BMI > 30 experience a dramatic, exponential surge in medical expenses compared to non-smokers, an interaction the Random Forest identified automatically.
