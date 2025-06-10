# 🛡️ Fraudulent Insurance Claim Detection

This project aims to build a machine learning pipeline to detect potentially fraudulent insurance claims using historical data. It was developed as part of a predictive analytics initiative for **Global Insure**, addressing a critical business challenge of financial loss due to undetected fraud.

---

## 📌 Project Overview

Fraudulent claims account for a significant portion of insurance payouts. This project explores and implements a **binary classification model** to predict whether an incoming insurance claim is likely fraudulent (`fraud_reported = Y`) or legitimate (`fraud_reported = N`).

---

## 🧠 Models Implemented

Two classification models were developed and compared:

- **Logistic Regression**
  - Optimal cutoff: `0.49`
  - Best validation performance across key fraud metrics
- **Random Forest (Tuned)**
  - Performed well on training data but showed overfitting on validation

✅ **Recommended Model:** Logistic Regression (due to better recall and F1 score on unseen data)

---

## 🔍 Key Features Used

- **Categorical Features**: `incident_severity`, `insured_occupation`, `insured_hobbies`, `collision_type`, etc.
- **Numerical Features**: `policy_deductable`, `vehicle_claim`, `injury_claim`, `witnesses`, etc.
- **Date-Derived Features**: `policy_age_at_incident_days`, `incident_month`, `incident_day_of_week`

---

## 📊 Dataset

- **File**: `insurance_claims.csv`
- **Rows**: 1000 records
- **Columns**: 40 original attributes
- **Target Variable**: `fraud_reported` (Y/N)

---

## 🛠️ Tech Stack

- **Language**: Python 3
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `imbalanced-learn`
  - `statsmodels`, `SHAP` (planned for future explainability)

---

## 📈 Evaluation Metrics

| Metric              | Logistic Regression | Random Forest |
|---------------------|---------------------|----------------|
| **Accuracy**        | 82.00%              | 76.67%         |
| **Recall (Fraud)**  | 71.62%              | 37.84%         |
| **Precision (Fraud)**| 61.63%             | 53.85%         |
| **F1 Score (Fraud)**| 0.6625              | 0.4444         |

---

## ✅ Key Insights

- Claims with **"Major Damage" severity** or insureds having hobbies like **“chess”** or **“cross-fit”** had significantly higher fraud likelihood.
- **Fewer witnesses** and **lower deductibles** were also strong fraud indicators.
- **Feature engineering** and **class balancing (RandomOverSampler)** were critical to model performance.

---

## ⚠️ Limitations & Ethical Considerations

- **Dataset size** is limited (1000 rows).
- Certain features (e.g., `insured_hobbies`) show strong correlation but **require ethical scrutiny** before use in production.
- Model performance should be monitored post-deployment for drift.

---

## 🔮 Future Work

- Incorporate **SHAP/LIME** for model explainability.
- Add **cost-sensitive learning** to better penalize false negatives.
- Explore **advanced models** like XGBoost or LightGBM.
- Integrate **claim text analysis** using NLP.

---
