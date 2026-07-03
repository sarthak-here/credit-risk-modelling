# Credit Risk Modelling — CredIQ Finance

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-4CAF50)
![Optuna](https://img.shields.io/badge/Optuna-3.x-6C63FF)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end credit risk prediction system for **CredIQ Finance** that estimates the probability of loan default, maps it to a credit score (300–900), and assigns a risk rating. Built on 50,000 customer records across three data sources, the final Logistic Regression model achieves **AUC = 0.98** and **Gini = 0.96**.

---

## Problem Statement

Lending institutions need to assess creditworthiness before approving loans. Manually reviewing applications is slow and inconsistent. This system automates the risk assessment pipeline — given a borrower's financial profile, it predicts the likelihood of default and outputs a structured credit score with a rating.

---

## Data Sources

Three relational tables merged on `cust_id`:

| Table | Rows | Columns | Description |
|---|---|---|---|
| `customers.csv` | 50,000 | 12 | Demographics: age, gender, income, employment, residence type |
| `loans.csv` | 50,000 | 15 | Loan details: amount, tenure, purpose, type, disbursal info, default flag |
| `bureau_data.csv` | 50,000 | 8 | Credit bureau: open/closed accounts, DPD, delinquency, credit utilization |

**Target**: `default` — binary (0 = repaid, 1 = defaulted)
**Class imbalance**: ~8.6% default rate

---

## ML Pipeline

### 1. Train-Test Split (Before EDA)
Stratified 75/25 split performed **before** any EDA to prevent data leakage.

### 2. Data Cleaning
- Imputed `residence_type` nulls with mode
- Removed processing fee outliers (fee > 3% of loan amount — business rule violation)
- Fixed typo: `Personaal` → `Personal` in loan_purpose
- Validated GST ≤ 20% and net disbursement ≤ loan amount

### 3. Feature Engineering

| Feature | Formula | Insight |
|---|---|---|
| `loan_to_income` | `loan_amount / income` | Higher LTI → higher default risk |
| `delinquency_ratio` | `delinquent_months / total_loan_months × 100` | % of loan life spent delinquent |
| `avg_dpd_per_delinquency` | `total_dpd / delinquent_months` (0 if none) | Severity of late payments |

### 4. Feature Selection

**VIF Analysis** — Removed multicollinear features:
`sanction_amount`, `processing_fee`, `gst`, `net_disbursement`, `principal_outstanding`

**Weight of Evidence (WOE) / Information Value (IV)** — Retained only features with IV > 0.02, removing low-signal columns like `city`, `state`, `gender`, `marital_status`, `employment_status`.

### 5. Encoding
One-hot encoding (`drop_first=True`) on remaining categorical features: `residence_type`, `loan_purpose`, `loan_type`.

### 6. Modeling — 4 Attempts

| Attempt | Model | Imbalance Handling | Tuning |
|---|---|---|---|
| 1 | LR / RF / XGBoost | None | None |
| 2 | LR + XGBoost | RandomUnderSampler | RandomizedSearchCV |
| 3 | Logistic Regression | SMOTETomek | Optuna (50 trials) |
| 4 | XGBoost | SMOTETomek | Optuna (50 trials) |

SMOTETomek combines oversampling of minority class with Tomek link removal to clean decision boundaries.

**Final model: Logistic Regression (Attempt 3)** — chosen for interpretability while matching XGBoost performance.

---

## Results

| Metric | Value |
|---|---|
| AUC-ROC | **0.98** |
| Gini Coefficient | **0.96** |
| KS Statistic | Strong rank ordering across all deciles |

**Rank ordering** confirmed — deciles with highest predicted default probability consistently show the highest actual default rates.

---

## Credit Score System

The model's default probability is mapped to a **300–900 credit score**:

```
credit_score = 300 + (1 - default_probability) × 600
```

| Score Range | Rating |
|---|---|
| 750 – 900 | Excellent |
| 650 – 749 | Good |
| 500 – 649 | Average |
| 300 – 499 | Poor |

---

## Streamlit App

```bash
cd app
streamlit run main.py
```

**Inputs**: Age, Income, Loan Amount, Loan Tenure, Avg DPD, Delinquency Ratio, Credit Utilization, Open Accounts, Residence Type, Loan Purpose, Loan Type

**Outputs**: Default Probability, Credit Score, Rating

The app auto-calculates the Loan-to-Income Ratio from the provided income and loan amount.

---

## Project Structure

```
├── credit_risk_model.ipynb     # Full pipeline: EDA → feature engineering → modeling → evaluation
├── dataset/
│   ├── customers.csv           # Customer demographics (50k)
│   ├── loans.csv               # Loan details and default label (50k)
│   └── bureau_data.csv         # Credit bureau data (50k)
├── artifacts/
│   └── model_data.joblib       # Serialized model, scaler, and feature metadata
├── app/
│   ├── main.py                 # Streamlit app
│   ├── prediction_helper.py    # Inference: preprocessing + credit score calculation
│   └── artifacts/
│       └── model_data.joblib   # Model artifacts used by app
```

---

## Setup

```bash
git clone https://github.com/sarthak-here/credit-risk-modelling.git
cd credit-risk-modelling
pip install -r requirements.txt

# Run the Streamlit app
cd app
streamlit run main.py
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| Hyperparameter Tuning | Optuna |
| Feature Selection | WOE/IV, VIF (statsmodels) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Serialization | Joblib |
