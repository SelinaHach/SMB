# SMB Dynamic Credit Line

Predicts whether a small business's credit line should be adjusted month-over-month — and if so, predicts the new amount.

---

## The Problem

Each month, lenders must decide: does this business's credit line need to change? This project automates that decision with a two-stage ML pipeline:

1. **Classification** — Will the credit line change this month? (Yes / No)
2. **Regression** — If yes, what should the new credit line amount be?

---

## Dataset

`SMB_Port_Dynamic_Line.csv` — 124,436 rows across 6,868 unique businesses, averaging 18 months per business.

---

## Key Findings

### Credit Line Adjustments Are the Norm
**83.9% of all monthly records show a credit line change.** Only 16.1% stay flat — meaning lenders are actively managing these lines far more often than leaving them static.

### Bank Balance Drives Everything
Correlation with `DynamicCreditLine`:

| Feature | Correlation |
|---------|-------------|
| Avg Monthly Bank Balance | **+0.84** |
| Max Monthly Bank Balance | **+0.84** |
| Min Monthly Bank Balance | **+0.83** |
| Annual Revenue | **+0.81** |
| Initial Credit Line | **+0.47** |
| FICO Score | **+0.24** |
| Has Collateral | **+0.22** |

Bank balance (avg, min, max) is the strongest signal by far — stronger than revenue, FICO score, or collateral. Lenders are essentially tracking cash flow consistency, not just credit score.

### Industry Has a 5× Range in Credit Lines
Average credit line by industry:

| Industry | Avg Credit Line |
|----------|----------------|
| Chemical Manufacturing | $84,733 |
| Wholesale Trade | $83,915 |
| Construction | $81,914 |
| Trucking | $81,700 |
| Real Estate & Rental | $76,718 |
| Restaurant | $32,234 |
| Health Care & Social Assistance | $25,523 |
| Educational Service | $17,345 |

Capital-intensive industries get nearly 5× more credit than education and healthcare.

### Risk Profile of the Portfolio
- **22.3%** of businesses have ever had a charge-off — meaningful default risk in this SMB portfolio
- **71.3%** have collateral backing their loan
- **79.6%** have a business checking account

### Data Quality
- Only one column had missing values: `Collateral_Type` — 35,654 missing (28.6%), imputed as `'Unknown'`
- Outliers removed via IQR method: ~25,000 rows total cleaned from the dataset

---

## Model Results

### Stage 1 — Classification (Will the credit line adjust?)

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | **91.5%** | **91.2%** |
| Precision (Adjust) | 0.91 | 0.91 |
| Recall (Adjust) | 0.99 | 0.99 |
| F1 (Adjust) | 0.95 | 0.95 |

The model is very strong at catching real adjustments (99% recall) — it rarely misses a case that should change.

### Stage 2 — Regression (How much should the new credit line be?)

| Metric | Validation | Test |
|--------|-----------|------|
| MAE | $1,183 | $1,176 |
| RMSE | $1,561 | $1,546 |
| R² | **0.9977** | **0.9977** |

R² of 0.9977 means the model explains 99.77% of the variance in credit line amounts — extremely tight predictions on a target that ranges from $1,000 to $150,000.

---

## Most Important Features

**For predicting IF a credit line changes:**

1. Initial Credit Line
2. Avg Monthly Bank Balance
3. Min Monthly Bank Balance
4. Max Monthly Bank Balance
5. Annual Revenue
6. Credit Utilization
7. Debt-to-Income Ratio
8. FICO Score
9. Month-on-Book (MOB)
10. Initial Loan Amount

**For predicting HOW MUCH the new credit line will be:**

1. Avg Monthly Bank Balance *(by far the most important — 59.6% importance)*
2. Initial Credit Line *(23.5%)*
3. Max Monthly Bank Balance
4. Min Monthly Bank Balance
5. Annual Revenue

---

## Pipeline

```
Raw Data (124,436 rows)
   ↓
EDA + Correlation Analysis
   ↓
Outlier Removal — IQR Method → 92,623 rows remaining
   ↓
Impute missing Collateral_Type → 'Unknown'
   ↓
Train / Val / Test Split (70 / 15 / 15)
   ↓
Feature Engineering: create AdjustCreditLine target
   ↓
XGBoost Classifier → predict if credit line changes
   ↓
XGBoost Regressor → predict new credit line amount
   ↓
Export: SMB_Dynamic_Credit_Line_Output.csv
```

---

## Output

Two new columns added to the exported CSV:

| Column | Description |
|--------|-------------|
| `Predicted_AdjustCreditLine` | 0 = no change, 1 = adjustment predicted |
| `Predicted_DynamicCreditLine` | Predicted new credit line amount |

---


**Dependencies:** `pandas` `numpy` `matplotlib` `seaborn` `scikit-learn` `xgboost`
