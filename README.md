# SMB Dynamic Credit Line

A machine learning pipeline that predicts whether a small business's credit line will be adjusted month-over-month — and if so, predicts the new credit line amount.

---

## The Problem

Every month, lenders must decide: should this business's credit line change? Too low and you limit a good customer. Too high and you take on unnecessary risk. This project automates that decision using two models working together:

- **Stage 1 — Classification**: Will the credit line change this month?
- **Stage 2 — Regression**: If yes, what should the new amount be?

---

## Dataset

**File:** `SMB_Port_Dynamic_Line.csv`

| Stat | Value |
|------|-------|
| Total rows | 124,436 |
| Unique businesses | 6,868 |
| Avg months per business | ~18 months |
| Total features | 25 |

Each row represents one business in one month (Month-on-Book = MOB). Features cover business profile, financial health, and credit behavior.

---

## Key Findings from EDA

### Credit lines span a wide range
- Average: **$38,285**
- Median: **$37,500**
- Range: **$1,000 → $111,000**
- 50% of businesses fall between $18,500 and $55,500

### Bank balance is the #1 predictor — not FICO
Pearson correlation with `DynamicCreditLine`:

| Feature | Correlation |
|---------|-------------|
| Avg Monthly Bank Balance | **+0.845** |
| Max Monthly Bank Balance | **+0.838** |
| Min Monthly Bank Balance | **+0.833** |
| Annual Revenue | **+0.807** |
| Initial Credit Line | **+0.470** |
| Has Biz Checking Account | **+0.240** |
| FICO Score | **+0.239** |
| Has Collateral | **+0.219** |
| Num Vendor Tradelines | **+0.204** |
| Credit Utilization | **-0.366** |
| Debt-to-Income Ratio | **-0.311** |
| Ever Had Charge-Off | **-0.203** |

Cash flow consistency is overwhelmingly the strongest signal — stronger than FICO score, collateral, or years in business.

### Industry drives a 3× gap in credit lines

| Industry | Avg Credit Line |
|----------|----------------|
| Wholesale Trade | $55,084 |
| Trucking | $53,554 |
| Chemical Manufacturing | $52,476 |
| Construction | $51,713 |
| Real Estate & Rental | $51,230 |
| Finance & Insurance | $41,839 |
| Restaurant | $33,704 |
| Health Care & Social Assistance | $27,102 |
| Educational Service | **$18,394** ← lowest |

### Risk signals in the portfolio
- **22.3%** of businesses have ever had a charge-off
- **71.3%** have collateral
- **79.6%** have a business checking account

---

## Data Cleaning

### Missing Values
Only one column had missing values: `Collateral_Type` — 35,654 missing (28.6% of rows). Imputed with `'Unknown'`.

### Outlier Removal — IQR Method
Removed rows where values fell below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR`.

| Column | Rows Removed |
|--------|-------------|
| Current_Avg_Days_Beyond_Terms | 11,521 |
| Current_Avg_Monthly_Bank_Balance | 7,324 |
| Current_Annual_Revenue | 4,607 |
| Current_Min_Monthly_Bank_Balance | 4,283 |
| Current_Max_Monthly_Bank_Balance | 1,833 |
| Current_Debt_To_Income_Ratio_SB | 1,626 |
| Current_Biz_Credit_Utilization | 249 |
| Num_Vendor_Tradelines | 156 |
| Current_Num_NSF_Last_12M | 108 |
| DynamicCreditLine | 106 |
| **Total removed** | **31,813 rows** |

**124,436 → 92,623 rows** after cleaning.

---

## Train / Val / Test Split

| Split | Rows | Share |
|-------|------|-------|
| Train | 64,836 | 70% |
| Validation | 13,893 | 15% |
| Test | 13,894 | 15% |

---

## Feature Engineering

Created `AdjustCreditLine` as the classification target:
- **1** = credit line changed from previous month
- **0** = stayed the same

`Industry_Type` was label-encoded into `industry_code`. Categorical columns (`Loan_Purpose`, `Collateral_Type`) were one-hot encoded. Numerical columns were standardized with `StandardScaler`.

---

## Model Results

### Stage 1 — XGBoost Classifier

Predicts whether the credit line will adjust this month.

**Validation Set:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| No Adjustment (0) | 0.95 | 0.49 | 0.64 |
| Adjustment (1) | 0.91 | **0.99** | **0.95** |
| **Accuracy** | | | **91.38%** |

**Test Set:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| No Adjustment (0) | 0.94 | 0.49 | 0.64 |
| Adjustment (1) | 0.91 | **0.99** | **0.95** |
| **Accuracy** | | | **91.20%** |

The model catches **99% of all real adjustments** — it almost never misses a month where the credit line actually changed. The lower recall on "no change" (49%) is an acceptable trade-off in a portfolio where adjustments are the norm.

---

### Stage 2 — XGBoost Regressor

Trained only on rows where the classifier predicted an adjustment. Dataset size: **113,837 rows**.

| Metric | Validation | Test |
|--------|-----------|------|
| MAE | $1,405 | **$1,416** |
| RMSE | $1,817 | **$1,825** |
| R² | 0.9968 | **0.9968** |

R² of **0.9968** means the model explains **99.68% of variance** in credit line amounts. On a target ranging from $1,000 to $111,000, an average error of ~$1,400 is extremely precise.

---

## Collateral Type — Adjustment Rate

| Collateral Type | Adjustment Rate |
|----------------|----------------|
| Accounts Receivable | **95.0%** |
| Inventory | 94.6% |
| Real Estate | 93.8% |
| Intellectual Property | 93.4% |
| Equipment | 93.3% |
| Cash | 91.5% |

Businesses with Accounts Receivable collateral see the highest rate of credit line changes — consistent with more dynamic, revenue-driven lending relationships.

---

## Pipeline Summary

```
Raw Data — 124,436 rows
        ↓
EDA + Correlation Analysis
        ↓
Outlier Removal (IQR) — 92,623 rows remaining
        ↓
Impute Collateral_Type → 'Unknown'
        ↓
Train / Val / Test Split  (70 / 15 / 15)
        ↓
Feature Engineering — create AdjustCreditLine target
        ↓
XGBoost Classifier ——— Accuracy: 91.2%  |  Recall (Adjust): 99%
        ↓
XGBoost Regressor  ——— R²: 0.9968  |  MAE: $1,416
        ↓
Export — SMB_Dynamic_Credit_Line_Output.csv
```

---

## Output File

`SMB_Dynamic_Credit_Line_Output.csv` — original data with two new columns:

| New Column | Description |
|------------|-------------|
| `Predicted_AdjustCreditLine` | 1 = adjustment predicted, 0 = no change |
| `Predicted_DynamicCreditLine` | Predicted credit line amount (only filled where adjustment predicted) |

---

**Dependencies:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost`
