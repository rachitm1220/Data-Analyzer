# ML Dataset Advisor

A Streamlit app that profiles an uploaded CSV, scores its "ML-readiness" out of 100,
and recommends which ML models are best suited to the dataset.

## What it checks

**Per-feature profiling**
- Data type detection: numeric, categorical, boolean, datetime, free text, ID-like
- Missing value count/percentage
- Cardinality (unique values)
- Value range, skewness, and outlier percentage (IQR method) for numeric columns

**Target analysis**
- Auto-detects regression vs. classification based on dtype and cardinality
- Class distribution and imbalance ratio for classification targets
- Missing target values

**Dataset score (0–100)**, weighted across:
| Category | Points |
|---|---|
| Missing data | 20 |
| Sample size adequacy | 20 |
| Dimensionality (features vs. rows) | 15 |
| Target quality / class balance | 15 |
| Duplicate rows | 10 |
| Outliers | 10 |
| Feature usability (junk/ID/text columns) | 10 |

**Model recommendations**
Rule-based scoring across common models (Linear/Logistic Regression, Random Forest,
Gradient Boosting/XGBoost, SVM/SVR, KNN) based on:
- Task type (regression vs. classification)
- Dataset size (small vs. large)
- Feature-to-row ratio (overfitting risk)
- Mix of categorical vs. numeric features
- Outlier prevalence
- Class imbalance (for classification)

Each recommendation includes a fit score and plain-language reasons, plus general
preprocessing tips (imputation, encoding, scaling, resampling, etc.).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501), upload a
CSV, and select the target column.

## Notes

This tool uses heuristics on dataset shape and statistics — it's a fast starting
point for model selection, not a replacement for proper cross-validation and
experimentation.
