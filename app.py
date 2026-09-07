"""
ML Dataset Advisor
-------------------
A Streamlit app that takes a CSV, profiles every column, scores the
dataset's "ML-readiness", and recommends suitable ML models based on
feature types, missingness, target type, class balance, dimensionality,
and other heuristics.

Run with:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML Dataset Advisor",
    page_icon="🧭",
    layout="wide",
)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

NUMERIC_KINDS = "iuf"


def detect_column_type(series: pd.Series) -> str:
    """Classify a column as numeric / categorical / boolean / datetime / text / id-like."""
    s = series.dropna()
    if s.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # try to parse object columns that look like dates
    if series.dtype == object:
        sample = s.sample(min(len(s), 25), random_state=0)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format=None)
            if parsed.notna().mean() > 0.9:
                return "datetime"
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(series):
        n_unique = s.nunique()
        # binary numeric (0/1) -> treat as categorical/boolean-like
        if n_unique == 2:
            return "binary_numeric"
        # low-cardinality integer columns often behave like categories
        if pd.api.types.is_integer_dtype(series) and n_unique <= 15 and n_unique / max(len(s), 1) < 0.05:
            return "categorical_numeric"
        return "numeric"

    # object / string columns
    n_unique = s.nunique()
    ratio_unique = n_unique / max(len(s), 1)
    avg_len = s.astype(str).str.len().mean()

    if ratio_unique > 0.9 and n_unique > 20:
        return "id_like"
    if n_unique == 2:
        return "binary_categorical"
    if avg_len > 30 and ratio_unique > 0.5:
        return "text"
    return "categorical"


def analyze_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        col_type = detect_column_type(s)
        missing = s.isna().sum()
        missing_pct = round(100 * missing / n, 2) if n else 0
        n_unique = s.nunique(dropna=True)

        val_range = ""
        skew = np.nan
        outlier_pct = np.nan
        if col_type == "numeric":
            numeric_s = pd.to_numeric(s, errors="coerce").dropna()
            if not numeric_s.empty:
                val_range = f"{numeric_s.min():.3g} to {numeric_s.max():.3g}"
                skew = numeric_s.skew()
                q1, q3 = numeric_s.quantile(0.25), numeric_s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outlier_pct = round(
                        100 * ((numeric_s < lo) | (numeric_s > hi)).sum() / len(numeric_s), 2
                    )
                else:
                    outlier_pct = 0.0

        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "detected_type": col_type,
                "missing_count": int(missing),
                "missing_pct": missing_pct,
                "unique_values": int(n_unique),
                "value_range": val_range,
                "skewness": round(skew, 2) if pd.notna(skew) else None,
                "outlier_pct": outlier_pct,
            }
        )
    return pd.DataFrame(rows)


def analyze_target(df: pd.DataFrame, target_col: str) -> dict:
    s = df[target_col]
    n = len(s)
    missing = int(s.isna().sum())
    n_unique = s.nunique(dropna=True)
    col_type = detect_column_type(s)

    is_numeric = pd.api.types.is_numeric_dtype(s) and col_type in ("numeric", "binary_numeric")

    if col_type in ("categorical", "binary_categorical", "boolean", "categorical_numeric", "binary_numeric") or (
        is_numeric and n_unique <= 20 and n_unique / max(n, 1) < 0.05
    ):
        task = "classification"
    elif is_numeric:
        task = "regression"
    else:
        task = "unknown"

    class_balance = None
    n_classes = None
    if task == "classification":
        counts = s.value_counts(dropna=True)
        n_classes = len(counts)
        if n_classes > 0:
            majority_pct = counts.iloc[0] / counts.sum()
            minority_pct = counts.iloc[-1] / counts.sum()
            class_balance = {
                "n_classes": n_classes,
                "majority_pct": round(100 * majority_pct, 2),
                "minority_pct": round(100 * minority_pct, 2),
                "imbalance_ratio": round(counts.iloc[0] / max(counts.iloc[-1], 1), 2),
                "counts": counts,
            }

    return {
        "column": target_col,
        "dtype": str(s.dtype),
        "detected_type": col_type,
        "task": task,
        "missing_count": missing,
        "missing_pct": round(100 * missing / n, 2) if n else 0,
        "n_unique": int(n_unique),
        "class_balance": class_balance,
    }


def compute_dataset_score(df: pd.DataFrame, feat_df: pd.DataFrame, target_info: dict):
    """Return overall score (0-100) plus a breakdown of sub-scores and messages."""
    n_rows, n_cols = df.shape
    n_features = max(n_cols - 1, 1)
    breakdown = {}
    notes = []

    # 1) Missingness (20 pts) - dataset-wide average missing pct across all columns
    overall_missing_pct = df.isna().mean().mean() * 100
    missing_score = max(0, 20 - overall_missing_pct * 0.8)
    breakdown["Missing data"] = round(min(missing_score, 20), 1)
    if overall_missing_pct > 20:
        notes.append(f"High overall missingness ({overall_missing_pct:.1f}%) — consider imputation or dropping sparse columns.")
    elif overall_missing_pct > 5:
        notes.append(f"Moderate missingness ({overall_missing_pct:.1f}%) — light imputation likely needed.")

    # 2) Duplicate rows (10 pts)
    dup_pct = 100 * df.duplicated().sum() / max(n_rows, 1)
    dup_score = max(0, 10 - dup_pct * 0.5)
    breakdown["Duplicate rows"] = round(min(dup_score, 10), 1)
    if dup_pct > 5:
        notes.append(f"{dup_pct:.1f}% duplicate rows detected — consider de-duplicating.")

    # 3) Sample size adequacy (20 pts)
    if n_rows >= 10000:
        size_score = 20
    elif n_rows >= 1000:
        size_score = 16
    elif n_rows >= 200:
        size_score = 11
    elif n_rows >= 50:
        size_score = 6
    else:
        size_score = 2
        notes.append("Very few rows — model reliability will be limited regardless of algorithm.")
    breakdown["Sample size"] = size_score

    # 4) Feature-to-row ratio / dimensionality (15 pts)
    ratio = n_features / max(n_rows, 1)
    if ratio < 0.02:
        dim_score = 15
    elif ratio < 0.1:
        dim_score = 11
    elif ratio < 0.3:
        dim_score = 7
        notes.append("Feature count is fairly high relative to rows — risk of overfitting; consider regularization or dimensionality reduction.")
    else:
        dim_score = 3
        notes.append("Very high dimensionality relative to sample size — strongly consider feature selection, PCA, or regularization.")
    breakdown["Dimensionality"] = dim_score

    # 5) Target quality / class balance (15 pts)
    target_score = 15
    if target_info["missing_pct"] > 0:
        target_score -= min(10, target_info["missing_pct"])
        notes.append(f"Target column has {target_info['missing_pct']}% missing values — rows with missing target should be dropped before training.")
    if target_info["task"] == "classification" and target_info["class_balance"]:
        imb = target_info["class_balance"]["imbalance_ratio"]
        if imb > 20:
            target_score -= 8
            notes.append(f"Severe class imbalance (majority:minority ≈ {imb}:1) — use resampling (SMOTE), class weights, or anomaly-detection framing.")
        elif imb > 5:
            target_score -= 4
            notes.append(f"Moderate class imbalance (majority:minority ≈ {imb}:1) — consider class weighting or resampling.")
    elif target_info["task"] == "unknown":
        target_score -= 8
        notes.append("Could not confidently determine whether the target is regression or classification — please verify the target column choice.")
    breakdown["Target quality"] = round(max(target_score, 0), 1)

    # 6) Outliers in numeric features (10 pts)
    numeric_feats = feat_df[feat_df["detected_type"] == "numeric"]
    if not numeric_feats.empty:
        avg_outlier_pct = numeric_feats["outlier_pct"].fillna(0).mean()
    else:
        avg_outlier_pct = 0
    outlier_score = max(0, 10 - avg_outlier_pct * 0.4)
    breakdown["Outliers"] = round(min(outlier_score, 10), 1)
    if avg_outlier_pct > 15:
        notes.append(f"Numeric features show notable outliers (avg {avg_outlier_pct:.1f}%) — consider robust scaling or tree-based models, which are less outlier-sensitive.")

    # 7) Feature diversity / redundancy (10 pts) - penalize too many id-like / text / empty columns
    junk_cols = feat_df["detected_type"].isin(["id_like", "empty", "text"]).sum()
    junk_ratio = junk_cols / max(len(feat_df), 1)
    diversity_score = max(0, 10 - junk_ratio * 20)
    breakdown["Feature usability"] = round(min(diversity_score, 10), 1)
    if junk_ratio > 0.3:
        notes.append("A large share of columns look like IDs, free text, or empty fields — these usually need dropping or special encoding (e.g. text embeddings) rather than direct use.")

    total = round(sum(breakdown.values()), 1)
    return total, breakdown, notes


def score_label(score: float) -> tuple:
    if score >= 80:
        return "Excellent", "🟢"
    elif score >= 60:
        return "Good", "🟡"
    elif score >= 40:
        return "Fair", "🟠"
    else:
        return "Poor", "🔴"


def suggest_models(df: pd.DataFrame, feat_df: pd.DataFrame, target_info: dict):
    """Rule-based model recommendation engine. Returns a ranked list of
    (model_name, score 0-100, reasons[]) plus general preprocessing tips."""
    n_rows, n_cols = df.shape
    n_features = max(n_cols - 1, 1)
    task = target_info["task"]

    numeric_ct = (feat_df["detected_type"] == "numeric").sum()
    categorical_ct = feat_df["detected_type"].isin(
        ["categorical", "binary_categorical", "categorical_numeric", "binary_numeric", "boolean"]
    ).sum()
    text_ct = (feat_df["detected_type"] == "text").sum()
    high_missing_ct = (feat_df["missing_pct"] > 30).sum()
    avg_outlier_pct = feat_df.loc[feat_df["detected_type"] == "numeric", "outlier_pct"].fillna(0).mean() if numeric_ct else 0
    dim_ratio = n_features / max(n_rows, 1)

    small_data = n_rows < 500
    large_data = n_rows > 20000
    high_dim = dim_ratio > 0.1
    mostly_categorical = categorical_ct > numeric_ct
    heavy_outliers = avg_outlier_pct > 15

    tips = []
    if high_missing_ct:
        tips.append(f"{high_missing_ct} column(s) have >30% missing data — impute (median/mode) or drop before modeling.")
    if text_ct:
        tips.append(f"{text_ct} free-text column(s) detected — needs vectorization (TF-IDF/embeddings) or should be dropped for classic ML.")
    if mostly_categorical:
        tips.append("Categorical features dominate — one-hot/target encoding needed for linear/SVM models; tree models can often use ordinal/label encoding directly.")
    if heavy_outliers:
        tips.append("Notable outliers present — consider robust scaling (RobustScaler) or use tree-based models, which are naturally robust to outliers.")
    if high_dim:
        tips.append("High feature-to-row ratio — favor regularized linear models, PCA, or feature selection to reduce overfitting risk.")

    candidates = []

    def add(name, base, reasons):
        candidates.append({"model": name, "score": base, "reasons": reasons})

    if task == "regression":
        # Linear Regression / Ridge / Lasso
        s = 55
        r = ["Baseline interpretable model for continuous targets."]
        if high_dim:
            s += 15
            r.append("High feature-to-row ratio favors regularization (Ridge/Lasso/ElasticNet) to control overfitting.")
        if small_data:
            s += 10
            r.append("Small dataset — simple linear models generalize better than complex ones and are fast to validate.")
        if heavy_outliers:
            s -= 10
            r.append("Outliers can distort ordinary least squares; consider Huber or robust regression variants.")
        add("Linear / Ridge / Lasso Regression", s, r)

        # Random Forest Regressor
        s = 60
        r = ["Handles non-linear relationships and mixed feature types without heavy preprocessing."]
        if mostly_categorical:
            s += 8
            r.append("Handles categorical/ordinal-encoded features well without scaling.")
        if heavy_outliers:
            s += 8
            r.append("Tree-based splits are robust to outliers.")
        if small_data:
            s -= 8
            r.append("With few rows, trees can overfit — use small depth and strong cross-validation.")
        if large_data:
            s += 5
        add("Random Forest Regressor", s, r)

        # Gradient Boosting (XGBoost/LightGBM)
        s = 58
        r = ["Strong performance on tabular data with complex feature interactions."]
        if large_data:
            s += 15
            r.append("Larger datasets let boosting models shine without overfitting as much.")
        if small_data:
            s -= 15
            r.append("Boosting models tend to overfit on very small datasets without careful tuning.")
        if numeric_ct + categorical_ct >= 8:
            s += 5
            r.append("Many features — boosting handles interactions and feature importance well.")
        add("Gradient Boosting (XGBoost / LightGBM)", s, r)

        # SVR
        s = 40
        r = ["Can capture non-linear patterns via kernels."]
        if large_data:
            s -= 15
            r.append("SVR scales poorly to large datasets.")
        if small_data:
            s += 10
            r.append("Works reasonably well on small-to-medium datasets after scaling.")
        add("Support Vector Regression (SVR)", s, r)

    elif task == "classification":
        n_classes = target_info["class_balance"]["n_classes"] if target_info["class_balance"] else 2
        imbalance = target_info["class_balance"]["imbalance_ratio"] if target_info["class_balance"] else 1
        binary = n_classes == 2

        # Logistic Regression
        s = 55
        r = ["Fast, interpretable baseline for classification."]
        if binary:
            s += 8
            r.append("Well suited to binary targets.")
        if high_dim:
            s += 10
            r.append("Regularized logistic regression handles high dimensionality well.")
        if imbalance > 10:
            s -= 8
            r.append("Needs class-weighting or resampling to handle severe imbalance.")
        add("Logistic Regression", s, r)

        # Random Forest Classifier
        s = 62
        r = ["Robust general-purpose classifier, handles mixed feature types and non-linearity."]
        if mostly_categorical:
            s += 8
            r.append("Handles categorical/ordinal features without heavy preprocessing.")
        if imbalance > 5:
            s += 5
            r.append("Can be combined with class_weight='balanced' to mitigate imbalance.")
        if small_data:
            s -= 5
        add("Random Forest Classifier", s, r)

        # Gradient Boosting Classifier
        s = 58
        r = ["Typically top performer on structured/tabular classification tasks."]
        if large_data:
            s += 15
            r.append("Scales well and captures complex patterns with enough data.")
        if small_data:
            s -= 12
            r.append("Prone to overfitting on very small datasets.")
        if imbalance > 5:
            s += 5
            r.append("Supports scale_pos_weight / class weighting for imbalance.")
        add("Gradient Boosting (XGBoost / LightGBM)", s, r)

        # SVM
        s = 42
        r = ["Effective for clear-margin classification, especially with scaled numeric features."]
        if large_data:
            s -= 15
            r.append("Training time grows quickly with dataset size.")
        if small_data:
            s += 12
            r.append("Performs well on small-to-medium, well-scaled datasets.")
        add("Support Vector Machine (SVM)", s, r)

        # KNN
        s = 38
        r = ["Simple, non-parametric method; sensitive to feature scaling and dimensionality."]
        if high_dim:
            s -= 12
            r.append("Suffers from the curse of dimensionality with many features.")
        if small_data:
            s += 8
            r.append("Reasonable choice on small, low-dimensional datasets.")
        add("K-Nearest Neighbors", s, r)

        if n_classes > 2:
            for c in candidates:
                c["reasons"].append(f"Multiclass target ({n_classes} classes) — ensure the chosen model/library supports multiclass natively (most listed here do).")

    else:
        add("Unable to auto-recommend", 0, ["Target type could not be determined — please double check the target column selection."])

    for c in candidates:
        c["score"] = round(max(0, min(100, c["score"])), 1)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates, tips


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🧭 ML Dataset Advisor")
st.caption(
    "Upload a CSV, pick your target column, and get a data-quality score "
    "plus model recommendations tailored to your dataset's characteristics."
)

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

try:
    raw_bytes = uploaded.read()
    df = pd.read_csv(io.BytesIO(raw_bytes))
except Exception as e:
    st.error(f"Could not read this CSV: {e}")
    st.stop()

if df.empty or df.shape[1] < 2:
    st.error("The uploaded CSV needs at least one feature column and one target column.")
    st.stop()

st.success(f"Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns")

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), width='stretch')

target_col = st.selectbox("Select the target column (what you want to predict)", options=df.columns.tolist())

feat_df = analyze_features(df, target_col)
target_info = analyze_target(df, target_col)
score, breakdown, notes = compute_dataset_score(df, feat_df, target_info)
label, emoji = score_label(score)

st.divider()

# --- Score summary ---
col1, col2 = st.columns([1, 2])
with col1:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            title={"text": f"Dataset Score — {emoji} {label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4C78A8"},
                "steps": [
                    {"range": [0, 40], "color": "#f8d7da"},
                    {"range": [40, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#d1ecf1"},
                    {"range": [80, 100], "color": "#d4edda"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Score breakdown")
    breakdown_df = pd.DataFrame(
        {"Category": list(breakdown.keys()), "Points": list(breakdown.values())}
    )
    max_points = {
        "Missing data": 20, "Duplicate rows": 10, "Sample size": 20,
        "Dimensionality": 15, "Target quality": 15, "Outliers": 10,
        "Feature usability": 10,
    }
    breakdown_df["Max"] = breakdown_df["Category"].map(max_points)
    bar = px.bar(
        breakdown_df, x="Points", y="Category", orientation="h",
        range_x=[0, breakdown_df["Max"].max()], text="Points",
        color="Points", color_continuous_scale="Blues",
    )
    bar.update_layout(height=280, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, width='stretch')

if notes:
    st.subheader("⚠️ Things to address")
    for n in notes:
        st.markdown(f"- {n}")
else:
    st.success("No major data-quality issues detected.")

st.divider()

# --- Target analysis ---
st.subheader(f"🎯 Target column: `{target_col}`")
tcol1, tcol2 = st.columns([1, 2])
with tcol1:
    st.metric("Detected task", target_info["task"].capitalize() if target_info["task"] != "unknown" else "Unclear")
    st.metric("Data type", target_info["detected_type"])
    st.metric("Missing values", f"{target_info['missing_count']} ({target_info['missing_pct']}%)")
    if target_info["task"] == "classification":
        st.metric("Number of classes", target_info["class_balance"]["n_classes"])

with tcol2:
    if target_info["task"] == "classification" and target_info["class_balance"]:
        counts = target_info["class_balance"]["counts"]
        cfig = px.bar(
            x=counts.index.astype(str), y=counts.values,
            labels={"x": target_col, "y": "count"}, title="Class distribution",
        )
        cfig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(cfig, width='stretch')
    elif target_info["task"] == "regression":
        numeric_target = pd.to_numeric(df[target_col], errors="coerce").dropna()
        hfig = px.histogram(numeric_target, nbins=40, title="Target distribution")
        hfig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(hfig, width='stretch')

st.divider()

# --- Feature analysis table ---
st.subheader("🔎 Feature profile")
st.dataframe(
    feat_df.style.background_gradient(subset=["missing_pct"], cmap="Reds", vmin=0, vmax=100),
    width='stretch',
    height=min(400, 40 + 35 * len(feat_df)),
)

type_counts = feat_df["detected_type"].value_counts()
tfig = px.pie(values=type_counts.values, names=type_counts.index, title="Feature type composition", hole=0.4)
tfig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(tfig, width='stretch')

st.divider()

# --- Model recommendations ---
st.subheader("🤖 Recommended models")

if target_info["task"] == "unknown":
    st.warning("Cannot recommend models confidently until the target column type is clear. "
               "Try selecting a different column, or verify this column's values.")
else:
    candidates, tips = suggest_models(df, feat_df, target_info)

    top = candidates[0]
    st.markdown(f"### 🏆 Top pick: **{top['model']}**  ·  fit score {top['score']}/100")
    for reason in top["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown("#### Other viable options")
    for c in candidates[1:]:
        with st.expander(f"{c['model']} — fit score {c['score']}/100"):
            for reason in c["reasons"]:
                st.markdown(f"- {reason}")

    rank_fig = px.bar(
        pd.DataFrame(candidates), x="score", y="model", orientation="h",
        range_x=[0, 100], text="score", title="Model fit scores",
        color="score", color_continuous_scale="Tealgrn",
    )
    rank_fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(rank_fig, width='stretch')

    if tips:
        st.markdown("#### 🛠️ Preprocessing tips")
        for t in tips:
            st.markdown(f"- {t}")

st.divider()
st.caption(
    "Note: this tool applies rule-of-thumb heuristics on dataset shape, missingness, "
    "feature types, and target characteristics. It is a starting point for model "
    "selection, not a substitute for proper experimentation and cross-validation."
)
