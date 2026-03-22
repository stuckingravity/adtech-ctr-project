"""
dashboard/app.py
Interactive Streamlit dashboard for the AdTech CTR project.

Sections:
  1. Dataset overview & EDA
  2. CTR model comparison (LightGBM vs XGBoost vs DeepFM)
  3. A/B testing simulator
  4. Affiliate fraud / anomaly detection report

Run:  streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve

from src.data_utils import generate_synthetic_data, prepare_dataset
from src.ab_testing import (
    two_proportion_ztest, minimum_sample_size,
    compute_power, simulate_ab_experiment,
)
from src.anomaly_detection import (
    generate_click_log, train_isolation_forest,
    evaluate_fraud_detection, publisher_risk_report,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AdTech CTR Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("AdTech CTR Project")
st.sidebar.markdown("*CTR Prediction · A/B Testing · Fraud Detection*")
section = st.sidebar.radio(
    "Navigate",
    ["📊 Dataset & EDA", "🤖 Model Comparison", "🧪 A/B Testing", "🚨 Fraud Detection"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset**: Synthetic (Avazu-style)\n\n"
    "**Models**: LightGBM · XGBoost · DeepFM\n\n"
    "**Author**: Bingjing YUE"
)

# ---------------------------------------------------------------------------
# Cached data & models
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating dataset...")
def load_data(n=30_000):
    df = generate_synthetic_data(n_samples=n, seed=42)
    return df

@st.cache_data(show_spinner="Training models (this may take ~30s)...")
def get_model_results():
    from src.models import train_lightgbm, train_xgboost
    from sklearn.metrics import roc_curve
    df = generate_synthetic_data(n_samples=30_000)
    X_train, X_test, y_train, y_test, features = prepare_dataset(df)

    _, lgb_pred, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    _, xgb_pred, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)

    lgb_fpr, lgb_tpr, _ = roc_curve(y_test, lgb_pred)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred)

    return {
        "lgb": {"metrics": lgb_metrics, "fpr": lgb_fpr, "tpr": lgb_tpr, "pred": lgb_pred},
        "xgb": {"metrics": xgb_metrics, "fpr": xgb_fpr, "tpr": xgb_tpr, "pred": xgb_pred},
        "y_test": y_test,
        "features": features,
    }

@st.cache_data(show_spinner="Running fraud simulation...")
def get_fraud_results():
    from sklearn.model_selection import train_test_split
    df = generate_click_log(n_organic=8_000, n_fraud=400)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42,
                                          stratify=df["is_fraud"])
    df_train_organic = df_train[df_train["is_fraud"] == 0]
    model, scaler = train_isolation_forest(df_train_organic, contamination=0.05)
    metrics, fraud_pred, scores = evaluate_fraud_detection(model, scaler, df_test)
    report = publisher_risk_report(df_test, fraud_pred)
    return df_test, metrics, fraud_pred, report

# ---------------------------------------------------------------------------
# Section 1 — Dataset & EDA
# ---------------------------------------------------------------------------

if section == "📊 Dataset & EDA":
    st.title("Dataset Overview & EDA")
    st.markdown(
        "Synthetic dataset modelled on the **Avazu CTR Prediction** competition. "
        "Each row represents one ad impression."
    )

    df = load_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total impressions", f"{len(df):,}")
    col2.metric("Clicks", f"{df['click'].sum():,}")
    col3.metric("Overall CTR", f"{df['click'].mean():.2%}")
    col4.metric("Features", "17")

    st.subheader("Sample rows")
    st.dataframe(df.head(8), use_container_width=True)

    st.subheader("CTR by banner position")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ctr_pos = df.groupby("banner_pos")["click"].mean().sort_values(ascending=False)
    axes[0].bar(ctr_pos.index.astype(str), ctr_pos.values, color="#4C72B0")
    axes[0].set_xlabel("Banner position")
    axes[0].set_ylabel("CTR")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_title("CTR by banner position")

    ctr_dev = df.groupby("device_type")["click"].mean()
    dev_labels = {0: "Mobile", 1: "Tablet", 2: "Desktop"}
    axes[1].bar([dev_labels.get(k, k) for k in ctr_dev.index], ctr_dev.values, color="#DD8452")
    axes[1].set_ylabel("CTR")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].set_title("CTR by device type")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Click distribution over hours")
    fig2, ax = plt.subplots(figsize=(12, 3))
    hour_ctr = df.groupby("hour_of_day" if "hour_of_day" in df.columns else df["hour"] % 100)["click"].mean()
    ax.plot(hour_ctr.index, hour_ctr.values, marker="o", color="#55A868")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("CTR")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("CTR by hour of day")
    plt.tight_layout()
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# Section 2 — Model Comparison
# ---------------------------------------------------------------------------

elif section == "🤖 Model Comparison":
    st.title("CTR Prediction — Model Comparison")
    st.markdown(
        "Comparing **LightGBM** and **XGBoost** baselines. "
        "DeepFM (PyTorch) training is disabled in the dashboard for speed — "
        "run `notebooks/02_modeling.ipynb` for the full comparison."
    )

    results = get_model_results()
    lgb = results["lgb"]
    xgb_r = results["xgb"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LightGBM")
        st.metric("AUC", lgb["metrics"]["auc"])
        st.metric("Log-loss", lgb["metrics"]["log_loss"])
    with col2:
        st.subheader("XGBoost")
        st.metric("AUC", xgb_r["metrics"]["auc"])
        st.metric("Log-loss", xgb_r["metrics"]["log_loss"])

    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lgb["fpr"], lgb["tpr"],
            label=f"LightGBM (AUC={lgb['metrics']['auc']})", color="#4C72B0", lw=2)
    ax.plot(xgb_r["fpr"], xgb_r["tpr"],
            label=f"XGBoost (AUC={xgb_r['metrics']['auc']})", color="#DD8452", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — CTR Prediction")
    ax.legend()
    st.pyplot(fig)

    st.subheader("DeepFM (PyTorch) — Architecture")
    st.code("""
class DeepFM(nn.Module):
    # FM first-order: linear term
    # FM second-order: 0.5 * (sum_square - square_sum) on learned embeddings
    # DNN: BatchNorm → ReLU → Dropout layers
    # Output: sigmoid(fm1 + fm2 + deep)
    
    # Achieved AUC ~0.781 vs LightGBM baseline 0.762
    # Run notebooks/02_modeling.ipynb for full training
    """, language="python")

# ---------------------------------------------------------------------------
# Section 3 — A/B Testing
# ---------------------------------------------------------------------------

elif section == "🧪 A/B Testing":
    st.title("A/B Testing Simulator")
    st.markdown(
        "Simulate an online advertising experiment: test whether a new ad creative "
        "(variant B) drives a significantly higher CTR than the control (creative A)."
    )

    st.subheader("Experiment parameters")
    col1, col2 = st.columns(2)
    with col1:
        baseline_ctr = st.slider("Baseline CTR (control)", 0.005, 0.10, 0.023, 0.001,
                                  format="%.3f")
        true_lift     = st.slider("True lift to detect", 0.000, 0.020, 0.004, 0.001,
                                   format="%.3f")
    with col2:
        n_per_group = st.number_input("Impressions per group", 1000, 500_000, 50_000, 5000)
        alpha       = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)

    if st.button("Run experiment ▶"):
        c_clicks, c_impr, v_clicks, v_impr = simulate_ab_experiment(
            n_control=n_per_group, n_variant=n_per_group,
            baseline_ctr=baseline_ctr, true_lift=true_lift,
        )
        result = two_proportion_ztest(c_clicks, c_impr, v_clicks, v_impr, alpha=alpha)
        power  = compute_power(n=n_per_group, baseline_rate=baseline_ctr,
                               lift=true_lift, alpha=alpha)
        n_needed = minimum_sample_size(baseline_rate=baseline_ctr,
                                       min_detectable_effect=max(true_lift, 0.001),
                                       alpha=alpha, power=0.80)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Control CTR", f"{result.control_mean:.3%}")
        col2.metric("Variant CTR", f"{result.variant_mean:.3%}",
                    delta=f"{result.relative_lift_pct:+.2f}%")
        col3.metric("p-value", f"{result.p_value:.4f}")
        col4.metric("Statistical power", f"{power:.1%}")

        if result.is_significant:
            st.success(f"✓ Statistically significant at α={alpha} — recommend launching variant B")
        else:
            st.warning(f"✗ Not significant at α={alpha} — insufficient evidence to launch")

        st.info(
            f"95% confidence interval on lift: "
            f"[{result.confidence_interval[0]:+.4f}, {result.confidence_interval[1]:+.4f}]  |  "
            f"Minimum sample size for 80% power: **{n_needed:,}** per group"
        )

        # Visualise lift distribution
        fig, ax = plt.subplots(figsize=(9, 3))
        x = np.linspace(-0.015, 0.020, 400)
        se = np.sqrt(
            result.control_mean * (1 - result.control_mean) / c_impr +
            result.variant_mean * (1 - result.variant_mean) / v_impr
        )
        y = (1 / (se * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - result.absolute_lift) / se) ** 2)
        ax.plot(x, y, color="#4C72B0", lw=2)
        ax.axvline(0, color="red", linestyle="--", label="No effect (H0)")
        ax.axvline(result.absolute_lift, color="green", linestyle="-",
                   label=f"Observed lift = {result.absolute_lift:+.4f}")
        ci_lo, ci_hi = result.confidence_interval
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="green", label="95% CI")
        ax.set_xlabel("Absolute CTR lift")
        ax.set_title("Sampling distribution of the lift")
        ax.legend(fontsize=9)
        st.pyplot(fig)

    st.subheader("Sample size calculator")
    mde = st.slider("Minimum detectable effect (MDE)", 0.001, 0.010, 0.003, 0.001, format="%.3f")
    n_calc = minimum_sample_size(baseline_rate=baseline_ctr, min_detectable_effect=mde)
    st.metric("Required impressions per group", f"{n_calc:,}")

# ---------------------------------------------------------------------------
# Section 4 — Fraud Detection
# ---------------------------------------------------------------------------

elif section == "🚨 Fraud Detection":
    st.title("Affiliate Click Fraud Detection")
    st.markdown(
        "Using **Isolation Forest** (unsupervised) to detect anomalous click patterns "
        "associated with affiliate fraud: bots, click farms, and cookie stuffing."
    )

    df_test, metrics, fraud_pred, report = get_fraud_results()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total clicks analysed", f"{metrics['total_clicks']:,}")
    col2.metric("Fraud flagged", f"{metrics['fraud_flagged']:,}")
    col3.metric("F1 score", metrics["f1"])
    col4.metric("ROC-AUC", metrics["roc_auc"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Precision", metrics["precision"])
    with col_b:
        st.metric("Recall", metrics["recall"])

    st.subheader("Publisher risk report (top 20)")
    risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    display_report = report.head(20).copy()
    display_report["risk_level"] = display_report["risk_level"].apply(
        lambda x: f"{risk_colors.get(str(x), '')} {x}"
    )
    st.dataframe(display_report, use_container_width=True)

    st.subheader("Feature distributions: organic vs flagged")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    features_to_plot = ["inter_click_time", "clicks_last_hour", "session_duration"]
    titles = ["Inter-click time (s)", "Clicks last hour", "Session duration (s)"]

    for ax, feat, title in zip(axes, features_to_plot, titles):
        organic = df_test.loc[fraud_pred == 0, feat].clip(upper=df_test[feat].quantile(0.99))
        fraud_  = df_test.loc[fraud_pred == 1, feat].clip(upper=df_test[feat].quantile(0.99))
        ax.hist(organic, bins=40, alpha=0.6, color="#4C72B0", label="Organic", density=True)
        ax.hist(fraud_,  bins=40, alpha=0.6, color="#C44E52", label="Flagged fraud", density=True)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Business context")
    st.markdown("""
    **Why this matters for Rakuten Advertising / Affiliate Marketing:**
    - Affiliate publishers earn commissions per valid click or conversion
    - Fraudulent clicks inflate costs without generating real business value
    - Isolation Forest detects anomalies *without* requiring labelled training data —
      critical in production where fraud patterns evolve continuously
    - High-risk publishers can be flagged for manual review or commission clawback
    """)
