# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 — A/B Testing & Anomaly Detection
#
# **Goal**: Demonstrate two critical capabilities for online advertising:
#
# **Part A — A/B Testing**: Statistical framework to evaluate whether a new ad creative
# drives a significant CTR uplift over the control.
#
# **Part B — Affiliate Fraud Detection**: Unsupervised anomaly detection (Isolation Forest)
# to identify fraudulent click patterns in affiliate marketing.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

from src.ab_testing import (
    two_proportion_ztest, ttest_continuous,
    minimum_sample_size, compute_power,
    bonferroni_correction, simulate_ab_experiment,
)
from src.anomaly_detection import (
    generate_click_log, train_isolation_forest,
    evaluate_fraud_detection, publisher_risk_report,
)

# %% [markdown]
# ---
# ## Part A — A/B Testing Framework
#
# ### Business context
#
# Rakuten Advertising runs experiments to test new ad creatives, landing pages,
# and audience segments. A statistically rigorous A/B testing framework is essential
# to avoid:
# - **False positives** (launching a variant that doesn't actually help)
# - **False negatives** (discarding a variant that would have improved performance)

# %% [markdown]
# ### 1. Sample size planning (pre-experiment)

# %%
baseline_ctr = 0.023    # 2.3% current CTR
mde          = 0.003    # minimum detectable effect: +0.3 pp uplift

n_required = minimum_sample_size(
    baseline_rate=baseline_ctr,
    min_detectable_effect=mde,
    alpha=0.05,
    power=0.80,
)
print(f"Minimum sample size per group: {n_required:,}")
print(f"(to detect a {mde:.1%} absolute lift with 80% power at α=0.05)")

# %% [markdown]
# ### 2. Simulate experiment results

# %%
# Scenario 1: real uplift (creative B genuinely better)
c1, i1, v1, j1 = simulate_ab_experiment(
    n_control=100_000, n_variant=100_000,
    baseline_ctr=0.023, true_lift=0.004, seed=42
)
result_sig = two_proportion_ztest(c1, i1, v1, j1, metric_name="CTR (creative B vs A)")
print(result_sig)

# Scenario 2: underpowered test (too few observations)
c2, i2, v2, j2 = simulate_ab_experiment(
    n_control=3_000, n_variant=3_000,
    baseline_ctr=0.023, true_lift=0.004, seed=42
)
result_insig = two_proportion_ztest(c2, i2, v2, j2, metric_name="CTR (underpowered)")
print(result_insig)

# %% [markdown]
# ### 3. Visualise lift distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, result, title in [
    (axes[0], result_sig,   "Scenario 1: Sufficient sample (n=100k)"),
    (axes[1], result_insig, "Scenario 2: Underpowered (n=3k)"),
]:
    lift = result.absolute_lift
    se   = abs(lift) / (abs(stats.norm.ppf(result.p_value / 2)) + 1e-9)
    x    = np.linspace(lift - 4*se, lift + 4*se, 400)
    y    = stats.norm.pdf(x, loc=lift, scale=se)

    ax.plot(x, y, color="#4C72B0", lw=2)
    ax.axvline(0,    color="red",   ls="--", label="H0: no lift")
    ax.axvline(lift, color="green", ls="-",  label=f"Observed lift={lift:+.4f}")
    ci_lo, ci_hi = result.confidence_interval
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="green", label="95% CI")
    ax.set_title(title)
    ax.set_xlabel("Absolute CTR lift")
    sig_label = "✓ Significant" if result.is_significant else "✗ Not significant"
    ax.set_ylabel(f"p={result.p_value:.4f}  {sig_label}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Power curve

# %%
sample_sizes = np.logspace(3, 6, 50).astype(int)
powers = [compute_power(n, baseline_rate=0.023, lift=0.003) for n in sample_sizes]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(sample_sizes, powers, color="#55A868", lw=2)
ax.axhline(0.80, color="red", ls="--", label="80% power threshold")
ax.axvline(n_required, color="gray", ls=":", label=f"Required n={n_required:,}")
ax.set_xscale("log")
ax.set_title("Statistical power vs sample size (MDE=0.3pp, α=0.05)")
ax.set_xlabel("Impressions per group")
ax.set_ylabel("Power (1 − β)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Multiple comparison correction (Bonferroni)

# %%
# Simulate testing 5 ad variants simultaneously
p_values = [0.03, 0.08, 0.001, 0.04, 0.12]
correction_df = bonferroni_correction(p_values, alpha=0.05)
print("Multiple comparison correction (Bonferroni):")
print(correction_df.to_string(index=False))

# %% [markdown]
# ---
# ## Part B — Affiliate Fraud Detection
#
# ### Business context
#
# In affiliate marketing, publishers earn a commission per valid click or conversion.
# This incentivises **click fraud**: bots or click farms generating fake impressions
# to inflate earnings. Isolation Forest detects these anomalies *without labelled data*.

# %% [markdown]
# ### 1. Generate click log (organic + fraud)

# %%
from sklearn.model_selection import train_test_split

df_clicks = generate_click_log(n_organic=15_000, n_fraud=750, seed=42)
print(f"Total clicks: {len(df_clicks):,} | Fraud rate: {df_clicks['is_fraud'].mean():.2%}")

df_train, df_test = train_test_split(df_clicks, test_size=0.3, random_state=42,
                                      stratify=df_clicks["is_fraud"])

# %% [markdown]
# ### 2. Train Isolation Forest on organic-only traffic

# %%
df_train_organic = df_train[df_train["is_fraud"] == 0]
model_if, scaler_if = train_isolation_forest(df_train_organic, contamination=0.05)
print("Isolation Forest trained on", len(df_train_organic), "organic clicks")

# %% [markdown]
# ### 3. Evaluate on labelled test set

# %%
metrics_fraud, fraud_pred, scores = evaluate_fraud_detection(model_if, scaler_if, df_test)
print("\n=== Fraud Detection Metrics ===")
for k, v in metrics_fraud.items():
    print(f"  {k:25s}: {v}")

# %% [markdown]
# ### 4. Feature distribution: organic vs detected fraud

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
features_plot = ["inter_click_time", "clicks_last_hour", "session_duration"]
titles = ["Inter-click time (s)", "Clicks in last hour", "Session duration (s)"]

for ax, feat, title in zip(axes, features_plot, titles):
    cap = df_test[feat].quantile(0.99)
    organic = df_test.loc[fraud_pred == 0, feat].clip(upper=cap)
    fraud_  = df_test.loc[fraud_pred == 1, feat].clip(upper=cap)
    ax.hist(organic, bins=40, alpha=0.6, color="#4C72B0", label="Organic", density=True)
    ax.hist(fraud_,  bins=40, alpha=0.6, color="#C44E52", label="Flagged", density=True)
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Publisher risk report

# %%
risk_report = publisher_risk_report(df_test, fraud_pred)
print("\n=== Top 15 High-Risk Publishers ===")
print(risk_report.head(15).to_string(index=False))

# %% [markdown]
# ## Summary
#
# ### A/B Testing
# - Proper pre-experiment power calculation prevents underpowered tests
# - Two-proportion z-test detects significant CTR uplift with correct Type I/II error control
# - Bonferroni correction required when testing multiple variants simultaneously
#
# ### Fraud Detection
# - Isolation Forest achieves strong recall on synthetic fraud without labelled training data
# - Key fraud signals: very short inter-click time, click bursts, near-zero session duration
# - Publisher-level aggregation enables actionable risk classification for affiliate management
