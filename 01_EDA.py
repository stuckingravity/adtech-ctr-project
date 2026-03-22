# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 — Exploratory Data Analysis
#
# **Goal**: Understand the CTR dataset structure, feature distributions, and key signals
# before modelling.
#
# **Dataset**: Synthetic Avazu-style click log (or real Avazu data if downloaded).
#
# **Key questions:**
# 1. What is the overall CTR and how imbalanced is the dataset?
# 2. Which features correlate most with clicks?
# 3. Are there temporal patterns in click behaviour?

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.data_utils import generate_synthetic_data, engineer_features

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## 1. Load data

# %%
# Use synthetic data — replace with pd.read_csv("../data/train.csv") for real Avazu
df = generate_synthetic_data(n_samples=100_000, seed=42)
df = engineer_features(df)
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Class balance

# %%
ctr = df["click"].mean()
print(f"Overall CTR: {ctr:.3%}")
print(f"Click : {df['click'].sum():,}  |  No-click : {(df['click']==0).sum():,}")

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(["No click (0)", "Click (1)"], df["click"].value_counts().sort_index(),
       color=["#4C72B0", "#DD8452"])
ax.set_title("Click label distribution")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. CTR by key categorical features

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for ax, col, title in zip(
    axes.flatten(),
    ["banner_pos", "device_type", "device_conn_type", "hour_of_day"],
    ["Banner position", "Device type", "Connection type", "Hour of day"],
):
    ctr_series = df.groupby(col)["click"].mean().sort_values(ascending=False)
    ax.bar(ctr_series.index.astype(str), ctr_series.values)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"CTR by {title}")
    ax.set_xlabel(col)
    ax.set_ylabel("CTR")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Temporal patterns

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# CTR by hour of day
hour_ctr = df.groupby("hour_of_day")["click"].mean()
axes[0].plot(hour_ctr.index, hour_ctr.values, marker="o", color="#55A868")
axes[0].set_title("CTR by hour of day")
axes[0].set_xlabel("Hour")
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

# Volume by hour
hour_vol = df.groupby("hour_of_day")["click"].count()
axes[1].bar(hour_vol.index, hour_vol.values, color="#4C72B0", alpha=0.7)
axes[1].set_title("Impression volume by hour")
axes[1].set_xlabel("Hour")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Correlation heatmap (numeric features)

# %%
num_cols = ["banner_pos", "device_type", "device_conn_type",
            "C1", "C14", "C15", "C16", "C17", "C18",
            "hour_of_day", "is_weekend", "click"]

corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Key findings
#
# | Finding | Business implication |
# |---------|----------------------|
# | Banner position 1 has the highest CTR | Premium placement pricing justified |
# | Mobile devices show higher CTR than desktop | Mobile-first ad strategy recommended |
# | CTR peaks at specific hours (7–9am, 7–9pm) | Bid higher during peak hours |
# | Weekend CTR marginally lower | Adjust campaign budgets on weekends |
#
# → These signals will serve as features in the CTR prediction model (notebook 02).
