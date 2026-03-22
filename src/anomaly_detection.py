"""
anomaly_detection.py
Affiliate click fraud detection using Isolation Forest.

In affiliate marketing, publishers earn a commission for each click or conversion.
This incentivises fraudulent behaviour: bots, click farms, or cookie stuffing that
generate fake clicks without real purchase intent.

This module:
  1. Engineers behavioural features that distinguish fraudulent from organic clicks
  2. Trains an Isolation Forest (unsupervised) on clean traffic
  3. Evaluates on a labelled dataset with synthetic fraud injected
  4. Provides a user-segment report flagging high-risk publishers / IPs
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)


# ---------------------------------------------------------------------------
# Synthetic click log generator
# ---------------------------------------------------------------------------

def generate_click_log(
    n_organic: int = 10_000,
    n_fraud: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic click log with organic and fraudulent entries.

    Organic click characteristics:
        - Spread across multiple publishers and IPs
        - Inter-click time follows exponential distribution (human pace)
        - Normal conversion rate (~2%)
        - Device mix: mobile-heavy

    Fraud characteristics (click farm / bot):
        - Concentrated on few publisher IDs
        - Very short inter-click times (< 1 second)
        - Near-zero conversion rate
        - Unrealistic click burst in short windows
    """
    rng = np.random.default_rng(seed)

    # --- Organic clicks ---
    organic = pd.DataFrame({
        "publisher_id":      rng.choice(range(1, 200), n_organic),
        "ip_hash":           rng.choice(range(1, 5000), n_organic),
        "inter_click_time":  rng.exponential(scale=120, size=n_organic),   # seconds
        "clicks_last_hour":  rng.poisson(lam=2, size=n_organic),
        "conversions_24h":   rng.binomial(10, 0.02, n_organic),
        "device_type":       rng.choice([0, 1, 2], n_organic, p=[0.6, 0.2, 0.2]),
        "unique_ips_pub_1h": rng.poisson(lam=15, size=n_organic),
        "session_duration":  rng.gamma(shape=3, scale=60, size=n_organic),  # seconds
        "is_fraud":          np.zeros(n_organic, dtype=int),
    })

    # --- Fraudulent clicks ---
    fraud = pd.DataFrame({
        "publisher_id":      rng.choice(range(1, 10), n_fraud),      # few publishers
        "ip_hash":           rng.choice(range(1, 30), n_fraud),       # few IPs
        "inter_click_time":  rng.exponential(scale=0.5, size=n_fraud),  # near-instant
        "clicks_last_hour":  rng.poisson(lam=150, size=n_fraud),      # burst
        "conversions_24h":   np.zeros(n_fraud, dtype=int),            # no conversions
        "device_type":       rng.choice([0, 1, 2], n_fraud, p=[0.98, 0.01, 0.01]),
        "unique_ips_pub_1h": rng.poisson(lam=2, size=n_fraud),        # few unique IPs
        "session_duration":  rng.exponential(scale=1.5, size=n_fraud),  # near-zero
        "is_fraud":          np.ones(n_fraud, dtype=int),
    })

    df = pd.concat([organic, fraud], ignore_index=True).sample(frac=1, random_state=seed)
    return df


# ---------------------------------------------------------------------------
# Feature engineering for fraud detection
# ---------------------------------------------------------------------------

FRAUD_FEATURES = [
    "inter_click_time",
    "clicks_last_hour",
    "conversions_24h",
    "unique_ips_pub_1h",
    "session_duration",
]


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that amplify fraud signal."""
    df = df.copy()

    # Conversion rate per publisher (low = suspicious)
    df["conversion_rate"] = df["conversions_24h"] / (df["clicks_last_hour"] + 1)

    # Click velocity: clicks per unit time (high = suspicious)
    df["click_velocity"] = df["clicks_last_hour"] / (df["inter_click_time"] + 1e-3)

    # IP concentration: low unique IPs relative to click volume = suspicious
    df["ip_concentration"] = df["clicks_last_hour"] / (df["unique_ips_pub_1h"] + 1)

    return df


EXTENDED_FRAUD_FEATURES = FRAUD_FEATURES + [
    "conversion_rate", "click_velocity", "ip_concentration"
]


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def train_isolation_forest(
    df_train: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple:
    """
    Train Isolation Forest on (presumed) clean traffic.

    contamination : expected proportion of anomalies (tune to your domain)
    Returns: model, scaler
    """
    df_train = engineer_fraud_features(df_train)
    X = df_train[EXTENDED_FRAUD_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler


def evaluate_fraud_detection(
    model: IsolationForest,
    scaler: StandardScaler,
    df_test: pd.DataFrame,
) -> dict:
    """
    Evaluate fraud detection on a labelled test set.

    Isolation Forest returns:
        +1 = inlier  (organic click)
        -1 = outlier (potential fraud)

    We map -1 → 1 (fraud predicted) for standard metric computation.
    """
    df_test = engineer_fraud_features(df_test)
    X = df_test[EXTENDED_FRAUD_FEATURES].copy()
    X_scaled = scaler.transform(X)

    raw_preds  = model.predict(X_scaled)          # +1 / -1
    scores     = -model.score_samples(X_scaled)   # higher = more anomalous
    fraud_pred = (raw_preds == -1).astype(int)

    y_true = df_test["is_fraud"].values

    metrics = {
        "precision": round(precision_score(y_true, fraud_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, fraud_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, fraud_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, scores), 4),
        "fraud_flagged": int(fraud_pred.sum()),
        "total_clicks":  len(fraud_pred),
    }
    return metrics, fraud_pred, scores


# ---------------------------------------------------------------------------
# Publisher risk report
# ---------------------------------------------------------------------------

def publisher_risk_report(df: pd.DataFrame, fraud_pred: np.ndarray) -> pd.DataFrame:
    """
    Aggregate fraud predictions at the publisher level to identify high-risk partners.

    Returns a DataFrame sorted by fraud_rate descending.
    """
    report_df = df.copy()
    report_df["fraud_predicted"] = fraud_pred

    agg = report_df.groupby("publisher_id").agg(
        total_clicks=("fraud_predicted", "count"),
        flagged_clicks=("fraud_predicted", "sum"),
        avg_inter_click_time=("inter_click_time", "mean"),
        avg_session_duration=("session_duration", "mean"),
    ).reset_index()

    agg["fraud_rate"] = (agg["flagged_clicks"] / agg["total_clicks"]).round(4)
    agg["risk_level"] = pd.cut(
        agg["fraud_rate"],
        bins=[-0.001, 0.05, 0.20, 1.001],
        labels=["Low", "Medium", "High"],
    )
    return agg.sort_values("fraud_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df = generate_click_log(n_organic=10_000, n_fraud=500)
    print(f"Dataset: {len(df):,} clicks | fraud rate: {df['is_fraud'].mean():.2%}")

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42,
                                         stratify=df["is_fraud"])

    # Train on organic-only data (unsupervised setting)
    df_train_organic = df_train[df_train["is_fraud"] == 0]
    model, scaler = train_isolation_forest(df_train_organic, contamination=0.05)

    metrics, fraud_pred, scores = evaluate_fraud_detection(model, scaler, df_test)
    print("\n=== Fraud Detection Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    report = publisher_risk_report(df_test, fraud_pred)
    print("\n=== Top 10 High-Risk Publishers ===")
    print(report.head(10).to_string(index=False))
