"""
data_utils.py
Data loading, synthetic data generation, and feature engineering for CTR prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Synthetic data generator (used when Avazu dataset is not downloaded)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic CTR dataset that mimics the structure of the Avazu dataset.
    Useful for quick testing without downloading the full 6GB Kaggle file.

    Columns produced:
        click          – binary label (0/1)
        hour           – YYMMDDHH format integer
        banner_pos     – ad banner position on page
        site_category  – encoded site category
        app_category   – encoded app category
        device_type    – device type (0=phone, 1=tablet, 2=desktop)
        device_conn_type – connection type
        C1, C14-C21    – anonymised categorical features (as in Avazu)
        user_click_history – engineered: historical CTR per user
        hour_of_day    – engineered: hour extracted from 'hour'
        is_weekend     – engineered: weekend flag
    """
    rng = np.random.default_rng(seed)

    hour_vals = rng.integers(14102100, 14103000, n_samples)
    banner_pos = rng.choice([0, 1, 2, 3, 5, 7], n_samples, p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
    site_category = rng.choice(list("ABCDEFGH"), n_samples)
    app_category  = rng.choice(list("WXYZ"), n_samples)
    device_type   = rng.choice([0, 1, 2], n_samples, p=[0.6, 0.2, 0.2])
    device_conn   = rng.choice([0, 2, 3, 5], n_samples)

    # Anonymised features
    C1  = rng.choice([1001, 1002, 1005, 1007, 1008, 1010], n_samples)
    C14 = rng.integers(100, 500, n_samples)
    C15 = rng.choice([50, 250, 320], n_samples)
    C16 = rng.choice([50, 90, 250, 480], n_samples)
    C17 = rng.integers(100, 3000, n_samples)
    C18 = rng.choice([0, 1, 2, 3], n_samples)
    C19 = rng.integers(0, 100, n_samples)
    C20 = rng.choice([-1, 100000, 100001, 100002], n_samples)
    C21 = rng.integers(0, 100, n_samples)

    # CTR influenced by banner position and device type (realistic signal)
    base_ctr = 0.17
    ctr_adj = (
        np.where(banner_pos == 1, 0.05, -0.02)
        + np.where(device_type == 0, 0.03, -0.01)
        + np.where(np.isin(site_category, list("AB")), 0.04, 0.0)
    )
    prob_click = np.clip(base_ctr + ctr_adj + rng.normal(0, 0.05, n_samples), 0.01, 0.99)
    click = rng.binomial(1, prob_click)

    df = pd.DataFrame({
        "click": click,
        "hour": hour_vals,
        "banner_pos": banner_pos,
        "site_category": site_category,
        "app_category": app_category,
        "device_type": device_type,
        "device_conn_type": device_conn,
        "C1": C1, "C14": C14, "C15": C15, "C16": C16,
        "C17": C17, "C18": C18, "C19": C19, "C20": C20, "C21": C21,
    })
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and interaction features to the raw dataframe.
    Works on both synthetic data and the real Avazu dataset.
    """
    df = df.copy()

    # Time features (Avazu 'hour' format: YYMMDDHH)
    df["hour_of_day"] = df["hour"] % 100
    day = (df["hour"] // 100) % 100
    df["is_weekend"] = (day % 7 >= 5).astype(int)
    df["day_of_week"] = day % 7

    # Interaction feature: banner position x device type
    df["banner_x_device"] = df["banner_pos"].astype(str) + "_" + df["device_type"].astype(str)

    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Label-encode a list of categorical columns in-place."""
    df = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Full preprocessing pipeline:
      1. Feature engineering
      2. Categorical encoding
      3. Train / test split (time-aware: last 20% of rows as test)

    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = engineer_features(df)

    cat_cols = ["site_category", "app_category", "banner_x_device"]
    df = encode_categoricals(df, cat_cols)

    feature_cols = [
        "banner_pos", "site_category", "app_category",
        "device_type", "device_conn_type",
        "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C21",
        "hour_of_day", "is_weekend", "day_of_week", "banner_x_device",
    ]
    # Remove features not present (e.g. C20 has -1 sentinel values)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["click"]

    # Time-aware split: keep temporal order
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=10_000)
    X_train, X_test, y_train, y_test, features = prepare_dataset(df)
    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")
    print(f"CTR (train) : {y_train.mean():.3f}")
    print(f"Features    : {features}")
