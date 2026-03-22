"""
models.py
CTR prediction models:
  - LightGBM baseline
  - XGBoost baseline
  - DeepFM (PyTorch) — deep learning model for sparse categorical ad features
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_test, y_test, params: dict = None):
    """Train a LightGBM classifier and return the model + metrics."""
    default_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                 lgb.log_evaluation(period=-1)]

    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_test)
    metrics = {
        "auc":      round(roc_auc_score(y_test, y_pred), 4),
        "log_loss": round(log_loss(y_test, y_pred), 4),
    }
    return model, y_pred, metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_test, y_test, params: dict = None):
    """Train an XGBoost classifier and return the model + metrics."""
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "verbosity": 0,
        "n_jobs": -1,
        "n_estimators": 300,
        "early_stopping_rounds": 50,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc":      round(roc_auc_score(y_test, y_pred), 4),
        "log_loss": round(log_loss(y_test, y_pred), 4),
    }
    return model, y_pred, metrics


# ---------------------------------------------------------------------------
# DeepFM (PyTorch)
# ---------------------------------------------------------------------------

class CTRDataset(Dataset):
    """PyTorch Dataset wrapper for numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepFM(nn.Module):
    """
    DeepFM: combines Factorization Machine (FM) for second-order feature
    interactions with a Deep Neural Network for higher-order interactions.

    Reference: Guo et al. (2017) — https://arxiv.org/abs/1703.04247

    Architecture:
        Input → [FM component] ─┐
                                ├─ Sigmoid → CTR probability
        Input → [DNN component]─┘
    """

    def __init__(self, input_dim: int, embed_dim: int = 8, hidden_units: list = None,
                 dropout: float = 0.2):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256, 128, 64]

        # FM: first-order linear term
        self.fm_linear = nn.Linear(input_dim, 1, bias=True)

        # FM: second-order embedding (V matrix)
        self.fm_embed = nn.Linear(input_dim, embed_dim, bias=False)

        # DNN
        layers = []
        in_dim = input_dim
        for h in hidden_units:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.dnn = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FM first-order
        fm1 = self.fm_linear(x)                       # (B, 1)

        # FM second-order: 0.5 * (sum_square - square_sum)
        v = self.fm_embed(x)                           # (B, embed_dim)
        sum_sq  = (v ** 2).sum(dim=1, keepdim=True)   # (B, 1)
        sq_sum  = v.sum(dim=1, keepdim=True) ** 2      # (B, 1)
        fm2     = 0.5 * (sum_sq - sq_sum)              # (B, 1)

        # Deep component
        deep = self.dnn(x)                             # (B, 1)

        logit = fm1 + fm2 + deep
        return torch.sigmoid(logit).squeeze(1)


def train_deepfm(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray,  y_test: np.ndarray,
                 embed_dim: int = 8,
                 hidden_units: list = None,
                 dropout: float = 0.2,
                 lr: float = 1e-3,
                 epochs: int = 10,
                 batch_size: int = 4096,
                 device: str = None):
    """
    Train the DeepFM model.

    Returns: model, y_pred (probabilities), metrics dict, training history
    """
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = X_train.shape[1]
    model = DeepFM(input_dim, embed_dim, hidden_units, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_loader = DataLoader(CTRDataset(X_train, y_train.values if hasattr(y_train, "values") else y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(CTRDataset(X_test,  y_test.values  if hasattr(y_test,  "values") else y_test),
                              batch_size=batch_size * 2, shuffle=False)

    history = {"train_loss": [], "val_auc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        val_auc = roc_auc_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(round(avg_loss, 4))
        history["val_auc"].append(round(val_auc, 4))
        print(f"Epoch {epoch+1:02d}/{epochs} | loss: {avg_loss:.4f} | val AUC: {val_auc:.4f}")

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    metrics = {
        "auc":      round(roc_auc_score(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_pred), 4),
    }
    return model, y_pred, metrics, history


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_utils import generate_synthetic_data, prepare_dataset

    df = generate_synthetic_data(n_samples=20_000)
    X_train, X_test, y_train, y_test, features = prepare_dataset(df)

    print("=== LightGBM ===")
    _, _, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    print(lgb_metrics)

    print("\n=== XGBoost ===")
    _, _, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    print(xgb_metrics)

    print("\n=== DeepFM ===")
    X_tr_np = X_train.values.astype(np.float32)
    X_te_np = X_test.values.astype(np.float32)
    _, _, dfm_metrics, _ = train_deepfm(X_tr_np, y_train, X_te_np, y_test, epochs=3)
    print(dfm_metrics)
