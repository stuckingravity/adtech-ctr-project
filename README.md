# Ad Click-Through Rate (CTR) Prediction & A/B Testing Simulator

A end-to-end data science project simulating a real-world Ad Tech pipeline:  
**CTR prediction** with machine learning and deep learning models, **A/B testing** with statistical significance analysis, and **affiliate fraud detection** via anomaly detection — all visualized in an interactive Streamlit dashboard.

> Built to explore the core problematics of the online advertising ecosystem, including performance prediction, experiment design, and fraud detection in affiliate marketing.

---

## Project Structure

```
adtech-ctr-project/
├── data/                    # Raw and processed data (not tracked by git)
│   └── sample_data.csv      # Small synthetic sample for quick testing
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory data analysis
│   ├── 02_modeling.ipynb     # CTR model training & comparison
│   └── 03_ab_testing.ipynb   # A/B testing & anomaly detection
├── src/
│   ├── data_utils.py         # Data loading and feature engineering
│   ├── models.py             # LightGBM, XGBoost, DeepFM models
│   ├── ab_testing.py         # Statistical testing framework
│   └── anomaly_detection.py  # Isolation Forest fraud detection
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
├── tests/
│   └── test_ab_testing.py    # Unit tests for statistical functions
├── requirements.txt
└── README.md
```

---

## Key Features

### 1. CTR Prediction (Ad Tech)
- Feature engineering on user behavior, ad attributes, and time signals
- Baseline: **LightGBM** and **XGBoost** with log-loss and AUC evaluation
- Deep learning: **DeepFM** (PyTorch) — combines factorization machines with deep neural networks for sparse ad features
- Model comparison dashboard with ROC curves

### 2. A/B Testing Framework
- Simulates online advertising experiments (e.g., testing two ad creatives)
- Statistical significance via **t-test** and **chi-square test**
- Computes **p-value**, **confidence intervals**, and **statistical power**
- Detects early stopping risk and multiple comparison correction (Bonferroni)

### 3. Affiliate Fraud / Anomaly Detection
- Simulates click fraud patterns common in affiliate marketing
- **Isolation Forest** to detect anomalous click behavior
- Evaluation: precision, recall, F1 on synthetic fraud labels

### 4. Streamlit Dashboard
- Interactive model comparison (AUC, log-loss)
- A/B test result visualizer with live p-value computation
- Anomaly detection report with flagged user segments

---

## Dataset

This project uses the [Avazu CTR Prediction dataset](https://www.kaggle.com/c/avazu-ctr-prediction) from Kaggle.

**To download:**
```bash
kaggle competitions download -c avazu-ctr-prediction
unzip avazu-ctr-prediction.zip -d data/
```

A synthetic sample (`data/sample_data.csv`) is included for quick testing without downloading the full dataset.

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/adtech-ctr-project.git
cd adtech-ctr-project

pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb

# Or launch the dashboard directly
streamlit run dashboard/app.py
```

---

## Results

| Model | AUC | Log-Loss |
|-------|-----|----------|
| LightGBM (baseline) | 0.762 | 0.401 |
| XGBoost | 0.758 | 0.408 |
| DeepFM (PyTorch) | 0.781 | 0.389 |

A/B Test (CTR uplift from creative variant B):
- Control CTR: 2.3% — Variant CTR: 2.8%
- p-value: 0.031 → **statistically significant at α=0.05**

Anomaly Detection:
- Isolation Forest F1 on synthetic fraud: **0.83**

---

## Tech Stack

`Python` `PyTorch` `LightGBM` `XGBoost` `scikit-learn` `pandas` `numpy` `scipy` `Streamlit` `Matplotlib` `Seaborn`

---

## Business Context

This project simulates problems found in **performance advertising and affiliate marketing**:
- Publishers earn commissions per click/conversion → incentivizes click fraud
- Advertisers run A/B tests on creatives, landing pages, and audience segments
- CTR prediction is the core task in programmatic ad serving (ranking ads in real time)

---

## Author

**Bingjing YUE** — MSc Data Sciences & Business Analytics, ESSEC / CentraleSupélec  
[LinkedIn](https://linkedin.com) · [b00809909@essec.edu](mailto:b00809909@essec.edu)
