"""
generate_sample.py
Run this script once to create data/sample_data.csv for quick dashboard testing.
Usage: python generate_sample.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_utils import generate_synthetic_data

df = generate_synthetic_data(n_samples=5_000, seed=42)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_data.csv", index=False)
print(f"Saved data/sample_data.csv ({len(df):,} rows)")
