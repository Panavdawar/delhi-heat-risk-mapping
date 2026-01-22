# === NEW: Vectorized risk labelling using np.select ===
import numpy as np

def assign_risk_labels(df):
    """
    Risk scheme (example thresholds) — change in config.yaml if needed.
    0 = low, 1 = moderate, 2 = high
    """
    conditions = [
        df['heat_index'] >= 45,
        (df['heat_index'] >= 38) & (df['heat_index'] < 45)
    ]
    choices = [2, 1]
    df['risk_label'] = np.select(conditions, choices, default=0)
    return df
# === END NEW ===
