# === NEW: Baseline training + feature importance plot ===
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging

def train_baseline_model(df, features, target, model_path="models/rf_baseline.joblib"):
    X = df[features]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    logging.info("Training RandomForest baseline...")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    logging.info("Saved model at %s", model_path)

    # === NEW: Feature importance plot ===
    importances = model.feature_importances_
    plt.figure(figsize=(6,3))
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.title("Feature importances (RF baseline)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance_rf.png")
    logging.info("Saved feature importance plot.")
    return model

def apply_risk_model(df):
    """Apply the trained risk model to predict risk scores."""
    model = joblib.load("rf_model.pkl")
    # Assuming features are temp, rh, heat_index
    features = ['temp', 'rh', 'heat_index']
    X = df[features]
    df['risk_score'] = model.predict_proba(X)[:, 1]  # probability of positive class
    logging.info("Applied risk model, added risk_score column")
    return df
# === END NEW ===
