# === NEW: Baseline training + feature importance plot ===
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_baseline_model(df, features, target, model_path="models/rf_baseline.joblib"):
    X = df[features]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    logging.info("Training RandomForest baseline...")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    logging.info("Saved model at %s", model_path)

    # === Feature importance plot (improved visuals) ===
    importances = model.feature_importances_
    sorted_idx = importances.argsort()
    palette = sns.color_palette("crest", len(features))
    plt.figure(figsize=(8, 4))
    plt.barh([features[i] for i in sorted_idx], importances[sorted_idx], color=palette)
    plt.xlabel("Relative importance")
    plt.title("Feature Importances — RF Baseline")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = Path("outputs/plots")
    output_path.mkdir(parents=True, exist_ok=True)
    feature_plot = output_path / "feature_importance_rf.png"
    plt.savefig(feature_plot, dpi=200)
    logging.info("Saved feature importance plot at %s", feature_plot)
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


def multivariate_analysis(df, features, target=None, output_dir="outputs/plots"):
    """Run a quick multivariate diagnostic: correlation heatmap and pairplot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the features (and optionally target).
    features : list[str]
        Feature column names to include in analysis.
    target : str, optional
        Target column to include for correlation context.
    output_dir : str, optional
        Directory to save plots into.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cols = features + ([target] if target and target in df.columns else [])
    data = df[cols].copy()

    # Correlation heatmap for quick readability
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    heatmap_path = output_path / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=240)
    logging.info("Saved correlation heatmap to %s", heatmap_path)

    # Pairplot (only on manageable feature counts)
    if len(features) <= 6:  # avoid huge grids
        pairplot = sns.pairplot(data, diag_kind="kde", corner=True, plot_kws={"alpha": 0.6, "s": 20})
        pairplot.fig.suptitle("Multivariate Pairwise Relationships", y=1.02)
        pairplot_path = output_path / "pairplot.png"
        pairplot.savefig(pairplot_path, dpi=200, bbox_inches="tight")
        logging.info("Saved pairplot to %s", pairplot_path)
    else:
        logging.info("Skipping pairplot (too many features: %d)", len(features))

    return {
        "correlation": corr,
        "heatmap_path": heatmap_path,
    }


def feature_target_correlation(df, features, target):
    """Compute correlations between each feature and the target for quick ranking."""
    available = [f for f in features if f in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    correlations = df[available + [target]].corr(numeric_only=True)[target].drop(target)
    sorted_corr = correlations.sort_values(ascending=False)
    logging.info("Computed feature-target correlations")
    return sorted_corr
# === END NEW ===
