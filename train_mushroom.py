from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
DATA = HERE / "data" / "mushrooms.csv"
MODELS = HERE / "models"

TARGET = "class"
ALL_FEATURES = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def train(data_path: Path = DATA, models_dir: Path = MODELS, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *args, **kwargs: None)
    show = step if verbose else (lambda title: None)

    show("1. Load the data & run essential EDA")
    df = pd.read_csv(data_path)
    say(f"Total records: {len(df)} mushrooms, {len(df.columns) - 1} raw features")

    # Target breakdown
    counts = df[TARGET].value_counts()
    edible_pct = (df[TARGET] == "e").mean()
    poison_pct = (df[TARGET] == "p").mean()
    say(
        f"Class breakdown: Edible (e) = {counts.get('e', 0)} ({edible_pct:.1%}) | "
        f"Poisonous (p) = {counts.get('p', 0)} ({poison_pct:.1%})"
    )

    # Missing values check
    missing_q = (df == "?").sum()
    cols_with_q = missing_q[missing_q > 0]
    if len(cols_with_q) > 0:
        say(f"Missing values ('?'): {dict(cols_with_q)} -> handled cleanly as distinct category in OneHotEncoder")
    else:
        say("No missing values ('?') found.")

    # Drop constant column
    active_features = [col for col in ALL_FEATURES if col in df.columns]
    if "veil-type" in active_features and df["veil-type"].nunique() <= 1:
        say(f"Dropping 'veil-type': constant value '{df['veil-type'].iloc[0]}' across 100% of rows (zero variance)")
        active_features.remove("veil-type")
    say(f"Active features for modeling: {len(active_features)}")

    # Odor inspection (the strongest individual cue)
    if "odor" in df.columns:
        say("\nOdor vs. Edibility breakdown (% edible vs poisonous):")
        odor_table = pd.crosstab(df["odor"], df[TARGET], normalize="index") * 100
        say(odor_table.round(1).to_string())

    show("2. Split: train on 80%, keep 20% aside for the final test")
    X = df[active_features]
    y = (df[TARGET] == "p").astype(int)  # 1 = poisonous, 0 = edible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    say(f"train: {len(X_train)} rows, test: {len(X_test)} rows (stratified equal proportion)")

    show("3. A baseline: always predict 'edible' (zero intelligence)")
    baseline = np.zeros(len(y_test), dtype=int)
    base_acc = accuracy_score(y_test, baseline)
    base_recall = recall_score(y_test, baseline, zero_division=0)
    say(f"Naive accuracy: {base_acc:.1%}, but poisonous recall: {base_recall:.0%}")
    say("-> Accuracy appears ~52%, but it catches ZERO poisonous mushrooms! This is why recall matters.")

    show("4. Compare two models with 5-fold cross-validation (ROC AUC)")
    candidates = {
        "logistic regression": Pipeline(
            [
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random forest": Pipeline(
            [
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ]
        ),
    }

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model_pipeline in candidates.items():
        auc = cross_val_score(model_pipeline, X_train, y_train, cv=folds, scoring="roc_auc")
        scores[name] = auc.mean()
        say(f"{name:<20} ROC AUC {auc.mean():.4f} (+/- {auc.std():.4f})")

    best_name = max(scores, key=scores.get)
    best_pipeline = candidates[best_name]
    say(f"-> best: {best_name}")

    show("5. Choose the decision threshold for safety (on training data, never test)")
    proba_cv = cross_val_predict(best_pipeline, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    thresholds = np.arange(0.01, 0.90, 0.01)

    # We want zero false negatives (fn == 0) while keeping false alarms low
    valid_thresholds = []
    for t in thresholds:
        preds = (proba_cv >= t).astype(int)
        tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_train, preds).ravel()
        if fn_cv == 0:  # zero poisonous mushrooms missed!
            valid_thresholds.append((t, fp_cv))

    if valid_thresholds:
        # Pick the highest threshold that still guarantees 0 false negatives
        threshold = float(round(max(valid_thresholds, key=lambda x: x[0])[0], 2))
        say(f"Selected safety threshold: {threshold} (guarantees ZERO false negatives during cross-validation)")
    else:
        # Fallback to threshold that maximizes F1
        f1s = [f1_score(y_train, proba_cv >= t) for t in thresholds]
        threshold = float(round(thresholds[int(np.argmax(f1s))], 2))
        say(f"F1-optimal threshold: {threshold}")

    show("6. Train on all training data, evaluate once on the test set")
    best_pipeline.fit(X_train, y_train)
    proba = best_pipeline.predict_proba(X_test)[:, 1]
    predicted = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted)), 4),
        "f1": round(float(f1_score(y_test, predicted)), 4),
    }

    for metric_name, val in metrics.items():
        say(f"{metric_name:<10} {val:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    say(f"Poisonous mushrooms caught: {tp} of {tp + fn} (Recall: {metrics['recall']:.1%})")
    safety_label = "PERFECT SAFETY!" if fn == 0 else "WARNING!"
    say(f"Missed poisonous (False Negatives): {fn} ({safety_label})")
    say(f"False alarms (Edible flagged as poison): {fp} of {fp + tn} edible mushrooms")

    show("7. Which features matter most (permutation importance on test set)")
    importance = permutation_importance(
        best_pipeline, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1
    )
    ranked = sorted(zip(active_features, importance.importances_mean, strict=True), key=lambda x: -x[1])
    for feature, val in ranked[:5]:
        say(f"{feature:<25} {val:+.4f}")

    show("8. Save the model bundle")
    bundle = {
        "model": best_pipeline,
        "algorithm": best_name,
        "features": active_features,
        "threshold": threshold,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "mushroom.joblib"
    joblib.dump(bundle, save_path)
    say(f"Model bundle successfully saved to {save_path}")
    return bundle


if __name__ == "__main__":
    train()
