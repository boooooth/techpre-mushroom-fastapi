"""Inference logic and model loader for Mushroom Toxicity."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))


def load(name: str = "mushroom") -> dict | None:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{path.name} was trained with scikit-learn {bundle.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed.",
            stacklevel=2,
        )
    return bundle


def risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def predict_mushroom(bundle: dict, specimens: list[dict]) -> list[dict]:
    # Normalize underscore keys to hyphenated keys matching bundle['features']
    cleaned_rows = []
    for item in specimens:
        row = {k.replace("_", "-"): v for k, v in item.items()}
        cleaned_rows.append(row)

    frame = pd.DataFrame(cleaned_rows, columns=bundle["features"])
    probabilities = bundle["model"].predict_proba(frame)[:, 1]
    threshold = bundle["threshold"]

    results = []
    for p in probabilities:
        prob = round(float(p), 4)
        is_poison = bool(prob >= threshold)
        risk = risk_level(prob, threshold)

        if is_poison:
            rec = "DO NOT EAT. Specimen is classified as poisonous/toxic."
        elif risk == "medium":
            rec = "CAUTION. Mild risk detected. Consumption not recommended without laboratory verification."
        else:
            rec = "EDIBLE. Specimen exhibits safe characteristics."

        results.append({
            "is_poisonous": is_poison,
            "poison_probability": prob,
            "risk": risk,
            "threshold": threshold,
            "recommendation": rec,
        })
    return results
