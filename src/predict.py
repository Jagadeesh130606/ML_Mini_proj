"""
predict.py
----------
Prediction interface for the UFC Fight Outcome Predictor.

predict_fight(red_stats, blue_stats, model_name)
  → returns {"winner": "Red"/"Blue", "red_prob": float, "blue_prob": float}

Also supports "Compare All Models" to get predictions from all four models.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

# Robust import: works whether called from project root or src/
_src_dir = os.path.dirname(__file__)
_project_root = os.path.dirname(_src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.feature_engineering import DIFF_STAT_PAIRS, STANCE_COLS


def build_fight_features(
    red_stats: dict,
    blue_stats: dict,
) -> pd.DataFrame:
    """
    Given two dictionaries of fighter stats, compute the comparative
    difference features expected by the trained pipeline.

    Parameters
    ----------
    red_stats  : dict  e.g. {"Height_cms": 180, "Weight_lbs": 155, ...}
    blue_stats : dict  same structure

    Returns
    -------
    pd.DataFrame with one row representing the fight-level feature vector.
    """
    row = {}

    for stat in DIFF_STAT_PAIRS:
        r_val = red_stats.get(stat, np.nan)
        b_val = blue_stats.get(stat, np.nan)
        try:
            row[f"diff_{stat}"] = float(r_val) - float(b_val)
        except (TypeError, ValueError):
            row[f"diff_{stat}"] = np.nan

    # Stance columns (categorical)
    for col, stats in zip(STANCE_COLS, [red_stats, blue_stats]):
        stance_key = col.replace("R_", "").replace("B_", "")  # → "Stance"
        row[col] = stats.get("Stance", "Orthodox")

    return pd.DataFrame([row])


def predict_fight(
    red_stats: dict,
    blue_stats: dict,
    trained_results: dict,
    model_name: Optional[str] = None,
) -> dict:
    """
    Predict the winner of a fight between two fighters.

    Parameters
    ----------
    red_stats, blue_stats : dicts of fighter pre-fight statistics
    trained_results       : output from train_all_models()
    model_name            : specific model to use, or None for all models

    Returns
    -------
    dict with:
      - "results": dict of {model_name: {"winner", "red_prob", "blue_prob"}}
      - "consensus": overall most-voted winner
    """
    X_fight = build_fight_features(red_stats, blue_stats)

    target_models = (
        {model_name: trained_results[model_name]}
        if model_name and model_name in trained_results
        else trained_results
    )

    output = {}
    red_votes = 0
    blue_votes = 0

    for name, info in target_models.items():
        pipeline = info["pipeline"]
        y_pred = pipeline.predict(X_fight)[0]

        red_prob = 0.5
        blue_prob = 0.5

        # Try predict_proba through the full pipeline first
        try:
            proba = pipeline.predict_proba(X_fight)[0]
            # sklearn classes_ is always sorted: [0=Blue, 1=Red]
            classes = pipeline.classes_
            red_idx  = list(classes).index(1) if 1 in classes else 1
            blue_idx = list(classes).index(0) if 0 in classes else 0
            red_prob  = float(proba[red_idx])
            blue_prob = float(proba[blue_idx])
        except Exception:
            # Fallback: decision_function → sigmoid
            try:
                score = pipeline.decision_function(X_fight)[0]
                red_prob  = float(1 / (1 + np.exp(-score)))
                blue_prob = 1.0 - red_prob
            except Exception:
                pass

        winner = "Red" if y_pred == 1 else "Blue"
        if winner == "Red":
            red_votes += 1
        else:
            blue_votes += 1

        output[name] = {
            "winner"  : winner,
            "red_prob": round(red_prob * 100, 1),
            "blue_prob": round(blue_prob * 100, 1),
        }

    consensus = "Red" if red_votes >= blue_votes else "Blue"

    return {"results": output, "consensus": consensus}
