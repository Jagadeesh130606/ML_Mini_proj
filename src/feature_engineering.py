"""
feature_engineering.py
-----------------------
Creates COMPARATIVE fight-level features from raw Red/Blue fighter columns.

Instead of feeding raw R_ and B_ statistics independently, we compute
the *difference* (Red − Blue) for each paired statistic. This directly
encodes which fighter has the advantage for each attribute and is the
approach used in the reference paper.

Key steps:
  1. Auto-detect paired R_ / B_ numeric columns.
  2. Compute diff features: diff_<stat> = R_<stat> - B_<stat>
  3. Encode stance: red_stance_enc - blue_stance_enc (ordinal) OR
     include both as one-hot (handled via ColumnTransformer in pipeline).
  4. Return:
       - X : feature DataFrame
       - feature_names : list of column names
       - categorical_cols : list of categorical columns for the pipeline
       - numerical_cols   : list of numerical columns for the pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Stat names that should be differenced (R_<name> vs B_<name>)
# Extend / trim this list to match your actual dataset columns.
# ---------------------------------------------------------------------------
DIFF_STAT_PAIRS = [
    "Height_cms",
    "Weight_lbs",
    "Reach_cms",
    "age",
    "wins",
    "losses",
    "current_win_streak",
    "avg_SIG_STR_landed",
    "avg_SIG_STR_pct",
    "avg_TD_landed",
    "avg_TD_pct",
    "avg_SUB_ATT",
    "avg_opp_SIG_STR_landed",
    "avg_opp_SIG_STR_pct",
    "avg_opp_TD_landed",
    "avg_opp_TD_pct",
    "longest_win_streak",
    "total_rounds_fought",
    "total_title_bouts",
    "win_by_Decision_Majority",
    "win_by_Decision_Split",
    "win_by_Decision_Unanimous",
    "win_by_KO",
    "win_by_Submission",
    "win_by_TKO_Doctor_Stoppage",
    "draw",
]

STANCE_COLS = ["R_Stance", "B_Stance"]


def _get_available_pairs(df: pd.DataFrame) -> list:
    """Return only those stat names where BOTH R_<stat> and B_<stat> exist."""
    available = []
    for stat in DIFF_STAT_PAIRS:
        r_col = f"R_{stat}"
        b_col = f"B_{stat}"
        if r_col in df.columns and b_col in df.columns:
            available.append(stat)
    return available


def create_diff_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    For each available stat pair, create: diff_<stat> = R_<stat> − B_<stat>.

    Returns a new DataFrame containing only the diff columns (and stance cols).
    """
    pairs = _get_available_pairs(df)

    if not pairs:
        raise ValueError(
            "[ERROR] No matching R_/B_ column pairs found in the dataset.\n"
            "Check that your CSV has columns named R_<stat> and B_<stat>.\n"
            f"Detected columns sample: {list(df.columns[:20])}"
        )

    if verbose:
        print(f"\n[FEAT] Creating {len(pairs)} difference features from column pairs.")

    feat_df = pd.DataFrame(index=df.index)

    for stat in pairs:
        r_col = f"R_{stat}"
        b_col = f"B_{stat}"
        # Convert to numeric (coerce errors → NaN)
        r_vals = pd.to_numeric(df[r_col], errors="coerce")
        b_vals = pd.to_numeric(df[b_col], errors="coerce")
        feat_df[f"diff_{stat}"] = r_vals - b_vals

    # Include stance as categorical columns if present
    for col in STANCE_COLS:
        if col in df.columns:
            feat_df[col] = df[col].astype(str).str.strip()

    if verbose:
        print(f"[FEAT] Feature matrix shape: {feat_df.shape}")

    return feat_df


def get_feature_columns(feat_df: pd.DataFrame) -> Tuple[list, list]:
    """
    Returns (numerical_cols, categorical_cols) from the feature DataFrame.
    """
    cat_cols = [c for c in STANCE_COLS if c in feat_df.columns]
    num_cols = [c for c in feat_df.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_feature_matrix(
    df: pd.DataFrame, verbose: bool = True
) -> Tuple[pd.DataFrame, list, list]:
    """
    Full feature engineering pipeline.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (comparative diff features + stance).
    numerical_cols : list
    categorical_cols : list
    """
    X = create_diff_features(df, verbose=verbose)
    num_cols, cat_cols = get_feature_columns(X)

    if verbose:
        print(f"[FEAT] Numerical features  : {len(num_cols)}")
        print(f"[FEAT] Categorical features: {len(cat_cols)} {cat_cols}")

    return X, num_cols, cat_cols
