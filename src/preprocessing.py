"""
preprocessing.py
----------------
Handles all data-cleaning steps:
  - Parsing dates
  - Removing draws / no-contests
  - Creating the binary target (1 = Red wins, 0 = Blue wins)
  - Dropping post-fight leakage columns
  - Deduplication
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Column-name conventions
# (adapt these lists if your CSV uses different names)
# ---------------------------------------------------------------------------

# Columns that describe the outcome of the fight – must NEVER be used as
# input features to avoid data leakage.
LEAKAGE_COLUMNS = [
    "winner",
    "R_fighter",
    "B_fighter",
    "Referee",
    "location",
    "country",
    "Fight_type",
    "title_bout",
    "weight_class",
    "no_of_rounds",
    "B_current_lose_streak",
    "R_current_lose_streak",
    # post-fight per-fight stats (if present)
    "R_SIG_STR_landed",
    "R_SIG_STR_attempted",
    "B_SIG_STR_landed",
    "B_SIG_STR_attempted",
    "R_TOTAL_STR_landed",
    "R_TOTAL_STR_attempted",
    "B_TOTAL_STR_landed",
    "B_TOTAL_STR_attempted",
    "R_TD_landed",
    "R_TD_attempted",
    "B_TD_landed",
    "B_TD_attempted",
    "R_SUB_ATT",
    "B_SUB_ATT",
    "R_REV",
    "B_REV",
    "R_CTRL_time_seconds",
    "B_CTRL_time_seconds",
    "last_round",
    "last_round_time",
    "finish",
    "finish_details",
    "finish_round_time",
]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'date' column to datetime. Drop rows with unparseable dates."""
    if "date" not in df.columns:
        print("[WARN] No 'date' column found – chronological split will be skipped.")
        df["date"] = pd.NaT
        return df

    original = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    dropped = original - len(df)
    if dropped:
        print(f"[CLEAN] Dropped {dropped} rows with invalid dates.")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target column 'target':
        1 = Red fighter wins
        0 = Blue fighter wins
    Draws / No-contests / missing results are removed.
    """
    if "winner" not in df.columns:
        raise ValueError(
            "[ERROR] Dataset must contain a 'winner' column with values "
            "'Red' or 'Blue' (case-insensitive)."
        )

    original = len(df)

    # Normalise to lowercase
    df["winner_lower"] = df["winner"].astype(str).str.strip().str.lower()

    # Keep only red / blue outcomes
    valid_mask = df["winner_lower"].isin(["red", "blue"])
    removed = (~valid_mask).sum()
    if removed:
        removed_vals = df.loc[~valid_mask, "winner_lower"].value_counts().to_dict()
        print(f"[CLEAN] Removed {removed} rows with non-binary outcomes: {removed_vals}")

    df = df[valid_mask].copy()
    df["target"] = (df["winner_lower"] == "red").astype(int)
    df = df.drop(columns=["winner_lower"])

    print(f"[TARGET] Binary target created: {len(df)} usable fights "
          f"(removed {original - len(df)} from {original})")
    print(f"[TARGET] Red wins: {df['target'].sum()} | "
          f"Blue wins: {(df['target'] == 0).sum()}")

    return df


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any post-fight / identifier columns that would cause data leakage."""
    to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")
    if to_drop:
        print(f"[CLEAN] Dropped {len(to_drop)} leakage/identifier columns.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[CLEAN] Removed {before - after} duplicate rows.")
    return df


def full_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete cleaning pipeline in order:
      0. Normalize column names to lowercase
      1. Parse dates
      2. Remove duplicates
      3. Create target
      4. Remove leakage columns
    """
    print("\n=== Data Cleaning Pipeline ===")

    # Normalize column names: 'Winner' -> 'winner', 'Date' -> 'date', etc.
    df.columns = [c.lower() if c.lower() in ("winner", "date") else c for c in df.columns]

    df = parse_dates(df)
    df = remove_duplicates(df)
    df = create_target(df)
    df = remove_leakage_columns(df)
    print(f"[CLEAN] Final cleaned shape: {df.shape}")
    return df
