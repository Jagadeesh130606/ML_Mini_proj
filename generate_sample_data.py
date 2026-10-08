"""
generate_sample_data.py
-----------------------
Creates a realistic-looking mock UFC fight dataset for demonstration.

This is NOT real UFC data. It is a synthetic dataset with plausible
statistical distributions so the ML pipeline can run end-to-end.

ACADEMIC NOTE:
  Results from this mock data are for demonstration only and do NOT
  reflect real UFC fight predictability. For the real project, replace
  data/ufc_fights.csv with actual UFC fight data from Kaggle or UFCStats.

Usage:
  python generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_FIGHTS = 1500  # number of mock fights (paper used ~3355)

STANCES = ["Orthodox", "Southpaw", "Switch", "Open Stance"]
WEIGHT_CLASSES = [
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
    "Lightweight", "Welterweight", "Middleweight",
    "Light Heavyweight", "Heavyweight",
]


def generate_fighter_stats(rng: np.random.Generator, n: int, prefix: str) -> pd.DataFrame:
    """Generate plausible fighter stats for n fights."""
    data = {
        f"{prefix}_fighter"          : [f"Fighter_{prefix}_{i}" for i in range(n)],
        f"{prefix}_Height_cms"       : rng.normal(178, 8, n).clip(155, 215),
        f"{prefix}_Weight_lbs"       : rng.normal(170, 35, n).clip(115, 265),
        f"{prefix}_Reach_cms"        : rng.normal(182, 10, n).clip(155, 220),
        f"{prefix}_Stance"           : rng.choice(STANCES, n, p=[0.60, 0.28, 0.10, 0.02]),
        f"{prefix}_age"              : rng.normal(29, 4, n).clip(18, 45),
        f"{prefix}_wins"             : rng.integers(0, 30, n),
        f"{prefix}_losses"           : rng.integers(0, 15, n),
        f"{prefix}_current_win_streak": rng.integers(0, 10, n),
        f"{prefix}_longest_win_streak": rng.integers(0, 15, n),
        f"{prefix}_avg_SIG_STR_landed": rng.normal(4.5, 1.5, n).clip(0, 12),
        f"{prefix}_avg_SIG_STR_pct"  : rng.normal(0.45, 0.10, n).clip(0.1, 0.9),
        f"{prefix}_avg_TD_landed"    : rng.normal(1.5, 1.2, n).clip(0, 8),
        f"{prefix}_avg_TD_pct"       : rng.normal(0.40, 0.15, n).clip(0.0, 0.9),
        f"{prefix}_avg_SUB_ATT"      : rng.normal(0.8, 0.8, n).clip(0, 5),
        f"{prefix}_avg_opp_SIG_STR_landed": rng.normal(4.2, 1.4, n).clip(0, 12),
        f"{prefix}_avg_opp_SIG_STR_pct"  : rng.normal(0.43, 0.10, n).clip(0.1, 0.9),
        f"{prefix}_avg_opp_TD_landed"    : rng.normal(1.2, 1.0, n).clip(0, 7),
        f"{prefix}_avg_opp_TD_pct"       : rng.normal(0.38, 0.14, n).clip(0.0, 0.9),
        f"{prefix}_total_rounds_fought"   : rng.integers(1, 60, n),
        f"{prefix}_total_title_bouts"     : rng.integers(0, 5, n),
        f"{prefix}_win_by_Decision_Majority" : rng.integers(0, 5, n),
        f"{prefix}_win_by_Decision_Split"    : rng.integers(0, 5, n),
        f"{prefix}_win_by_Decision_Unanimous": rng.integers(0, 15, n),
        f"{prefix}_win_by_KO"               : rng.integers(0, 10, n),
        f"{prefix}_win_by_Submission"       : rng.integers(0, 10, n),
        f"{prefix}_win_by_TKO_Doctor_Stoppage": rng.integers(0, 3, n),
        f"{prefix}_draw"                    : rng.integers(0, 2, n),
    }
    return pd.DataFrame(data)


def simulate_winner(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate realistic outcomes based on feature differences.
    This adds a weak but non-random signal for the ML models to learn.
    """
    # Use a simple logistic model of the "true" advantage
    red_advantage = (
        0.3 * (df["R_wins"] - df["B_wins"]) / 30
        + 0.2 * (df["R_current_win_streak"] - df["B_current_win_streak"]) / 10
        + 0.15 * (df["R_avg_SIG_STR_landed"] - df["B_avg_SIG_STR_landed"]) / 10
        + 0.1 * (df["R_avg_TD_landed"] - df["B_avg_TD_landed"]) / 8
        + 0.05 * (df["R_Reach_cms"] - df["B_Reach_cms"]) / 30
    )
    prob_red = 1 / (1 + np.exp(-red_advantage))
    # Add noise to keep it realistic (~60% max signal)
    prob_red = 0.4 + 0.2 * prob_red  # squash to [0.4, 0.6] range
    winners = np.where(rng.random(len(df)) < prob_red, "Red", "Blue")
    return winners


def generate_mock_dataset(output_path: str) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    n = N_FIGHTS

    # Generate dates spanning 2010-2024
    start = pd.Timestamp("2010-01-01")
    end   = pd.Timestamp("2024-12-31")
    dates = pd.to_datetime(
        rng.integers(start.value, end.value, n)
    ).normalize().sort_values()

    red_df  = generate_fighter_stats(rng, n, "R")
    blue_df = generate_fighter_stats(rng, n, "B")

    df = pd.concat([red_df, blue_df], axis=1)
    df["date"]         = dates.values
    df["weight_class"] = rng.choice(WEIGHT_CLASSES, n)
    df["title_bout"]   = rng.choice([True, False], n, p=[0.08, 0.92])
    df["no_of_rounds"] = rng.choice([3, 5], n, p=[0.75, 0.25])

    df["winner"] = simulate_winner(df, rng)

    # Introduce ~5% missing values realistically
    for col in df.select_dtypes(include=[np.number]).columns:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[MOCK DATA] Generated {n} fights -> {output_path}")
    print(f"[MOCK DATA] Shape: {df.shape}")
    print(f"[MOCK DATA] Winner distribution:\n{df['winner'].value_counts()}")
    print("\n[NOTE] This is synthetic data for demonstration.")
    print("       Replace with real UFC data from Kaggle for research use.")


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(__file__), "data", "ufc_fights.csv"
    )
    generate_mock_dataset(out)
