"""
data_loader.py
--------------
Responsible for loading the UFC fight dataset from the data/ directory.
Automatically detects the CSV file and reports basic info.
"""

import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def find_dataset() -> str | None:
    """
    Look for any CSV file inside the data/ directory.
    Returns the path to the first CSV found, or None.
    """
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".csv"):
            return os.path.join(DATA_DIR, fname)
    return None


def load_dataset(path: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load the UFC fight dataset.

    Parameters
    ----------
    path : str, optional
        Explicit path to CSV. If None, auto-detects from data/.
    verbose : bool
        Print summary info when True.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    if path is None:
        path = find_dataset()

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "\n[ERROR] No UFC dataset found in the data/ directory.\n"
            "Please place your UFC CSV file (e.g. ufc_fights.csv) inside:\n"
            f"  {os.path.abspath(DATA_DIR)}\n\n"
            "Expected columns include: date, winner, R_fighter, B_fighter,\n"
            "R_Height_cms, R_Weight_lbs, R_Reach_cms, R_Stance, R_age,\n"
            "R_wins, R_losses, R_current_win_streak, R_avg_SIG_STR_landed,\n"
            "R_avg_SIG_STR_pct, R_avg_TD_landed, R_avg_TD_pct,\n"
            "and equivalent B_ columns.\n"
            "A sample dataset is provided as data/ufc_fights.csv (mock data).\n"
        )

    df = pd.read_csv(path, low_memory=False)

    if verbose:
        print(f"\n[DATA] Loaded: {os.path.basename(path)}")
        print(f"[DATA] Shape : {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"[DATA] Columns ({len(df.columns)}): {list(df.columns[:10])} ...")

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print("\n--- Dataset Overview ---")
    print(f"Rows     : {len(df)}")
    print(f"Columns  : {len(df.columns)}")
    print(f"Missing  : {df.isnull().sum().sum()} total NaN values")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        valid_dates = df["date"].dropna()
        if not valid_dates.empty:
            print(f"Date range: {valid_dates.min().date()} -> {valid_dates.max().date()}")
    if "winner" in df.columns:
        print(f"Winner distribution:\n{df['winner'].value_counts()}")
