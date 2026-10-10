"""
train.py
--------
Main training script for the UFC Fight Outcome Predictor.

This script runs the complete ML pipeline:
  1. Load dataset (auto-detects CSV in data/)
  2. Inspect & clean data
  3. Create binary target
  4. Feature engineering (comparative diff features)
  5. Chronological train/test split
  6. Train 4 ML models with TimeSeriesSplit CV
  7. Evaluate on held-out test set
  8. Save trained models (.pkl)
  9. Save evaluation results (CSV)
  10. Save feature importance (CSV)
  11. Print summary

Usage:
  python train.py

If no dataset exists, a mock dataset is generated automatically.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset, describe_dataset, find_dataset
from src.preprocessing import full_clean
from src.feature_engineering import build_feature_matrix
from src.train_models import train_all_models
from src.evaluate_models import evaluate_on_test_set, get_feature_importance, print_comparison_table


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
MODELS_PATH  = os.path.join(MODELS_DIR, "trained_models.pkl")
RESULTS_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")
FI_PATH      = os.path.join(RESULTS_DIR, "feature_importance.csv")
META_PATH    = os.path.join(RESULTS_DIR, "training_meta.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_FRACTION = 0.20   # last 20% of sorted fights used as test set
CV_SPLITS     = 5


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  UFC Fight Outcome Predictor — Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Load dataset
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading UFC dataset...")

    dataset_path = find_dataset()
    if dataset_path is None:
        print("[INFO] No dataset found in data/. Generating mock dataset...")
        from generate_sample_data import generate_mock_dataset
        mock_path = os.path.join(DATA_DIR, "ufc_fights.csv")
        generate_mock_dataset(mock_path)
        dataset_path = mock_path

    df_raw = load_dataset(dataset_path, verbose=True)
    describe_dataset(df_raw)

    # ------------------------------------------------------------------
    # STEP 2: Clean data + create target
    # ------------------------------------------------------------------
    print("\n[Step 2] Cleaning data and creating target variable...")
    df_clean = full_clean(df_raw)

    # ------------------------------------------------------------------
    # STEP 3: Sort chronologically & report dataset statistics
    # ------------------------------------------------------------------
    print("\n[Step 3] Sorting fights chronologically & reporting stats...")
    if "date" in df_clean.columns:
        df_clean = df_clean.sort_values("date").reset_index(drop=True)
        print(f"[INFO] Date range: {df_clean['date'].min().date()} "
              f"-> {df_clean['date'].max().date()}")

    red_rate_overall = df_clean["target"].mean()
    print(f"[STATS] Cleaned Fights Count  : {len(df_clean)}")
    print(f"[STATS] Date Range            : {df_clean['date'].min().date()} -> {df_clean['date'].max().date()}")
    print(f"[STATS] Red Win Rate (Overall): {red_rate_overall:.4f} ({red_rate_overall*100:.2f}%)")

    # ------------------------------------------------------------------
    # STEP 4: Feature engineering
    # ------------------------------------------------------------------
    print("\n[Step 4] Engineering comparative features...")
    X, numerical_cols, categorical_cols = build_feature_matrix(df_clean, verbose=True)
    y = df_clean["target"]

    print(f"\n[INFO] Final feature matrix: {X.shape}")
    print(f"[INFO] Target distribution — Red: {y.sum()} | Blue: {(y==0).sum()}")

    # ------------------------------------------------------------------
    # STEP 5: Chronological train/test split
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Chronological train/test split ({int((1-TEST_FRACTION)*100)}% / {int(TEST_FRACTION*100)}%)...")
    split_idx = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    red_rate_train = y_train.mean()
    red_rate_test  = y_test.mean()
    print(f"[INFO] Train: {len(X_train)} fights | Red Win Rate: {red_rate_train:.4f} ({red_rate_train*100:.2f}%)")
    print(f"[INFO] Test : {len(X_test)} fights | Red Win Rate: {red_rate_test:.4f} ({red_rate_test*100:.2f}%)")
    print("[INFO] WHY TIME-BASED SPLIT? Future fight data must never leak")
    print("       into training. We train on older fights and test on newer ones.")

    # ------------------------------------------------------------------
    # STEP 6: Train all four models
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Training 4 models with TimeSeriesSplit (k={CV_SPLITS})...")
    trained_results = train_all_models(
        X_train, y_train, numerical_cols, categorical_cols,
        n_splits=CV_SPLITS, verbose=True,
    )

    # ------------------------------------------------------------------
    # STEP 7: Evaluate on test set
    # ------------------------------------------------------------------
    print("\n[Step 7] Evaluating on held-out test set (including Always Red baseline)...")
    results_df = evaluate_on_test_set(
        trained_results, X_test, y_test, y_train=y_train, n_splits=CV_SPLITS, verbose=True
    )
    print_comparison_table(results_df)


    # ------------------------------------------------------------------
    # STEP 8: Feature importance
    # ------------------------------------------------------------------
    print("\n[Step 8] Extracting feature importance (tree-based models)...")
    # Get feature names after preprocessing
    sample_pipeline = list(trained_results.values())[0]["pipeline"]
    try:
        preprocessor = sample_pipeline.named_steps["preprocessor"]
        ohe_cols = []
        if categorical_cols:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            ohe_cols = list(ohe.get_feature_names_out(categorical_cols))
        all_feature_names = numerical_cols + ohe_cols
    except Exception:
        all_feature_names = numerical_cols + categorical_cols

    fi_df = get_feature_importance(trained_results, all_feature_names)
    print(f"[INFO] Top 10 features (Gradient Boosting):")
    if "Gradient Boosting" in fi_df.columns:
        print(fi_df[["Feature", "Gradient Boosting"]].head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # STEP 9: Save everything
    # ------------------------------------------------------------------
    print("\n[Step 9] Saving models and results...")

    with open(MODELS_PATH, "wb") as f:
        pickle.dump(trained_results, f)
    print(f"[SAVED] Models -> {MODELS_PATH}")

    results_save = results_df.drop(columns=["Confusion Matrix"], errors="ignore")
    results_save.to_csv(RESULTS_PATH, index=False)
    print(f"[SAVED] Comparison -> {RESULTS_PATH}")

    fi_df.to_csv(FI_PATH, index=False)
    print(f"[SAVED] Feature importance -> {FI_PATH}")

    meta = {
        "numerical_cols"  : numerical_cols,
        "categorical_cols": categorical_cols,
        "feature_names"   : all_feature_names,
        "test_size"       : len(X_test),
        "train_size"      : len(X_train),
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"[SAVED] Training meta -> {META_PATH}")

    # ------------------------------------------------------------------
    # STEP 10: Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Dataset  : {len(df_clean)} fights used")
    print(f"  Features : {X.shape[1]} comparative features")
    print(f"  Train    : {len(X_train)} | Test: {len(X_test)}")
    print()
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<22} Accuracy: {row['Accuracy']*100:.2f}%")
    print()
    print("  To launch the Streamlit app, run:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
