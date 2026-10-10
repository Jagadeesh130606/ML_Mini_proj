"""
evaluate_models.py
------------------
Computes and stores evaluation metrics for all trained models:
  - Accuracy, Precision, Recall, F1-score (weighted)
  - Confusion matrix
  - Cross-validation mean / std (passed in from training)
  - Feature importance (for tree-based models)

Uses a held-out chronological test set (last portion of sorted data).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


from sklearn.model_selection import TimeSeriesSplit


def evaluate_on_test_set(
    trained_results: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series = None,
    n_splits: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate each trained pipeline on the held-out test set,
    and include the Always Red (baseline).

    Parameters
    ----------
    trained_results : dict from train_models.train_all_models()
    X_test, y_test  : chronologically later test split
    y_train         : training set target (used for baseline CV score computation)
    n_splits        : number of splits for baseline CV evaluation

    Returns
    -------
    pd.DataFrame with one row per model containing all metrics.
    """
    rows = []

    # 1. Evaluate trained models
    for name, info in trained_results.items():
        pipeline = info["pipeline"]
        y_pred = pipeline.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0, average="weighted")
        rec  = recall_score(y_test, y_pred, zero_division=0, average="weighted")
        f1   = f1_score(y_test, y_pred, zero_division=0, average="weighted")
        cm   = confusion_matrix(y_test, y_pred)

        rows.append({
            "Model"        : name,
            "Accuracy"     : acc,
            "Precision"    : prec,
            "Recall"       : rec,
            "F1 Score"     : f1,
            "CV Mean"      : info["cv_mean"],
            "CV Std"       : info["cv_std"],
            "Confusion Matrix": cm.tolist(),
        })

        if verbose:
            print(f"\n--- {name} ---")
            print(f"  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
            print(f"  Precision     : {prec:.4f}")
            print(f"  Recall        : {rec:.4f}")
            print(f"  F1 Score      : {f1:.4f}")
            print(f"  CV Accuracy   : {info['cv_mean']:.4f} ± {info['cv_std']:.4f}")
            print(f"  Confusion Matrix:\n{cm}")

    # 2. Evaluate Always Red baseline
    y_pred_base = np.ones(len(y_test), dtype=int)
    acc_b  = accuracy_score(y_test, y_pred_base)
    prec_b = precision_score(y_test, y_pred_base, zero_division=0, average="weighted")
    rec_b  = recall_score(y_test, y_pred_base, zero_division=0, average="weighted")
    f1_b   = f1_score(y_test, y_pred_base, zero_division=0, average="weighted")
    cm_b   = confusion_matrix(y_test, y_pred_base, labels=[0, 1])

    cv_mean_b, cv_std_b = np.nan, np.nan
    if y_train is not None:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores_b = [
            accuracy_score(y_train.iloc[val_idx], np.ones(len(val_idx), dtype=int))
            for _, val_idx in tscv.split(y_train)
        ]
        cv_mean_b = float(np.mean(cv_scores_b))
        cv_std_b  = float(np.std(cv_scores_b))

    rows.append({
        "Model"        : "Always Red (baseline)",
        "Accuracy"     : acc_b,
        "Precision"    : prec_b,
        "Recall"       : rec_b,
        "F1 Score"     : f1_b,
        "CV Mean"      : cv_mean_b,
        "CV Std"       : cv_std_b,
        "Confusion Matrix": cm_b.tolist(),
    })

    if verbose:
        print(f"\n--- Always Red (baseline) ---")
        print(f"  Test Accuracy : {acc_b:.4f} ({acc_b*100:.2f}%)")
        print(f"  Precision     : {prec_b:.4f}")
        print(f"  Recall        : {rec_b:.4f}")
        print(f"  F1 Score      : {f1_b:.4f}")
        if not np.isnan(cv_mean_b):
            print(f"  CV Accuracy   : {cv_mean_b:.4f} ± {cv_std_b:.4f}")
        print(f"  Confusion Matrix:\n{cm_b}")

    results_df = pd.DataFrame(rows)
    return results_df



def get_feature_importance(
    trained_results: dict,
    feature_names: list,
) -> pd.DataFrame:
    """
    Extract feature importances for tree-based models.
    SGDClassifier uses coefficients; MLP has no direct importance.

    Returns
    -------
    pd.DataFrame with columns: Feature, <model_name_1>, <model_name_2>, ...
    """
    importance_df = pd.DataFrame({"Feature": feature_names})

    tree_models = ["Decision Tree", "Gradient Boosting"]

    for name in tree_models:
        if name not in trained_results:
            continue
        clf = trained_results[name]["pipeline"].named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            # feature_names might be longer if OHE expanded cat cols
            if len(importances) == len(feature_names):
                importance_df[name] = importances
            else:
                # Pad or truncate silently
                padded = np.zeros(len(feature_names))
                padded[:len(importances)] = importances[:len(feature_names)]
                importance_df[name] = padded

    # Sort by Gradient Boosting importance if available
    sort_col = "Gradient Boosting" if "Gradient Boosting" in importance_df.columns else None
    if sort_col:
        importance_df = importance_df.sort_values(sort_col, ascending=False)

    return importance_df.reset_index(drop=True)


def print_comparison_table(results_df: pd.DataFrame) -> None:
    """Pretty-print the model comparison table to the terminal."""
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    fmt = "{:<22} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Model", "Accuracy", "Precision", "Recall", "F1"))
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(fmt.format(
            row["Model"],
            f"{row['Accuracy']*100:.2f}%",
            f"{row['Precision']*100:.2f}%",
            f"{row['Recall']*100:.2f}%",
            f"{row['F1 Score']*100:.2f}%",
        ))
    print("=" * 70)
