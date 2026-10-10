"""
train_models.py
---------------
Defines and trains all four ML models used in this project, matching
the models from the reference paper:

  1. SGDClassifier      → represents the Generalized Linear / SGD approach
  2. MLPClassifier      → Multilayer Perceptron
  3. DecisionTreeClassifier
  4. GradientBoostingClassifier

Each model is wrapped in a scikit-learn Pipeline that includes:
  - Preprocessing (ColumnTransformer with imputation + scaling / encoding)
  - The classifier itself

WHY TIME-BASED SPLIT?
UFC fights happen chronologically. If we randomly shuffle the data, the
model may "see" future fights during training, which is unrealistic.
TimeSeriesSplit ensures that training always uses only PAST data relative
to the validation fold — the same condition a real predictor would face.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Preprocessing sub-pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes + scales numerical features
      - Imputes + one-hot encodes categorical features

    Fitting is done ONLY on training data inside the Pipeline.
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", numerical_transformer, numerical_cols)]
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_model_definitions(
    numerical_cols: list, categorical_cols: list
) -> dict:
    """
    Returns a dictionary of {model_name: sklearn Pipeline}.
    Each pipeline includes preprocessing + the classifier.
    """
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    models = {

        # ------------------------------------------------------------------
        # 1. SGD Classifier (Generalized Linear Model)
        # log_loss gives probabilistic output (logistic regression via SGD).
        # ------------------------------------------------------------------
        "SGD Classifier": Pipeline([
            ("preprocessor", build_preprocessor(numerical_cols, categorical_cols)),
            ("classifier", SGDClassifier(
                loss="log_loss",
                max_iter=1000,
                random_state=42,
                tol=1e-3,
                class_weight="balanced",
            )),
        ]),

        # ------------------------------------------------------------------
        # 2. MLP Classifier (Multilayer Perceptron)
        # Small architecture suitable for a mini-project dataset.
        # ------------------------------------------------------------------
        "MLP Classifier": Pipeline([
            ("preprocessor", build_preprocessor(numerical_cols, categorical_cols)),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )),
        ]),

        # ------------------------------------------------------------------
        # 3. Decision Tree
        # max_depth limits the tree to prevent severe overfitting.
        # ------------------------------------------------------------------
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor(numerical_cols, categorical_cols)),
            ("classifier", DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight="balanced",
            )),
        ]),

        # ------------------------------------------------------------------
        # 4. Gradient Boosting
        # Reference paper found this performed marginally best.
        # ------------------------------------------------------------------
        "Gradient Boosting": Pipeline([
            ("preprocessor", build_preprocessor(numerical_cols, categorical_cols)),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )),
        ]),
    }

    return models


# ---------------------------------------------------------------------------
# Training with TimeSeriesSplit cross-validation
# ---------------------------------------------------------------------------

def check_and_get_valid_tscv(
    X: pd.DataFrame, y: pd.Series, requested_splits: int = 5, verbose: bool = True
) -> TimeSeriesSplit:
    """
    Ensure each TimeSeriesSplit fold contains both binary target classes (0 and 1)
    in both training and validation slices. If any fold contains a single class,
    dynamically decrease the number of splits until all folds are multi-class.

    This guarantees valid cross-validation scores without NaN values.
    """
    for splits in range(requested_splits, 1, -1):
        tscv = TimeSeriesSplit(n_splits=splits)
        valid = True
        for train_idx, val_idx in tscv.split(X):
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            if len(np.unique(y_train_fold)) < 2 or len(np.unique(y_val_fold)) < 2:
                valid = False
                break
        if valid:
            if splits < requested_splits and verbose:
                print(
                    f"[CV NOTE] Reduced TimeSeriesSplit from {requested_splits} to {splits} "
                    f"to ensure both classes exist in all training/validation folds."
                )
            return tscv

    # Fallback minimum 2 splits
    return TimeSeriesSplit(n_splits=2)


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    numerical_cols: list,
    categorical_cols: list,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Train all four models using TimeSeriesSplit cross-validation.

    Parameters
    ----------
    X : pd.DataFrame  – feature matrix (already sorted chronologically)
    y : pd.Series     – binary target (1=Red wins, 0=Blue wins)
    numerical_cols, categorical_cols : column lists for the preprocessor
    n_splits : int    – number of TimeSeriesSplit folds

    Returns
    -------
    dict mapping model_name → {
        "pipeline"  : fitted Pipeline,
        "cv_scores" : np.array of per-fold accuracies,
        "cv_mean"   : float,
        "cv_std"    : float,
    }
    """
    models = get_model_definitions(numerical_cols, categorical_cols)
    tscv = check_and_get_valid_tscv(X, y, requested_splits=n_splits, verbose=verbose)
    results = {}

    total = len(models)
    for i, (name, pipeline) in enumerate(models.items(), 1):
        if verbose:
            print(f"\n[{i}/{total}] Training: {name} ...")

        cv_scores = cross_val_score(
            pipeline, X, y, cv=tscv, scoring="accuracy", n_jobs=-1
        )

        # Fit on the full training set for deployment
        pipeline.fit(X, y)

        results[name] = {
            "pipeline": pipeline,
            "cv_scores": cv_scores,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        }

        if verbose:
            print(f"       CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results

