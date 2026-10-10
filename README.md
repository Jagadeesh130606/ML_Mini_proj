# Applying Machine Learning Algorithms to Predict UFC Fight Outcomes

> **Machine Learning Subject — Mini Project**
> Department of Computer Science / Information Technology
> Academic Year: 2026

---

## Abstract

This mini project implements a binary classification system to predict the winner
of Ultimate Fighting Championship (UFC) bouts using pre-fight fighter statistics.
We replicate and adapt the methodology of the reference paper
*"Applying Machine Learning Algorithms to Predict UFC Fight Outcomes"*,
comparing four machine learning algorithms:
Stochastic Gradient Descent (SGD), Multilayer Perceptron (MLP),
Decision Tree, and Gradient Boosting.

The key insight is that instead of using raw Red and Blue fighter statistics
independently, we engineer *comparative* features (Red − Blue differences),
which directly encode the relative advantage of each fighter.
Time-series cross-validation is used to respect the chronological nature of fights.

---

## Problem Statement

Given pre-fight statistics for two UFC fighters (Fighter Red and Fighter Blue),
predict which fighter will win the bout.

This is a **binary classification** problem:
- **Class 1** → Red Fighter wins
- **Class 0** → Blue Fighter wins

---

## Objectives

1. Implement a complete ML pipeline: data loading → preprocessing → feature engineering → training → evaluation → prediction.
2. Compare four ML algorithms on the same real dataset against an Always Red baseline.
3. Use time-aware validation to avoid data leakage and temporal lookahead.
4. Build a Streamlit web interface for live demonstration.
5. Report honest, reproducible evaluation metrics.

---

## Dataset

| Property | Value |
|---|---|
| Source | Real UFC historical fight dataset |
| Location | `data/ufc_fights.csv` |
| Raw Dataset Rows | 6,012 fights (144 columns, dates 1994-03-11 to 2021-03-20) |
| Draws Removed | 110 fights (non-binary outcomes) |
| Pre-2010 Fights Removed | 1,244 fights (dropped due to early dataset label bias) |
| Final Cleaned Fights | 4,673 fights |
| Final Date Range | 2010-01-02 to 2021-03-20 |
| Features per fight | 28 comparative features (26 numerical diffs + 2 stance categoricals) |
| Target | Winner: Red (1) / Blue (0) |
| Red Win Rate (Overall) | 58.85% (2,750 Red wins / 1,923 Blue wins) |
| Train Set (80%) | 3,738 fights (2010-01-02 to 2019-04-13 \| 58.64% Red wins) |
| Test Set (20%) | 935 fights (2019-04-13 to 2021-03-20 \| 59.68% Red wins) |

---

## Data Issue We Found and Fixed

During initial data inspection, we identified a severe **label collection bias in pre-2010 fights**:
- Out of 1,244 fights before year 2010, **1,229 fights (98.8%) were recorded as Red wins** and 0 as Blue wins (with 15 draws).
- This synthetic artifact taught models a fake pattern and caused the first fold of `TimeSeriesSplit` cross-validation (1994–2009) to contain only a single class (Red wins), resulting in `NaN` cross-validation scores for SGD Classifier and Gradient Boosting.
- **Fix**: We set a named cutoff constant `MIN_YEAR = 2010` in `src/preprocessing.py`. Filtering out pre-2010 fights restored a realistic outcome distribution (~58.85% Red wins) and completely resolved the single-class fold issue, filling all cross-validation scores with valid metrics.

---

## Features Used

All numerical features are **comparative differences** (Red − Blue):

| Feature | Description |
|---|---|
| `diff_Height_cms` | Height difference (cm) |
| `diff_Reach_cms` | Reach difference (cm) |
| `diff_Weight_lbs` | Weight difference (lbs) |
| `diff_age` | Age difference (years) |
| `diff_wins` | Total wins difference |
| `diff_losses` | Total losses difference |
| `diff_current_win_streak` | Current win streak difference |
| `diff_longest_win_streak` | Longest win streak difference |
| `diff_total_rounds_fought` | Total rounds fought difference |
| `diff_total_title_bouts` | Total title bouts difference |
| `diff_avg_SIG_STR_landed` | Avg significant strikes/min difference |
| `diff_avg_SIG_STR_pct` | Strike accuracy difference |
| `diff_avg_TD_landed` | Avg takedowns/15min difference |
| `diff_avg_TD_pct` | Takedown accuracy difference |
| `diff_avg_SUB_ATT` | Submission attempts difference |
| `diff_win_by_KO/TKO` | KO/TKO wins difference |
| `diff_win_by_Submission` | Submission wins difference |
| `diff_win_by_Decision_Unanimous` | Unanimous decision wins difference |
| `R_Stance`, `B_Stance` | Fighter stance (categorical) |

---

## Methodology

```
Real UFC Dataset (6,012 raw rows)
   ↓
Data Cleaning (remove 110 draws, parse dates, drop 1,244 pre-2010 fights, drop duplicates)
   ↓
Target Creation (1 = Red wins, 0 = Blue wins)
   ↓
Chronological Sort (2010-01-02 to 2021-03-20)
   ↓
Feature Engineering (28 Red − Blue comparative diff & stance features)
   ↓
Train/Test Split (80% older fights train [3,738] | 20% newer fights test [935])
   ↓
Preprocessing Pipeline (Median Imputation → StandardScaler → OneHotEncoder)
   ↓
┌────────────────────────┐
│ 1. SGD Classifier      │
│ 2. MLP Classifier      │  ← TimeSeriesSplit CV (k=5)
│ 3. Decision Tree       │
│ 4. Gradient Boosting   │
└────────────────────────┘
   ↓
Evaluation (Accuracy, Precision, Recall, F1, CV Mean ± Std vs Always Red Baseline)
   ↓
Streamlit UI (live prediction)
```

---

## Algorithms

### 1. SGD Classifier (Generalized Linear Model)
- Loss: `log_loss` (logistic regression via SGD)
- Represents the linear boundary approach from the reference paper

### 2. MLP Classifier (Multilayer Perceptron)
- Architecture: (64, 32) hidden layers, ReLU activation
- Early stopping to prevent overfitting

### 3. Decision Tree
- Max depth: 6 (regularized to prevent overfitting)
- Min samples per leaf: 10

### 4. Gradient Boosting
- 100 estimators, learning rate 0.1, max depth 4
- Reference paper found this performed marginally best

---

## System Architecture

```
ufc_fight_prediction/
│
├── data/
│   └── ufc_fights.csv          ← Real UFC dataset
│
├── src/
│   ├── data_loader.py           ← Auto-detect & load CSV
│   ├── preprocessing.py         ← Clean, filter pre-2010, create target
│   ├── feature_engineering.py   ← Red−Blue difference features
│   ├── train_models.py          ← 4 ML pipelines + TimeSeriesSplit CV
│   ├── evaluate_models.py       ← Metrics + baseline + feature importance
│   └── predict.py               ← Prediction interface
│
├── models/
│   └── trained_models.pkl       ← Saved pipelines
│
├── results/
│   ├── model_comparison_before.csv ← Historical baseline comparison before fix
│   ├── model_comparison.csv     ← Accuracy, Precision, Recall, F1, CV Mean/Std
│   └── feature_importance.csv   ← Tree-based importances
│
├── app.py                        ← Streamlit web app
├── train.py                      ← Training pipeline
├── generate_sample_data.py       ← Mock dataset generator
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

- **Accuracy** — Overall correct predictions / total predictions
- **Precision** — Weighted precision across classes
- **Recall** — Weighted recall across classes
- **F1 Score** — Weighted harmonic mean of Precision and Recall
- **CV Accuracy** — TimeSeriesSplit cross-validation (mean ± std)

---

## Results

Evaluation on held-out chronological test set (935 fights, 2019-04-13 to 2021-03-20):

| Model | Accuracy | Precision | Recall | F1 Score | CV Mean | CV Std |
|---|---|---|---|---|---|---|
| **SGD Classifier** | 54.44% | 55.95% | 54.44% | 54.86% | 55.54% | ±2.69% |
| **MLP Classifier** | **62.35%** | **61.02%** | **62.35%** | **60.57%** | **59.94%** | ±2.39% |
| **Decision Tree** | 59.14% | 59.14% | 59.14% | 59.14% | 53.16% | ±2.26% |
| **Gradient Boosting** | 59.89% | 58.51% | 59.89% | 58.58% | 58.72% | ±2.38% |
| **Always Red (baseline)** | 59.68% | 35.62% | 59.68% | 44.61% | 58.27% | ±3.02% |

The best performing model, **MLP Classifier**, achieved a test accuracy of **62.35%** (and **60.57% F1 score**), outperforming the Always Red baseline (59.68% accuracy, 44.61% F1 score) by **2.67 percentage points** in accuracy and **15.96 percentage points** in F1 score.

---

## How to Run

### Prerequisites
- Python 3.9+
- pip

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the models
```bash
python train.py
```

### Step 3: Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## Future Scope

1. **Advanced features**: Days since last fight, fighter Elo rating, weight-class specific trends
2. **Better models**: XGBoost, LightGBM, Random Forest ensemble
3. **Hyperparameter tuning**: Optuna / GridSearchCV
4. **Imbalance handling**: Threshold tuning for decision boundaries

---

## Academic Note

This project is a **Machine Learning Subject Mini Project**. It is NOT a final-year or major project.

- Results are evaluated on real UFC fight statistics from 2010 to 2021
- This system should NOT be used for betting or real-world decision making
- The reference paper is used as inspiration for algorithm selection and methodology

---

## References

1. *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes* (Reference Paper)
2. Scikit-learn documentation: https://scikit-learn.org
3. Streamlit documentation: https://streamlit.io
4. UFC fight data: https://www.kaggle.com/rajeevw/ufcdata
