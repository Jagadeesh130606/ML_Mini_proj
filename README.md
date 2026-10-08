# Applying Machine Learning Algorithms to Predict UFC Fight Outcomes

> **Machine Learning Subject — Mini Project**
> Department of Computer Science / Information Technology
> Academic Year: 2025-2026

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
2. Compare four ML algorithms on the same dataset.
3. Use time-aware validation to avoid data leakage.
4. Build a simple Streamlit web interface for live demonstration.
5. Report honest, reproducible evaluation metrics.

---

## Dataset

| Property | Value |
|---|---|
| Source | UFC fight statistics (CSV format) |
| Location | `data/ufc_fights.csv` |
| Expected rows | ~1500–3355 fights |
| Features per fight | ~26 comparative features (after engineering) |
| Target | Winner: Red (1) / Blue (0) |
| Date range | 2010–2024 |

**To use a real dataset:**
Place your UFC CSV file in the `data/` directory.
The system auto-detects and adapts to the column structure.

Recommended source: [Kaggle — UFC Fight Historical Data](https://www.kaggle.com/rajeevw/ufcdata)

If no dataset is found, a synthetic mock dataset is generated automatically.

---

## Features Used

All features are **comparative differences** (Red − Blue):

| Feature | Description |
|---|---|
| `diff_Height_cms` | Height difference (cm) |
| `diff_Reach_cms` | Reach difference (cm) |
| `diff_Weight_lbs` | Weight difference (lbs) |
| `diff_age` | Age difference (years) |
| `diff_wins` | Total wins difference |
| `diff_losses` | Total losses difference |
| `diff_current_win_streak` | Current win streak difference |
| `diff_avg_SIG_STR_landed` | Avg significant strikes/min difference |
| `diff_avg_SIG_STR_pct` | Strike accuracy difference |
| `diff_avg_TD_landed` | Avg takedowns/15min difference |
| `diff_avg_TD_pct` | Takedown accuracy difference |
| `diff_avg_SUB_ATT` | Submission attempts difference |
| `R_Stance`, `B_Stance` | Fighter stance (categorical) |
| + more... | Additional win-method and round statistics |

---

## Methodology

```
Dataset
   ↓
Data Cleaning (remove draws, parse dates, drop duplicates)
   ↓
Target Creation (1 = Red wins, 0 = Blue wins)
   ↓
Chronological Sort (by fight date)
   ↓
Feature Engineering (Red − Blue differences)
   ↓
Train/Test Split (80% older fights train | 20% newer fights test)
   ↓
Preprocessing Pipeline (Imputation → Scaling → OHE)
   ↓
┌────────────────────────┐
│ 1. SGD Classifier      │
│ 2. MLP Classifier      │  ← TimeSeriesSplit CV (k=5)
│ 3. Decision Tree       │
│ 4. Gradient Boosting   │
└────────────────────────┘
   ↓
Evaluation (Accuracy, Precision, Recall, F1)
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
- Reference paper found this performed marginally best (~60% accuracy)

---

## System Architecture

```
ufc_fight_prediction/
│
├── data/
│   └── ufc_fights.csv          ← Place real dataset here
│
├── src/
│   ├── data_loader.py           ← Auto-detect & load CSV
│   ├── preprocessing.py         ← Clean, create target
│   ├── feature_engineering.py   ← Red−Blue difference features
│   ├── train_models.py          ← 4 ML pipelines + TimeSeriesSplit CV
│   ├── evaluate_models.py       ← Metrics + feature importance
│   └── predict.py               ← Prediction interface
│
├── models/
│   └── trained_models.pkl       ← Saved pipelines
│
├── results/
│   ├── model_comparison.csv     ← Accuracy, Precision, Recall, F1
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
- **Precision** — Of all predicted Red wins, how many were actually Red wins
- **Recall** — Of all actual Red wins, how many did we correctly predict
- **F1 Score** — Harmonic mean of Precision and Recall
- **CV Accuracy** — TimeSeriesSplit cross-validation (mean ± std)

---

## Results

> **Note**: Results below are from the synthetic mock dataset.
> Replace `data/ufc_fights.csv` with real UFC data for research-grade results.
> Results will differ from the reference paper's reported ~60% accuracy since we use a different dataset.

After running `python train.py`, results are saved to `results/model_comparison.csv`.

---

## Screenshots

*(Add screenshots of the Streamlit app here for your college submission.)*

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
cd ufc_fight_prediction
python train.py
```

### Step 3: Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

### Using a Real Dataset
1. Download the UFC dataset from Kaggle (see link above)
2. Place the CSV file in `data/` directory (rename to `ufc_fights.csv`)
3. Run `python train.py` again

---

## Future Scope

1. **More data**: Use the full historical UFC dataset (3000+ fights)
2. **Advanced features**: Fighter age at time of fight, days since last fight, weight-class-specific models
3. **Better models**: XGBoost, Random Forest, ensemble voting
4. **Hyperparameter tuning**: GridSearchCV / Optuna
5. **Real-time data**: Integrate with UFCStats API for live predictions
6. **Imbalance handling**: SMOTE or class weighting for better minority-class recall

---

## Academic Note

This project is a **Machine Learning Subject Mini Project**. It is NOT a final-year or major project.

- Results are from our own dataset and implementation
- We do NOT claim to match or replicate the exact results of the reference paper
- This system should NOT be used for betting or real-world decision making
- The reference paper is used as inspiration for algorithm selection and methodology only

---

## References

1. *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes* (Reference Paper)
2. Scikit-learn documentation: https://scikit-learn.org
3. Streamlit documentation: https://streamlit.io
4. UFC fight data: https://www.kaggle.com/rajeevw/ufcdata
5. Brownlee, J. (2020). *Imbalanced Classification with Python*. Machine Learning Mastery.
6. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.)
