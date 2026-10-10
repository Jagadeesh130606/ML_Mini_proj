# Report Outline
## "Applying Machine Learning Algorithms to Predict UFC Fight Outcomes"
### Machine Learning Subject — Mini Project Report

---

## CHAPTER 1: Introduction

### 1.1 Background
The Ultimate Fighting Championship (UFC) is one of the world's premier combat sports
organizations, featuring mixed martial arts (MMA) competitions across multiple weight classes.
Each fight involves two athletes with highly varied physical attributes, combat experience,
and technical skill sets.

Predicting the outcome of sporting events has long attracted interest from both the academic
and commercial communities. In recent years, machine learning has emerged as a powerful tool
for extracting patterns from historical sports data and building predictive models.

### 1.2 Motivation
UFC fight prediction presents a challenging classification problem because:
- Fights are influenced by many interacting factors (striking, grappling, experience, physique)
- Outcomes are inherently uncertain due to the nature of combat sports
- Historical statistics provide only an indirect signal about future performance

### 1.3 Scope
This project implements a complete machine learning pipeline to predict UFC fight winners
using pre-fight fighter statistics. It is a mini project for the Machine Learning subject
and is academic in nature.

---

## CHAPTER 2: Problem Statement

### 2.1 Problem Definition
Given a pair of UFC fighters (designated Red and Blue per UFC convention) and their
historical pre-fight statistics, predict whether the **Red fighter** or **Blue fighter**
will win the bout.

**This is a binary classification problem:**
- Output Class 1: Red fighter wins
- Output Class 0: Blue fighter wins

### 2.2 Constraints
- Only **pre-fight information** may be used as input (no post-fight statistics)
- Results must be **evaluated chronologically** (no future data leakage)
- Draws and No-Contests are excluded from binary classification
- Fights prior to 2010 are excluded due to historical label collection bias in early data

### 2.3 Challenges
1. **High uncertainty**: Even professional analysts cannot reliably predict MMA outcomes
2. **Class imbalance**: The Red fighter (favoured by UFC matchmakers) wins ~58.85% of bouts post-2010
3. **Historical label collection artifact**: Pre-2010 data in the dataset records ~98.8% Red wins with 0 Blue wins
4. **Missing data**: Historical stats are incomplete for newer fighters
5. **Temporal dependency**: Fighter performance changes over time

---

## CHAPTER 3: Objectives

1. Build a complete binary classification pipeline for UFC fight prediction
2. Implement four machine learning algorithms matching the reference paper
3. Apply proper time-series cross-validation to prevent data leakage
4. Engineer comparative features (Red − Blue differences) for better generalization
5. Evaluate and compare all models honestly using standard metrics against an Always Red baseline
6. Build a Streamlit web interface for demonstration

---

## CHAPTER 4: Dataset and Features

### 4.1 Dataset Overview
The project uses UFC historical fight data containing:
- Fight metadata: date, weight class, title bout status, number of rounds
- Red fighter stats: height, weight, reach, stance, age, wins, losses, win streak,
  significant strikes landed/attempted, takedowns, submission attempts, etc.
- Blue fighter stats: equivalent columns with B_ prefix

**Dataset characteristics:**
| Property | Value |
|---|---|
| Raw dataset rows | 6,012 fights (144 columns, 1994-03-11 to 2021-03-20) |
| Draws removed | 110 fights |
| Pre-2010 fights removed | 1,244 fights (label bias artifact) |
| Cleaned dataset fights | 4,673 fights (2010-01-02 to 2021-03-20) |
| Engineered features | 28 comparative features (26 numerical diffs + 2 stance) |
| Binary target | 1 (Red wins), 0 (Blue wins) |
| Train set (80%) | 3,738 fights (2010-01-02 to 2019-04-13 \| 58.64% Red wins) |
| Test set (20%) | 935 fights (2019-04-13 to 2021-03-20 \| 59.68% Red wins) |

### 4.2 Feature Description
Key raw features per fighter:
- `Height_cms`: Fighter height in centimetres
- `Reach_cms`: Fighter reach in centimetres
- `Weight_lbs`: Fighter weight in pounds
- `Stance`: Fighting stance (Orthodox, Southpaw, Switch, Open Stance)
- `age`: Fighter age at time of bout
- `wins` / `losses`: Career record
- `current_win_streak`: Current consecutive wins
- `avg_SIG_STR_landed`: Average significant strikes landed per minute (career average)
- `avg_SIG_STR_pct`: Significant strike accuracy (career average)
- `avg_TD_landed`: Average takedowns per 15 minutes
- `avg_TD_pct`: Takedown accuracy
- `avg_SUB_ATT`: Average submission attempts per 15 minutes

---

## CHAPTER 5: Data Preprocessing

### 5.1 Overview
Raw data must be cleaned and standardized before model training. The following
steps are applied in order:

### 5.2 Date Parsing & Pre-2010 Filtering
- The `date` column is parsed into Python `datetime` objects
- Rows with invalid or missing dates are dropped
- Fights prior to **2010-01-01** are dropped via `MIN_YEAR = 2010` to eliminate a historical label collection bias where pre-2010 fights recorded 1,229 Red wins and 0 Blue wins
- Fights are **sorted chronologically** to enable time-based evaluation

### 5.3 Duplicate Removal
- Exact duplicate rows are identified and removed

### 5.4 Target Creation
- The `winner` column contains 'Red', 'Blue', or other values
- Rows where outcome is 'Draw', 'No Contest', or missing are **removed** (110 total rows)
- Binary target: 1 = Red wins, 0 = Blue wins

### 5.5 Leakage Prevention
- All post-fight statistics are removed (e.g., strikes landed in this specific fight)
- Fighter names, referee names, and location metadata are removed
- Only career-average statistics (computed before each fight) are used as features

### 5.6 Imputation
- **Numerical features**: Missing values imputed with the **median** of the training set
- **Categorical features**: Missing values imputed with the **most frequent** value
- Imputation is fitted on training data only and applied to test data

### 5.7 Scaling
- **StandardScaler** applied to all numerical features
- Centres data to mean=0, std=1
- Required for SGD Classifier and MLP Classifier

### 5.8 Encoding
- **OneHotEncoder** applied to stance columns (R_Stance, B_Stance)
- `handle_unknown='ignore'` prevents errors on unseen values

### 5.9 Pipeline
All preprocessing steps are encapsulated in a scikit-learn `Pipeline` + `ColumnTransformer`:
```
ColumnTransformer(
  numerical → SimpleImputer(median) → StandardScaler
  categorical → SimpleImputer(most_frequent) → OneHotEncoder
)
```

---

## CHAPTER 6: Feature Engineering

### 6.1 Motivation
The reference paper highlights that a key design decision is how to represent
the "matchup" between two fighters as a single feature vector.

A naive approach would concatenate all Red and Blue features independently.
However, this makes it difficult for the model to learn the relative advantage.

### 6.2 Comparative Difference Features
For every shared statistic S, we create:
```
diff_S = R_S − B_S
```

Examples:
| Engineered Feature | Interpretation |
|---|---|
| `diff_wins` > 0 | Red fighter has more wins |
| `diff_current_win_streak` > 0 | Red fighter is on a longer current streak |
| `diff_Reach_cms` > 0 | Red fighter has longer reach |
| `diff_avg_SIG_STR_pct` > 0 | Red fighter is more accurate striker |

### 6.3 Name Mismatch Resolution
We verified feature engineering against the raw CSV column headers. The KO/TKO win count feature in the CSV is named `R_win_by_KO/TKO` and `B_win_by_KO/TKO`. We updated `DIFF_STAT_PAIRS` in `src/feature_engineering.py` from `win_by_KO` to `win_by_KO/TKO`, ensuring all 26 comparative numeric pairs were correctly matched and engineered without silent dropping.

### 6.4 Advantages
- The model directly sees the **advantage gap**, not just raw numbers
- Reduces dimensionality compared to using all R_ and B_ columns independently
- Symmetric: swapping fighters negates all features, which is logically consistent

### 6.5 Categorical Features
Stance (Orthodox / Southpaw / Switch / Open Stance) is kept as separate Red and Blue categorical
columns and one-hot encoded, since stance is a style rather than a numeric quantity
and subtraction does not apply.

---

## CHAPTER 7: Machine Learning Algorithms

### 7.1 SGD Classifier (Stochastic Gradient Descent / Generalized Linear Model)
**Algorithm**: Logistic Regression optimised with Stochastic Gradient Descent
**Loss**: log_loss (cross-entropy)
**Hyperparameters**: max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42
**Why**: Represents the linear model baseline. Fast training, interpretable linear decision boundary.

### 7.2 MLP Classifier (Multilayer Perceptron)
**Architecture**: Input → Dense(64, ReLU) → Dense(32, ReLU) → Output(Sigmoid)
**Solver**: Adam
**Regularisation**: Early stopping with 20 iterations patience, random_state=42
**Why**: Captures non-linear relationships between features.

### 7.3 Decision Tree Classifier
**Max depth**: 6 (limits tree growth to prevent overfitting)
**Min samples per split / leaf**: 20 / 10, class_weight='balanced', random_state=42
**Why**: Highly interpretable tree-based classifier.

### 7.4 Gradient Boosting Classifier
**Estimators**: 100
**Learning rate**: 0.1
**Max depth**: 4 (per tree), subsample=0.8, random_state=42
**Why**: Sequential boosting ensemble.

---

## CHAPTER 8: Implementation

### 8.1 Project Structure
```
ufc_fight_prediction/
├── data/ufc_fights.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict.py
├── models/trained_models.pkl
├── results/
│   ├── model_comparison_before.csv
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

### 8.2 Training Script (train.py)
The training script runs the complete pipeline:
1. Load dataset (6,012 raw fights)
2. Clean data (filter pre-2010 label artifact, remove 110 draws → 4,673 clean fights)
3. Sort fights chronologically (2010-01-02 to 2021-03-20)
4. Engineer comparative features (28 features)
5. Chronological train/test split (80% train: 3,738 | 20% test: 935)
6. Train 4 models with TimeSeriesSplit cross-validation (k=5)
7. Evaluate on held-out test set alongside Always Red baseline
8. Save trained models and result CSVs

### 8.3 Time-Series Validation & Fold Protection
TimeSeriesSplit is used instead of random K-Fold cross-validation because UFC fights are temporal events. We implemented `check_and_get_valid_tscv` in `src/train_models.py` to verify that every validation fold contains both target classes, preventing NaN cross-validation scores.

### 8.4 Streamlit Application (app.py)
A web application featuring:
- **About**: Project overview, methodology, ethics note
- **Fight Prediction**: Live prediction between Red and Blue fighters with probability bars
- **Model Comparison**: Metric summary cards, accuracy comparison, CV error bars, multi-metric grouped bars
- **Feature Importance**: Top-N feature importance charts from tree models

---

## CHAPTER 9: Results and Discussion

### 9.1 Reference Paper Results
The reference paper (*Applying ML Algorithms to Predict UFC Fight Outcomes*) reports:
- Dataset: 3,355 samples, 134 features
- Gradient Boosting: ~60% accuracy (highest)
- MLP: similar accuracy
- Decision Tree: comparable performance
- GLM/SGD: comparable linear baseline

### 9.2 Our Implementation Results

Evaluated on the 935-fight held-out test set (2019-04-13 to 2021-03-20):

| Model | Test Accuracy | Precision | Recall | F1 Score | CV Mean | CV Std |
|---|---|---|---|---|---|---|
| **SGD Classifier** | 54.44% | 55.95% | 54.44% | 54.86% | 55.54% | ±2.69% |
| **MLP Classifier** | **62.35%** | **61.02%** | **62.35%** | **60.57%** | **59.94%** | ±2.39% |
| **Decision Tree** | 59.14% | 59.14% | 59.14% | 59.14% | 53.16% | ±2.26% |
| **Gradient Boosting** | 59.89% | 58.51% | 59.89% | 58.58% | 58.72% | ±2.38% |
| **Always Red (baseline)** | 59.68% | 35.62% | 59.68% | 44.61% | 58.27% | ±3.02% |

### 9.3 Discussion
- **Best Model**: MLP Classifier achieved the highest test accuracy of **62.35%** and highest F1 score of **60.57%**, outperforming the Always Red baseline (59.68% accuracy, 44.61% F1 score) by **2.67 percentage points** in accuracy and **15.96 percentage points** in F1 score.
- **Why ~60–62% accuracy?** Combat sports outcome prediction contains substantial inherent noise. A 62.35% test accuracy with 60.57% F1 score is a solid result on temporal test data.
- **Baseline Comparison**: The Always Red baseline achieves 59.68% accuracy simply due to matchmaker favoring Red corner fighters, but suffers a very low F1 score (44.61%) because it never predicts Blue wins (0% recall for Blue). The trained models achieve balanced predictions across both Red and Blue outcomes.

### 9.4 Comparison with Reference Paper
| Aspect | Reference Paper | Our Implementation |
|---|---|---|
| Raw dataset | 3,355 fights | 6,012 fights raw (4,673 cleaned post-2010) |
| Feature engineering | 134 features | 28 comparative diff & stance features |
| Best model | Gradient Boosting ~60% | MLP Classifier 62.35% (GB: 59.89%) |
| Baseline | Not explicitly detailed | Always Red (59.68% Acc, 44.61% F1) |
| Validation | Time-series k-fold | TimeSeriesSplit (k=5, protected against NaN) |
| Framework | Not specified | scikit-learn + Streamlit |

---

## CHAPTER 10: Conclusion

This mini project successfully demonstrates a complete machine learning pipeline
applied to UFC fight outcome prediction on real UFC data.

Key achievements:
1. **Complete pipeline**: Data loading → preprocessing → pre-2010 filter → feature engineering → training → evaluation → prediction
2. **4 algorithms + Baseline**: SGD, MLP, Decision Tree, Gradient Boosting compared against Always Red baseline
3. **Data Quality Fix**: Identified and filtered 1,244 pre-2010 fights suffering from 98.8% Red label collection bias, resolving single-class fold errors and filling all CV metrics
4. **Comparative features**: 28 Red − Blue difference features directly capture fighter advantage
5. **Interactive UI**: Streamlit application displaying real-time predictions and performance comparisons

---

## CHAPTER 11: Future Scope

1. **Additional Features**: Fighter days since last fight, career Elo rating trajectory, weight class specific trends
2. **Advanced Modeling**: XGBoost, LightGBM, Random Forest ensemble voting
3. **Hyperparameter Optimization**: Systematic tuning via Optuna / GridSearchCV
4. **Betting Market Comparison**: Benchmarking predictions against closing bookmaker odds

---

## References

1. *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes* (Course Reference Paper)
2. Scikit-learn: Machine Learning in Python. Pedregosa et al. (2011). JMLR 12, pp. 2825-2830.
3. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
4. Mitchell, T. (1997). *Machine Learning*. McGraw-Hill.
5. UFC Historical Fight Data. Available at: https://www.kaggle.com/rajeevw/ufcdata
6. Streamlit. (2023). Documentation. https://docs.streamlit.io
7. Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5).
8. Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123-140.
