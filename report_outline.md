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

### 2.3 Challenges
1. **High uncertainty**: Even professional analysts cannot reliably predict MMA outcomes
2. **Class imbalance**: The Red fighter (favoured by UFC matchmakers) tends to win more often
3. **Missing data**: Historical stats are incomplete for newer fighters
4. **Temporal dependency**: Fighter performance changes over time

---

## CHAPTER 3: Objectives

1. Build a complete binary classification pipeline for UFC fight prediction
2. Implement four machine learning algorithms matching the reference paper
3. Apply proper time-series cross-validation to prevent data leakage
4. Engineer comparative features (Red − Blue differences) for better generalization
5. Evaluate and compare all models honestly using standard metrics
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
| Total fights | ~1500+ (real dataset: ~3355 per reference paper) |
| Original features | 50+ raw columns |
| After feature engineering | ~26 comparative features |
| Binary target | 1 (Red wins), 0 (Blue wins) |

### 4.2 Feature Description
Key raw features per fighter:
- `Height_cms`: Fighter height in centimetres
- `Reach_cms`: Fighter reach in centimetres
- `Weight_lbs`: Fighter weight in pounds
- `Stance`: Fighting stance (Orthodox, Southpaw, Switch)
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

### 5.2 Date Parsing
- The `date` column is parsed into Python `datetime` objects
- Rows with invalid or missing dates are dropped
- Fights are **sorted chronologically** to enable time-based evaluation

### 5.3 Duplicate Removal
- Exact duplicate rows are identified and removed

### 5.4 Target Creation
- The `winner` column contains 'Red', 'Blue', or other values
- Rows where outcome is 'Draw', 'No Contest', or missing are **removed**
- Number of removed records is reported explicitly
- Binary target: 1 = Red wins, 0 = Blue wins

### 5.5 Leakage Prevention
- All post-fight statistics are removed (e.g., strikes landed in this specific fight)
- Fighter names and referee names are removed
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

### 6.3 Advantages
- The model directly sees the **advantage gap**, not just raw numbers
- Reduces dimensionality compared to using all R_ and B_ columns independently
- Symmetric: swapping fighters negates all features, which is logically consistent

### 6.4 Categorical Features
Stance (Orthodox / Southpaw / Switch) is kept as separate Red and Blue categorical
columns and one-hot encoded, since stance is a style rather than a numeric quantity
and subtraction does not apply.

---

## CHAPTER 7: Machine Learning Algorithms

### 7.1 SGD Classifier (Stochastic Gradient Descent / Generalized Linear Model)
**Algorithm**: Logistic Regression optimised with Stochastic Gradient Descent
**Loss**: log_loss (cross-entropy)
**Hyperparameters**: max_iter=1000, class_weight='balanced'
**Why**: Represents the linear model baseline. Fast training, interpretable linear decision boundary.
The reference paper includes a Generalized Linear Model as one of its four algorithms.

### 7.2 MLP Classifier (Multilayer Perceptron)
**Architecture**: Input → Dense(64, ReLU) → Dense(32, ReLU) → Output(Sigmoid)
**Solver**: Adam
**Regularisation**: Early stopping with 20 iterations patience
**Why**: Captures non-linear relationships between features. Represents neural networks.
The reference paper also uses an MLP classifier.

### 7.3 Decision Tree Classifier
**Max depth**: 6 (limits tree growth to prevent overfitting)
**Min samples per leaf**: 10
**Why**: Highly interpretable. Feature importances show which attributes matter most.
The reference paper includes a Decision Tree.

### 7.4 Gradient Boosting Classifier
**Estimators**: 100
**Learning rate**: 0.1
**Max depth**: 4 (per tree)
**Subsample**: 0.8 (stochastic boosting)
**Why**: The reference paper reports Gradient Boosting achieves marginally the highest accuracy
among the four algorithms (~60%).

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
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── app.py
├── train.py
└── requirements.txt
```

### 8.2 Training Script (train.py)
The training script runs the complete 9-step pipeline:
1. Load and inspect dataset
2. Clean data and create binary target
3. Sort fights chronologically
4. Engineer comparative features
5. Chronological train/test split (80/20)
6. Train 4 models with TimeSeriesSplit cross-validation (k=5)
7. Evaluate on held-out chronological test set
8. Save trained models as pickle
9. Save results to CSV

### 8.3 Time-Series Validation
TimeSeriesSplit is used instead of random K-Fold cross-validation because:
- UFC fights are **temporal events** — a model trained in 2022 should only use fights from before 2022
- Random shuffling would allow the model to train on "future" fights
- TimeSeriesSplit always trains on the past k-1 folds and validates on fold k

### 8.4 Streamlit Application (app.py)
A four-page web application:
- **About**: Project overview, pipeline, algorithms
- **Fight Prediction**: Enter Red/Blue fighter stats → get winner prediction with probabilities
- **Model Comparison**: Bar charts, metrics table, CV scores
- **Feature Importance**: Top-N features from tree-based models

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

> **Note**: Results shown here are from the synthetic demo dataset.
> Run `python train.py` and check `results/model_comparison.csv` for actual numbers.

| Model | Test Accuracy | CV Accuracy | F1 Score |
|---|---|---|---|
| SGD Classifier | ~XX% | ~XX% | ~XX% |
| MLP Classifier | ~XX% | ~XX% | ~XX% |
| Decision Tree | ~XX% | ~XX% | ~XX% |
| Gradient Boosting | ~XX% | ~XX% | ~XX% |

*(Replace XX% with actual values from your training run.)*

### 9.3 Discussion
- **Why ~60% accuracy?** UFC outcomes have significant randomness (any fighter can win on a given day). 60% is a meaningful result above the 50% random baseline.
- **Gradient Boosting** typically performs best due to its ability to capture complex feature interactions while controlling overfitting.
- **Decision Trees** provide interpretability but may underfit or overfit without careful regularization.
- **MLP** requires careful tuning and may not generalize well on small datasets.
- **SGD** provides a fast linear baseline and shows whether the problem has a linear structure.

### 9.4 Comparison with Reference Paper
| Aspect | Reference Paper | Our Implementation |
|---|---|---|
| Dataset size | 3,355 fights | ~1,500 (mock/demo) |
| Feature engineering | 134 features | ~26 diff features |
| Best model | Gradient Boosting ~60% | (see results/) |
| Validation | Time-series k-fold | TimeSeriesSplit k=5 |
| Framework | Not specified | scikit-learn |

---

## CHAPTER 10: Conclusion

This mini project successfully demonstrates a complete machine learning pipeline
applied to the problem of UFC fight outcome prediction.

Key achievements:
1. **Complete pipeline**: Data loading → preprocessing → feature engineering → training → evaluation → prediction
2. **4 algorithms**: SGD, MLP, Decision Tree, Gradient Boosting — matching the reference paper
3. **No data leakage**: Chronological sorting, time-based split, only pre-fight features used
4. **Comparative features**: Red − Blue difference approach reduces dimensionality while preserving relative information
5. **Working web app**: Streamlit interface for live demonstration and model comparison

The reference paper's finding that accuracy plateaus around 60% is consistent with the inherent unpredictability of MMA combat. Our implementation provides a replicable, educational demonstration of these concepts at undergraduate level.

---

## CHAPTER 11: Future Scope

1. **Real UFC dataset**: Use the full UFCStats dataset (Kaggle) for research-grade results
2. **More features**: Include days since last fight, fight weight class, title-fight flag, opponent quality metrics (e.g. Elo rating)
3. **Better models**: XGBoost, LightGBM, Random Forest with voting ensemble
4. **Hyperparameter tuning**: Systematic tuning with Optuna or GridSearchCV
5. **Fighter embeddings**: Use fighter career trajectories as sequences (LSTM/RNN approach)
6. **Odds comparison**: Compare predictions against bookmaker odds as a baseline
7. **Weight-class specific models**: Train separate models per weight class since fighter styles differ significantly

---

## References

1. [Reference Paper] *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes* (course reference)
2. Scikit-learn: Machine Learning in Python. Pedregosa et al. (2011). JMLR 12, pp. 2825-2830.
3. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
4. Mitchell, T. (1997). *Machine Learning*. McGraw-Hill.
5. UFC Historical Fight Data. Available at: https://www.kaggle.com/rajeevw/ufcdata
6. Streamlit. (2023). Documentation. https://docs.streamlit.io
7. Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5).
8. Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123-140.
