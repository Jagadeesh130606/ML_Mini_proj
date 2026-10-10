"""
app.py  —  UFC Fight Outcome Predictor
Streamlit web application.

Sections:
  A. About Project
  B. Fighter Input (Red vs Blue)
  C. Prediction (winner + probabilities)
  D. Model Comparison (accuracy, precision, recall, F1)
  E. Feature Importance
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict_fight, build_fight_features

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UFC Fight Outcome Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – clean dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Main background */
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* Red / Blue fighter cards */
.red-card  { border-left: 4px solid #ff4757; }
.blue-card { border-left: 4px solid #2979ff; }

/* Hero title */
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ff4757, #a855f7, #2979ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Prediction result boxes */
.pred-red  { background: linear-gradient(135deg,#ff4757,#c0392b); border-radius:16px;
              padding:2rem; text-align:center; color:white; }
.pred-blue { background: linear-gradient(135deg,#2979ff,#1a237e); border-radius:16px;
              padding:2rem; text-align:center; color:white; }
.pred-label { font-size:1rem; opacity:0.8; text-transform:uppercase; letter-spacing:2px; }
.pred-name  { font-size:2rem; font-weight:900; margin:0.5rem 0; }
.pred-prob  { font-size:1.5rem; font-weight:700; }

/* Metric pills */
.metric-pill {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    margin: 0.2rem;
    font-size: 0.85rem;
    color: white;
}

/* Section headers */
h2 { color: white !important; }
h3 { color: rgba(255,255,255,0.9) !important; }
label { color: rgba(255,255,255,0.85) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load trained models & results
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
MODELS_PATH  = os.path.join(BASE_DIR, "models", "trained_models.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "model_comparison.csv")
FI_PATH      = os.path.join(BASE_DIR, "results", "feature_importance.csv")
META_PATH    = os.path.join(BASE_DIR, "results", "training_meta.pkl")


@st.cache_resource(show_spinner=False)
def load_models():
    with open(MODELS_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_data(show_spinner=False)
def load_fi():
    return pd.read_csv(FI_PATH)


models_loaded = os.path.exists(MODELS_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥊 UFC Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 About Project", "⚔️ Fight Prediction", "📊 Model Comparison", "📈 Feature Importance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Models Used**")
    st.markdown("• SGD Classifier (Linear)\n• MLP Classifier\n• Decision Tree\n• Gradient Boosting")
    st.markdown("---")
    st.markdown("**Reference Paper**")
    st.markdown("*Applying Machine Learning Algorithms to Predict UFC Fight Outcomes*")
    st.markdown("---")
    if not models_loaded:
        st.error("⚠️ Models not trained yet!\nRun: `python train.py`")
    else:
        st.success("✅ Models loaded")

# ─────────────────────────────────────────────────────────────────────────────
# A. ABOUT PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 About Project":
    st.markdown('<div class="hero-title">🥊 UFC Fight Outcome Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Machine Learning Mini Project — Predicting UFC Fight Winners</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="card">
        <h4 style="color:#ff4757">🎯 Objective</h4>
        <p style="color:rgba(255,255,255,0.75)">
        Predict the winner of a UFC fight using pre-fight fighter statistics,
        without using any post-fight information.
        </p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card">
        <h4 style="color:#a855f7">📚 Reference Paper</h4>
        <p style="color:rgba(255,255,255,0.75)">
        <em>Applying Machine Learning Algorithms to Predict UFC Fight Outcomes</em>
        — inspiration for algorithm selection and methodology.
        </p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="card">
        <h4 style="color:#2979ff">🔬 Approach</h4>
        <p style="color:rgba(255,255,255,0.75)">
        Comparative feature engineering: compute Red − Blue differences for each
        fighter stat, then classify with 4 ML models.
        </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🔄 ML Pipeline")
        steps = [
            ("1", "Dataset Loading", "Auto-detects CSV in data/"),
            ("2", "Data Cleaning", "Remove draws, NaN dates, duplicates"),
            ("3", "Target Creation", "1 = Red Wins, 0 = Blue Wins"),
            ("4", "Feature Engineering", "Red − Blue comparative differences"),
            ("5", "Chronological Split", "Train on older, test on newer fights"),
            ("6", "Preprocessing", "Imputation + Scaling + OneHotEncoding"),
            ("7", "Model Training", "4 models with TimeSeriesSplit CV"),
            ("8", "Evaluation", "Accuracy, Precision, Recall, F1"),
            ("9", "Prediction", "Enter fighter stats → predict winner"),
        ]
        for num, title, desc in steps:
            st.markdown(
                f'<div style="display:flex;align-items:start;margin:0.5rem 0;">'
                f'<span style="background:#a855f7;color:white;border-radius:50%;'
                f'width:28px;height:28px;display:flex;align-items:center;justify-content:center;'
                f'font-weight:700;flex-shrink:0;margin-right:12px;">{num}</span>'
                f'<div><strong style="color:white">{title}</strong>'
                f'<br><span style="color:rgba(255,255,255,0.6);font-size:0.85rem">{desc}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown("### 🤖 Algorithms")
        algos = [
            ("SGD Classifier", "#ff4757", "Generalized Linear Model via Stochastic Gradient Descent. Fast, interpretable linear boundary."),
            ("MLP Classifier", "#a855f7", "Multilayer Perceptron with (64,32) hidden layers. Captures non-linear patterns."),
            ("Decision Tree", "#ffa502", "Tree-based model with depth=6 regularization to prevent overfitting."),
            ("Gradient Boosting", "#2979ff", "Ensemble of trees boosted sequentially. Reference paper found this marginally best."),
        ]
        for name, color, desc in algos:
            st.markdown(
                f'<div class="card" style="border-left:4px solid {color};margin:0.5rem 0;">'
                f'<strong style="color:{color}">{name}</strong>'
                f'<p style="color:rgba(255,255,255,0.7);margin:0.3rem 0 0;font-size:0.88rem">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### ⚠️ Academic Note")
    st.info(
        "This is a **Machine Learning Subject Mini Project**. Results are based on the dataset "
        "used and are reported honestly. We do NOT claim to reliably predict real UFC fights. "
        "The reference paper achieved ~60% accuracy; our results reflect our own dataset and implementation. "
        "This system should not be used for betting or real-world decision making."
    )

# ─────────────────────────────────────────────────────────────────────────────
# B. FIGHT PREDICTION PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚔️ Fight Prediction":
    st.markdown('<div class="hero-title">⚔️ Fight Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter pre-fight statistics for both fighters</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("🚨 Models not trained. Run `python train.py` first.")
        st.stop()

    trained_results = load_models()
    model_names = list(trained_results.keys())

    # Model selector
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_model = st.selectbox(
            "🤖 Select Model",
            ["Compare All Models"] + model_names,
            index=0,
            key="model_selector",
        )
    with col_sel2:
        st.markdown('<br>', unsafe_allow_html=True)
        predict_btn = st.button("🥊 PREDICT FIGHT", use_container_width=True, type="primary")

    st.markdown("---")

    # Fighter input columns
    col_red, col_divider, col_blue = st.columns([5, 1, 5])

    def fighter_inputs(prefix: str, color: str):
        """Render all slider/number inputs for one fighter."""
        st.markdown(f"<h3 style='color:{color};text-align:center'>{'🔴 RED FIGHTER' if prefix=='R' else '🔵 BLUE FIGHTER'}</h3>", unsafe_allow_html=True)
        name = st.text_input(f"Fighter Name", value=f"{'Red Fighter' if prefix=='R' else 'Blue Fighter'}", key=f"{prefix}_name")
        stats = {}
        with st.expander("📏 Physical Attributes", expanded=True):
            stats["Height_cms"]  = st.slider("Height (cm)", 155, 215, 178, key=f"{prefix}_height")
            stats["Weight_lbs"]  = st.slider("Weight (lbs)", 115, 270, 170, key=f"{prefix}_weight")
            stats["Reach_cms"]   = st.slider("Reach (cm)", 155, 220, 183, key=f"{prefix}_reach")
            stats["Stance"]      = st.selectbox("Stance", ["Orthodox", "Southpaw", "Switch", "Open Stance"], key=f"{prefix}_stance")
            stats["age"]         = st.slider("Age", 18, 45, 29, key=f"{prefix}_age")

        with st.expander("🏆 Fight Record", expanded=True):
            stats["wins"]               = st.number_input("Total Wins", 0, 50, 10, key=f"{prefix}_wins")
            stats["losses"]             = st.number_input("Total Losses", 0, 30, 3, key=f"{prefix}_losses")
            stats["current_win_streak"] = st.number_input("Current Win Streak", 0, 15, 2, key=f"{prefix}_streak")
            stats["longest_win_streak"] = st.number_input("Longest Win Streak", 0, 20, 4, key=f"{prefix}_lstreak")
            stats["total_rounds_fought"]= st.number_input("Total Rounds Fought", 0, 100, 20, key=f"{prefix}_rounds")

        with st.expander("👊 Striking Stats", expanded=False):
            stats["avg_SIG_STR_landed"]     = st.slider("Avg Sig. Strikes Landed/min", 0.0, 12.0, 4.5, 0.1, key=f"{prefix}_ssl")
            stats["avg_SIG_STR_pct"]        = st.slider("Sig. Strike Accuracy %", 0, 90, 45, key=f"{prefix}_ssa")
            stats["avg_opp_SIG_STR_landed"] = st.slider("Avg Opp Sig. Strikes Abs./min", 0.0, 12.0, 4.0, 0.1, key=f"{prefix}_ossl")
            stats["avg_opp_SIG_STR_pct"]    = st.slider("Opp Sig. Strike Acc % (absorbed)", 0, 90, 43, key=f"{prefix}_nossa")

        with st.expander("🤼 Grappling Stats", expanded=False):
            stats["avg_TD_landed"]   = st.slider("Avg Takedowns/15min", 0.0, 8.0, 1.5, 0.1, key=f"{prefix}_td")
            stats["avg_TD_pct"]      = st.slider("Takedown Accuracy %", 0, 90, 40, key=f"{prefix}_tdpct")
            stats["avg_SUB_ATT"]     = st.slider("Avg Submission Attempts/15min", 0.0, 5.0, 0.8, 0.1, key=f"{prefix}_sub")
            stats["avg_opp_TD_landed"]= st.slider("Avg Opp Takedowns Abs./15min", 0.0, 7.0, 1.2, 0.1, key=f"{prefix}_otd")

        return name, stats

    with col_red:
        red_name, red_stats = fighter_inputs("R", "#ff4757")
    with col_divider:
        st.markdown('<div style="text-align:center;padding-top:4rem;font-size:2rem;color:rgba(255,255,255,0.4)">VS</div>', unsafe_allow_html=True)
    with col_blue:
        blue_name, blue_stats = fighter_inputs("B", "#2979ff")

    # ── Prediction output ──────────────────────────────────────────────────
    if predict_btn:
        st.markdown("---")
        st.markdown("## 🏆 Prediction Results")

        use_model = None if selected_model == "Compare All Models" else selected_model
        pred_output = predict_fight(red_stats, blue_stats, trained_results, use_model)
        results = pred_output["results"]
        consensus = pred_output["consensus"]

        # Consensus banner
        winner_name = red_name if consensus == "Red" else blue_name
        winner_color = "#ff4757" if consensus == "Red" else "#2979ff"
        st.markdown(
            f'<div style="background:linear-gradient(135deg,{winner_color}33,{winner_color}11);'
            f'border:2px solid {winner_color};border-radius:16px;padding:1.5rem;text-align:center;margin-bottom:1.5rem">'
            f'<div style="font-size:0.9rem;color:rgba(255,255,255,0.6);letter-spacing:3px;text-transform:uppercase">🏆 Predicted Winner</div>'
            f'<div style="font-size:2.5rem;font-weight:900;color:{winner_color};margin:0.5rem 0">{winner_name}</div>'
            f'<div style="color:rgba(255,255,255,0.5)">Consensus across all models</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Per-model results
        cols = st.columns(len(results))
        for i, (model_name, res) in enumerate(results.items()):
            with cols[i]:
                w_color = "#ff4757" if res["winner"] == "Red" else "#2979ff"
                w_name  = red_name if res["winner"] == "Red" else blue_name
                st.markdown(
                    f'<div class="card" style="border-top:3px solid {w_color};text-align:center">'
                    f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.5);margin-bottom:0.5rem">{model_name}</div>'
                    f'<div style="font-size:1.2rem;font-weight:700;color:{w_color}">{w_name}</div>'
                    f'<div style="margin:0.8rem 0">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">'
                    f'<span style="color:#ff4757">🔴 {res["red_prob"]}%</span>'
                    f'<span style="color:#2979ff">🔵 {res["blue_prob"]}%</span></div>'
                    f'<div style="background:rgba(255,255,255,0.1);border-radius:10px;height:8px;overflow:hidden">'
                    f'<div style="width:{res["red_prob"]}%;height:100%;'
                    f'background:linear-gradient(90deg,#ff4757,#c0392b)"></div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # Probability chart
        st.markdown("### 📊 Prediction Probability by Model")
        model_list = list(results.keys())
        red_probs  = [results[m]["red_prob"] for m in model_list]
        blue_probs = [results[m]["blue_prob"] for m in model_list]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=f"🔴 {red_name}",  x=model_list, y=red_probs,
                             marker_color="#ff4757", opacity=0.85))
        fig.add_trace(go.Bar(name=f"🔵 {blue_name}", x=model_list, y=blue_probs,
                             marker_color="#2979ff", opacity=0.85))
        fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                      annotation_text="50% threshold")
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# C. MODEL COMPARISON PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown('<div class="hero-title">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Evaluation metrics on the chronological test set</div>', unsafe_allow_html=True)

    if not os.path.exists(RESULTS_PATH):
        st.error("Results not found. Run `python train.py` first.")
        st.stop()

    results_df = load_results()

    # Metric summary cards
    trained_models_df = results_df[results_df["Model"] != "Always Red (baseline)"]
    best_model = trained_models_df.loc[trained_models_df["Accuracy"].idxmax()]
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("🏆 Best Model",     best_model["Model"],               "#a855f7"),
        ("🎯 Best Accuracy",  f"{best_model['Accuracy']*100:.1f}%", "#2979ff"),
        ("📏 Best F1 Score",  f"{results_df['F1 Score'].max()*100:.1f}%", "#ff4757"),
        ("🔄 Models Trained", str(len(trained_models_df)),       "#ffa502"),
    ]
    for col, (label, val, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f'<div class="card" style="text-align:center;border-top:3px solid {color}">'
                f'<div style="color:rgba(255,255,255,0.6);font-size:0.8rem">{label}</div>'
                f'<div style="font-size:1.8rem;font-weight:700;color:{color}">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Metrics table
    st.markdown("### 📋 Detailed Results")
    display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Mean", "CV Std"]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "CV Mean"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    display_df["CV Std"] = results_df["CV Std"].apply(lambda x: f"±{x*100:.2f}%" if pd.notnull(x) else "N/A")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Accuracy comparison bar chart
    st.markdown("### 📊 Accuracy Comparison")
    colors = ["#ff4757", "#a855f7", "#ffa502", "#2979ff", "#718093"]
    fig = go.Figure(go.Bar(
        x=results_df["Model"],
        y=results_df["Accuracy"] * 100,
        marker_color=colors[:len(results_df)],
        text=[f"{v*100:.1f}%" for v in results_df["Accuracy"]],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(title="Accuracy (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=380, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Multi-metric comparison
    st.markdown("### 🎯 Multi-Metric Comparison")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig2 = go.Figure()
    for i, row in results_df.iterrows():
        color_idx = i if i < len(colors) else len(colors) - 1
        fig2.add_trace(go.Bar(
            name=row["Model"],
            x=metric_cols,
            y=[row[m] * 100 for m in metric_cols],
            marker_color=colors[color_idx],
            opacity=0.85,
        ))
    fig2.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # CV scores with error bars
    st.markdown("### 🔄 Cross-Validation Accuracy (TimeSeriesSplit)")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=results_df["Model"],
        y=results_df["CV Mean"] * 100,
        error_y=dict(type="data", array=results_df["CV Std"] * 100, visible=True),
        marker_color=colors[:len(results_df)],
        text=[f"{v*100:.1f}%" for v in results_df["CV Mean"]],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(title="CV Accuracy (%)", gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=350, showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)


    st.info(
        "**Why TimeSeriesSplit?** UFC fights happen chronologically. Using a random split "
        "would allow the model to 'see' future fights during training, which is unrealistic. "
        "TimeSeriesSplit always trains on older fights and validates on newer ones, "
        "mimicking the real-world prediction scenario."
    )

# ─────────────────────────────────────────────────────────────────────────────
# D. FEATURE IMPORTANCE PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Feature Importance":
    st.markdown('<div class="hero-title">📈 Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Which fighter attributes matter most?</div>', unsafe_allow_html=True)

    if not os.path.exists(FI_PATH):
        st.error("Feature importance not found. Run `python train.py` first.")
        st.stop()

    fi_df = load_fi()

    # Clean feature names for display
    fi_df["Feature_Display"] = fi_df["Feature"].str.replace("diff_", "Δ ").str.replace("_", " ").str.title()

    tree_cols = [c for c in ["Decision Tree", "Gradient Boosting"] if c in fi_df.columns]
    if not tree_cols:
        st.warning("No tree-based feature importance available.")
        st.stop()

    col_tabs = st.tabs(tree_cols)
    colors_map = {"Decision Tree": "#ffa502", "Gradient Boosting": "#2979ff"}

    for tab, model_col in zip(col_tabs, tree_cols):
        with tab:
            top_n = 15
            plot_df = fi_df[["Feature_Display", model_col]].dropna()
            plot_df = plot_df.sort_values(model_col, ascending=False).head(top_n)

            fig = go.Figure(go.Bar(
                x=plot_df[model_col],
                y=plot_df["Feature_Display"],
                orientation="h",
                marker=dict(
                    color=plot_df[model_col],
                    colorscale=[[0, "rgba(255,255,255,0.1)"], [1, colors_map[model_col]]],
                    showscale=False,
                ),
                text=[f"{v:.4f}" for v in plot_df[model_col]],
                textposition="auto",
                textfont=dict(color="white", size=11),
            ))
            fig.update_layout(
                title=f"Top {top_n} Features — {model_col}",
                title_font=dict(color="white", size=16),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
                height=500,
                margin=dict(l=180),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 What do Δ (difference) features mean?")
    st.markdown("""
    All features are computed as **Red Fighter value − Blue Fighter value**.

    | Feature | Positive means | Negative means |
    |---------|---------------|----------------|
    | **Δ wins** | Red has more wins | Blue has more wins |
    | **Δ current_win_streak** | Red on longer streak | Blue on longer streak |
    | **Δ Reach_cms** | Red has longer reach | Blue has longer reach |
    | **Δ avg_SIG_STR_pct** | Red more accurate striker | Blue more accurate |
    | **Δ age** | Red fighter is older | Blue fighter is older |

    A large positive importance score means that *knowing the difference in this attribute* 
    strongly helps the model decide who wins.
    """)

    st.info(
        "**Academic Note**: High feature importance does NOT imply causation. "
        "It means the feature was useful for classification in this dataset."
    )
