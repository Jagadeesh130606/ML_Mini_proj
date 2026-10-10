"""
app.py  —  UFC Fight Outcome Predictor (Fight Night Edition)
Streamlit web application.

UI/UX: Fight Night Arena Theme (Dark #0b0b0b, Soft Gold #e0a93b, Red #ff3d2e, Blue #3d6bff)
Fonts: Anton (Headings) & Archivo (Body text)
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

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict_fight, build_fight_features

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UFC Fight Outcome Predictor — Fight Night Edition",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom CSS — Fight Night Theme System
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Archivo:ital,wght@0,400;0,600;0,700;0,900;1,400&display=swap');

/* Global Reset & Dark Theme */
html, body, [class*="css"] {
    font-family: 'Archivo', Arial, sans-serif !important;
    background-color: #0b0b0b !important;
    color: #f7f7ea !important;
}

.stApp {
    background-color: #0b0b0b !important;
}

/* Typography Headings */
h1, h2, h3, .fn-heading {
    font-family: 'Anton', Impact, sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: #e0a93b !important;
}

/* Header Banner */
.banner-container {
    text-align: center;
    padding: 1.8rem 1rem 0.8rem 1rem;
    background: #1c1c1a;
    border-bottom: 3px solid #e0a93b;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.8);
}
.banner-title {
    font-family: 'Anton', Impact, sans-serif;
    font-size: 3.5rem;
    color: #e0a93b;
    margin: 0;
    line-height: 1.0;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.banner-sub {
    font-family: 'Archivo', sans-serif;
    font-size: 1.05rem;
    color: #f7f7ea;
    opacity: 0.85;
    letter-spacing: 3px;
    margin-top: 0.4rem;
    text-transform: uppercase;
    font-weight: 700;
}
.split-bar {
    display: flex;
    height: 6px;
    width: 100%;
    margin-top: 1.2rem;
}
.split-red { background: #ff3d2e; flex: 1; }
.split-blue { background: #3d6bff; flex: 1; }

/* Corner Cards (Sharp Edges, Thick Borders) */
.corner-card-red {
    background: #1c1c1a;
    border-top: 5px solid #ff3d2e;
    border-radius: 2px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    border-left: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.corner-card-blue {
    background: #1c1c1a;
    border-top: 5px solid #3d6bff;
    border-radius: 2px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    border-left: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.corner-title-red {
    color: #ff3d2e;
    font-size: 2rem;
    font-family: 'Anton', Impact, sans-serif;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.corner-title-blue {
    color: #3d6bff;
    font-size: 2rem;
    font-family: 'Anton', Impact, sans-serif;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Sharp Content Cards */
.fn-card {
    background: #1c1c1a;
    border: 1px solid rgba(235, 230, 218, 0.12);
    border-radius: 2px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.fn-card-gold {
    background: #1c1c1a;
    border-left: 4px solid #e0a93b;
    border-radius: 2px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.fn-card-bone {
    background: #ebe6da;
    color: #0b0b0b !important;
    border-left: 4px solid #e0a93b;
    border-radius: 2px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.fn-card-bone h3, .fn-card-bone h4, .fn-card-bone p, .fn-card-bone strong, .fn-card-bone td, .fn-card-bone th {
    color: #0b0b0b !important;
}

/* Streamlit Button Styling */
div.stButton > button {
    background-color: #e0a93b !important;
    color: #0b0b0b !important;
    font-family: 'Anton', Impact, sans-serif !important;
    font-size: 1.8rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 15px rgba(224, 169, 59, 0.3) !important;
    transition: all 0.15s ease-in-out !important;
}
div.stButton > button:hover {
    background-color: #f5bd4f !important;
    transform: translateY(-2px) !important;
    color: #0b0b0b !important;
    box-shadow: 0 6px 20px rgba(224, 169, 59, 0.5) !important;
}

/* Decision Box */
.decision-box {
    background: #1c1c1a;
    border: 2px solid #e0a93b;
    border-radius: 2px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.decision-sub {
    font-size: 0.95rem;
    color: #e0a93b;
    letter-spacing: 4px;
    text-transform: uppercase;
    font-weight: 700;
}
.decision-winner-red {
    font-family: 'Anton', Impact, sans-serif;
    font-size: 3.2rem;
    color: #ff3d2e;
    margin: 0.4rem 0;
    text-transform: uppercase;
}
.decision-winner-blue {
    font-family: 'Anton', Impact, sans-serif;
    font-size: 3.2rem;
    color: #3d6bff;
    margin: 0.4rem 0;
    text-transform: uppercase;
}

/* Tug of War Probability Bar */
.tug-container {
    display: flex;
    height: 32px;
    width: 100%;
    background: #0b0b0b;
    border-radius: 2px;
    overflow: hidden;
    margin: 1.2rem 0;
    border: 1px solid rgba(255,255,255,0.15);
}
.tug-red {
    background: #ff3d2e;
    color: #ffffff;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding-left: 12px;
    font-size: 1rem;
}
.tug-blue {
    background: #3d6bff;
    color: #ffffff;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 12px;
    font-size: 1rem;
}

/* Verdict Cards */
.verdict-chip-red {
    background: #1c1c1a;
    border: 1px solid #ff3d2e;
    border-left: 4px solid #ff3d2e;
    color: #f7f7ea;
    padding: 0.8rem;
    border-radius: 2px;
    text-align: center;
}
.verdict-chip-blue {
    background: #1c1c1a;
    border: 1px solid #3d6bff;
    border-left: 4px solid #3d6bff;
    color: #f7f7ea;
    padding: 0.8rem;
    border-radius: 2px;
    text-align: center;
}

/* Tale of the Tape */
.tot-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}
.tot-table td {
    padding: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.tot-val-red {
    color: #ff3d2e;
    font-weight: 700;
    font-size: 1.05rem;
    width: 25%;
    text-align: right;
}
.tot-label {
    color: #e0a93b;
    text-transform: uppercase;
    font-weight: 700;
    font-size: 0.9rem;
    width: 50%;
    text-align: center;
    letter-spacing: 1.5px;
}
.tot-val-blue {
    color: #3d6bff;
    font-weight: 700;
    font-size: 1.05rem;
    width: 25%;
    text-align: left;
}

/* Streamlit Tabs Customization */
button[data-baseweb="tab"] {
    font-family: 'Anton', Impact, sans-serif !important;
    font-size: 1.3rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: rgba(247, 247, 234, 0.6) !important;
    padding: 0.8rem 1.5rem !important;
}
button[aria-selected="true"] {
    color: #e0a93b !important;
    border-bottom-color: #e0a93b !important;
}

/* Footer */
.fn-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid rgba(235, 230, 218, 0.15);
    color: rgba(247, 247, 234, 0.6);
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load Trained Models & Results
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
MODELS_PATH  = os.path.join(BASE_DIR, "models", "trained_models.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "model_comparison.csv")
FI_PATH      = os.path.join(BASE_DIR, "results", "feature_importance.csv")


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
# 4. Header Banner Component
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner-container">
    <div class="banner-title">UFC FIGHT PREDICTOR</div>
    <div class="banner-sub">ML Mini Project: Fight Night Edition</div>
    <div class="split-bar">
        <div class="split-red"></div>
        <div class="split-blue"></div>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Trained models not found. Please run `python train.py` first.")
    st.stop()

trained_results = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Navigation Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_fight, tab_score, tab_matters, tab_about = st.tabs([
    "🥊 FIGHT CARD",
    "📊 SCORECARDS",
    "📈 WHAT MATTERS",
    "ℹ️ ABOUT",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: FIGHT CARD (Input & Prediction)
# ─────────────────────────────────────────────────────────────────────────────
with tab_fight:
    st.markdown("### ⚔️ ENTER FIGHTER WEIGH-IN STATS")
    
    col_red, col_blue = st.columns(2)

    def render_fighter_inputs(prefix: str, corner_name: str, border_class: str, title_class: str):
        """Render fighter input fields organized in sharp cards."""
        st.markdown(f'<div class="{border_class}"><div class="{title_class}">{corner_name}</div>', unsafe_allow_html=True)
        
        name = st.text_input("Fighter Name", value=f"{'Red Fighter' if prefix=='R' else 'Blue Fighter'}", key=f"{prefix}_name")
        stats = {}
        
        with st.expander("📏 Body & Physique", expanded=True):
            stats["Height_cms"] = st.slider("Height (cm)", 155, 215, 178, key=f"{prefix}_height")
            stats["Weight_lbs"] = st.slider("Weight (lbs)", 115, 270, 170, key=f"{prefix}_weight")
            stats["Reach_cms"]  = st.slider("Reach (cm)", 155, 220, 183, key=f"{prefix}_reach")
            stats["Stance"]     = st.selectbox("Stance", ["Orthodox", "Southpaw", "Switch", "Open Stance"], key=f"{prefix}_stance")
            stats["age"]        = st.slider("Age (years)", 18, 45, 29, key=f"{prefix}_age")

        with st.expander("🏆 Record & Experience", expanded=True):
            stats["wins"]               = st.number_input("Career Wins", 0, 50, 12, key=f"{prefix}_wins")
            stats["losses"]             = st.number_input("Career Losses", 0, 30, 3, key=f"{prefix}_losses")
            stats["current_win_streak"] = st.number_input("Current Win Streak", 0, 15, 3, key=f"{prefix}_streak")
            stats["longest_win_streak"] = st.number_input("Longest Win Streak", 0, 20, 5, key=f"{prefix}_lstreak")
            stats["total_rounds_fought"]= st.number_input("Total Rounds Fought", 0, 100, 25, key=f"{prefix}_rounds")
            stats["total_title_bouts"]  = st.number_input("Total Title Bouts", 0, 10, 1, key=f"{prefix}_title")

        with st.expander("👊 Striking & Grappling Stats", expanded=False):
            stats["avg_SIG_STR_landed"]     = st.slider("Avg Sig. Strikes Landed/min", 0.0, 12.0, 4.5, 0.1, key=f"{prefix}_ssl")
            stats["avg_SIG_STR_pct"]        = st.slider("Sig. Strike Accuracy %", 0, 90, 45, key=f"{prefix}_ssa") / 100.0
            stats["avg_opp_SIG_STR_landed"] = st.slider("Avg Opp Sig. Strikes Abs/min", 0.0, 12.0, 3.8, 0.1, key=f"{prefix}_ossl")
            stats["avg_opp_SIG_STR_pct"]    = st.slider("Opp Sig. Strike Acc %", 0, 90, 42, key=f"{prefix}_nossa") / 100.0
            stats["avg_TD_landed"]          = st.slider("Avg Takedowns/15min", 0.0, 8.0, 1.8, 0.1, key=f"{prefix}_td")
            stats["avg_TD_pct"]             = st.slider("Takedown Accuracy %", 0, 90, 40, key=f"{prefix}_tdpct") / 100.0
            stats["avg_SUB_ATT"]            = st.slider("Avg Sub Attempts/15min", 0.0, 5.0, 0.8, 0.1, key=f"{prefix}_sub")
            stats["avg_opp_TD_landed"]       = st.slider("Avg Opp Takedowns Abs/15min", 0.0, 7.0, 1.2, 0.1, key=f"{prefix}_otd")
            stats["avg_opp_TD_pct"]          = st.slider("Opp Takedown Acc %", 0, 90, 35, key=f"{prefix}_otdpct") / 100.0

        st.markdown('</div>', unsafe_allow_html=True)
        return name, stats

    with col_red:
        red_name, red_stats = render_fighter_inputs("R", "🔴 RED CORNER", "corner-card-red", "corner-title-red")
    
    with col_blue:
        blue_name, blue_stats = render_fighter_inputs("B", "🔵 BLUE CORNER", "corner-card-blue", "corner-title-blue")

    st.markdown("<br>", unsafe_allow_html=True)
    ring_bell = st.button("🔔 RING THE BELL — PREDICT WINNER")

    if ring_bell:
        st.markdown("---")
        st.markdown("## 🏆 THE DECISION")

        pred_output = predict_fight(red_stats, blue_stats, trained_results)
        results = pred_output["results"]
        consensus = pred_output["consensus"]

        winner_name = red_name if consensus == "Red" else blue_name
        winner_color_class = "decision-winner-red" if consensus == "Red" else "decision-winner-blue"

        # Calculate average Red probability across models
        avg_red_prob = np.mean([r["red_prob"] for r in results.values()])
        avg_blue_prob = 100.0 - avg_red_prob

        # Consensus Box
        st.markdown(f"""
        <div class="decision-box">
            <div class="decision-sub">OFFICIAL CONSENSUS DECISION</div>
            <div class="{winner_color_class}">{winner_name}</div>
            <div style="color:#f7f7ea;font-weight:600;font-size:1.05rem;">
                PREDICTED WINNER VIA CONSENSUS OF ALL 4 ML MODELS
            </div>
            
            <div class="tug-container">
                <div class="tug-red" style="width:{avg_red_prob:.1f}%;">🔴 RED {avg_red_prob:.1f}%</div>
                <div class="tug-blue" style="width:{avg_blue_prob:.1f}%;">BLUE {avg_blue_prob:.1f}% 🔵</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-model verdict chips
        st.markdown("### 📋 MODEL JUDGES SCORECARDS")
        cols_verdicts = st.columns(len(results))
        red_votes = 0
        for i, (m_name, res) in enumerate(results.items()):
            w_corner = res["winner"]
            if w_corner == "Red":
                red_votes += 1
            chip_class = "verdict-chip-red" if w_corner == "Red" else "verdict-chip-blue"
            w_fighter = red_name if w_corner == "Red" else blue_name
            with cols_verdicts[i]:
                st.markdown(f"""
                <div class="{chip_class}">
                    <div style="font-size:0.8rem;text-transform:uppercase;opacity:0.8">{m_name}</div>
                    <div style="font-family:'Anton';font-size:1.3rem;margin:0.2rem 0">{w_fighter}</div>
                    <div style="font-size:0.85rem">🔴 {res['red_prob']}% | 🔵 {res['blue_prob']}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.info(f"**Consensus Breakdown**: {red_votes} out of {len(results)} models voted for **{red_name}**.")

        # Tale of the Tape Comparison
        st.markdown("---")
        st.markdown("### 📊 TALE OF THE TAPE COMPARISON")
        
        tot_stats = [
            ("Height", f"{red_stats['Height_cms']} cm", f"{blue_stats['Height_cms']} cm"),
            ("Reach", f"{red_stats['Reach_cms']} cm", f"{blue_stats['Reach_cms']} cm"),
            ("Weight", f"{red_stats['Weight_lbs']} lbs", f"{blue_stats['Weight_lbs']} lbs"),
            ("Age", f"{red_stats['age']} yrs", f"{blue_stats['age']} yrs"),
            ("Stance", f"{red_stats['Stance']}", f"{blue_stats['Stance']}"),
            ("Career Wins", f"{red_stats['wins']}", f"{blue_stats['wins']}"),
            ("Career Losses", f"{red_stats['losses']}", f"{blue_stats['losses']}"),
            ("Win Streak", f"{red_stats['current_win_streak']}", f"{blue_stats['current_win_streak']}"),
            ("Sig. Strike Acc", f"{red_stats['avg_SIG_STR_pct']*100:.0f}%", f"{blue_stats['avg_SIG_STR_pct']*100:.0f}%"),
            ("Takedown Acc", f"{red_stats['avg_TD_pct']*100:.0f}%", f"{blue_stats['avg_TD_pct']*100:.0f}%"),
        ]

        tot_html = '<table class="tot-table">'
        for label, r_val, b_val in tot_stats:
            tot_html += f'''
            <tr>
                <td class="tot-val-red">{r_val}</td>
                <td class="tot-label">{label}</td>
                <td class="tot-val-blue">{b_val}</td>
            </tr>
            '''
        tot_html += '</table>'
        st.markdown(tot_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SCORECARDS (Model Performance)
# ─────────────────────────────────────────────────────────────────────────────
with tab_score:
    st.markdown("### 📊 MODEL EVALUATION SCORECARDS")
    st.markdown("Evaluated on the held-out test set (935 fights, 2019-04-13 to 2021-03-20)")

    if os.path.exists(RESULTS_PATH):
        results_df = load_results()
        
        # Summary Metric Cards
        trained_df = results_df[results_df["Model"] != "Always Red (baseline)"]
        best_row = trained_df.loc[trained_df["Accuracy"].idxmax()]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="fn-card-gold"><div style="font-size:0.8rem;text-transform:uppercase;color:#e0a93b">TOP MODEL</div><div style="font-family:\'Anton\';font-size:1.6rem;color:#f7f7ea">{best_row["Model"]}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="fn-card-gold"><div style="font-size:0.8rem;text-transform:uppercase;color:#e0a93b">BEST ACCURACY</div><div style="font-family:\'Anton\';font-size:1.6rem;color:#e0a93b">{best_row["Accuracy"]*100:.2f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="fn-card-gold"><div style="font-size:0.8rem;text-transform:uppercase;color:#e0a93b">BEST F1 SCORE</div><div style="font-family:\'Anton\';font-size:1.6rem;color:#ff3d2e">{best_row["F1 Score"]*100:.2f}%</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="fn-card-gold"><div style="font-size:0.8rem;text-transform:uppercase;color:#e0a93b">MODELS TRAINED</div><div style="font-family:\'Anton\';font-size:1.6rem;color:#3d6bff">{len(trained_df)}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 DETAILED MODEL COMPARISON TABLE")
        
        disp_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Mean", "CV Std"]].copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score", "CV Mean"]:
            disp_df[col] = disp_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        disp_df["CV Std"] = results_df["CV Std"].apply(lambda x: f"±{x*100:.2f}%" if pd.notnull(x) else "N/A")
        
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📊 ACCURACY COMPARISON VS ALWAYS RED BASELINE")
        
        # Accuracy chart
        fig_acc = go.Figure()
        
        # Extract baseline accuracy for horizontal line
        baseline_row = results_df[results_df["Model"] == "Always Red (baseline)"]
        baseline_acc = baseline_row["Accuracy"].values[0] * 100 if not baseline_row.empty else 59.68

        # Plot model bars
        colors_list = ["#e0a93b" if m == best_row["Model"] else "#3d6bff" if m != "Always Red (baseline)" else "#718093" for m in results_df["Model"]]
        
        fig_acc.add_trace(go.Bar(
            x=results_df["Model"],
            y=results_df["Accuracy"] * 100,
            marker_color=colors_list,
            text=[f"{v*100:.2f}%" for v in results_df["Accuracy"]],
            textposition="outside",
            textfont=dict(color="#f7f7ea", size=12),
        ))

        fig_acc.add_hline(
            y=baseline_acc, line_dash="dash", line_color="#ff3d2e", line_width=2,
            annotation_text=f"Always Red Baseline ({baseline_acc:.2f}%)",
            annotation_position="top right", annotation_font_color="#ff3d2e"
        )

        fig_acc.update_layout(
            paper_bgcolor="#1c1c1a",
            plot_bgcolor="#1c1c1a",
            font=dict(color="#f7f7ea"),
            yaxis=dict(title="Test Accuracy (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔄 TIME-SERIES CROSS-VALIDATION SCORES (k=5)")
        
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=results_df["Model"],
            y=results_df["CV Mean"] * 100,
            error_y=dict(type="data", array=results_df["CV Std"] * 100, visible=True, color="#e0a93b"),
            marker_color="#e0a93b",
            text=[f"{v*100:.2f}%" if pd.notnull(v) else "N/A" for v in results_df["CV Mean"]],
            textposition="outside",
            textfont=dict(color="#f7f7ea"),
        ))
        fig_cv.update_layout(
            paper_bgcolor="#1c1c1a",
            plot_bgcolor="#1c1c1a",
            font=dict(color="#f7f7ea"),
            yaxis=dict(title="CV Accuracy (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=380,
        )
        st.plotly_chart(fig_cv, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: WHAT MATTERS (Feature Importance)
# ─────────────────────────────────────────────────────────────────────────────
with tab_matters:
    st.markdown("### 📈 WHAT MATTERS — FEATURE IMPORTANCE")
    st.markdown("Which fighter attributes provide the strongest predictive signal?")

    if os.path.exists(FI_PATH):
        fi_df = load_fi()
        fi_df["Feature_Display"] = fi_df["Feature"].str.replace("diff_", "Δ ").str.replace("_", " ").str.title()
        
        tree_cols = [c for c in ["Gradient Boosting", "Decision Tree"] if c in fi_df.columns]
        
        if tree_cols:
            col_tab_m = st.tabs(tree_cols)
            for tab_m, m_col in zip(col_tab_m, tree_cols):
                with tab_m:
                    top_df = fi_df[["Feature_Display", m_col]].dropna().sort_values(m_col, ascending=False).head(15)
                    
                    fig_fi = go.Figure(go.Bar(
                        x=top_df[m_col],
                        y=top_df["Feature_Display"],
                        orientation="h",
                        marker=dict(color="#e0a93b"),
                        text=[f"{v:.4f}" for v in top_df[m_col]],
                        textposition="auto",
                        textfont=dict(color="#0b0b0b", size=11),
                    ))
                    fig_fi.update_layout(
                        paper_bgcolor="#1c1c1a",
                        plot_bgcolor="#1c1c1a",
                        font=dict(color="#f7f7ea"),
                        xaxis=dict(title="Importance Score", gridcolor="rgba(255,255,255,0.1)"),
                        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
                        height=520,
                        margin=dict(l=180, r=40, t=30, b=40),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div class="fn-card-gold">
            <h3 style="color:#e0a93b;margin-top:0">🔍 UNDERSTANDING COMPARATIVE DELTA (Δ) FEATURES</h3>
            <p>All numeric features are computed as <strong>Red Fighter value − Blue Fighter value</strong>:</p>
            <ul>
                <li><strong>Δ wins > 0</strong>: Red fighter has more career wins.</li>
                <li><strong>Δ age > 0</strong>: Red fighter is older.</li>
                <li><strong>Δ Reach_cms > 0</strong>: Red fighter has a reach advantage in centimetres.</li>
                <li><strong>Δ avg_SIG_STR_pct > 0</strong>: Red fighter is more accurate with significant strikes.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ABOUT (Project & Data Insights)
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### ℹ️ ABOUT THE PROJECT")

    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        <div class="fn-card">
            <h3 style="color:#e0a93b">🎯 1. THE PROBLEM</h3>
            <p style="color:#f7f7ea">
                Given pre-fight statistics for two UFC athletes (Red Corner vs Blue Corner),
                predict which fighter wins the bout. This is a <strong>binary classification</strong>
                problem using strictly pre-fight information.
            </p>
        </div>
        
        <div class="fn-card">
            <h3 style="color:#e0a93b">📊 2. THE DATASET</h3>
            <p style="color:#f7f7ea">
                <strong>Source</strong>: Real UFC historical fight data (1994–2021).<br>
                <strong>Raw Fights</strong>: 6,012 fights (144 raw columns).<br>
                <strong>Cleaned Fights</strong>: 4,673 fights (2010–2021) after filtering draws and pre-2010 label bias.<br>
                <strong>Features</strong>: 28 comparative difference features (Red − Blue).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="fn-card">
            <h3 style="color:#e0a93b">⚙️ 3. THE FIGHT PLAN (ML PIPELINE)</h3>
            <p style="color:#f7f7ea">
                <strong>1. Data Cleaning</strong>: Drop pre-2010 fights, remove draws, drop leakage.<br>
                <strong>2. Feature Engineering</strong>: Compute Red − Blue stat gaps.<br>
                <strong>3. Temporal Split</strong>: 80% train (3,738 fights) / 20% test (935 fights).<br>
                <strong>4. Validation</strong>: TimeSeriesSplit cross-validation (k=5).
            </p>
        </div>
        
        <div class="fn-card-bone">
            <h3>🛠️ 4. DATA ISSUE FOUND & FIXED</h3>
            <p>
                <strong>The Pre-2010 Label Bias Artifact</strong>: Out of 1,244 fights prior to 2010,
                <strong>1,229 fights (98.8%) were recorded as Red wins</strong> and 0 as Blue wins.
                This taught models fake patterns and caused single-class fold errors in cross-validation.
            </p>
            <p>
                <strong>Fix</strong>: Established <code>MIN_YEAR = 2010</code> cutoff in preprocessing,
                restoring a realistic 58.85% Red win rate and eliminating NaN cross-validation scores.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Footer Component
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fn-footer">
    🥊 Learning project only. Not for betting. — Academic Year 2026
</div>
""", unsafe_allow_html=True)
