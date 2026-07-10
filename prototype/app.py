"""
prototype/app.py
----------------
ThreatSense — Insider Threat Detection Platform
COS720 | University of Pretoria
"""

import os, sys, time
from datetime import datetime
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import matplotlib, warnings
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

from src.models.predict   import load_models, predict_record, predict_dataframe
from src.explainability.explain import (
    get_local_explanation, generate_forensic_report,
    get_risk_score_breakdown, get_global_feature_importance,
)
from src.data.preprocess  import FEATURE_COLS, preprocess_single_record
from src.utils.helpers    import load_sample_profiles
from src.utils.pdf_report import generate_pdf_report

st.set_page_config(
    page_title="ThreatSense",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "theme"       not in st.session_state: st.session_state.theme = "dark"
if "history"     not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None

T    = st.session_state.theme
DARK = T == "dark"

# ── Theme tokens — all colours are valid CSS or hex ───────────────────────────
if DARK:
    C = dict(
        app_bg        = "#0f1117",
        sidebar_bg    = "#13161f",
        card_bg       = "#161b22",
        border        = "#21262d",
        text_pri      = "#e6edf3",
        text_sec      = "#8b949e",
        text_ter      = "#c9d1d9",
        accent        = "#06b6d4",
        accent_muted  = "#0e7490",   # hex — safe for matplotlib
        red           = "#ef4444",
        red_muted     = "#7f1d1d",
        green         = "#22c55e",
        green_muted   = "#14532d",
        orange        = "#f97316",
        orange_muted  = "#7c2d12",
        input_bg      = "#0d1117",
        btn_bg        = "#06b6d4",
        btn_text      = "#0f1117",
        chart_bg      = "#161b22",
        chart_grid    = "#21262d",   # hex — safe for matplotlib
        # CSS-only (rgba) — never passed to matplotlib
        border_css    = "rgba(255,255,255,0.06)",
        accent_dim_css= "rgba(6,182,212,0.12)",
        red_dim_css   = "rgba(239,68,68,0.08)",
        green_dim_css = "rgba(34,197,94,0.06)",
        orange_dim_css= "rgba(249,115,22,0.1)",
        toggle_icon   = "☀️",
        toggle_label  = "Light mode",
    )
else:
    C = dict(
        app_bg        = "#f8fafc",
        sidebar_bg    = "#ffffff",
        card_bg       = "#ffffff",
        border        = "#e2e8f0",
        text_pri      = "#0f172a",
        text_sec      = "#64748b",
        text_ter      = "#334155",
        accent        = "#0891b2",
        accent_muted  = "#bae6fd",
        red           = "#dc2626",
        red_muted     = "#fecaca",
        green         = "#16a34a",
        green_muted   = "#bbf7d0",
        orange        = "#ea580c",
        orange_muted  = "#fed7aa",
        input_bg      = "#f1f5f9",
        btn_bg        = "#0891b2",
        btn_text      = "#ffffff",
        chart_bg      = "#f8fafc",
        chart_grid    = "#e2e8f0",
        border_css    = "rgba(0,0,0,0.07)",
        accent_dim_css= "rgba(8,145,178,0.08)",
        red_dim_css   = "rgba(220,38,38,0.06)",
        green_dim_css = "rgba(22,163,74,0.06)",
        orange_dim_css= "rgba(234,88,12,0.08)",
        toggle_icon   = "🌙",
        toggle_label  = "Dark mode",
    )

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after {{ font-family:'Inter',sans-serif !important; box-sizing:border-box; }}

/* App background — covers the black header bar in light mode */
.stApp, .stApp > div, header[data-testid="stHeader"] {{
    background:{C['app_bg']} !important;
}}
section[data-testid="stSidebar"] {{
    background:{C['sidebar_bg']} !important;
    border-right:1px solid {C['border']} !important;
}}
section[data-testid="stSidebar"] > div {{ padding:0; }}
.main .block-container {{ padding:0 2rem 2rem; max-width:1440px; }}

/* Text */
.stMarkdown p {{ color:{C['text_sec']}; font-size:14px; }}
h1,h2,h3 {{ color:{C['text_pri']} !important; font-weight:600 !important; }}
label, .stCheckbox label span {{ color:{C['text_sec']} !important; }}

/* Metrics */
[data-testid="metric-container"] {{
    background:{C['card_bg']} !important;
    border:1px solid {C['border']} !important;
    border-radius:10px !important;
    padding:1rem 1.25rem !important;
}}
[data-testid="stMetricLabel"] > div {{
    color:{C['text_sec']} !important; font-size:11px !important;
    text-transform:uppercase; letter-spacing:0.5px;
}}
[data-testid="stMetricValue"] {{
    color:{C['text_pri']} !important; font-size:26px !important; font-weight:600 !important;
}}

/* Buttons */
.stButton > button[kind="primary"] {{
    background:{C['btn_bg']} !important; color:{C['btn_text']} !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:14px !important; height:42px !important;
}}
.stButton > button[kind="primary"]:hover {{ opacity:0.88 !important; }}
.stButton > button:not([kind="primary"]) {{
    background:{C['card_bg']} !important; color:{C['text_sec']} !important;
    border:1px solid {C['border']} !important; border-radius:8px !important; font-size:13px !important;
}}
.stButton > button:not([kind="primary"]):hover {{
    color:{C['text_pri']} !important; border-color:{C['accent']} !important;
}}

/* Download button */
[data-testid="stDownloadButton"] > button {{
    background:{C['card_bg']} !important; color:{C['accent']} !important;
    border:1px solid {C['accent']} !important; border-radius:8px !important;
    font-size:13px !important; font-weight:500 !important;
}}

/* Inputs */
.stSelectbox > div > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background:{C['input_bg']} !important; border:1px solid {C['border']} !important;
    color:{C['text_pri']} !important; border-radius:8px !important; font-size:14px !important;
}}
.stSelectbox label, .stTextInput label, .stNumberInput label,
.stSlider label, .stRadio label {{
    color:{C['text_sec']} !important; font-size:11px !important;
    text-transform:uppercase; letter-spacing:0.5px; font-weight:500 !important;
}}
/* Selectbox dropdown list */
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{
    background:{C['card_bg']} !important; border:1px solid {C['border']} !important;
}}
[data-baseweb="menu"] li {{ color:{C['text_ter']} !important; }}
[data-baseweb="menu"] li:hover {{ background:{C['input_bg']} !important; }}

/* Radio */
.stRadio [data-testid="stMarkdownContainer"] p {{
    color:{C['text_sec']} !important; font-size:13px !important;
    text-transform:none !important; letter-spacing:0 !important;
}}

/* Checkbox */
.stCheckbox > label {{ color:{C['text_ter']} !important; font-size:13px !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background:transparent; border-bottom:1px solid {C['border']}; gap:0;
}}
.stTabs [data-baseweb="tab"] {{
    color:{C['text_sec']} !important; font-size:13px !important;
    padding:8px 16px !important; border-bottom:2px solid transparent !important;
    background:transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color:{C['accent']} !important; border-bottom-color:{C['accent']} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border:1px solid {C['border']} !important; border-radius:10px !important;
}}
[data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {{
    color:{C['text_sec']} !important; font-size:13px !important;
    background:{C['card_bg']} !important;
}}

/* Code blocks */
.stCode, pre {{
    background:{C['card_bg']} !important; border:1px solid {C['border']} !important;
    border-radius:10px !important; font-size:12px !important; color:{C['accent']} !important;
}}

/* Expander */
.streamlit-expanderHeader {{
    background:{C['card_bg']} !important; border:1px solid {C['border']} !important;
    color:{C['text_sec']} !important; border-radius:8px !important; font-size:13px !important;
}}
.streamlit-expanderContent {{
    background:{C['card_bg']} !important; border:1px solid {C['border']} !important;
    border-top:none !important;
}}

/* File uploader */
[data-testid="stFileUploader"] {{
    background:{C['card_bg']} !important; border:1px dashed {C['border']} !important;
    border-radius:10px !important;
}}
[data-testid="stFileUploader"] label {{ color:{C['text_pri']} !important; }}

/* Progress bar */
[data-testid="stProgressBar"] > div {{ background:{C['accent']} !important; }}

/* Divider */
hr {{ border-color:{C['border']} !important; margin:1rem 0 !important; }}

/* Alert / toast */
[data-testid="stAlert"] {{ background:{C['card_bg']} !important; border-radius:8px !important; }}

/* Radio button circles — fix black fill in light mode */
[data-testid="stRadio"] [role="radio"] {{
    border-color:{C['accent']} !important;
}}
[data-testid="stRadio"] [role="radio"][aria-checked="true"] {{
    background-color:{C['accent']} !important;
    border-color:{C['accent']} !important;
}}

/* Navigation radio labels */
section[data-testid="stSidebar"] [data-testid="stRadio"] label span {{
    color:{C['text_sec']} !important; font-size:13px !important;
}}

/* Checkbox accent colour */
input[type="checkbox"] {{ accent-color:{C['accent']} !important; }}

/* Number input arrow buttons */
[data-testid="stNumberInput"] button {{
    background:{C['input_bg']} !important;
    border:1px solid {C['border']} !important;
    color:{C['text_sec']} !important;
}}

/* Selectbox chevron icon */
[data-testid="stSelectbox"] svg {{ color:{C['text_sec']} !important; }}

/* Tooltip */
[data-testid="stTooltipIcon"] svg {{ color:{C['text_sec']} !important; }}

/* File uploader — fix "uploadupload" double text bug */
[data-testid="stFileUploader"] section {{
    padding:0 !important;
}}
[data-testid="stFileUploader"] button {{
    background:{C['accent']} !important;
    color:{C['btn_text']} !important;
    border:none !important;
    border-radius:6px !important;
    font-size:13px !important;
    font-weight:500 !important;
    padding:6px 16px !important;
}}
[data-testid="stFileUploader"] button span {{
    display:none !important;
}}
[data-testid="stFileUploader"] button::after {{
    content:"Browse files" !important;
    display:block !important;
    color:{C['btn_text']} !important;
    font-size:13px !important;
    font-weight:500 !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background:{C['card_bg']} !important;
    border:1px dashed {C['border']} !important;
    border-radius:10px !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{
    color:{C['text_sec']} !important;
}}

</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pill(label, level="Medium"):
    configs = {
        "Critical": (C["red_dim_css"],    f"1px solid {C['red']}",    C["red"]),
        "High":     (C["orange_dim_css"], f"1px solid {C['orange']}",  C["orange"]),
        "Medium":   (C["accent_dim_css"], f"1px solid {C['accent']}",  C["accent"]),
        "Low":      (C["green_dim_css"],  f"1px solid {C['green']}",   C["green"]),
    }
    bg, border, text = configs.get(level, configs["Medium"])
    return f'<span style="background:{bg};border:{border};color:{text};padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:0.5px;">{label}</span>'

def status_dot(colour):
    return f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{colour};box-shadow:0 0 5px {colour};margin-right:8px;flex-shrink:0;"></span>'

def section_label(text):
    st.markdown(f'<p style="font-size:11px;color:{C["text_sec"]};letter-spacing:1px;text-transform:uppercase;margin:1.25rem 0 0.5rem;font-weight:500;">{text}</p>', unsafe_allow_html=True)

def kv_row(key, val, val_col=None):
    vc = val_col or C["text_ter"]
    return f'<div style="display:flex;justify-content:space-between;padding:9px 14px;border-bottom:1px solid {C["border"]};"><span style="color:{C["text_sec"]};font-size:13px;">{key}</span><span style="color:{vc};font-size:13px;font-weight:500;">{val}</span></div>'

def chart_setup(fig, ax):
    """Apply consistent dark/light styling to a matplotlib figure."""
    fig.patch.set_facecolor(C["chart_bg"])
    ax.set_facecolor(C["chart_bg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["border"])
    ax.tick_params(colors=C["text_sec"])
    ax.xaxis.label.set_color(C["text_sec"])
    ax.yaxis.label.set_color(C["text_sec"])
    return fig, ax

def gauge_chart(score, risk_level):
    risk_colours = {
        "Critical": C["red"], "High": C["orange"],
        "Medium": C["accent"], "Low": C["green"],
    }
    colour = risk_colours.get(risk_level, C["accent"])

    fig, ax = plt.subplots(figsize=(3.2, 2.0), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(C["chart_bg"])
    ax.set_facecolor(C["chart_bg"])
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.15, 1.3); ax.axis("off")

    # Background arc
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            color=C["border"], linewidth=10, solid_capstyle="round")

    # Value arc
    end = np.pi - (score / 100) * np.pi
    theta_val = np.linspace(np.pi, end, 200)
    ax.plot(np.cos(theta_val), np.sin(theta_val),
            color=colour, linewidth=10, solid_capstyle="round")

    ax.text(0, 0.28, f"{score:.0f}", ha="center", va="center",
            fontsize=22, fontweight="700", color=C["text_pri"])
    ax.text(0, 0.08, "/ 100", ha="center", va="center",
            fontsize=9, color=C["text_sec"])
    ax.text(0, -0.08, risk_level.upper(), ha="center", va="center",
            fontsize=8, fontweight="600", color=colour)
    plt.tight_layout(pad=0)
    return fig


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    return load_models()

try:
    xgb, rf, iso, encoders, dept_stats, threshold = load_resources()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1.5rem 1.25rem 0.75rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
            <div style="width:34px;height:34px;background:{C['accent_dim_css']};
                border:1px solid {C['accent']};border-radius:8px;
                display:flex;align-items:center;justify-content:center;font-size:16px;">🛡</div>
            <div>
                <div style="color:{C['text_pri']};font-weight:700;font-size:15px;line-height:1.2;">ThreatSense</div>
                <div style="color:{C['text_sec']};font-size:11px;">v1.0 · COS720 UP</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"{C['toggle_icon']}  {C['toggle_label']}", use_container_width=True):
        st.session_state.theme = "light" if DARK else "dark"
        st.rerun()

    st.markdown(f"""
    <div style="padding:0.75rem 1.25rem 0;">
        <p style="font-size:11px;color:{C['text_sec']};letter-spacing:1px;text-transform:uppercase;margin:0.75rem 0 0.5rem;font-weight:500;">Engine Status</p>
        <div style="background:{C['app_bg']};border:1px solid {C['border']};border-radius:8px;padding:12px;margin-bottom:1rem;">
            <div style="font-size:13px;color:{C['text_ter']};margin-bottom:6px;">{status_dot(C['green'])}XGBoost Classifier</div>
            <div style="font-size:13px;color:{C['text_ter']};margin-bottom:6px;">{status_dot(C['green'])}Isolation Forest</div>
            <div style="font-size:13px;color:{C['text_ter']};margin-bottom:6px;">{status_dot(C['green'])}Forensic Engine</div>
            <div style="font-size:13px;color:{C['text_ter']};">{status_dot(C['accent'])}Threshold: {threshold if model_loaded else 'N/A'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("",
        ["Analyse Record", "Batch Processing", "Model Overview", "Analysis History"],
        label_visibility="collapsed")

    if st.session_state.history:
        st.markdown(f'<div style="padding:0 1.25rem;"><p style="font-size:12px;color:{C["text_sec"]};">Session: <span style="color:{C["accent"]};font-weight:600;">{len(st.session_state.history)}</span> analyses</p></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:0 1.25rem;margin-top:1rem;">
        <p style="font-size:11px;color:{C['text_sec']};letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;font-weight:500;">Detection Engine</p>
        <div style="font-size:12px;color:{C['text_sec']};line-height:2.2;">
            XGBoost (n=300, depth=6)<br>
            Isolation Forest (n=200)<br>
            Composite Risk Score 0–100<br>
            5-fold stratified CV<br>
            118,614 labelled records
        </div>
    </div>
    """, unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Engine offline. Run `python src/models/train.py` first.\n\n{model_error}")
    st.stop()


# ── Top bar ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%d %b %Y · %H:%M")
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
    padding:1rem 0;border-bottom:1px solid {C['border']};margin-bottom:1.5rem;">
    <div>
        <span style="color:{C['text_pri']};font-size:18px;font-weight:600;">{page}</span>
        <span style="color:{C['text_sec']};font-size:13px;margin-left:12px;">
            Hybrid Detection Engine · University of Pretoria COS720
        </span>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
        <span style="font-size:12px;color:{C['text_sec']};margin-right:4px;">{ts}</span>
        <span style="background:{C['accent_dim_css']};border:1px solid {C['accent']};
            color:{C['accent']};font-size:11px;padding:2px 10px;border-radius:20px;font-weight:500;">
            118,614 records
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analyse Record
# ════════════════════════════════════════════════════════════════════════════
if page == "Analyse Record":
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        section_label("Input Method")
        method = st.radio("", ["Sample profile", "Manual entry"],
                          horizontal=True, label_visibility="collapsed")
        record = {}

        if method == "Sample profile":
            section_label("Profile")
            profiles = load_sample_profiles()
            sel = st.selectbox("", [p["name"] for p in profiles],
                               label_visibility="collapsed")
            profile = next(p for p in profiles if p["name"] == sel)
            record  = {k: v for k, v in profile.items() if k != "name"}
            show_raw = st.checkbox("Show raw record", value=False)
            if show_raw:
                st.json(record)
        else:
            section_label("Employee")
            record["employee_department"] = st.selectbox("Department", [
                "Engineering Department","R&D Department","Operations and Manufacturing",
                "Finance Department","Information Technology","Security and Information Security",
                "Legal and Regulation","Project Management"])
            record["employee_campus"]   = st.selectbox("Campus",
                ["Campus A","Campus B","Campus C"])
            record["employee_position"] = st.selectbox("Position", [
                "Software Engineer","Systems Engineer","Test Engineer",
                "Data Scientist","Security Officer","Project Engineer","Algorithm Engineer"])
            c1, c2 = st.columns(2)
            record["employee_seniority_years"] = c1.number_input("Seniority (yrs)", 0, 40, 5)
            record["employee_classification"]  = c2.slider("Clearance level", 1, 5, 2)
            record["employee_origin_country"]  = st.text_input("Country of origin", "South Africa")
            c3, c4, c5 = st.columns(3)
            record["is_contractor"]           = int(c3.checkbox("Contractor"))
            record["has_foreign_citizenship"] = int(c4.checkbox("Foreign citizenship"))
            record["has_criminal_record"]     = int(c5.checkbox("Criminal record"))
            record["has_medical_history"]     = int(st.checkbox("Medical history"))

            section_label("Behavioural Indicators")
            c6, c7 = st.columns(2)
            record["total_printed_pages"]         = c6.number_input("Pages printed", 0, 1000, 0)
            record["num_printed_pages_off_hours"] = c7.number_input("Off-hours pages", 0, 500, 0)
            c8, c9 = st.columns(2)
            record["total_files_burned"] = c8.number_input("Files to ext. media", 0, 200, 0)
            record["burned_from_other"]  = int(c9.checkbox("Non-standard source"))

            section_label("Physical & Travel")
            c10, c11, c12 = st.columns(3)
            record["num_entries"]             = c10.number_input("Building entries", 0, 20, 1)
            record["num_unique_campus"]       = c11.number_input("Campuses visited", 1, 5, 1)
            record["hostility_country_level"] = c12.slider("Country risk", 0, 3, 0)
            record["is_abroad"]               = int(st.checkbox("Currently abroad"))
            record["trip_day_number"]         = float(st.number_input("Trip day", 0, 30, 0))
            c13, c14 = st.columns(2)
            record["late_exit_flag"]          = int(c13.checkbox("Late exit"))
            record["entry_during_weekend"]    = int(c14.checkbox("Weekend entry"))

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        run = st.button("Run Analysis", type="primary", use_container_width=True)

    with right:
        if run:
            with st.spinner("Running hybrid detection engine..."):
                t0       = time.time()
                result   = predict_record(record, xgb, rf, iso, encoders, dept_stats, threshold)
                X        = preprocess_single_record(record, encoders, dept_stats)
                local_df = get_local_explanation(rf, X, FEATURE_COLS)
                report   = generate_forensic_report(local_df, result)
                breakdown= get_risk_score_breakdown(local_df)
                elapsed  = time.time() - t0

            # Save to history
            st.session_state.history.append({
                "time":   datetime.now().strftime("%H:%M:%S"),
                "label":  result["label"],
                "score":  result["composite_risk_score"],
                "risk":   result["risk_level"],
            })

            is_mal  = result["prediction"] == 1
            v_col   = C["red"]   if is_mal else C["green"]
            v_dim   = C["red_dim_css"]   if is_mal else C["green_dim_css"]
            v_icon  = "⚠" if is_mal else "✓"
            verdict = "Threat Detected" if is_mal else "No Threat Detected"
            sub     = "Immediate review recommended · Do not alert employee" \
                      if is_mal else "Behaviour consistent with normal departmental activity"

            # ── OUTPUT 1: Classification Result ──────────────────────────────
            section_label("① Classification Result")
            st.markdown(f"""
            <div style="background:{v_dim};border:1px solid {v_col}33;
                border-left:3px solid {v_col};border-radius:10px;
                padding:14px 18px;margin-bottom:1rem;
                display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="color:{v_col};font-weight:700;font-size:16px;">{v_icon} {verdict}</div>
                    <div style="color:{C['text_sec']};font-size:13px;margin-top:3px;">{sub}</div>
                </div>
                <div style="text-align:right;">
                    {pill(result["risk_level"].upper(), result["risk_level"])}
                    <div style="color:{C['text_sec']};font-size:11px;margin-top:5px;">
                        Analysed in {elapsed:.2f}s
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── OUTPUT 2: Confidence Score ────────────────────────────────────
            section_label("② Confidence Score")
            g_col, k1, k2, k3 = st.columns([1.2, 1, 1, 1])
            with g_col:
                fig_g = gauge_chart(result["composite_risk_score"], result["risk_level"])
                st.pyplot(fig_g)
                plt.close()
            k1.metric("XGB Confidence",  f"{result['rf_confidence']}%",
                      help="XGBoost probability of the predicted class")
            k2.metric("Malicious Prob",  f"{result['prob_malicious']}%",
                      help="Raw probability of malicious classification")
            k3.metric("Normal Prob",     f"{result['prob_normal']}%",
                      help="Raw probability of normal classification")

            if_col = C["orange"] if result["isolation_forest_anomaly"] else C["green"]
            if_txt = "Isolation Forest: anomaly detected — statistically unusual behaviour" \
                     if result["isolation_forest_anomaly"] \
                     else "Isolation Forest: within normal range"
            st.markdown(f"""
            <div style="font-size:13px;color:{if_col};margin:6px 0 0;
                display:flex;align-items:center;gap:6px;">
                {status_dot(if_col)}{if_txt}
            </div>
            """, unsafe_allow_html=True)

            # ── OUTPUT 3: Key Risk Indicators ─────────────────────────────────
            # ── OUTPUT 3: AI Explainability ───────────────────────────────────
            section_label("③ AI Explainability")

            risk_colour_map = {
                "Critical": C["red"], "High": C["orange"],
                "Medium": C["accent"], "Low": C["green"],
            }

            # All indicators ranked
            all_indicators = breakdown[~breakdown["description"].isin([
                "Department","Campus","Job position","Country of origin"
            ])].head(8)
            top_indicators = all_indicators[all_indicators["risk_level"].isin(["High","Medium"])]
            normal_indicators = all_indicators[all_indicators["risk_level"] == "Low"].head(3)

            # ── Manager summary ───────────────────────────────────────────────
            # Build a plain-English verdict that a non-technical manager can act on
            top_names = [r["description"] for _, r in top_indicators.head(3).iterrows()]
            top_vals  = []
            for _, r in top_indicators.head(3).iterrows():
                v = r["value"]
                top_vals.append(str(int(v)) if isinstance(v, (int,float)) and v == int(v) else f"{v:.1f}")

            if is_mal:
                if len(top_names) >= 2:
                    reason_str = f"{top_names[0]} ({top_vals[0]}) and {top_names[1]} ({top_vals[1]})"
                elif len(top_names) == 1:
                    reason_str = f"{top_names[0]} ({top_vals[0]})"
                else:
                    reason_str = "multiple elevated behavioural indicators"
                manager_text = (
                    f"<b>Why was this flagged?</b> This employee was flagged as a potential "
                    f"insider threat primarily because of elevated <b>{reason_str}</b>. "
                    f"These values are significantly higher than what is normal for their "
                    f"department, which is a recognised indicator of data exfiltration or "
                    f"unauthorised activity. The AI model is <b>{result['rf_confidence']}% confident</b> "
                    f"in this classification."
                )
                action_text = (
                    "Recommended action: Escalate to your security team for review. "
                    "Preserve logs before contacting the employee."
                )
                action_col = C["red"]
            elif result["isolation_forest_anomaly"]:
                manager_text = (
                    f"<b>Why was this flagged?</b> The primary classifier considers this employee "
                    f"<b>Normal</b>, however the anomaly detection system identified behaviour that "
                    f"is statistically unusual compared to peers in the same department. "
                    f"No single indicator is alarming on its own, but the combination is atypical."
                )
                action_text = (
                    "Recommended action: No immediate escalation required. "
                    "Schedule a routine check-in and continue monitoring."
                )
                action_col = C["orange"]
            else:
                manager_text = (
                    f"<b>Why was this cleared?</b> This employee's activity across all 25 behavioural "
                    f"indicators is consistent with normal patterns for their role and department. "
                    f"Both the supervised and anomaly detection models agree on this assessment "
                    f"with <b>{result['prob_normal']}% confidence</b>."
                )
                action_text = "Recommended action: No action required. Routine monitoring continues."
                action_col = C["green"]

            # Build as plain string — avoids Streamlit treating # colours as headings
            summary_html = (
                f'<div style="background:{C["card_bg"]};border:1px solid {C["border"]};'
                f'border-radius:10px;padding:16px;margin-bottom:12px;">'
                f'<div style="font-size:13px;color:{C["text_ter"]};line-height:1.7;margin-bottom:12px;">'
                f'{manager_text}'
                f'</div>'
                f'<div style="background:transparent;border:1px solid {action_col};'
                f'border-left:3px solid {action_col};border-radius:6px;'
                f'padding:10px 14px;font-size:13px;color:{action_col};font-weight:500;">'
                f'{action_text}'
                f'</div>'
                f'</div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)

            # ── Feature importance ranking ────────────────────────────────────
            # ── Feature importance ranking — single HTML block (no split divs) ───
            st.markdown(f'<p style="font-size:12px;color:{C["text_sec"]};font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin:4px 0 8px;">Feature Importance Ranking</p>', unsafe_allow_html=True)

            # Build entire ranking as one HTML string to avoid unclosed div issues
            rows_html = ""
            for i, (_, row) in enumerate(all_indicators.iterrows()):
                rc    = risk_colour_map.get(row["risk_level"], C["accent"])
                val   = row["value"]
                val_s = str(int(val)) if isinstance(val, (int, float)) and val == int(val) else f"{val:.2f}"
                pct   = min(int(row["contribution"] * 2000), 100)
                is_abnormal = row["risk_level"] in ("Critical", "High", "Medium")
                flag_icon   = "⚠" if row["risk_level"] in ("Critical", "High") else "▲" if row["risk_level"] == "Medium" else "✓"
                flag_col    = rc if is_abnormal else C["green"]
                mb          = "12px" if i < len(all_indicators) - 1 else "0"
                rows_html += f"""
                <div style="margin-bottom:{mb};">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="color:{flag_col};font-size:12px;width:14px;">{flag_icon}</span>
                            <span style="color:{C['text_ter']};font-size:13px;">{row['description']}</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px;">
                            <span style="color:{C['text_sec']};font-size:12px;">
                                Recorded: <b style="color:{rc if is_abnormal else C['text_ter']}">{val_s}</b>
                            </span>
                            <span style="background:{rc}20;border:1px solid {rc}50;color:{rc};
                                font-size:10px;padding:1px 8px;border-radius:10px;font-weight:600;
                                min-width:52px;text-align:center;">{row['risk_level']}</span>
                        </div>
                    </div>
                    <div style="background:{C['border']};border-radius:3px;height:3px;">
                        <div style="background:{rc};border-radius:3px;height:3px;width:{pct}%;"></div>
                    </div>
                </div>"""

            st.markdown(
                f'<div style="background:{C["card_bg"]};border:1px solid {C["border"]};'
                f'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
                f'{rows_html}</div>',
                unsafe_allow_html=True,
            )


            # ── Feature importance global chart (inline, small) ───────────────
            st.markdown(f'<p style="font-size:12px;color:{C["text_sec"]};font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin:4px 0 8px;">Global Model Feature Importance</p>', unsafe_allow_html=True)
            imp_df = get_global_feature_importance(rf, FEATURE_COLS).head(8)
            fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
            fig_imp, ax_imp = chart_setup(fig_imp, ax_imp)
            imp_cols = [C["accent"] if v >= imp_df["importance"].max()*0.5 else C["accent_muted"]
                        for v in imp_df["importance"].iloc[::-1].values]
            ax_imp.barh(imp_df["description"].iloc[::-1],
                        imp_df["importance"].iloc[::-1],
                        color=imp_cols, height=0.5)
            ax_imp.set_xlabel("Importance", fontsize=9)
            ax_imp.grid(axis="x", color=C["chart_grid"], linewidth=0.5)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_imp)
            plt.close()

            # PDF download
            pdf_bytes = generate_pdf_report(result, local_df, breakdown, record, theme=T)
            fname = f"threatsense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                "⬇  Download Forensic Report (PDF)",
                data=pdf_bytes, file_name=fname,
                mime="application/pdf", use_container_width=True,
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Detailed tabs — for technical reviewers
            section_label("Technical Detail")
            t1, t2, t3, t4 = st.tabs(["Full Forensic Report", "Risk Breakdown", "Probability", "Record Inspector"])



            with t1:
                st.code(report, language=None)

            with t2:
                df_show = breakdown.copy()
                df_show.columns = ["Indicator", "Value", "Contribution", "Risk Level"]
                st.dataframe(df_show, use_container_width=True, hide_index=True)

                # Contribution bar chart — safe hex colours only
                top8   = breakdown.head(8)
                risk_colour_map = {
                    "Critical": C["red"], "High": C["orange"],
                    "Medium":   C["accent"], "Low": C["green"],
                }
                bar_colours = [risk_colour_map.get(r, C["accent"]) for r in top8["risk_level"]]

                fig2, ax2 = plt.subplots(figsize=(5, 3))
                fig2, ax2 = chart_setup(fig2, ax2)
                ax2.barh(top8["description"].iloc[::-1],
                         top8["contribution"].iloc[::-1],
                         color=list(reversed(bar_colours)), height=0.5)
                ax2.set_xlabel("Contribution", fontsize=9)
                ax2.grid(axis="x", color=C["chart_grid"], linewidth=0.8)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig2)
                plt.close()

            with t3:
                fig3, ax3 = plt.subplots(figsize=(5, 1.8))
                fig3, ax3 = chart_setup(fig3, ax3)
                bars = ax3.barh(
                    ["Normal", "Malicious"],
                    [result["prob_normal"], result["prob_malicious"]],
                    color=[C["green"], C["red"]], height=0.45,
                )
                for bar, v in zip(bars, [result["prob_normal"], result["prob_malicious"]]):
                    ax3.text(min(v + 1, 92), bar.get_y() + bar.get_height() / 2,
                             f"{v:.1f}%", va="center",
                             color=C["text_pri"], fontsize=11, fontweight="500")
                ax3.set_xlim(0, 100)
                ax3.set_xlabel("Probability (%)", fontsize=10)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig3)
                plt.close()

            with t4:
                section_label("Raw Employee Record")
                LABELS = {
                    "employee_department":"Department","employee_campus":"Campus",
                    "employee_position":"Position","employee_seniority_years":"Seniority (yrs)",
                    "is_contractor":"Contractor","employee_classification":"Clearance level",
                    "has_foreign_citizenship":"Foreign citizenship",
                    "has_criminal_record":"Criminal record",
                    "has_medical_history":"Medical history",
                    "employee_origin_country":"Country of origin",
                    "total_printed_pages":"Pages printed",
                    "num_printed_pages_off_hours":"Off-hours pages",
                    "total_files_burned":"Files burned",
                    "burned_from_other":"Non-standard source",
                    "is_abroad":"Currently abroad","trip_day_number":"Trip day",
                    "hostility_country_level":"Country risk","num_entries":"Building entries",
                    "num_unique_campus":"Campuses visited","late_exit_flag":"Late exit",
                    "entry_during_weekend":"Weekend entry",
                }
                st.markdown(f'<div style="background:{C["card_bg"]};border:1px solid {C["border"]};border-radius:10px;overflow:hidden;">', unsafe_allow_html=True)
                for k, v in record.items():
                    flag_red = k in ("has_criminal_record","has_medical_history") and v == 1
                    vc = C["red"] if flag_red else C["text_ter"]
                    st.markdown(kv_row(LABELS.get(k, k), str(v), vc), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background:{C['card_bg']};border:1px dashed {C['border']};
                border-radius:12px;padding:4rem 2rem;text-align:center;margin-top:0.5rem;">
                <div style="font-size:36px;margin-bottom:14px;">🛡</div>
                <div style="color:{C['text_pri']};font-size:15px;font-weight:500;margin-bottom:6px;">
                    Ready to analyse
                </div>
                <div style="color:{C['text_sec']};font-size:13px;">
                    Configure a profile on the left, then click Run Analysis
                </div>
                <div style="color:{C['text_sec']};font-size:12px;margin-top:6px;">
                    Results include a downloadable PDF forensic report
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Processing
# ════════════════════════════════════════════════════════════════════════════
elif page == "Batch Processing":
    section_label("Upload Dataset")
    st.markdown(f'<p style="color:{C['text_sec']};font-size:13px;margin-bottom:8px;">Drop a CSV file containing employee behavioural records to analyse in bulk.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        prog = st.progress(0, text="Preprocessing...")
        prog.progress(30, text="Running hybrid detection engine...")
        results = predict_dataframe(df, xgb, rf, iso, encoders, dept_stats, threshold)
        prog.progress(100, text="Complete")
        prog.empty()
        st.toast(f"Processed {len(results):,} records", icon="✅")

        mal       = int((results["prediction"] == 1).sum())
        norm      = int((results["prediction"] == 0).sum())
        critical  = int((results["risk_level"] == "Critical").sum())
        anomalies = int(results["if_anomaly"].sum())

        section_label("Batch Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",       f"{len(results):,}")
        c2.metric("Threats",     f"{mal:,}")
        c3.metric("Clear",       f"{norm:,}")
        c4.metric("Critical",    f"{critical:,}")
        c5.metric("IF Anomalies",f"{anomalies:,}")

        col_a, col_b = st.columns([1, 2])

        with col_a:
            section_label("Risk Distribution")
            counts = results["risk_level"].value_counts()
            labels = ["Critical", "High", "Medium", "Low"]
            sizes  = [counts.get(l, 0) for l in labels]
            colours_pie = [C["red"], C["orange"], C["accent"], C["green"]]

            fig_d, ax_d = plt.subplots(figsize=(3.5, 3.5))
            fig_d.patch.set_facecolor(C["chart_bg"])
            ax_d.set_facecolor(C["chart_bg"])
            wedges, texts, autos = ax_d.pie(
                sizes, labels=labels, colors=colours_pie,
                autopct="%1.0f%%", startangle=90,
                wedgeprops=dict(width=0.55, edgecolor=C["chart_bg"], linewidth=2),
                textprops=dict(color=C["text_sec"], fontsize=9),
            )
            for a in autos:
                a.set_color(C["text_pri"]); a.set_fontsize(9); a.set_fontweight("600")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig_d)
            plt.close()

        with col_b:
            section_label("Results")
            show = results[["label","composite_risk_score","risk_level",
                             "xgb_confidence","if_anomaly","prob_malicious"]]
            st.dataframe(show, use_container_width=True, height=260)

        col_c, col_d = st.columns(2)
        with col_c:
            csv = results.to_csv(index=False).encode()
            st.download_button("⬇  Export Results CSV", csv,
                               "threatsense_batch.csv", "text/csv",
                               use_container_width=True)
        with col_d:
            section_label("Score Distribution")
            mal_s  = results[results["prediction"]==1]["composite_risk_score"]
            norm_s = results[results["prediction"]==0]["composite_risk_score"]
            fig_h, ax_h = plt.subplots(figsize=(4, 2))
            fig_h, ax_h = chart_setup(fig_h, ax_h)
            ax_h.hist(norm_s, bins=30, color=C["green"], alpha=0.65, label="Normal")
            ax_h.hist(mal_s,  bins=30, color=C["red"],   alpha=0.75, label="Malicious")
            ax_h.set_xlabel("Composite Risk Score", fontsize=9)
            ax_h.grid(axis="y", color=C["chart_grid"], linewidth=0.8)
            ax_h.legend(fontsize=8, labelcolor=C["text_sec"],
                        facecolor=C["chart_bg"], edgecolor=C["border"])
            plt.tight_layout(pad=0.4)
            st.pyplot(fig_h)
            plt.close()
    else:
        st.markdown(f"""
        <div style="background:{C['card_bg']};border:1px dashed {C['border']};
            border-radius:12px;padding:3rem;text-align:center;">
            <div style="color:{C['text_pri']};font-size:15px;font-weight:500;margin-bottom:6px;">
                Upload an employee dataset
            </div>
            <div style="color:{C['text_sec']};font-size:13px;">
                CSV format · Same columns as the training dataset
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Overview
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Overview":
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        section_label("Performance Metrics")
        metrics = [
            ("Accuracy",           "97.23%", C["text_ter"]),
            ("Precision",          "67.61%", C["accent"]),
            ("Recall",             "93.19%", C["green"]),
            ("F1-Score",           "78.37%", C["accent"]),
            ("False Positive Rate","2.54%",  C["orange"]),
            ("False Negative Rate","6.81%",  C["orange"]),
        ]
        st.markdown(f'<div style="background:{C["card_bg"]};border:1px solid {C["border"]};border-radius:10px;overflow:hidden;">', unsafe_allow_html=True)
        for name, val, col in metrics:
            st.markdown(kv_row(name, val, col), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        section_label("Architecture")
        rows = [
            ("Primary classifier",  "XGBoost (n=300, depth=6)"),
            ("Anomaly detector",    "Isolation Forest (n=200)"),
            ("Explainability",      "Random Forest importances"),
            ("Imbalance handling",  "scale_pos_weight = 17"),
            ("Threshold",           "0.65 (F1-optimised)"),
            ("Validation",          "5-fold stratified CV"),
            ("Training samples",    "94,891"),
            ("Test samples",        "23,723"),
            ("Features",            "25 (21 raw + 4 deviation)"),
        ]
        st.markdown(f'<div style="background:{C["card_bg"]};border:1px solid {C["border"]};border-radius:10px;overflow:hidden;">', unsafe_allow_html=True)
        for k, v in rows:
            st.markdown(kv_row(k, v), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        section_label("Gap Addressed")
        st.markdown(f"""
        <div style="background:{C['card_bg']};border:1px solid {C['border']};
            border-left:2px solid {C['accent']};border-radius:10px;
            padding:14px 16px;font-size:13px;color:{C['text_sec']};line-height:1.7;">
            Shoderu et al. (2025) validated DFR-BUST using only unsupervised detection on a
            simulated single-user scenario with no labelled ground truth, and explicitly
            recommended hybrid models and real-dataset validation as future work (§F.3).
            This system implements that recommendation: XGBoost + Isolation Forest validated
            on 118,614 labelled records with 5-fold cross-validation.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        section_label("Feature Importance (Top 15)")
        imp   = get_global_feature_importance(rf, FEATURE_COLS).head(15)
        vals  = imp["importance"].iloc[::-1].values
        descs = imp["description"].iloc[::-1].values
        max_v = vals.max()
        # Safe hex colours only — no rgba for matplotlib
        bar_c = [C["accent"] if v >= max_v * 0.5 else C["accent_muted"] for v in vals]

        fig4, ax4 = plt.subplots(figsize=(6, 6.5))
        fig4, ax4 = chart_setup(fig4, ax4)
        ax4.barh(range(len(vals)), vals, color=bar_c, height=0.55)
        ax4.set_yticks(range(len(descs)))
        ax4.set_yticklabels(descs, fontsize=10, color=C["text_sec"])
        ax4.set_xlabel("Importance", fontsize=10)
        ax4.set_xlim(0, max_v * 1.15)
        ax4.grid(axis="x", color=C["chart_grid"], linewidth=0.8)
        plt.tight_layout(pad=0.8)
        st.pyplot(fig4)
        plt.close()

        section_label("Cross-Validation (5-fold)")
        cv_data = [
            ("Accuracy",  0.9705, 0.0009),
            ("Precision", 0.6550, 0.0078),
            ("Recall",    0.9550, 0.0081),
            ("F1-Score",  0.7770, 0.0052),
        ]
        for name, mean, std in cv_data:
            pct = mean * 100
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:{C['text_ter']};font-size:12px;">{name}</span>
                    <span style="color:{C['text_sec']};font-size:12px;">
                        {pct:.2f}% ± {std*100:.2f}%
                    </span>
                </div>
                <div style="background:{C['border']};border-radius:4px;height:5px;">
                    <div style="background:{C['accent']};border-radius:4px;height:5px;width:{pct:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Analysis History
# ════════════════════════════════════════════════════════════════════════════
elif page == "Analysis History":
    if not st.session_state.history:
        st.markdown(f"""
        <div style="background:{C['card_bg']};border:1px dashed {C['border']};
            border-radius:12px;padding:3rem;text-align:center;">
            <div style="color:{C['text_pri']};font-size:15px;font-weight:500;margin-bottom:6px;">
                No analyses yet
            </div>
            <div style="color:{C['text_sec']};font-size:13px;">
                Run analyses in the Analyse Record page to see them here
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        section_label(f"Session History — {len(st.session_state.history)} analyses")
        for h in reversed(st.session_state.history):
            is_m   = h["label"] == "Malicious"
            h_col  = C["red"] if is_m else C["green"]
            h_icon = "⚠" if is_m else "✓"
            st.markdown(f"""
            <div style="background:{C['card_bg']};border:1px solid {C['border']};
                border-radius:10px;padding:12px 16px;margin-bottom:8px;
                display:flex;align-items:center;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="color:{h_col};font-size:18px;">{h_icon}</span>
                    <div>
                        <div style="color:{C['text_pri']};font-size:13px;font-weight:500;">
                            {h['label']}
                        </div>
                        <div style="color:{C['text_sec']};font-size:12px;">
                            Analysed at {h['time']}
                        </div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="text-align:right;">
                        <div style="color:{C['text_ter']};font-size:14px;font-weight:600;">
                            {h['score']}/100
                        </div>
                        <div style="color:{C['text_sec']};font-size:11px;">Composite score</div>
                    </div>
                    {pill(h['risk'].upper(), h['risk'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()