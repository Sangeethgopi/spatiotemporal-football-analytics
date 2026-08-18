import streamlit as st
import os

st.set_page_config(
    page_title="Spatiotemporal Football Analytics Framework",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Design tokens
#   bg           #050b16   near-black navy pitch-at-night
#   bg-glow      #16326b   floodlight glow behind the hero
#   accent-blue  #3b82f6   primary action / tracking-vector color
#   accent-pitch #22c55e   single football accent, used sparingly (live dot,
#                          card top-rule, hover) so it reads as "the pitch"
#   text-primary #f5f7fa
#   text-muted   #8291a8
#   line         rgba(255,255,255,0.07)
#
# Typography:
#   Display  -> Space Grotesk  (technical, slightly sporty geometric grotesk)
#   Body     -> Inter
#   Data/UI  -> JetBrains Mono (badge, nav eyebrow, coordinates strip — reads
#               like telemetry / tracking-data readout, tying back to the
#               thesis subject: high-frequency tracking coordinates)
#
# Signature element:
#   A faint animated "vector field" behind the hero — short dashed arrows
#   drifting like player-motion vectors on a tracking frame, plus a hairline
#   pitch arc in the corner. This is the one visual risk, kept quiet.
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #050b16;
        --bg-glow: #16326b;
        --accent-blue: #3b82f6;
        --accent-blue-dim: #2563eb;
        --accent-pitch: #22c55e;
        --text-primary: #f5f7fa;
        --text-muted: #8291a8;
        --line: rgba(255,255,255,0.07);
    }

    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none !important; }
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: var(--bg) !important;
        background-image:
            radial-gradient(circle at 50% -10%, var(--bg-glow) 0%, transparent 55%),
            repeating-linear-gradient(120deg, rgba(59,130,246,0.05) 0px, rgba(59,130,246,0.05) 1px, transparent 1px, transparent 90px),
            repeating-linear-gradient(60deg, rgba(59,130,246,0.05) 0px, rgba(59,130,246,0.05) 1px, transparent 1px, transparent 90px);
        color: var(--text-primary) !important;
    }

    .block-container { padding-top: 0 !important; max-width: 1280px; }

    @media (prefers-reduced-motion: no-preference) {
        .vector-field { animation: driftField 26s linear infinite; }
        .badge-dot { animation: pulseDot 2.2s ease-in-out infinite; }
    }
    @keyframes driftField {
        0%   { background-position: 0px 0px, 0px 0px; }
        100% { background-position: 120px 60px, -90px 40px; }
    }
    @keyframes pulseDot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.45); }
        50%      { opacity: 0.75; box-shadow: 0 0 0 5px rgba(34,197,94,0); }
    }

    /* ---------------- Top navbar ---------------- */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 26px 48px;
        position: relative;
        border-bottom: 1px solid var(--line);
    }
    .top-nav-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .top-nav-links {
        display: flex;
        gap: 34px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
    }
    .top-nav-links span {
        position: relative;
        padding-bottom: 6px;
        cursor: default;
    }
    .top-nav-links span.active {
        color: var(--text-primary);
    }
    .top-nav-links span.active::after {
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 2px;
        background: var(--accent-blue);
        border-radius: 2px;
    }

    /* ---------------- Hero ---------------- */
    .hero-container {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 62vh;
        text-align: center;
        padding: 90px 24px 20px 24px;
        overflow: hidden;
    }
    .vector-field {
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: 0.55;
        background-image:
            linear-gradient(100deg, transparent 46%, rgba(59,130,246,0.5) 47%, rgba(59,130,246,0.5) 48%, transparent 49%),
            linear-gradient(20deg, transparent 46%, rgba(34,197,94,0.35) 47%, rgba(34,197,94,0.35) 48%, transparent 49%);
        background-size: 140px 140px, 170px 170px;
        mask-image: radial-gradient(circle at 50% 30%, black 0%, transparent 70%);
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        background: rgba(59,130,246,0.10);
        border: 1px solid rgba(59,130,246,0.25);
        color: #93c5fd;
        padding: 7px 18px 7px 14px;
        border-radius: 99px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 28px;
        z-index: 1;
    }
    .badge-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent-pitch);
        flex-shrink: 0;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.4rem;
        font-weight: 700;
        line-height: 1.12;
        letter-spacing: -0.02em;
        margin-bottom: 22px;
        max-width: 880px;
        z-index: 1;
        background: linear-gradient(180deg, #ffffff 0%, #c9d6ea 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.05rem;
        max-width: 700px;
        line-height: 1.7;
        margin-bottom: 38px;
        z-index: 1;
    }
    .hero-subtitle b { color: #c3d0e4; font-weight: 500; }

    /* ---------------- CTA button ---------------- */
    div.stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-dim)) !important;
        color: white !important;
        border: none !important;
        padding: 10px 40px !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: -0.01em;
        box-shadow: 0 10px 28px rgba(37,99,235,0.35) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        min-height: 54px !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 16px 34px rgba(37,99,235,0.45) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    .cta-caption {
        text-align: center;
        margin-top: 16px;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .cta-caption span { color: var(--accent-pitch); }

    /* ---------------- Feature grid ---------------- */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        max-width: 1200px;
        margin: 64px auto 40px auto;
        padding: 0 24px;
    }
    .feature-card {
        position: relative;
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--line);
        border-top: 2px solid transparent;
        border-radius: 10px;
        padding: 26px 22px;
        text-align: left;
        transition: border-color 0.2s ease, transform 0.2s ease, background 0.2s ease;
    }
    .feature-card:hover {
        border-color: rgba(59,130,246,0.35);
        border-top-color: var(--accent-pitch);
        background: rgba(255,255,255,0.035);
        transform: translateY(-3px);
    }
    .feature-icon {
        color: var(--accent-blue);
        font-size: 1.5rem;
        margin-bottom: 16px;
    }
    .feature-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 9px;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }
    .feature-desc {
        color: var(--text-muted);
        font-size: 0.85rem;
        line-height: 1.6;
    }

    @media (max-width: 900px) {
        .top-nav-links { display: none; }
        .hero-title { font-size: 2.3rem; }
        .features-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 480px) {
        .features-grid { grid-template-columns: 1fr; }
        .top-nav { padding: 20px 22px; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-nav">
    <div class="top-nav-logo">⚽ Spatiotemporal Football Analytics</div>
    <div class="top-nav-links">
        <span class="active">Home</span>
        <span>About</span>
        <span>Framework</span>
        <span>Methodology</span>
        <span>Dashboard</span>
    </div>
</div>

<div class="hero-container">
    <div class="vector-field"></div>
    <div class="badge"><span class="badge-dot"></span>MSc Data Science &amp; AI · Thesis Project</div>
    <div class="hero-title">A framework for reading the press before it happens</div>
    <div class="hero-subtitle">
        A temporally causal, data-driven framework for estimating and detecting turnover-associated
        pressing dynamics in football, built on <b>high-frequency tracking data</b> and <b>computer vision</b>.<br><br>
        Standardized tracking datasets and computer vision pipelines, unified in one platform to surface
        what drives team pressing behavior.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Enter Dashboard  →", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")

st.markdown("""
<div class="cta-caption"><span>●</span> Secure&nbsp;&nbsp;·&nbsp;&nbsp;Scalable&nbsp;&nbsp;·&nbsp;&nbsp;Research-Driven</div>

<div class="features-grid">
    <div class="feature-card">
        <div class="feature-icon">🗄️</div>
        <div class="feature-title">Data Integration</div>
        <div class="feature-desc">Metrica tracking data and computer vision video analysis, unified in one platform.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Spatiotemporal Modeling</div>
        <div class="feature-desc">Temporally causal models that capture pressing dynamics as they unfold around turnovers.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">AI &amp; CV Pipelines</div>
        <div class="feature-desc">State-of-the-art computer vision for player and ball tracking, and event detection.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Actionable Insights</div>
        <div class="feature-desc">Visual analytics and metrics that turn pressing behavior into decisions coaches can use.</div>
    </div>
</div>
""", unsafe_allow_html=True)
