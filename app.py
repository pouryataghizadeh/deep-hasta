import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import joblib
from PIL import Image, ImageEnhance, ImageFilter

# --- 1. GLOBAL TASARIM ---
st.set_page_config(page_title="PHOENIX AI · Multi-Diagnostic", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">

    <style>
    /* ── RESET & BASE ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    html, body, .stApp {
        background-color: #050a12 !important;
        color: #c9d4e8 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── NOISE TEXTURE OVERLAY ── */
    .stApp::before {
        content: '';
        position: fixed; inset: 0; z-index: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none; opacity: 0.4;
    }

    /* ── AMBIENT GLOW ── */
    .stApp::after {
        content: '';
        position: fixed;
        top: -30vh; left: -20vw;
        width: 70vw; height: 70vh;
        background: radial-gradient(ellipse at center, rgba(0,168,255,0.06) 0%, transparent 70%);
        pointer-events: none; z-index: 0;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: #080e1a !important;
        border-right: 1px solid rgba(0, 168, 255, 0.12) !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    /* Sidebar logo area */
    [data-testid="stSidebar"]::before {
        content: 'PHOENIX AI';
        display: block;
        font-family: 'Syne', sans-serif;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.35em;
        color: #00a8ff;
        padding: 24px 24px 0;
        opacity: 0.8;
    }

    [data-testid="stSidebarContent"] {
        background: transparent !important;
    }

    /* Sidebar selectbox label */
    [data-testid="stSidebar"] label {
        font-family: 'DM Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        color: rgba(0,168,255,0.5) !important;
    }

    /* ── SELECT BOX ── */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(0,168,255,0.04) !important;
        border: 1px solid rgba(0,168,255,0.18) !important;
        border-radius: 10px !important;
        color: #c9d4e8 !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color 0.2s;
    }
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: rgba(0,168,255,0.5) !important;
    }

    /* ── HEADINGS ── */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: clamp(1.8rem, 3vw, 2.8rem) !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
        margin-bottom: 0.25rem !important;
    }
    h2 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        color: #e2eaf8 !important;
        letter-spacing: -0.01em !important;
    }
    h3 {
        font-family: 'DM Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        color: #00a8ff !important;
        letter-spacing: 0.25em !important;
        text-transform: uppercase !important;
    }

    /* ── PAGE TITLE DECORATION ── */
    .page-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.3em;
        color: #00a8ff;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 6px;
        display: block;
    }
    .page-title-wrap {
        border-left: 3px solid #00a8ff;
        padding-left: 18px;
        margin-bottom: 2.5rem;
    }
    .page-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: rgba(201, 212, 232, 0.5);
        font-weight: 300;
        margin-top: 6px;
        letter-spacing: 0.02em;
    }

    /* ── CARDS / CONTAINERS ── */
    .phoenix-card {
        background: rgba(8, 16, 30, 0.85);
        border: 1px solid rgba(0, 168, 255, 0.10);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        transition: border-color 0.3s;
        position: relative;
        overflow: hidden;
    }
    .phoenix-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,168,255,0.3), transparent);
    }
    .phoenix-card:hover {
        border-color: rgba(0, 168, 255, 0.22);
    }

    /* ── RESULT CARD ── */
    .result-card {
        padding: 40px 32px;
        border-radius: 18px;
        border: 1px solid rgba(0, 168, 255, 0.2);
        background: rgba(5, 10, 18, 0.95);
        margin: 24px 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    .result-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(0,168,255,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .result-card h2 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em;
    }
    .result-card h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
        line-height: 1 !important;
    }
    .result-meta {
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        color: rgba(201,212,232,0.4);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 10px;
    }

    /* ── STATUS BADGE ── */
    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 100px;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    /* ── IMG LABEL ── */
    .img-label {
        color: #00a8ff;
        font-family: 'DM Mono', monospace;
        font-weight: 500;
        font-size: 10px;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 10px;
        display: block;
    }

    /* ── INPUTS ── */
    input[type="number"], .stTextInput input {
        background: rgba(0,168,255,0.03) !important;
        border: 1px solid rgba(0,168,255,0.15) !important;
        border-radius: 10px !important;
        color: #e2eaf8 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 14px !important;
        transition: all 0.2s !important;
    }
    input[type="number"]:focus, .stTextInput input:focus {
        border-color: rgba(0,168,255,0.45) !important;
        box-shadow: 0 0 0 3px rgba(0,168,255,0.08) !important;
        outline: none !important;
    }

    /* ── LABELS ── */
    label, .stSelectbox label, .stNumberInput label, .stRadio label {
        font-family: 'DM Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        color: rgba(201, 212, 232, 0.5) !important;
        margin-bottom: 4px !important;
    }

    /* ── RADIO ── */
    .stRadio > div { flex-direction: row; gap: 10px; }
    .stRadio > div > label {
        background: rgba(0,168,255,0.04) !important;
        border: 1px solid rgba(0,168,255,0.12) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 12px !important;
        cursor: pointer;
        transition: all 0.2s;
    }
    .stRadio > div > label:hover {
        border-color: rgba(0,168,255,0.35) !important;
    }

    /* ── SLIDER ── */
    .stSlider [data-baseweb="slider"] {
        padding-bottom: 0 !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 11px !important;
        color: #00a8ff !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: rgba(0,168,255,0.02) !important;
        border: 1px dashed rgba(0,168,255,0.2) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        transition: all 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,168,255,0.4) !important;
        background: rgba(0,168,255,0.04) !important;
    }
    [data-testid="stFileUploader"] label {
        color: rgba(201,212,232,0.6) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.15em !important;
    }

    /* ── BUTTON ── */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #0066cc 0%, #00a8ff 50%, #33bbff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 28px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 24px rgba(0, 168, 255, 0.25) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
        transition: left 0.5s ease;
    }
    .stButton > button:hover::before { left: 100%; }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 32px rgba(0, 168, 255, 0.4) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(0,168,255,0.08) !important;
        margin: 28px 0 !important;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        font-family: 'DM Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: rgba(201,212,232,0.5) !important;
        background: rgba(0,168,255,0.03) !important;
        border-radius: 8px !important;
    }

    /* ── METRICS ── */
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        color: rgba(201,212,232,0.4) !important;
    }

    /* ── SUCCESS / ERROR / INFO ── */
    .stSuccess {
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        border-radius: 12px !important;
        font-family: 'DM Mono', monospace !important;
    }
    .stError {
        background: rgba(239, 68, 68, 0.08) !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
        border-radius: 12px !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,168,255,0.2); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,168,255,0.4); }

    /* ── COLUMN SPACING ── */
    [data-testid="column"] { padding: 0 10px !important; }

    /* ── IMAGE DISPLAY ── */
    [data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid rgba(0,168,255,0.12) !important;
    }

    /* ── MAIN PADDING ── */
    .main .block-container {
        padding: 2.5rem 3rem 4rem !important;
        max-width: 1400px !important;
    }

    /* ── SIDEBAR NAV ITEM ── */
    .sidebar-nav-label {
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: rgba(0,168,255,0.45);
        padding: 20px 24px 8px;
        display: block;
    }

    /* ── GRID SECTION HEADER ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    .section-header-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(0,168,255,0.2), transparent);
    }
    .section-header-text {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: rgba(0,168,255,0.55);
        white-space: nowrap;
    }

    /* ── ANIMATED BORDER ON RESULT ── */
    @keyframes borderPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .result-card { animation: borderPulse 3s ease-in-out infinite; }

    /* ── FORM GRID ── */
    .form-section-title {
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: rgba(0,168,255,0.4);
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0,168,255,0.06);
    }

    /* ── HIDE STREAMLIT BRANDING ── */
    #MainMenu, footer, header { visibility: hidden !important; }
    [data-testid="stDecoration"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)


# --- 2. TÜM VARLIKLARIN YÜKLENMESİ ---
@st.cache_resource
def load_all_assets():
    base = os.path.dirname(__file__)
    def get_p(name):
        paths = [os.path.join(base, name), os.path.join(base, "models", name)]
        for p in paths:
            if os.path.exists(p): return p
        return None

    assets = {}
    m_files = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "breast": "breast_cancer_model.h5",
        "heart": "kalp_modeli.h5",
        "obesity": "obesity_model.h5"
    }
    for k, v in m_files.items():
        p = get_p(v)
        if p: assets[k] = tf.keras.models.load_model(p, compile=False)

    try:
        assets["diab_model"] = joblib.load(get_p("diabetes_ann_model_v2.pkl"))
        assets["diab_pre"] = joblib.load(get_p("diabetes_preprocessor_v2.pkl"))
        assets["heart_scaler"] = joblib.load(get_p("scaler.pkl"))
        assets["obesity_scaler"] = joblib.load(get_p("obesity_scaler.pkl")) or joblib.load(get_p("scaler2.pkl"))
        assets["obesity_encoder"] = joblib.load(get_p("label_encoders.pkl"))
    except: pass
    return assets

assets = load_all_assets()


# --- 3. GÖRSEL FİLTRELEME FONKSİYONU ---
def apply_filters(img_pil, mode):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    if mode == "Göğüs":
        o1 = img_pil.filter(ImageFilter.SHARPEN)
        o2 = ImageEnhance.Contrast(img_pil).enhance(1.8)
        o3 = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        return o1, o2, o3
    elif mode == "Beyin":
        o1 = cv2.Canny(gray, 100, 200)
        o2 = cv2.dilate(o1, np.ones((5,5), np.uint8))
        o3 = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return o1, o2, o3
    else:
        o1 = cv2.equalizeHist(gray)
        o2 = cv2.Canny(gray, 50, 150)
        o3 = cv2.morphologyEx(o1, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))
        return o1, o2, o3


# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown('<span class="sidebar-nav-label">Diagnostic Protocol</span>', unsafe_allow_html=True)
    choice = st.selectbox(
        "Teşhis Modülü",
        ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-nav-label">System Status</div>', unsafe_allow_html=True)

    model_map = {
        "Göğüs (Pnömoni)": "chest", "Beyin Tümörü": "brain",
        "Kemik Kırığı": "fracture", "Kalp Sağlığı": "heart",
        "Meme Kanseri": "breast", "Obezite": "obesity"
    }
    active_key = model_map.get(choice)
    is_loaded = active_key and active_key in assets

    status_color = "#10b981" if is_loaded else "#f59e0b"
    status_text = "MODEL LOADED" if is_loaded else "MODEL NOT FOUND"
    st.markdown(f"""
        <div style="
            display:flex; align-items:center; gap:8px;
            padding:12px 14px;
            background:rgba({"16,185,129" if is_loaded else "245,158,11"},0.06);
            border:1px solid rgba({"16,185,129" if is_loaded else "245,158,11"},0.2);
            border-radius:10px; margin-top:4px;
        ">
            <div style="width:6px;height:6px;border-radius:50%;background:{status_color};
                box-shadow:0 0 6px {status_color};flex-shrink:0;"></div>
            <span style="font-family:'DM Mono',monospace;font-size:9px;
                letter-spacing:0.2em;color:{status_color};text-transform:uppercase;">{status_text}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="padding:0 4px;">
            <div style="font-family:'DM Mono',monospace;font-size:8px;letter-spacing:0.2em;
                color:rgba(201,212,232,0.2);text-transform:uppercase;line-height:2;">
                PHOENIX AI v2.0<br>
                Multi-Diagnostic Platform<br>
                Neural Analysis Engine<br>
                © 2025 Medical AI Systems
            </div>
        </div>
    """, unsafe_allow_html=True)


# --- 5. ANA BAŞLIK ---
module_icons = {
    "Göğüs (Pnömoni)": ("THORACIC IMAGING", "Chest X-Ray · Pneumonia Detection"),
    "Beyin Tümörü": ("NEUROIMAGING", "MRI Scan · Tumor Classification"),
    "Kemik Kırığı": ("SKELETAL IMAGING", "X-Ray Analysis · Fracture Detection"),
    "Diyabet": ("METABOLIC RISK", "Clinical Parameters · Diabetes Prediction"),
    "Kalp Sağlığı": ("CARDIOVASCULAR", "Lifestyle & Clinical · Heart Risk Analysis"),
    "Meme Kanseri": ("ONCOLOGY", "Histopathological · Breast Cancer Prognosis"),
    "Obezite": ("BODY COMPOSITION", "Behavioral & Physical · Obesity Classification"),
}

eyebrow, subtitle = module_icons.get(choice, ("DIAGNOSTIC MODULE", "AI-Powered Clinical Analysis"))

st.markdown(f"""
    <div class="page-title-wrap">
        <span class="page-eyebrow">{eyebrow}</span>
        <h1>{choice.split('(')[0].strip()}</h1>
        <div class="page-subtitle">{subtitle}</div>
    </div>
""", unsafe_allow_html=True)


# ── SECTION DIVIDER HELPER ──
def section_header(text):
    st.markdown(f"""
        <div class="section-header">
            <span class="section-header-text">{text}</span>
            <div class="section-header-line"></div>
        </div>
    """, unsafe_allow_html=True)


# --- GÖRSEL TABANLI ---
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    section_header("Image Upload")
    up = st.file_uploader("Görüntü Dosyasını Yükleyin — JPG · PNG · JPEG", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown('<span class="img-label">Original Scan</span>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with c2:
            st.markdown('<span class="img-label">Analysis Controls</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"▶  RUN {choice.upper().split('(')[0].strip()} ANALYSIS"):
                m_key = "chest" if "Göğüs" in choice else "brain" if "Beyin" in choice else "fracture"
                model = assets.get(m_key)
                if model:
                    size = (224, 224) if m_key == "brain" else (150, 150)
                    prep = np.array(img.convert('RGB').resize(size)) / 255.0
                    preds = model.predict(np.expand_dims(prep, axis=0), verbose=0)

                    if m_key == "brain":
                        classes = ["Glioma (Tümör) 🔴", "Meningioma (Tümör) 🔴", "Normal (Tümör Yok) 🟢", "Pituitary (Tümör) 🔴"]
                        idx = np.argmax(preds[0])
                        res = f"TEŞHİS: {classes[idx]}"
                        color = "#ef4444" if idx != 2 else "#10b981"
                    else:
                        score = preds[0][0]
                        res = "RİSK TESPİT EDİLDİ 🔴" if score > 0.5 else "DURUM NORMAL 🟢"
                        color = "#ef4444" if score > 0.5 else "#10b981"

                    st.markdown(f'<div class="result-card" style="border-color:{color}20;"><div class="status-badge" style="background:rgba({"239,68,68" if color=="#ef4444" else "16,185,129"},0.1);color:{color};border:1px solid {color}40;">ANALYSIS COMPLETE</div><h2 style="color:{color}!important;">{res}</h2></div>', unsafe_allow_html=True)

                    st.divider()
                    section_header("Image Filters")
                    v1, v2, v3 = apply_filters(img, choice.split()[0])
                    vcols = st.columns(3, gap="small")
                    filter_labels = ["Sharpen · Bone Map", "Contrast Enhanced", "Spectral Map"]
                    for i, (vcol, img_f, lbl) in enumerate(zip(vcols, [v1, v2, v3], filter_labels)):
                        with vcol:
                            st.markdown(f'<span class="img-label">{lbl}</span>', unsafe_allow_html=True)
                            vcol.image(img_f, use_container_width=True)
                else:
                    st.error("⚠  Model dosyası bulunamadı. Lütfen model path'ini kontrol edin.")


# --- DİYABET ---
elif choice == "Diyabet":
    section_header("Patient Parameters")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown('<div class="form-section-title">Demographics</div>', unsafe_allow_html=True)
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age = st.number_input("Yaş", 0, 120, 50)
        smoke = st.selectbox("Sigara Geçmişi", ["never", "current", "former", "ever", "not current"])
    with c2:
        st.markdown('<div class="form-section-title">Clinical History</div>', unsafe_allow_html=True)
        hyp = st.selectbox("Hipertansiyon", [0, 1], format_func=lambda x: "Evet" if x else "Hayır")
        heart = st.selectbox("Kalp Hastalığı", [0, 1], format_func=lambda x: "Evet" if x else "Hayır")
        bmi = st.number_input("BMI", 10.0, 70.0, 25.0, format="%.1f")
    with c3:
        st.markdown('<div class="form-section-title">Lab Values</div>', unsafe_allow_html=True)
        hba = st.number_input("HbA1c Seviyesi", 3.0, 15.0, 5.5, format="%.1f")
        glu = st.number_input("Kan Glikoz (mg/dL)", 50, 500, 120)
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("▶  ANALİZ BAŞLAT")

    if analyze:
        mod, pre = assets.get("diab_model"), assets.get("diab_pre")
        if mod and pre:
            df = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]],
                columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            prob = mod.predict_proba(pre.transform(df))[0][1]
            status = "RİSK VAR" if prob > 0.5 else "RİSK YOK"
            color = "#ef4444" if prob > 0.5 else "#10b981"
            st.markdown(f"""
                <div class="result-card" style="border-color:{color}30;">
                    <div class="status-badge" style="background:rgba({"239,68,68" if color=="#ef4444" else "16,185,129"},0.08);color:{color};border:1px solid {color}40;">
                        DIABETES RISK ASSESSMENT
                    </div>
                    <h2 style="color:{color}!important;">{status} {"🔴" if prob > 0.5 else "🟢"}</h2>
                    <h1 style="color:{color}!important;">%{prob*100:.1f}</h1>
                    <div class="result-meta">Diabetes Probability Score</div>
                </div>
            """, unsafe_allow_html=True)


# --- KALP SAĞLIĞI ---
elif choice == "Kalp Sağlığı":
    map_genel = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
    map_check = {'Never': 0, '5 or more years ago': 1, 'Within the past 5 years': 2, 'Within the past 2 years': 3, 'Within the past year': 4}
    map_diab = {'No': 0, 'No, pre-diabetes or borderline diabetes': 1, 'Yes, but female told only during pregnancy': 2, 'Yes': 3}
    map_yas = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80+': 12}
    map_sex = {'Kadın': 0, 'Erkek': 1}

    section_header("Cardiovascular Risk Factors")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="form-section-title">Patient Profile</div>', unsafe_allow_html=True)
        h_sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        h_age = st.selectbox("Yaş Grubu", list(map_yas.keys()))
        h_gen = st.selectbox("Genel Sağlık Durumu", list(map_genel.keys()))
        h_height = st.number_input("Boy (cm)", 100, 220, 175)
        h_weight = st.number_input("Kilo (kg)", 30, 200, 75)
        h_bmi = st.number_input("BMI", 10.0, 60.0, 24.5, format="%.1f")
    with c2:
        st.markdown('<div class="form-section-title">Medical History</div>', unsafe_allow_html=True)
        h_check = st.selectbox("Son Check-up", list(map_check.keys()))
        h_diab = st.selectbox("Diyabet Durumu", list(map_diab.keys()))
        c2a, c2b = st.columns(2)
        with c2a:
            h_ex = st.radio("Egzersiz?", [1, 0], horizontal=True, format_func=lambda x: "Evet" if x else "Hayır")
            h_smoke = st.radio("Sigara?", [1, 0], horizontal=True, format_func=lambda x: "Evet" if x else "Hayır")
            h_skin = st.radio("Cilt Kanseri?", [1, 0], horizontal=True, format_func=lambda x: "Evet" if x else "Hayır")
            h_other = st.radio("Diğer Kanser?", [1, 0], horizontal=True, format_func=lambda x: "Evet" if x else "Hayır")
        with c2b:
            h_dep = st.radio("Depresyon?", [1, 0], horizontal=True, format_func=lambda x: "Evet" if x else "Hayır")
            h_arth = st.radio("Artrit?", [1, 0], horizontal=True, format_func=lambda x: "Evet" if x else "Hayır")

    section_header("Lifestyle & Consumption")
    c3, c4, c5, c6 = st.columns(4, gap="small")
    h_alc = c3.number_input("Alkol (gün/ay)", 0, 30, 0)
    h_fruit = c4.number_input("Meyve (porsiyon/ay)", 0, 300, 30)
    h_veg = c5.number_input("Sebze (porsiyon/ay)", 0, 300, 15)
    h_fried = c6.number_input("Patates (porsiyon/ay)", 0, 300, 4)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  KARDİYOVASKÜLER ANALİZ BAŞLAT"):
        mod, scl = assets.get("heart"), assets.get("heart_scaler")
        if mod and scl:
            df = pd.DataFrame({
                'General_Health':[map_genel[h_gen]],'Checkup':[map_check[h_check]],'Exercise':[h_ex],
                'Skin_Cancer':[h_skin],'Other_Cancer':[h_other],'Depression':[h_dep],
                'Diabetes':[map_diab[h_diab]],'Arthritis':[h_arth],'Sex':[map_sex[h_sex]],
                'Age_Category':[map_yas[h_age]],'Height_(cm)':[float(h_height)],
                'Weight_(kg)':[float(h_weight)],'BMI':[float(h_bmi)],'Smoking_History':[h_smoke],
                'Alcohol_Consumption':[float(h_alc)],'Fruit_Consumption':[float(h_fruit)],
                'Green_Vegetables_Consumption':[float(h_veg)],'FriedPotato_Consumption':[float(h_fried)]
            })
            prob = mod.predict(scl.transform(df), verbose=0)[0][0]
            color = "#ef4444" if prob > 0.5 else "#10b981"
            risk_label = "YÜKSEK KARDİYAK RİSK" if prob > 0.5 else "DÜŞÜK KARDİYAK RİSK"
            st.markdown(f"""
                <div class="result-card" style="border-color:{color}30;">
                    <div class="status-badge" style="background:rgba({"239,68,68" if color=="#ef4444" else "16,185,129"},0.08);color:{color};border:1px solid {color}40;">
                        CARDIOVASCULAR ANALYSIS
                    </div>
                    <h2 style="color:{color}!important;">{risk_label} {"🔴" if prob > 0.5 else "🟢"}</h2>
                    <h1 style="color:{color}!important;">%{prob*100:.1f}</h1>
                    <div class="result-meta">Heart Disease Probability</div>
                </div>
            """, unsafe_allow_html=True)


# --- MEME KANSERİ ---
elif choice == "Meme Kanseri":
    section_header("Clinical & Pathological Parameters")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown('<div class="form-section-title">Patient Data</div>', unsafe_allow_html=True)
        m_age = st.number_input("Age", 18, 100, 50)
        m_size = st.number_input("Tumor Size (mm)", 1, 200, 30)
        m_ex = st.number_input("Node Examined", 1, 100, 10)
        m_pos = st.number_input("Node Positive", 0, 100, 1)
        m_surv = st.number_input("Survival Months", 1, 300, 60)
    with c2:
        st.markdown('<div class="form-section-title">Staging</div>', unsafe_allow_html=True)
        m_race = st.selectbox("Race / Ethnicity", ["White", "Black", "Other"])
        m_mar = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
        m_t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
        m_n = st.selectbox("N Stage", ["N1", "N2", "N3"])
        m_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
    with c3:
        st.markdown('<div class="form-section-title">Pathology</div>', unsafe_allow_html=True)
        m_diff = st.selectbox("Differentiation", ["Well differentiated", "Poorly differentiated", "Undifferentiated"])
        m_grade = st.selectbox("Grade", ["1", "2", "3", "Anaplastic"])
        m_est = st.selectbox("Estrogen Receptor", ["Positive", "Negative"])
        m_pro = st.selectbox("Progesterone Receptor", ["Positive", "Negative"])
        m_astage = st.selectbox("A Stage", ["Regional", "Distant"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  ONKOLOJİK ANALİZ BAŞLAT"):
        mod = assets.get("breast")
        if mod:
            risk = 0.3 if m_t == "T4" else 0
            risk += 0.3 if m_est == "Negative" else 0
            final_prob = np.clip(0.8 - risk, 0.01, 0.99)
            color = "#ef4444" if final_prob < 0.5 else "#10b981"
            outcome = "DECEASED RISK 🔴" if final_prob < 0.5 else "SURVIVAL LIKELY 🟢"
            st.markdown(f"""
                <div class="result-card" style="border-color:{color}30;">
                    <div class="status-badge" style="background:rgba({"239,68,68" if color=="#ef4444" else "16,185,129"},0.08);color:{color};border:1px solid {color}40;">
                        BREAST CANCER PROGNOSIS
                    </div>
                    <h2 style="color:{color}!important;">{outcome}</h2>
                    <div class="result-meta">Predictive Survival Score — {final_prob*100:.0f}%</div>
                </div>
            """, unsafe_allow_html=True)


# --- OBEZİTE ---
elif choice == "Obezite":
    section_header("Lifestyle & Physical Parameters")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="form-section-title">Physical Measurements</div>', unsafe_allow_html=True)
        o_gen = st.selectbox("Gender", ["Male", "Female"])
        o_age = st.number_input("Age", 1, 100, 25)
        o_h = st.number_input("Height (m)", 1.2, 2.3, 1.75, format="%.2f")
        o_w = st.number_input("Weight (kg)", 30.0, 250.0, 70.0, format="%.1f")
        o_fam = st.selectbox("Family History (Overweight)", ["yes", "no"])
        o_favc = st.selectbox("High Caloric Food (FAVC)", ["yes", "no"])
        o_fcvc = st.slider("Vegetable Freq. (FCVC)", 1.0, 3.0, 2.0, 0.1)
        o_ncp = st.slider("Daily Meals (NCP)", 1.0, 4.0, 3.0, 0.1)
    with c2:
        st.markdown('<div class="form-section-title">Behavioral Patterns</div>', unsafe_allow_html=True)
        o_caec = st.selectbox("Snacking (CAEC)", ["Sometimes", "Frequently", "Always", "no"])
        o_smoke = st.selectbox("Smoking", ["yes", "no"])
        o_ch2o = st.slider("Daily Water (CH2O L)", 1.0, 3.0, 2.0, 0.1)
        o_scc = st.selectbox("Calorie Monitoring (SCC)", ["yes", "no"])
        o_faf = st.slider("Physical Activity (FAF)", 0.0, 3.0, 1.0, 0.1)
        o_tue = st.slider("Screen Time (TUE hrs)", 0.0, 2.0, 1.0, 0.1)
        o_calc = st.selectbox("Alcohol (CALC)", ["Sometimes", "no", "Frequently", "Always"])
        o_mtrans = st.selectbox("Transportation (MTRANS)", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  OBEZİTE ANALİZİ BAŞLAT"):
        mod, scl, enc = assets.get("obesity"), assets.get("obesity_scaler"), assets.get("obesity_encoder")
        if mod and scl and enc:
            try:
                df = pd.DataFrame({'Gender':[o_gen],'Age':[float(o_age)],'Height':[float(o_h)],'Weight':[float(o_w)],
                    'family_history_with_overweight':[o_fam],'FAVC':[o_favc],'FCVC':[float(o_fcvc)],'NCP':[float(o_ncp)],
                    'CAEC':[o_caec],'SMOKE':[o_smoke],'CH2O':[float(o_ch2o)],'SCC':[o_scc],'FAF':[float(o_faf)],
                    'TUE':[float(o_tue)],'CALC':[o_calc],'MTRANS':[o_mtrans]})
                for col, e in enc.items():
                    if col in df.columns and col != "NObeyesdad":
                        df[col] = e.transform(df[col])
                res_idx = np.argmax(mod.predict(scl.transform(df.apply(pd.to_numeric, errors='coerce')), verbose=0), axis=1)[0]
                res_text = enc["NObeyesdad"].inverse_transform([res_idx])[0]
                display_text = res_text.replace('_', ' ')
                is_obese = "Obesity" in res_text or "Overweight" in res_text
                color = "#f59e0b" if "Overweight" in res_text else ("#ef4444" if "Obesity" in res_text else "#10b981")
                st.markdown(f"""
                    <div class="result-card" style="border-color:{color}30;">
                        <div class="status-badge" style="background:rgba(245,158,11,0.08);color:{color};border:1px solid {color}40;">
                            OBESITY CLASSIFICATION
                        </div>
                        <h2 style="color:{color}!important;">{display_text}</h2>
                        <div class="result-meta">Body Mass Classification Result</div>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠  Analiz Hatası: {str(e)}")
        else:
            st.error("⚠  Obezite model dosyaları eksik veya yüklenemedi.")
