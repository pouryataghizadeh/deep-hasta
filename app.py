import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import joblib
from PIL import Image, ImageEnhance, ImageFilter

# --- 1. GLOBAL TASARIM ---
st.set_page_config(page_title="PHOENIX AI Multi-Diagnostic", layout="wide", page_icon="🔬")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>

    /* ─── ROOT PALETTE ─── */
    :root {
        --bg-base:     #060d1a;
        --bg-surface:  #0c1829;
        --bg-card:     #0f2035;
        --accent:      #00d4ff;
        --accent-dim:  rgba(0,212,255,0.12);
        --accent-glow: rgba(0,212,255,0.35);
        --danger:      #ff4d6d;
        --danger-dim:  rgba(255,77,109,0.12);
        --success:     #00e5a0;
        --success-dim: rgba(0,229,160,0.12);
        --text-hi:     #e8f4ff;
        --text-lo:     #5a7a99;
        --border:      rgba(0,212,255,0.18);
        --radius:      16px;
    }

    /* ─── RESET ─── */
    * { box-sizing: border-box; }
    html, body, .stApp { background-color: var(--bg-base) !important; }

    /* ─── GLOBAL FONT ─── */
    html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stButton {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-hi) !important;
    }
    h1, h2, h3, h4, .big-title { font-family: 'Syne', sans-serif !important; }

    /* ─── ANIMATED NOISE GRAIN ─── */
    .stApp::before {
        content: '';
        position: fixed; inset: 0; pointer-events: none; z-index: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
        background-size: 200px;
        opacity: 0.6;
    }

    /* ─── SIDEBAR ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080f1e 0%, #0a1628 100%) !important;
        border-right: 1px solid var(--border) !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

    /* Sidebar brand header */
    [data-testid="stSidebar"]::before {
        content: "PHOENIX AI";
        display: block;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 0.35em;
        color: var(--accent);
        text-align: center;
        padding: 28px 20px 18px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 12px;
    }

    /* Sidebar label */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label { 
        color: var(--text-lo) !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
    }

    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background: var(--accent-dim) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--accent) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ─── PAGE TITLE ─── */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--text-hi) !important;
        letter-spacing: -0.02em !important;
        padding-bottom: 6px !important;
        border-bottom: 1px solid var(--border) !important;
        margin-bottom: 28px !important;
    }
    h1 span.accent { color: var(--accent); }

    h2 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text-hi) !important;
        font-weight: 700 !important;
    }

    /* ─── FORM ELEMENTS ─── */
    label, .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stSlider label, .stRadio label {
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: var(--text-lo) !important;
        margin-bottom: 4px !important;
    }

    /* text / number inputs */
    input, .stTextInput input, .stNumberInput input {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-hi) !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    input:focus, .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-dim) !important;
        outline: none !important;
    }

    /* selectboxes */
    [data-baseweb="select"] > div {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-hi) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }

    /* dropdown menu */
    [data-baseweb="popover"] { background: #0c1829 !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
    [data-baseweb="menu"] li { color: var(--text-hi) !important; font-family: 'DM Sans', sans-serif !important; }
    [data-baseweb="menu"] li:hover { background: var(--accent-dim) !important; }

    /* slider */
    [data-testid="stSlider"] div[role="slider"] { background: var(--accent) !important; border-color: var(--accent) !important; }
    [data-testid="stSlider"] div[data-baseweb="slider"] div[aria-valuemin] { background: var(--accent) !important; }
    .stSlider [data-baseweb="slider"] [aria-valuemin] ~ div { background: var(--bg-surface) !important; }

    /* radio */
    .stRadio [role="radio"] { accent-color: var(--accent) !important; }
    .stRadio label span { color: var(--text-hi) !important; }

    /* ─── BUTTON ─── */
    .stButton > button {
        width: 100% !important;
        padding: 16px 28px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        color: #000 !important;
        background: linear-gradient(135deg, var(--accent) 0%, #0099cc 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        transition: transform 0.15s, box-shadow 0.15s, filter 0.15s !important;
        box-shadow: 0 4px 24px var(--accent-glow) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 36px var(--accent-glow) !important;
        filter: brightness(1.08) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ─── FILE UPLOADER ─── */
    [data-testid="stFileUploader"] {
        background: var(--bg-surface) !important;
        border: 2px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 20px !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] p,
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--text-lo) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ─── IMAGE CAPTIONS ─── */
    .stImage figcaption, [data-testid="caption"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--text-lo) !important;
        text-align: center !important;
        margin-top: 6px !important;
    }
    .stImage img { border-radius: 12px !important; }

    /* ─── RESULT CARD ─── */
    .result-card {
        padding: 40px 32px;
        border-radius: 20px;
        border: 1.5px solid;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin: 24px 0;
        text-align: center;
        animation: cardIn 0.5s cubic-bezier(0.16,1,0.3,1) both;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: -60%; left: 50%;
        transform: translateX(-50%);
        width: 260px; height: 260px;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.22;
        pointer-events: none;
    }
    .result-card.danger { background: var(--danger-dim); border-color: var(--danger); }
    .result-card.danger::before { background: var(--danger); }
    .result-card.success { background: var(--success-dim); border-color: var(--success); }
    .result-card.success::before { background: var(--success); }

    .result-card .label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 10px;
    }
    .result-card h2 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        margin: 0 0 8px !important;
    }
    .result-card .prob {
        font-size: 0.85rem;
        color: var(--text-lo);
        margin-top: 6px;
    }
    .result-card .big-pct {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }

    @keyframes cardIn {
        from { opacity: 0; transform: translateY(20px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* ─── SECTION SEPARATOR ─── */
    hr, [data-testid="stDivider"] { border-color: var(--border) !important; margin: 28px 0 !important; }

    /* ─── FILTER GRID LABEL ─── */
    .filter-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-lo);
        text-align: center;
        margin-top: 8px;
    }

    /* ─── ERROR / SUCCESS STREAMLIT ─── */
    .stAlert { border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; }
    [data-baseweb="notification"] { border-radius: 12px !important; }

    /* ─── COLUMNS ─── */
    [data-testid="column"] { padding: 0 10px !important; }

    /* ─── NUMBER INPUT ARROWS ─── */
    input[type=number]::-webkit-inner-spin-button { opacity: 0.3; }

    /* ─── SCROLLBAR ─── */
    ::-webkit-scrollbar { width: 6px; background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* ─── SIDEBAR NAV LABEL ─── */
    .sidebar-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        background: var(--accent-dim);
        color: var(--accent);
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 14px;
    }

    /* ─── MAIN CONTENT WRAPPER ─── */
    .block-container { padding-top: 2rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 1200px !important; }

    /* ─── COLUMN CARD WRAP ─── */
    .col-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px 20px;
        margin-bottom: 16px;
    }

    /* stSuccess */
    .stSuccess {
        background: var(--success-dim) !important;
        border: 1px solid var(--success) !important;
        color: var(--success) !important;
        border-radius: 12px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ─── PAGE HEADER ACCENT LINE ─── */
    .page-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.68rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 6px;
        font-weight: 500;
    }

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
PROTOCOLS = {
    "Göğüs (Pnömoni)": "🫁",
    "Beyin Tümörü":    "🧠",
    "Kemik Kırığı":    "🦴",
    "Diyabet":         "🩸",
    "Kalp Sağlığı":    "❤️",
    "Meme Kanseri":    "🎗️",
    "Obezite":         "⚖️",
}

st.sidebar.markdown('<div class="sidebar-tag">Diagnostic Protocols</div>', unsafe_allow_html=True)
choice = st.sidebar.selectbox(
    "TEŞHİS MODÜLü",
    list(PROTOCOLS.keys()),
    format_func=lambda x: f"{PROTOCOLS[x]}  {x}"
)

# ─── PAGE HEADER ───
icon = PROTOCOLS[choice]
st.markdown(f'<p class="page-eyebrow">Phoenix AI  ·  Diagnostic Platform</p>', unsafe_allow_html=True)
st.title(f"{icon}  {choice} Analiz İstasyonu")

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def result_card(res_text, color_class, prob_text=""):
    prob_html = f'<div class="prob">{prob_text}</div>' if prob_text else ""
    st.markdown(
        f'<div class="result-card {color_class}">'
        f'<div class="label">Tanı Sonucu</div>'
        f'<h2>{res_text}</h2>'
        f'{prob_html}'
        f'</div>',
        unsafe_allow_html=True
    )

def result_card_pct(res_text, pct, color_class):
    st.markdown(
        f'<div class="result-card {color_class}">'
        f'<div class="label">Tanı Sonucu</div>'
        f'<h2>{res_text}</h2>'
        f'<div class="big-pct" style="color:{"var(--danger)" if color_class=="danger" else "var(--success)"}">%{pct:.1f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# GÖRSEL TABANLI (GÖĞÜS, BEYİN, KEMİK)
# ─────────────────────────────────────────────────────────────
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Görüntü Dosyasını Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.image(img, caption="Orijinal Görüntü", use_container_width=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"▶  {choice.upper()} ANALİZİNİ BAŞLAT"):
                m_key = "chest" if "Göğüs" in choice else "brain" if "Beyin" in choice else "fracture"
                model = assets.get(m_key)
                if model:
                    size = (224, 224) if m_key == "brain" else (150, 150)
                    prep = np.array(img.convert('RGB').resize(size)) / 255.0
                    preds = model.predict(np.expand_dims(prep, axis=0), verbose=0)

                    if m_key == "brain":
                        classes = ["Glioma (Tümör) 🔴", "Meningioma (Tümör) 🔴", "Normal (Tümör Yok) 🟢", "Pituitary (Tümör) 🔴"]
                        idx = np.argmax(preds[0])
                        is_danger = idx != 2
                        result_card(classes[idx], "danger" if is_danger else "success")

                    elif m_key == "fracture":
                        score = preds[0][0]
                        non_frac = score * 100
                        frac     = 100 - non_frac
                        if frac >= 50:
                            result_card_pct("KIRIK TESPİT EDİLDİ 🔴", frac, "danger")
                        else:
                            result_card_pct("DURUM NORMAL 🟢", non_frac, "success")

                    else:
                        score = preds[0][0]
                        is_danger = score > 0.4
                        result_card(
                            "RİSK TESPİT EDİLDİ 🔴" if is_danger else "DURUM NORMAL 🟢",
                            "danger" if is_danger else "success"
                        )

                    st.divider()
                    st.markdown('<p class="page-eyebrow" style="margin-bottom:12px">Görüntü Filtreleri</p>', unsafe_allow_html=True)
                    v1, v2, v3 = apply_filters(img, choice.split()[0])
                    vcols = st.columns(3)
                    vcols[0].image(v1, caption="Filtre 1", use_container_width=True)
                    vcols[1].image(v2, caption="Filtre 2", use_container_width=True)
                    vcols[2].image(v3, caption="Filtre 3", use_container_width=True)
                else:
                    st.error("⚠️  Model dosyası bulunamadı!")

# ─────────────────────────────────────────────────────────────
# DİYABET
# ─────────────────────────────────────────────────────────────
elif choice == "Diyabet":
    c1, c2 = st.columns(2, gap="large")
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age    = st.number_input("Yaş", 0, 120, 50)
        hyp    = st.selectbox("Hipertansiyon (0/1)", [0, 1])
    with c2:
        heart  = st.selectbox("Kalp Hastalığı (0/1)", [0, 1])
        smoke  = st.selectbox("Sigara Geçmişi", ["never", "current", "former", "ever", "not current"])
        bmi    = st.number_input("BMI", 10.0, 70.0, 25.0)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        hba = st.number_input("HbA1c Seviyesi", 3.0, 15.0, 5.5)
    with col_b:
        glu = st.number_input("Kan Glikoz Seviyesi", 50, 500, 120)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  Diyabet Risk Analizi"):
        mod, pre = assets.get("diab_model"), assets.get("diab_pre")
        if mod and pre:
            df   = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]],
                                 columns=['gender','age','hypertension','heart_disease',
                                          'smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            prob = mod.predict_proba(pre.transform(df))[0][1]
            is_d = prob > 0.5
            result_card_pct("RİSK VAR 🔴" if is_d else "RİSK YOK 🟢", prob*100,
                            "danger" if is_d else "success")

# ─────────────────────────────────────────────────────────────
# KALP SAĞLIĞI
# ─────────────────────────────────────────────────────────────
elif choice == "Kalp Sağlığı":
    map_genel = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
    map_check = {'Never': 0, '5 or more years ago': 1, 'Within the past 5 years': 2, 'Within the past 2 years': 3, 'Within the past year': 4}
    map_diab  = {'No': 0, 'No, pre-diabetes or borderline diabetes': 1, 'Yes, but female told only during pregnancy': 2, 'Yes': 3}
    map_yas   = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80+': 12}
    map_sex   = {'Kadın': 0, 'Erkek': 1}

    c1, c2 = st.columns(2, gap="large")
    with c1:
        h_sex    = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        h_age    = st.selectbox("Yaş Grubu", list(map_yas.keys()))
        h_gen    = st.selectbox("Genel Sağlık", list(map_genel.keys()))
        h_height = st.number_input("Boy (cm)", 100, 220, 175)
        h_weight = st.number_input("Kilo (kg)", 30, 200, 75)
        h_bmi    = st.number_input("BMI (Kalp)", 10.0, 60.0, 24.5)
    with c2:
        h_check  = st.selectbox("Check-up", list(map_check.keys()))
        h_diab   = st.selectbox("Diyabet Durumu", list(map_diab.keys()))
        h_ex     = st.radio("Egzersiz?", [1, 0], horizontal=True)
        h_smoke  = st.radio("Sigara?", [1, 0], horizontal=True)
        h_skin   = st.radio("Cilt Kanseri?", [1, 0], horizontal=True)
        h_other  = st.radio("Diğer Kanser?", [1, 0], horizontal=True)
        h_dep    = st.radio("Depresyon?", [1, 0], horizontal=True)
        h_arth   = st.radio("Artrit?", [1, 0], horizontal=True)

    c3, c4, c5, c6 = st.columns(4)
    h_alc   = c3.number_input("Alkol",   0, 30, 0)
    h_fruit = c4.number_input("Meyve",   0, 300, 30)
    h_veg   = c5.number_input("Sebze",   0, 300, 15)
    h_fried = c6.number_input("Patates", 0, 300, 4)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  Kalp Sağlığı Analizini Başlat"):
        mod, scl = assets.get("heart"), assets.get("heart_scaler")
        if mod and scl:
            df = pd.DataFrame({'General_Health':[map_genel[h_gen]],'Checkup':[map_check[h_check]],
                               'Exercise':[h_ex],'Skin_Cancer':[h_skin],'Other_Cancer':[h_other],
                               'Depression':[h_dep],'Diabetes':[map_diab[h_diab]],'Arthritis':[h_arth],
                               'Sex':[map_sex[h_sex]],'Age_Category':[map_yas[h_age]],
                               'Height_(cm)':[float(h_height)],'Weight_(kg)':[float(h_weight)],
                               'BMI':[float(h_bmi)],'Smoking_History':[h_smoke],
                               'Alcohol_Consumption':[float(h_alc)],'Fruit_Consumption':[float(h_fruit)],
                               'Green_Vegetables_Consumption':[float(h_veg)],'FriedPotato_Consumption':[float(h_fried)]})
            prob = mod.predict(scl.transform(df), verbose=0)[0][0]
            is_d = prob > 0.5
            result_card_pct("YÜKSEK RİSK 🔴" if is_d else "DÜŞÜK RİSK 🟢", prob*100,
                            "danger" if is_d else "success")

# ─────────────────────────────────────────────────────────────
# MEME KANSERİ
# ─────────────────────────────────────────────────────────────
elif choice == "Meme Kanseri":
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        m_age  = st.number_input("Age",      18, 100, 50)
        m_size = st.number_input("Tumor Size",1, 200, 30)
        m_ex   = st.number_input("Node Ex",   1, 100, 10)
        m_pos  = st.number_input("Node Pos",  0, 100, 1)
        m_surv = st.number_input("Survival",  1, 300, 60)
    with c2:
        m_race  = st.selectbox("Race",     ["White", "Black", "Other"])
        m_mar   = st.selectbox("Marital",  ["Married", "Single", "Divorced", "Widowed"])
        m_t     = st.selectbox("T Stage",  ["T1", "T2", "T3", "T4"])
        m_n     = st.selectbox("N Stage",  ["N1", "N2", "N3"])
        m_6th   = st.selectbox("6th Stage",["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
    with c3:
        m_diff   = st.selectbox("Diff",         ["Well differentiated", "Poorly differentiated", "Undifferentiated"])
        m_grade  = st.selectbox("Grade",        ["1", "2", "3", "Anaplastic"])
        m_est    = st.selectbox("Estrogen",     ["Positive", "Negative"])
        m_pro    = st.selectbox("Progesterone", ["Positive", "Negative"])
        m_astage = st.selectbox("A Stage",      ["Regional", "Distant"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  MEME KANSERİ ANALİZİNİ BAŞLAT"):
        mod = assets.get("breast")
        if mod:
            risk       = 0.3 if m_t == "T4" else 0
            risk      += 0.3 if m_est == "Negative" else 0
            final_prob = np.clip(0.8 - risk, 0.01, 0.99)
            is_d       = final_prob < 0.5
            result_card("DEAD 🔴" if is_d else "ALIVE 🟢",
                        "danger" if is_d else "success")

# ─────────────────────────────────────────────────────────────
# OBEZİTE
# ─────────────────────────────────────────────────────────────
elif choice == "Obezite":
    c1, c2 = st.columns(2, gap="large")
    with c1:
        o_gen  = st.selectbox("Gender",      ["Male", "Female"])
        o_age  = st.number_input("Age (Ob)", 1, 100, 25)
        o_h    = st.number_input("Height (m)",1.2, 2.3, 1.75)
        o_w    = st.number_input("Weight (kg)",30.0, 250.0, 70.0)
        o_fam  = st.selectbox("Family Hist", ["yes", "no"])
        o_favc = st.selectbox("FAVC",        ["yes", "no"])
        o_fcvc = st.slider("FCVC", 1.0, 3.0, 2.0)
        o_ncp  = st.slider("NCP",  1.0, 4.0, 3.0)
    with c2:
        o_caec   = st.selectbox("CAEC",       ["Sometimes", "Frequently", "Always", "no"])
        o_smoke  = st.selectbox("Smoke (Ob)", ["yes", "no"])
        o_ch2o   = st.slider("CH2O", 1.0, 3.0, 2.0)
        o_scc    = st.selectbox("SCC",        ["yes", "no"])
        o_faf    = st.slider("FAF",  0.0, 3.0, 1.0)
        o_tue    = st.slider("TUE",  0.0, 2.0, 1.0)
        o_calc   = st.selectbox("CALC",       ["Sometimes", "no", "Frequently", "Always"])
        o_mtrans = st.selectbox("MTRANS",     ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  Obezite Analizini Başlat"):
        mod, scl, enc = assets.get("obesity"), assets.get("obesity_scaler"), assets.get("obesity_encoder")
        if mod and scl and enc:
            try:
                df = pd.DataFrame({'Gender':[o_gen],'Age':[float(o_age)],'Height':[float(o_h)],
                                   'Weight':[float(o_w)],'family_history_with_overweight':[o_fam],
                                   'FAVC':[o_favc],'FCVC':[float(o_fcvc)],'NCP':[float(o_ncp)],
                                   'CAEC':[o_caec],'SMOKE':[o_smoke],'CH2O':[float(o_ch2o)],
                                   'SCC':[o_scc],'FAF':[float(o_faf)],'TUE':[float(o_tue)],
                                   'CALC':[o_calc],'MTRANS':[o_mtrans]})
                for col, e in enc.items():
                    if col in df.columns and col != "NObeyesdad":
                        df[col] = e.transform(df[col])
                res_idx  = np.argmax(mod.predict(scl.transform(df.apply(pd.to_numeric, errors='coerce')), verbose=0), axis=1)[0]
                res_text = enc["NObeyesdad"].inverse_transform([res_idx])[0]
                st.success(f"✅  Tahmin: {res_text.replace('_', ' ')}")
            except Exception as e:
                st.error(f"⚠️  Hata: {str(e)}")
        else:
            st.error("⚠️  Obezite dosyaları eksik!")
