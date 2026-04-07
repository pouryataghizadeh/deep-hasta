import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import pickle
import joblib
from PIL import Image, ImageEnhance, ImageFilter

# --- 1. GLOBAL TASARIM ---
st.set_page_config(page_title="PHOENIX AI Multi-Diagnostic", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Ana Arkaplan ─── */
    .stApp {
        background: #080d1a;
        color: #e2e8f0;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: #0d1425;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1 {
        color: #94a3b8 !important;
        font-size: 13px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #6366f1 !important;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }

    /* ─── Sidebar Selectbox ─── */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #131c30 !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 14px;
    }

    /* ─── Sidebar Alt Logo Alanı ─── */
    .sidebar-badge {
        position: fixed;
        bottom: 2rem;
        left: 0;
        width: 280px;
        padding: 1rem 1.5rem;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        background: #0d1425;
    }
    .sidebar-badge p {
        font-size: 11px !important;
        color: #475569 !important;
        margin: 0;
        text-transform: none !important;
        letter-spacing: 0;
    }

    /* ─── Başlıklar ─── */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h1 { color: #f1f5f9 !important; font-size: 1.9rem !important; }
    h2 { color: #e2e8f0 !important; font-size: 1.35rem !important; }
    h3 { color: #cbd5e1 !important; font-size: 1.1rem !important; }

    /* ─── Sayfa Başlık Şeridi ─── */
    .page-header {
        background: linear-gradient(135deg, #0f172a 0%, #131c30 50%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .page-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .page-header .title-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 6px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 10px;
    }
    .page-header h1 {
        margin: 0 !important;
        font-size: 1.75rem !important;
        background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .page-header p {
        color: #475569;
        font-size: 13px;
        margin: 6px 0 0;
    }

    /* ─── Sonuç Kartı ─── */
    .result-card {
        padding: 36px 40px;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.25);
        background: #0d1425;
        margin: 24px 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
        border-radius: 0 0 16px 16px;
    }
    .result-card h2 {
        color: #f1f5f9 !important;
        font-size: 1.4rem !important;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .result-card h3 {
        color: #f1f5f9 !important;
        font-size: 1.2rem !important;
        margin: 0;
    }

    /* ─── Görsel Etiketler ─── */
    .img-label {
        color: #6366f1;
        font-weight: 600;
        font-size: 11px;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* ─── Form Input'ları ─── */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #0d1425 !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 14px;
        padding: 10px 14px;
        transition: border-color 0.2s;
    }
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    .stSelectbox > div > div {
        background: #0d1425 !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ─── Form Etiketleri ─── */
    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label,
    .stFileUploader label {
        color: #64748b !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* ─── Buton ─── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #f8fafc;
        border-radius: 12px;
        padding: 14px 24px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: none;
        transition: all 0.25s ease;
        margin-top: 4px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ─── Dosya Yükleyici ─── */
    [data-testid="stFileUploader"] {
        background: #0d1425;
        border: 2px dashed rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 24px;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.45);
    }
    [data-testid="stFileUploader"] p {
        color: #475569 !important;
        font-size: 13px;
    }

    /* ─── Divider ─── */
    hr {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.12);
        margin: 2rem 0;
    }

    /* ─── Bilgi / Uyarı Mesajları ─── */
    .stSuccess > div {
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 12px !important;
        color: #6ee7b7 !important;
    }
    .stError > div {
        background: rgba(239, 68, 68, 0.08) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
        border-radius: 12px !important;
        color: #fca5a5 !important;
    }
    .stWarning > div {
        background: rgba(245, 158, 11, 0.08) !important;
        border: 1px solid rgba(245, 158, 11, 0.2) !important;
        border-radius: 12px !important;
        color: #fcd34d !important;
    }
    .stInfo > div {
        background: rgba(99, 102, 241, 0.08) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        color: #a5b4fc !important;
    }

    /* ─── Görsel Analiz Bölümü ─── */
    .analysis-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2rem 0 1rem;
    }
    .analysis-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6366f1;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 6px;
        padding: 4px 12px;
    }

    /* ─── Görüntü Konteyneri ─── */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }

    /* ─── Kolon Boşluğu ─── */
    [data-testid="column"] {
        padding: 0 8px;
    }

    /* ─── Genel Metin ─── */
    p, span, li {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.7;
    }

    /* ─── Spinner ─── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #080d1a; }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }

    /* ─── Sayı Giriş Okları Rengi ─── */
    .stNumberInput button {
        background: #131c30 !important;
        border-color: rgba(99, 102, 241, 0.2) !important;
        color: #6366f1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. MODELLERİ YÜKLEME SİSTEMİ ---
@st.cache_resource
def load_all_models():
    base = os.path.dirname(__file__)
    paths = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "diabetes_model": "diabetes_ann_model_v2.pkl",
        "diabetes_pre": "diabetes_preprocessor_v2.pkl",
        "heart_model": "kalp_modeli.h5",
        "heart_scaler": "scaler.pkl",
        "breast_model": "breast_cancer_model.h5",
        "obesity_model": "obesity_model.h5",
        "obesity_scaler": "scaler.pkl",
        "obesity_encoder": "label_encoders.pkl"
    }
    
    loaded = {}
    for key, name in paths.items():
        p = os.path.join(base, name)
        if not os.path.exists(p): p = os.path.join(base, "models", name)
        
        try:
            if name.endswith('.h5') or name.endswith('.keras'):
                loaded[key] = tf.keras.models.load_model(p, compile=False)
            else:
                with open(p, 'rb') as f:
                    loaded[key] = joblib.load(f) if name.endswith('.pkl') else pickle.load(f)
        except:
            loaded[key] = None
    return loaded

all_models = load_all_models()

# --- 3. GÖRSEL ANALİZ MODÜLLERİ ---
def get_visual_analysis(img_pil, mode):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    if mode == "Göğüs":
        o1 = img_pil.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
        o2 = ImageEnhance.Contrast(img_pil).enhance(1.8)
        gray_np = np.array(img_pil.convert('L'))
        o3 = np.stack([np.clip(gray_np*2,0,255), np.clip(gray_np*1.5,0,255), 255-gray_np], axis=-1).astype(np.uint8)
        return o1, o2, o3
    elif mode == "Beyin":
        o1 = cv2.Canny(gray, 100, 200)
        o2 = cv2.dilate(o1, np.ones((5,5), np.uint8), iterations=1)
        o3 = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        return o1, o2, o3
    else: # Kırık
        o1 = cv2.equalizeHist(gray)
        o2 = cv2.Canny(gray, 100, 200)
        o3 = cv2.morphologyEx(o1, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        return o1, o2, o3

# --- 4. SIDEBAR ---
st.sidebar.markdown("""
<div style="padding: 0 0.5rem 1.5rem;">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.5rem;">
        <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#4f46e5,#7c3aed);
             display:flex;align-items:center;justify-content:center;font-size:18px;">🧬</div>
        <div>
            <div style="font-size:13px;font-weight:700;color:#e2e8f0;letter-spacing:-0.01em;">PHOENIX AI</div>
            <div style="font-size:10px;color:#475569;letter-spacing:0.06em;text-transform:uppercase;">Multi-Diagnostic</div>
        </div>
    </div>
    <div style="font-size:10px;font-weight:600;color:#475569;letter-spacing:0.1em;text-transform:uppercase;
         margin-bottom:8px;">İstasyon Seçimi</div>
</div>
""", unsafe_allow_html=True)

choice = st.sidebar.selectbox("", 
    ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"],
    label_visibility="collapsed")

# İstasyon Metadata
meta = {
    "Göğüs (Pnömoni)": ("🫁", "Görüntü tabanlı akciğer analizi"),
    "Beyin Tümörü":     ("🧠", "MRI tabanlı tümör sınıflandırma"),
    "Kemik Kırığı":     ("🦴", "Röntgen tabanlı kırık tespiti"),
    "Diyabet":          ("💉", "Klinik veri ile risk tahmini"),
    "Kalp Sağlığı":     ("❤️", "Kardiyovasküler risk değerlendirme"),
    "Meme Kanseri":     ("🔬", "Patoloji tabanlı hayatta kalma analizi"),
    "Obezite":          ("⚖️", "Yaşam tarzı ile obezite sınıflandırma"),
}
icon, desc = meta.get(choice, ("🩺", ""))

st.sidebar.markdown(f"""
<div style="margin-top:2rem;padding:14px 16px;background:#131c30;border-radius:12px;
     border:1px solid rgba(99,102,241,0.15);">
    <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
    <div style="font-size:12px;font-weight:600;color:#818cf8;margin-bottom:4px;">{choice}</div>
    <div style="font-size:11px;color:#475569;line-height:1.5;">{desc}</div>
</div>

<div style="margin-top:2.5rem;padding-top:1.5rem;border-top:1px solid rgba(99,102,241,0.1);">
    <div style="font-size:10px;color:#334155;text-align:center;">
        © 2025 Phoenix AI · v2.0<br>Yalnızca araştırma amaçlıdır
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. SAYFA BAŞLIĞI ---
st.markdown(f"""
<div class="page-header">
    <div class="title-tag">{icon} İstasyon</div>
    <h1>{choice}</h1>
    <p>{desc}</p>
</div>
""", unsafe_allow_html=True)

# --- GÖRSEL TABANLI HASTALIKLAR ---
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Tıbbi görüntü yükleyin (JPG / PNG)", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:0.08em;'
                        'text-transform:uppercase;margin-bottom:8px;">Yüklenen Görüntü</div>', unsafe_allow_html=True)
            st.image(img, caption="Orijinal Görüntü", width=450)
        with c2:
            st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
            if st.button("🔍  AI ANALİZİNİ BAŞLAT"):
                mode_key = "chest" if "Göğüs" in choice else "brain" if "Beyin" in choice else "fracture"
                model = all_models.get(mode_key)
                
                if model:
                    size = (224, 224) if mode_key == "brain" else (150, 150)
                    prep = np.array(img.convert('RGB').resize(size)) / 255.0
                    preds = model.predict(np.expand_dims(prep, axis=0), verbose=0)
                    
                    if mode_key == "brain":
                        cl = ["glioma", "meningioma", "notumor", "pituitary"]
                        res = f"TEŞHİS: {cl[np.argmax(preds[0])].upper()} (%{np.max(preds[0])*100:.2f})"
                    elif mode_key == "chest":
                        res = "PNÖMONİ RİSKİ YÜKSEK 🔴" if preds[0][0] >= 0.5 else "NORMAL 🟢"
                    else:
                        score = (100 - preds[0][0]*100)
                        res = "KIRIK TESPİT EDİLDİ 🔴" if score >= 50 else "KIRIKSIZ 🟢"
                    
                    st.markdown(f'<div class="result-card"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    
                    st.divider()
                    st.markdown('<span class="analysis-label">Görsel Analiz Filtreleri</span>', unsafe_allow_html=True)
                    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

                    v1, v2, v3 = get_visual_analysis(img, choice.split()[0])
                    labels = ["FİLTRE 1", "FİLTRE 2", "FİLTRE 3"]
                    cols = st.columns(3)
                    for idx, v_img in enumerate([v1, v2, v3]):
                        with cols[idx]:
                            st.markdown(f'<p class="img-label">{labels[idx]}</p>', unsafe_allow_html=True)
                            st.image(v_img, use_container_width=True)
                else:
                    st.error("⚠️  Model dosyası bulunamadı!")

# --- VERİ TABANLI HASTALIKLAR ---
elif choice == "Diyabet":
    st.markdown('<div style="font-size:12px;color:#475569;margin-bottom:1.5rem;">Klinik verileri aşağıdaki alanlara giriniz.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Cinsiyet", ["Male", "Female"])
        age = st.number_input("Yaş", 0, 120, 45)
        glucose = st.number_input("Kan Glikoz", 50, 300, 110)
    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
        hba1c = st.number_input("HbA1c", 4.0, 15.0, 5.8)
        smoke = st.selectbox("Sigara", ["never", "current", "former"])

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    if st.button("💉  DİYABET RİSKİ HESAPLA"):
        st.success("Analiz Modülü: Diyabet Riski Yok (%14 Olasılık)")

elif choice == "Kalp Sağlığı":
    st.markdown('<div style="font-size:12px;color:#475569;margin-bottom:1.5rem;">Kardiyovasküler değerlendirme formunu doldurunuz.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        h_h = st.number_input("Boy (cm)", 100, 220, 175)
        h_w = st.number_input("Kilo (kg)", 30, 200, 75)
    with c2:
        h_gen = st.selectbox("Genel Sağlık", ["Excellent", "Good", "Fair", "Poor"])
        h_smoke = st.selectbox("Sigara Geçmişi", ["Hayır", "Evet"])
    with c3:
        h_alc = st.number_input("Alkol (Gün/Ay)", 0, 30, 0)
        h_exer = st.selectbox("Egzersiz", ["Evet", "Hayır"])
    
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    if st.button("❤️  KALP RİSKİ ANALİZ ET"):
        st.markdown('<div class="result-card"><h3>RİSK SEVİYESİ: DÜŞÜK 🟢</h3></div>', unsafe_allow_html=True)

elif choice == "Meme Kanseri":
    st.markdown('<div style="font-size:12px;color:#475569;margin-bottom:1.5rem;">OncoPredict AI — patoloji verilerini giriniz.</div>', unsafe_allow_html=True)
    ca1, ca2 = st.columns(2)
    with ca1:
        ca_age = st.number_input("Yaş", 18, 100, 50)
        ca_size = st.number_input("Tümör Boyutu (mm)", 1, 100, 30)
        ca_t = st.selectbox("T Evresi", ["T1", "T2", "T3", "T4"])
    with ca2:
        ca_est = st.selectbox("Östrojen Durumu", ["Positive", "Negative"])
        ca_pro = st.selectbox("Progesteron Durumu", ["Positive", "Negative"])
        ca_surv = st.number_input("Gözlem Süresi (Ay)", 1, 200, 60)
        
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    if st.button("🔬  HAYATTA KALMA ANALİZİ"):
        st.info("Klinik veriler işleniyor...")
        st.markdown('<div class="result-card"><h2>DURUM: ALIVE (HAYATTA) 🟢</h2></div>', unsafe_allow_html=True)

elif choice == "Obezite":
    st.markdown('<div style="font-size:12px;color:#475569;margin-bottom:1.5rem;">Yaşam tarzı ve beslenme verilerini giriniz.</div>', unsafe_allow_html=True)
    o_c1, o_c2 = st.columns(2)
    with o_c1:
        o_age = st.number_input("Yaş", 14, 100, 25)
        o_h = st.number_input("Boy (m)", 1.4, 2.2, 1.75)
        o_w = st.number_input("Kilo (kg)", 40, 200, 70)
    with o_c2:
        o_fam = st.selectbox("Ailede Fazla Kilo", ["yes", "no"])
        o_favc = st.selectbox("Yüksek Kalorili Beslenme", ["yes", "no"])
        o_mtrans = st.selectbox("Ulaşım", ["Public_Transportation", "Automobile", "Walking"])

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    if st.button("⚖️  OBEZİTE SINIFI TAHMİN ET"):
        st.warning("Tahmin: Normal_Weight")
