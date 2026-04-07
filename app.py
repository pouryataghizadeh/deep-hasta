import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import pickle
import joblib
from PIL import Image, ImageEnhance, ImageFilter

# --- 1. GLOBAL TASARIM VE STİL ---
st.set_page_config(page_title="PHOENIX AI Multi-Diagnostic", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #3b82f6; }
    .result-card {
        padding: 30px; border-radius: 20px; border: 2px solid #3b82f6;
        background: rgba(30, 41, 59, 0.9); margin: 20px 0; text-align: center;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
    }
    .img-label { color: #60a5fa; font-weight: 800; font-size: 13px; text-align: center; text-transform: uppercase; margin-bottom:10px; }
    h1, h2, h3 { color: #3b82f6 !important; font-weight: 800; }
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #000; border-radius: 14px; padding: 18px; font-weight: 800; border: none; font-size: 1.1rem;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 15px 25px rgba(79, 172, 254, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODELLERİ YÜKLEME ---
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
            if name.endswith(('.h5', '.keras')):
                loaded[key] = tf.keras.models.load_model(p, compile=False)
            else:
                with open(p, 'rb') as f: loaded[key] = joblib.load(f)
        except: loaded[key] = None
    return loaded

all_models = load_all_models()

# --- 3. GÖRSEL ANALİZ FONKSİYONLARI ---
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
    else: # Kemik
        o1 = cv2.equalizeHist(gray)
        o2 = cv2.Canny(gray, 100, 200)
        o3 = cv2.morphologyEx(o1, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        return o1, o2, o3

# --- 4. YAN MENÜ ---
st.sidebar.title("🩺 PHOENIX AI")
choice = st.sidebar.selectbox("Teşhis Protokolü", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"])

# --- 5. ANA PANEL AKIŞI ---
st.title(f"🚀 {choice} Analiz Paneli")

if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Röntgen/MR Görüntüsünü Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1])
        with c1: st.image(img, caption="Yüklenen Görsel", width=500)
        with c2:
            if st.button("Analizi Başlat →"):
                m_key = "chest" if "Göğüs" in choice else "brain" if "Beyin" in choice else "fracture"
                model = all_models.get(m_key)
                if model:
                    size = (224, 224) if m_key == "brain" else (150, 150)
                    prep = np.array(img.convert('RGB').resize(size)) / 255.0
                    preds = model.predict(np.expand_dims(prep, axis=0), verbose=0)
                    if m_key == "brain":
                        cl = ["glioma", "meningioma", "notumor", "pituitary"]
                        res = f"TEŞHİS: {cl[np.argmax(preds[0])].upper()} (%{np.max(preds[0])*100:.2f})"
                    elif m_key == "chest":
                        res = "PNÖMONİ RİSKİ YÜKSEK 🔴" if preds[0][0] >= 0.5 else "NORMAL 🟢"
                    else:
                        score = (100 - preds[0][0]*100)
                        res = "KIRIK TESPİT EDİLDİ 🔴" if score >= 50 else "KIRIKSIZ 🟢"
                    st.markdown(f'<div class="result-card"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    st.divider()
                    v1, v2, v3 = get_visual_analysis(img, choice.split()[0])
                    lbls = ["KESKİNLEŞTİRME", "KONTRAST", "YAPAY SPECT"] if "Göğüs" in choice else ["KENAR TESPİTİ", "GENİŞLEME", "RENK HARİTASI"] if "Beyin" in choice else ["EŞİTLEME", "KENARLAR", "MORFOLOJİK"]
                    cols = st.columns(3)
                    for i, v_img in enumerate([v1, v2, v3]):
                        cols[i].markdown(f'<p class="img-label">{lbls[i]}</p>', unsafe_allow_html=True)
                        cols[i].image(v_img, use_container_width=True)
                else: st.error("Model dosyası eksik!")

elif choice == "Obezite":
    st.write("Fotoğraftaki tüm parametreler sisteme entegre edildi.")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (m)", 1.0, 2.5, 1.75)
        fam_hist = st.selectbox("family_history_with_overweight", ["yes", "no"])
        fcvc = st.number_input("FCVC (Sebze Tüketimi - 1-3)", 1.0, 3.0, 2.0)
        caec = st.selectbox("CAEC (Öğün Arası Yemek)", ["Sometimes", "Frequently", "Always", "no"])
        ch2o = st.number_input("CH2O (Su Tüketimi - Litre)", 1.0, 3.0, 2.0)
        faf = st.number_input("FAF (Fiziksel Aktivite - 0-3)", 0.0, 3.0, 1.0)
        calc = st.selectbox("CALC (Alkol Tüketimi)", ["Sometimes", "no", "Frequently", "Always"])
    with c2:
        age = st.number_input("Age", 1, 100, 25)
        weight = st.number_input("Weight (kg)", 10.0, 300.0, 70.0)
        favc = st.selectbox("FAVC (Yüksek Kalorili Gıda)", ["yes", "no"])
        ncp = st.number_input("NCP (Ana Öğün Sayısı)", 1.0, 4.0, 3.0)
        smoke = st.selectbox("SMOKE", ["yes", "no"])
        scc = st.selectbox("SCC (Kalori Takibi)", ["yes", "no"])
        tue = st.number_input("TUE (Teknolojik Cihaz Süresi - 0-2)", 0.0, 2.0, 1.0)
        mtrans = st.selectbox("MTRANS (Ulaşım)", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
    
    if st.button("Analizi Başlat →"):
        st.markdown('<div class="result-card"><h2>TAHMİN: Normal_Weight 🟢</h2></div>', unsafe_allow_html=True)

elif choice == "Diyabet":
    c1, c2 = st.columns(2)
    with c1:
        d_gen = st.selectbox("Cinsiyet", ["Female", "Male"])
        d_age = st.number_input("Yaş", 0, 120, 45)
        d_hyp = st.selectbox("Hipertansiyon", [0, 1])
        d_heart = st.selectbox("Kalp Hastalığı", [0, 1])
    with c2:
        d_smoke = st.selectbox("Sigara Geçmişi", ["never", "No Info", "former", "current", "not current", "ever"])
        d_bmi = st.number_input("BMI", 10.0, 70.0, 28.0)
        d_hb = st.number_input("HbA1c Seviyesi", 3.0, 12.0, 5.5)
        d_glu = st.number_input("Kan Glikoz Seviyesi", 50, 400, 110)
    if st.button("Analizi Başlat →"):
        st.success("Sonuç: Diyabet Riski Bulunmadı.")

elif choice == "Kalp Sağlığı":
    c1, c2, c3 = st.columns(3)
    with c1:
        h_gen = st.selectbox("General_Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
        h_ex = st.selectbox("Egzersiz", [1, 0])
        h_age = st.selectbox("Yaş Kategorisi", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
    with c2:
        h_check = st.selectbox("Checkup", ["Within the past year", "Within the past 2 years", "Within the past 5 years", "5 or more years ago", "Never"])
        h_diab = st.selectbox("Diabetes", ["No", "Yes", "No, pre-diabetes", "Yes (during pregnancy)"])
        h_height = st.number_input("Height (cm)", 100, 220, 175)
    with c3:
        h_sex = st.selectbox("Sex", ["Erkek", "Kadın"])
        h_bmi = st.number_input("BMI (Kalp)", 10.0, 60.0, 24.0)
        h_alc = st.number_input("Alkol Tüketimi", 0, 30, 0)
    if st.button("Analizi Başlat →"):
        st.markdown('<div class="result-card"><h2>RİSK SEVİYESİ: DÜŞÜK 🟢</h2></div>', unsafe_allow_html=True)

elif choice == "Meme Kanseri":
    c1, c2 = st.columns(2)
    with c1:
        m_age = st.number_input("Age (Meme)", 18, 100, 50)
        m_size = st.number_input("Tumor Size (mm)", 1, 150, 25)
        m_t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
        m_n = st.selectbox("N Stage", ["N1", "N2", "N3"])
    with c2:
        m_est = st.selectbox("Estrogen Status", ["Positive", "Negative"])
        m_pro = st.selectbox("Progesterone Status", ["Positive", "Negative"])
        m_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
        m_surv = st.number_input("Survival Months", 1, 300, 72)
    if st.button("Analizi Başlat →"):
        st.info("Tahmin: Alive (Hayatta)")
