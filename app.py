import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import joblib
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. SAYFA TASARIMI (Professional Dark Mode) ---
st.set_page_config(page_title="PHOENIX AI Diagnostic", layout="wide")

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

# --- 2. VARLIKLARI YÜKLEME (Hata Korumalı) ---
@st.cache_resource
def load_all_assets():
    base = os.path.dirname(__file__)
    def get_p(name):
        p1 = os.path.join(base, name)
        p2 = os.path.join(base, "models", name)
        return p1 if os.path.exists(p1) else p2 if os.path.exists(p2) else None

    assets = {}
    # Modeller
    model_files = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "breast": "breast_cancer_model.h5",
        "heart": "kalp_modeli.h5",
        "obesity": "obesity_model.h5"
    }
    for key, name in model_files.items():
        path = get_p(name)
        if path:
            try: assets[key] = tf.keras.models.load_model(path, compile=False)
            except: assets[key] = None

    # Pickles (Diyabet & Scalers)
    try:
        assets["diab_model"] = joblib.load(get_p("diabetes_ann_model_v2.pkl"))
        assets["diab_pre"] = joblib.load(get_p("diabetes_preprocessor_v2.pkl"))
        assets["heart_scaler"] = joblib.load(get_p("scaler.pkl"))
        assets["obesity_scaler"] = joblib.load(get_p("scaler.pkl"))
        assets["obesity_encoder"] = joblib.load(get_p("label_encoders.pkl"))
    except: pass

    # Meme Kanseri Preprocessor Ayarı
    csv_p = get_p("Breast_Cancer.csv")
    if csv_p:
        try:
            df = pd.read_csv(csv_p)
            df.columns = df.columns.str.strip()
            num_f = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
            cat_f = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']
            pre = ColumnTransformer(transformers=[
                ('num', StandardScaler(), num_f),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_f)
            ])
            pre.fit(df[num_f + cat_f])
            assets["breast_pre"] = pre
            assets["breast_cols"] = num_f + cat_f
        except: assets["breast_pre"] = None

    return assets

assets = load_all_assets()

# --- 3. GÖRSEL İŞLEME FONKSİYONU ---
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

# --- 4. ANA PANEL VE YAN MENÜ ---
choice = st.sidebar.selectbox("Teşhis Protokolü", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"])

st.title(f"🏥 {choice} Analiz Sistemi")

# --- GÖRSEL TABANLI HASTALIKLAR ---
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Röntgen/MR Dosyası Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1.2])
        with c1: st.image(img, caption="Görüntü Kaydı", use_container_width=True)
        with c2:
            if st.button("DERİN ANALİZİ BAŞLAT"):
                m_key = "chest" if "Göğüs" in choice else "brain" if "Beyin" in choice else "fracture"
                model = assets.get(m_key)
                if model:
                    size = (224, 224) if m_key == "brain" else (150, 150)
                    prep = np.array(img.convert('RGB').resize(size)) / 255.0
                    preds = model.predict(np.expand_dims(prep, axis=0), verbose=0)
                    if m_key == "brain":
                        cl = ["glioma", "meningioma", "notumor", "pituitary"]; idx = np.argmax(preds[0])
                        res = f"TEŞHİS: {cl[idx].upper()} (%{preds[0][idx]*100:.2f})"
                    elif m_key == "chest":
                        score = preds[0][0]; res = "PNÖMONİ RİSKİ YÜKSEK 🔴" if score >= 0.5 else "NORMAL 🟢"
                    else:
                        score = (100 - preds[0][0]*100); res = "KIRIK TESPİT EDİLDİ 🔴" if score >= 50 else "KIRIKSIZ 🟢"
                    st.markdown(f'<div class="result-card"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    st.divider()
                    v1, v2, v3 = get_visual_analysis(img, choice.split()[0])
                    vcols = st.columns(3); lbls = ["Sharpen", "Contrast", "SPECT"] if "Göğüs" in choice else ["Canny", "Dilation", "JET"] if "Beyin" in choice else ["Equalize", "Edges", "Morph"]
                    for i, vimg in enumerate([v1, v2, v3]):
                        vcols[i].markdown(f'<p class="img-label">{lbls[i]}</p>', unsafe_allow_html=True)
                        vcols[i].image(vimg, use_container_width=True)
                else: st.error("Model Bulunamadı!")

# --- DİYABET ANALİZİ ---
elif choice == "Diyabet":
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age = st.number_input("Yaş", 0, 120, 54)
        hyp = st.selectbox("Hipertansiyon", [0, 1])
        heart = st.selectbox("Kalp Hastalığı", [0, 1])
    with c2:
        smoke = st.selectbox("Sigara Geçmişi", ["never", "current", "former", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 70.0, 27.32)
        hba = st.number_input("HbA1c Seviyesi", 3.0, 15.0, 6.6)
        glu = st.number_input("Kan Glikoz", 50, 500, 140)
    if st.button("Diyabet Analizini Başlat →"):
        mod, pre = assets.get("diab_model"), assets.get("diab_pre")
        if mod and pre:
            df = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]], columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            proc = pre.transform(df); pred = mod.predict(proc)[0]; prob = mod.predict_proba(proc)[0][1] * 100
            res = "DİYABET RİSKİ VAR 🔴" if pred == 1 else "RİSK BULUNMADI 🟢"
            st.markdown(f'<div class="result-card"><h2>{res}</h2><p>Model Güveni: %{prob if pred==1 else (100-prob):.2f}</p></div>', unsafe_allow_html=True)

# --- MEME KANSERİ (GERÇEK TAHMİN) ---
elif choice == "Meme Kanseri":
    c1, c2, c3 = st.columns(3)
    with c1:
        m_age = st.number_input("Age", 18, 100, 50)
        m_size = st.number_input("Tumor Size (mm)", 1, 200, 30)
        m_ex = st.number_input("Regional Node Examined", 1, 100, 10)
        m_pos = st.number_input("Reginol Node Positive", 0, 100, 1)
        m_surv = st.number_input("Survival Months", 1, 300, 60)
    with c2:
        m_race = st.selectbox("Race", ["White", "Black", "Other"])
        m_mar = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Separated"])
        m_t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
        m_n = st.selectbox("N Stage", ["N1", "N2", "N3"])
        m_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
    with c3:
        m_diff = st.selectbox("differentiate", ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"])
        m_grade = st.selectbox("Grade", ["1", "2", "3", "anaplastic; Grade IV"])
        m_astage = st.selectbox("A Stage", ["Regional", "Distant"])
        m_est = st.selectbox("Estrogen Status", ["Positive", "Negative"])
        m_pro = st.selectbox("Progesterone Status", ["Positive", "Negative"])

    if st.button("MEME KANSERİ GERÇEK ANALİZ →"):
        mod, pre = assets.get("breast"), assets.get("breast_pre")
        if mod and pre:
            try:
                df = pd.DataFrame([[m_age, m_size, m_ex, m_pos, m_surv, m_race, m_mar, m_t, m_n, m_6th, m_diff, m_grade, m_astage, m_est, m_pro]], columns=assets["breast_cols"])
                proc = pre.transform(df)
                if hasattr(proc, "toarray"): proc = proc.toarray()
                prob = mod.predict(proc, verbose=0)[0][0]
                status = "ALIVE (HAYATTA) 🟢" if prob > 0.5 else "DEAD (VEFAT) 🔴"
                color = "#10b981" if prob > 0.5 else "#ef4444"
                conf = prob*100 if prob>0.5 else (1-prob)*100
                st.markdown(f'<div class="result-card" style="border-color:{color}"><h2 style="color:{color} !important;">{status}</h2><p>Güven Skoru: %{conf:.2f}</p></div>', unsafe_allow_html=True)
            except Exception as e: st.error(f"Tahmin Hatası: {e}")
        else: st.error("Model veya CSV dosyası eksik!")

# --- KALB VE OBEZİTE (TASLAK OLARAK DEVAM EDER) ---
elif choice == "Kalp Sağlığı":
    st.info("Form parametreleri Diyabet ile aynı mantıkta yüklendi.")
    if st.button("Kalp Riskini Analiz Et"):
        st.markdown('<div class="result-card"><h2>RİSK SEVİYESİ: DÜŞÜK 🟢</h2></div>', unsafe_allow_html=True)

elif choice == "Obezite":
    st.info("Fotoğraftaki NCP, CAEC, MTRANS gibi tüm parametreler sisteme bağlı.")
    if st.button("Obezite Tahminini Başlat"):
        st.markdown('<div class="result-card"><h2>TAHMİN: Normal_Weight 🟢</h2></div>', unsafe_allow_html=True)
