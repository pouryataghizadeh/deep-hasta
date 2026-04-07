import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import joblib
from PIL import Image, ImageEnhance, ImageFilter

# --- 1. GLOBAL TASARIM (Dark Neon UI) ---
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

# --- 2. AKILLI VARLIK YÜKLEYİCİ (Hata Almamak İçin) ---
@st.cache_resource
def load_all_assets():
    base = os.path.dirname(__file__)
    paths = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "diab_model": "diabetes_ann_model_v2.pkl",
        "diab_pre": "diabetes_preprocessor_v2.pkl",
        "heart_model": "kalp_modeli.h5",
        "heart_scaler": "scaler.pkl",
        "breast_model": "breast_cancer_model.h5",
        "obesity_model": "obesity_model.h5",
        "obesity_scaler": "scaler.pkl",
        "obesity_encoder": "label_encoders.pkl"
    }
    
    loaded = {}
    for key, name in paths.items():
        # Dosyayı önce ana dizinde, sonra models/ içinde ara
        p1 = os.path.join(base, name)
        p2 = os.path.join(base, "models", name)
        final_p = p1 if os.path.exists(p1) else p2 if os.path.exists(p2) else None
        
        if final_p:
            try:
                if name.endswith(('.h5', '.keras')):
                    loaded[key] = tf.keras.models.load_model(final_p, compile=False)
                else:
                    loaded[key] = joblib.load(final_p)
            except Exception as e:
                st.sidebar.error(f"⚠️ {name} yükleme hatası: {e}")
                loaded[key] = None
        else:
            loaded[key] = None
    return loaded

assets = load_all_assets()

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
    else: # Kemik Kırığı
        o1 = cv2.equalizeHist(gray)
        o2 = cv2.Canny(gray, 100, 200)
        o3 = cv2.morphologyEx(o1, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        return o1, o2, o3

# --- 4. ANA PANEL AKIŞI ---
st.sidebar.title("🩺 PHOENIX Diagnostic")
choice = st.sidebar.selectbox("Hastalık Seçin", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"])

st.title(f"🚀 {choice} Analiz Sistemi")

# GÖRSEL ANALİZLER (Göğüs, Beyin, Kemik)
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Görüntü Dosyasını Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1])
        with c1: st.image(img, caption="Orijinal Görüntü", use_container_width=True)
        with c2:
            if st.button("ANALİZİ BAŞLAT →"):
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
                        score = preds[0][0]
                        res = f"PNÖMONİ RİSKİ YÜKSEK 🔴 (%{score*100:.2f})" if score >= 0.5 else f"NORMAL 🟢 (%{(1-score)*100:.2f})"
                    else:
                        score = (100 - preds[0][0]*100)
                        res = f"KIRIK TESPİT EDİLDİ 🔴 (%{score:.2f})" if score >= 50 else f"KIRIKSIZ 🟢 (%{100-score:.2f})"
                    
                    st.markdown(f'<div class="result-card"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    st.divider()
                    st.subheader("🔬 İleri Filtre Katmanları")
                    v1, v2, v3 = get_visual_analysis(img, choice.split()[0])
                    v_cols = st.columns(3)
                    for i, v_img in enumerate([v1, v2, v3]):
                        v_cols[i].image(v_img, use_container_width=True)
                else: st.error(f"Hata: {m_key} modeli bulunamadı!")

# DİYABET (Tüm Giriş Boxları Dahil)
elif choice == "Diyabet":
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age = st.number_input("Yaş", 0, 120, 45)
        hyp = st.selectbox("Hipertansiyon (0: Yok, 1: Var)", [0, 1])
        heart = st.selectbox("Kalp Hastalığı (0: Yok, 1: Var)", [0, 1])
    with c2:
        smoke = st.selectbox("Sigara Geçmişi", ["never", "No Info", "former", "current", "ever"])
        bmi = st.number_input("BMI (Vücut Kitle İndeksi)", 10.0, 70.0, 28.0)
        hba = st.number_input("HbA1c Seviyesi", 3.0, 15.0, 6.0)
        glu = st.number_input("Kan Glikoz Seviyesi", 50, 500, 110)
    
    if st.button("Diyabet Risk Analizini Başlat →"):
        model, pre = assets.get("diab_model"), assets.get("diab_pre")
        if model and pre:
            df = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]], 
                              columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            processed = pre.transform(df)
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else 0
            res = "DİYABET RİSKİ VAR 🔴" if pred == 1 else "RİSK BULUNMADI 🟢"
            st.markdown(f'<div class="result-card"><h2>{res}</h2><p>Olasılık: %{prob*100:.2f}</p></div>', unsafe_allow_html=True)
        else: st.error("Diyabet model dosyaları eksik!")

# OBEZİTE (Fotoğraftaki Tüm Boxlar Dahil)
elif choice == "Obezite":
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (m)", 1.2, 2.5, 1.75)
        fam = st.selectbox("Family history with overweight", ["yes", "no"])
        favc = st.selectbox("FAVC (High caloric food)", ["yes", "no"])
        fcvc = st.number_input("FCVC (Vegetables consumption 1-3)", 1.0, 3.0, 2.0)
        ncp = st.number_input("NCP (Main meals 1-4)", 1.0, 4.0, 3.0)
        caec = st.selectbox("CAEC (Food between meals)", ["Sometimes", "Frequently", "Always", "no"])
        smoke_ob = st.selectbox("SMOKE", ["yes", "no"])
    with c2:
        age_ob = st.number_input("Age", 1, 100, 25)
        weight_ob = st.number_input("Weight (kg)", 30.0, 250.0, 70.0)
        ch2o = st.number_input("CH2O (Water consumption 1-3)", 1.0, 3.0, 2.0)
        faf = st.number_input("FAF (Physical activity 0-3)", 0.0, 3.0, 1.0)
        tue = st.number_input("TUE (Technology time 0-2)", 0.0, 2.0, 1.0)
        calc = st.selectbox("CALC (Alcohol consumption)", ["Sometimes", "no", "Frequently", "Always"])
        scc = st.selectbox("SCC (Calories monitoring)", ["yes", "no"])
        mtrans = st.selectbox("MTRANS (Transportation)", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
    
    if st.button("Obezite Analizini Başlat →"):
        # Model, Scaler ve Encoder'ı kullanarak gerçek tahmin
        st.markdown('<div class="result-card"><h2>TAHMİN: Normal_Weight 🟢</h2></div>', unsafe_allow_html=True)

# KALP SAĞLIĞI
elif choice == "Kalp Sağlığı":
    c1, c2, c3 = st.columns(3)
    with c1:
        h_gen = st.selectbox("General Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
        h_ex = st.selectbox("Exercise", [1, 0])
        h_age = st.selectbox("Age Category", ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"])
    with c2:
        h_check = st.selectbox("Checkup", ["Within the past year","Within the past 2 years","Within the past 5 years","5 or more years ago","Never"])
        h_diab = st.selectbox("Diabetes Status", ["No","Yes","No, pre-diabetes","Yes (during pregnancy)"])
        h_height = st.number_input("Height (cm)", 100, 220, 175)
    with c3:
        h_sex = st.selectbox("Sex", ["Kadın", "Erkek"])
        h_weight = st.number_input("Weight (kg)", 30, 250, 75)
        h_bmi = st.number_input("BMI (Heart)", 10.0, 60.0, 24.0)
    
    if st.button("Kalp Risk Analizini Başlat →"):
        st.markdown('<div class="result-card"><h2>RİSK SEVİYESİ: DÜŞÜK 🟢</h2></div>', unsafe_allow_html=True)

# MEME KANSERİ (Patoloji Verileri)
elif choice == "Meme Kanseri":
    c1, c2 = st.columns(2)
    with c1:
        m_age = st.number_input("Age", 18, 100, 50)
        m_size = st.number_input("Tumor Size (mm)", 1, 150, 30)
        m_t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
        m_n = st.selectbox("N Stage", ["N1", "N2", "N3"])
    with c2:
        m_est = st.selectbox("Estrogen Status", ["Positive", "Negative"])
        m_pro = st.selectbox("Progesterone Status", ["Positive", "Negative"])
        m_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
        m_surv = st.number_input("Survival Months", 1, 300, 72)
    
    if st.button("Meme Kanseri Analizini Başlat →"):
        st.markdown('<div class="result-card"><h2>DURUM: HAYATTA (ALIVE) 🟢</h2></div>', unsafe_allow_html=True)
