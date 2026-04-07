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

# --- 1. GLOBAL TASARIM ---
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

# --- 2. AKILLI VARLIK YÜKLEYİCİ ---
@st.cache_resource
def load_all_assets():
    base = os.path.dirname(__file__)
    def get_p(name):
        p1 = os.path.join(base, name)
        p2 = os.path.join(base, "models", name)
        return p1 if os.path.exists(p1) else p2 if os.path.exists(p2) else None

    assets = {}
    
    # H5 / Keras Modelleri
    model_list = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "breast_model": "breast_cancer_model.h5"
    }
    
    for key, name in model_list.items():
        path = get_p(name)
        if path:
            try:
                assets[key] = tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                st.sidebar.error(f"❌ {name} yüklenemedi: {e}")
                assets[key] = None
        else:
            assets[key] = None

    # Diyabet (PKL)
    try:
        assets["diab_model"] = joblib.load(get_p("diabetes_ann_model_v2.pkl"))
        assets["diab_pre"] = joblib.load(get_p("diabetes_preprocessor_v2.pkl"))
    except:
        assets["diab_model"] = None

    # Meme Kanseri Preprocessor (CSV tabanlı)
    try:
        csv_path = get_p("Breast_Cancer.csv")
        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            num_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
            cat_cols = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']
            
            pre = ColumnTransformer(transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ])
            pre.fit(df[num_cols + cat_cols])
            assets["breast_pre"] = pre
            assets["breast_cols"] = num_cols + cat_cols
        else:
            assets["breast_pre"] = None
    except Exception as e:
        st.sidebar.error(f"⚠️ Meme Kanseri CSV hatası: {e}")
        assets["breast_pre"] = None

    return assets

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
    else: # Kemik
        o1 = cv2.equalizeHist(gray)
        o2 = cv2.Canny(gray, 100, 200)
        o3 = cv2.morphologyEx(o1, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        return o1, o2, o3

# --- 4. YAN MENÜ VE PANEL ---
st.sidebar.title("🩺 PHOENIX Diagnostic")
choice = st.sidebar.selectbox("Teşhis Protokolü", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Meme Kanseri"])
st.title(f"🚀 {choice} Analiz Paneli")

# --- GÖRSEL TABANLI ---
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Görüntü Dosyasını Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1])
        with c1: st.image(img, caption="Orijinal Görüntü", use_container_width=True)
        with c2:
            if st.button("ANALİZİ BAŞLAT"):
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
                        res = "PNÖMONİ RİSKİ YÜKSEK 🔴" if score >= 0.5 else "NORMAL 🟢"
                    else:
                        score = (100 - preds[0][0]*100)
                        res = "KIRIK TESPİT EDİLDİ 🔴" if score >= 50 else "KIRIKSIZ 🟢"
                    st.markdown(f'<div class="result-card"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    st.divider()
                    v1, v2, v3 = get_visual_analysis(img, choice.split()[0])
                    v_cols = st.columns(3)
                    for i, v_img in enumerate([v1, v2, v3]): v_cols[i].image(v_img, use_container_width=True)
                else: st.error("Model yüklenemedi!")

# --- DİYABET ---
elif choice == "Diyabet":
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age = st.number_input("Yaş", 0, 120, 45)
        hyp = st.selectbox("Hipertansiyon", [0, 1])
        heart = st.selectbox("Kalp Hastalığı", [0, 1])
    with c2:
        smoke = st.selectbox("Sigara Geçmişi", ["never", "current", "former", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 70.0, 28.0)
        hba = st.number_input("HbA1c", 3.0, 15.0, 6.0)
        glu = st.number_input("Kan Glikoz", 50, 400, 110)
    
    if st.button("Diyabet Analizini Başlat"):
        model, pre = assets.get("diab_model"), assets.get("diab_pre")
        if model and pre:
            df = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]], columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            processed = pre.transform(df)
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1] * 100
            res = "DİYABET RİSKİ VAR 🔴" if pred == 1 else "RİSK BULUNMADI 🟢"
            st.markdown(f'<div class="result-card"><h2>{res}</h2><p>Olasılık: %{prob:.2f}</p></div>', unsafe_allow_html=True)
        else: st.error("Diyabet model dosyaları eksik!")

# --- MEME KANSERİ (Çökme Engellenmiş Versiyon) ---
elif choice == "Meme Kanseri":
    c1, c2 = st.columns(2)
    with c1:
        m_age = st.number_input("Age", 18, 100, 50)
        m_t_size = st.number_input("Tumor Size (mm)", 1, 200, 30)
        m_ex = st.number_input("Regional Node Examined", 1, 100, 10)
        m_pos = st.number_input("Reginol Node Positive", 0, 100, 1)
        m_surv = st.number_input("Survival Months", 1, 300, 60)
        m_race = st.selectbox("Race", ["White", "Black", "Other"])
        m_mar = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Separated"])
    with c2:
        m_t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
        m_n = st.selectbox("N Stage", ["N1", "N2", "N3"])
        m_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
        m_diff = st.selectbox("differentiate", ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"])
        m_grade = st.selectbox("Grade", ["1", "2", "3", "anaplastic; Grade IV"])
        m_astage = st.selectbox("A Stage", ["Regional", "Distant"])
        m_est = st.selectbox("Estrogen Status", ["Positive", "Negative"])
        m_pro = st.selectbox("Progesterone Status", ["Positive", "Negative"])

    if st.button("Meme Kanseri Analizini Başlat"):
        # assets.get() kullanarak KeyError'dan kurtuluyoruz
        b_model = assets.get("breast_model")
        b_pre = assets.get("breast_pre")
        b_cols = assets.get("breast_cols")

        if b_model and b_pre:
            input_df = pd.DataFrame([[m_age, m_t_size, m_ex, m_pos, m_surv, m_race, m_mar, m_t, m_n, m_6th, m_diff, m_grade, m_astage, m_est, m_pro]], 
                                    columns=b_cols)
            processed = b_pre.transform(input_df)
            prob = b_model.predict(processed)[0][0]
            status = "DEAD (VEFAT) 🔴" if prob < 0.5 else "ALIVE (HAYATTA) 🟢"
            color = "#ef4444" if prob < 0.5 else "#10b981"
            st.markdown(f'<div class="result-card" style="border-color:{color}"><h2 style="color:{color} !important;">{status}</h2><p>Güven: %{prob*100 if prob>0.5 else (1-prob)*100:.2f}</p></div>', unsafe_allow_html=True)
        else:
            st.error("Hata: Meme kanseri modeli veya 'Breast_Cancer.csv' bulunamadı. Lütfen GitHub'a yüklediğinizden emin olun!")
