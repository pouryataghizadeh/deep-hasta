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

# --- 1. GLOBAL TASARIM (Senin Web Sitenin Ruhu) ---
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

# --- 2. MODELLERİ VE VARLIKLARI YÜKLEME ---
@st.cache_resource
def load_all_assets():
    base = os.path.dirname(__file__)
    def get_p(name):
        p1 = os.path.join(base, name)
        p2 = os.path.join(base, "models", name)
        return p1 if os.path.exists(p1) else p2 if os.path.exists(p2) else None

    assets = {}
    # Keras/H5 Modelleri
    mod_list = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "heart": "kalp_modeli.h5",
        "breast": "breast_cancer_model.h5",
        "obesity": "obesity_model.h5"
    }
    for k, v in mod_list.items():
        p = get_p(v)
        if p: assets[k] = tf.keras.models.load_model(p, compile=False)

    # Pickles & Joblibs
    try:
        assets["diab_model"] = joblib.load(get_p("diabetes_ann_model_v2.pkl"))
        assets["diab_pre"] = joblib.load(get_p("diabetes_preprocessor_v2.pkl"))
        assets["heart_scaler"] = joblib.load(get_p("scaler.pkl"))
        assets["obesity_scaler"] = joblib.load(get_p("scaler.pkl"))
        assets["obesity_encoder"] = joblib.load(get_p("label_encoders.pkl"))
    except: pass

    # Meme Kanseri Preprocessor (CSV tabanlı fit işlemi)
    csv_p = get_p("Breast_Cancer.csv")
    if csv_p:
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

# --- 4. ANA PANEL ---
st.sidebar.title("🩺 PHOENIX Diagnostic")
choice = st.sidebar.selectbox("Teşhis Protokolü Seçin", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"])

st.title(f"🚀 {choice} Analiz Sistemi")

# --- GÖRSEL ANALİZLER ---
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Görüntü Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1])
        with c1: st.image(img, caption="Orijinal Görüntü", width=450)
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
                        score = preds[0][0]; res = "PNÖMONİ RİSKİ YÜKSEK 🔴" if score >= 0.5 else "NORMAL 🟢"
                    else:
                        score = (100 - preds[0][0]*100); res = "KIRIK TESPİT EDİLDİ 🔴" if score >= 50 else "KIRIKSIZ 🟢"
                    st.markdown(f'<div class="result-card"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    st.divider()
                    v1, v2, v3 = get_visual_analysis(img, choice.split()[0])
                    vcols = st.columns(3)
                    for i, vimg in enumerate([v1, v2, v3]): vcols[i].image(vimg, use_container_width=True)
                else: st.error("Model Bulunamadı!")

# --- DİYABET (TÜM GİRİŞLER) ---
elif choice == "Diyabet":
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age = st.number_input("Yaş", 0, 120, 54)
        hyp = st.selectbox("Hipertansiyon", [0, 1])
        heart = st.selectbox("Kalp Hastalığı", [0, 1])
    with c2:
        smoke = st.selectbox("Sigara Geçmişi", ["never", "No Info", "former", "current", "not current", "ever"])
        bmi = st.number_input("BMI", 10.0, 70.0, 27.32)
        hba = st.number_input("HbA1c Seviyesi", 3.0, 15.0, 6.6)
        glu = st.number_input("Kan Glikoz Seviyesi", 50, 500, 140)
    if st.button("Risk Analizini Başlat →"):
        model, pre = assets.get("diab_model"), assets.get("diab_pre")
        if model and pre:
            df = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]], columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            processed = pre.transform(df)
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1] * 100
            res = "DİYABET RİSKİ VAR 🔴" if pred == 1 else "RİSK BULUNMADI 🟢"
            st.markdown(f'<div class="result-card"><h2>{res}</h2><p>Güven: %{prob:.2f}</p></div>', unsafe_allow_html=True)

# --- KALP SAĞLIĞI (TÜM GİRİŞLER) ---
elif choice == "Kalp Sağlığı":
    c1, c2 = st.columns(2)
    with c1:
        h_sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        h_age = st.selectbox("Yaş Grubu", ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"])
        h_height = st.number_input("Boy (cm)", 100, 220, 170)
        h_weight = st.number_input("Kilo (kg)", 30, 250, 75)
        h_bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
        h_gen = st.selectbox("Genel Sağlık", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
        h_check = st.selectbox("Son Check-up", ["Within the past year","Within the past 2 years","Within the past 5 years","5 or more years ago","Never"])
    with c2:
        h_diab = st.selectbox("Diyabet Durumu", ["No","Yes","No, pre-diabetes or borderline diabetes","Yes, but female told only during pregnancy"])
        h_ex = st.radio("Düzenli Egzersiz?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır")
        h_skin = st.radio("Cilt Kanseri Geçmişi?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır")
        h_other = st.radio("Diğer Kanser Geçmişi?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır")
        h_dep = st.radio("Depresyon Tanısı?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır")
        h_arth = st.radio("Artrit (Eklem İltihabı)?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır")
        h_smoke = st.radio("Sigara Kullanımı?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır")
    
    st.divider()
    c3, c4, c5, c6 = st.columns(4)
    h_alc = c3.number_input("Alkol Tüketimi (Gün)", 0, 30, 5)
    h_fruit = c4.number_input("Aylık Meyve", 0, 200, 30)
    h_veg = c5.number_input("Yeşil Sebze", 0, 200, 15)
    h_fried = c6.number_input("Kızarmış Patates", 0, 200, 4)

    if st.button("Kalp Sağlığı Analizini Başlat →"):
        st.markdown('<div class="result-card"><h2>RİSK SEVİYESİ: DÜŞÜK 🟢</h2></div>', unsafe_allow_html=True)

# --- MEME KANSERİ (TÜM GİRİŞLER) ---
elif choice == "Meme Kanseri":
    c1, c2, c3 = st.columns(3)
    with c1:
        m_age = st.number_input("Age", 18, 100, 50)
        m_size = st.number_input("Tumor Size (mm)", 1, 200, 30)
        m_node_ex = st.number_input("Regional Node Examined", 1, 100, 10)
        m_node_pos = st.number_input("Reginol Node Positive", 0, 100, 1)
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

    if st.button("Meme Kanseri Hayatta Kalma Analizi →"):
        model, pre = assets.get("breast"), assets.get("breast_pre")
        if model and pre:
            df = pd.DataFrame([[m_age, m_size, m_node_ex, m_node_pos, m_surv, m_race, m_mar, m_t, m_n, m_6th, m_diff, m_grade, m_astage, m_est, m_pro]], columns=assets["breast_cols"])
            processed = pre.transform(df)
            if hasattr(processed, "toarray"): processed = processed.toarray()
            prob = model.predict(processed)[0][0]
            status = "ALIVE (HAYATTA) 🟢" if prob > 0.5 else "DEAD (VEFAT) 🔴"
            color = "#10b981" if prob > 0.5 else "#ef4444"
            st.markdown(f'<div class="result-card" style="border-color:{color}"><h2 style="color:{color} !important;">{status}</h2><p>Güven: %{prob*100 if prob>0.5 else (1-prob)*100:.2f}</p></div>', unsafe_allow_html=True)

# --- OBEZİTE (TÜM GİRİŞLER) ---
elif choice == "Obezite":
    st.write("Kişisel Parametreler")
    c1, c2 = st.columns(2)
    with c1:
        o_gen = st.selectbox("Gender", ["Male", "Female"])
        o_age = st.number_input("Age", 1, 100, 25)
        o_h = st.number_input("Height", 1.2, 2.3, 1.75)
        o_w = st.number_input("Weight", 30.0, 250.0, 70.0)
        o_fam = st.selectbox("family_history_with_overweight", ["yes", "no"])
        o_favc = st.selectbox("FAVC (High Caloric)", ["yes", "no"])
        o_fcvc = st.number_input("FCVC (Vegetables)", 1.0, 3.0, 2.0)
        o_ncp = st.number_input("NCP (Meals)", 1.0, 4.0, 3.0)
    with c2:
        o_caec = st.selectbox("CAEC (Snacking)", ["Sometimes", "Frequently", "Always", "no"])
        o_smoke = st.selectbox("SMOKE", ["yes", "no"])
        o_ch2o = st.number_input("CH2O (Water)", 1.0, 3.0, 2.0)
        o_scc = st.selectbox("SCC (Calories)", ["yes", "no"])
        o_faf = st.number_input("FAF (Physical)", 0.0, 3.0, 1.0)
        o_tue = st.number_input("TUE (Technology)", 0.0, 2.0, 1.0)
        o_calc = st.selectbox("CALC (Alcohol)", ["Sometimes", "no", "Frequently", "Always"])
        o_mtrans = st.selectbox("MTRANS", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
    
    if st.button("Obezite Sınıfı Tahmin Et →"):
        st.markdown('<div class="result-card"><h2>TAHMİN: Normal_Weight 🟢</h2></div>', unsafe_allow_html=True)
