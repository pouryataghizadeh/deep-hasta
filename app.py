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
    h1, h2, h3 { color: #3b82f6 !important; font-weight: 800; }
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #000; border-radius: 14px; padding: 18px; font-weight: 800; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODELLERİ VE PREPROCESSOR'LARI YÜKLEME ---
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    def get_p(name):
        p = os.path.join(base, name)
        return p if os.path.exists(p) else os.path.join(base, "models", name)

    assets = {}
    # Modeller
    assets["chest"] = tf.keras.models.load_model(get_p("chest_xray_pneumonia_model.h5"), compile=False)
    assets["brain"] = tf.keras.models.load_model(get_p("brain_tumor_model.h5"), compile=False)
    assets["fracture"] = tf.keras.models.load_model(get_p("best_fracture_detector_model.keras"), compile=False)
    assets["diab_model"] = joblib.load(get_p("diabetes_ann_model_v2.pkl"))
    assets["diab_pre"] = joblib.load(get_p("diabetes_preprocessor_v2.pkl"))
    assets["breast_model"] = tf.keras.models.load_model(get_p("breast_cancer_model.h5"), compile=False)
    
    # Meme Kanseri Ön İşlemcisi İçin Eğitim Verisi Gereklidir (Flask kodundaki mantık)
    csv_p = get_p("Breast_Cancer.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        df.columns = df.columns.str.strip()
        num_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
        cat_cols = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ])
        preprocessor.fit(df[num_cols + cat_cols])
        assets["breast_pre"] = preprocessor
        assets["breast_cols"] = num_cols + cat_cols
    
    return assets

assets = load_assets()

# --- 3. ANA PANEL ---
st.sidebar.title("🩺 PHOENIX Diagnostic")
choice = st.sidebar.selectbox("Teşhis Protokolü", ["Meme Kanseri", "Diyabet", "Beyin Tümörü", "Göğüs (Pnömoni)", "Kemik Kırığı"])

st.title(f"🚀 {choice} Analizi")

# --- MEME KANSERİ (GERÇEK TAHMİN) ---
if choice == "Meme Kanseri":
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 100, 50)
        t_size = st.number_input("Tumor Size (mm)", 1, 200, 30)
        r_node_ex = st.number_input("Regional Node Examined", 1, 100, 10)
        r_node_pos = st.number_input("Reginol Node Positive", 0, 100, 1)
        surv_months = st.number_input("Survival Months", 1, 300, 60)
        race = st.selectbox("Race", ["White", "Black", "Other"])
        marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Separated"])
    with c2:
        t_stage = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
        n_stage = st.selectbox("N Stage", ["N1", "N2", "N3"])
        stage_6 = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
        diff = st.selectbox("differentiate", ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"])
        grade = st.selectbox("Grade", ["1", "2", "3", "anaplastic; Grade IV"])
        a_stage = st.selectbox("A Stage", ["Regional", "Distant"])
        estrogen = st.selectbox("Estrogen Status", ["Positive", "Negative"])
        progesterone = st.selectbox("Progesterone Status", ["Positive", "Negative"])

    if st.button("GERÇEK ANALİZİ BAŞLAT"):
        if "breast_model" in assets and "breast_pre" in assets:
            # Girdiyi DataFrame yap
            input_data = pd.DataFrame([[age, t_size, r_node_ex, r_node_pos, surv_months, race, marital, t_stage, n_stage, stage_6, diff, grade, a_stage, estrogen, progesterone]], 
                                     columns=assets["breast_cols"])
            
            # Ön işlemden geçir ve tahmin yap
            processed = assets["breast_pre"].transform(input_data)
            prob = assets["breast_model"].predict(processed)[0][0]
            
            # Olasılığa göre gerçek sonuç
            status = "ALIVE (HAYATTA) 🟢" if prob > 0.5 else "DEAD (VEFAT) 🔴"
            color = "#10b981" if prob > 0.5 else "#ef4444"
            
            st.markdown(f"""
                <div class="result-card" style="border-color: {color};">
                    <h2 style="color: {color} !important;">{status}</h2>
                    <p>Güven Skoru: %{prob*100 if prob > 0.5 else (1-prob)*100:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Meme kanseri modeli veya veri dosyası (.csv) bulunamadı!")

# --- DİYABET (GERÇEK TAHMİN) ---
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
        hba = st.number_input("HbA1c Seviyesi", 3.0, 15.0, 5.5)
        glu = st.number_input("Kan Glikoz Seviyesi", 50, 400, 110)

    if st.button("DİYABET ANALİZİ BAŞLAT"):
        model, pre = assets.get("diab_model"), assets.get("diab_pre")
        if model and pre:
            df = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]], 
                              columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            processed = pre.transform(df)
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else 0
            
            res = "DİYABET RİSKİ VAR 🔴" if pred == 1 else "RİSK BULUNMADI 🟢"
            color = "#ef4444" if pred == 1 else "#10b981"
            st.markdown(f'<div class="result-card" style="border-color:{color};"><h2 style="color:{color} !important;">{res}</h2><p>Olasılık: %{prob*100 if pred==1 else (1-prob)*100:.2f}</p></div>', unsafe_allow_html=True)

# Not: Diğer görsel analiz bölümleri (Beyin, Göğüs, Kırık) bir önceki hatasız kodundaki 3'lü resim mantığıyla buraya eklenebilir.
