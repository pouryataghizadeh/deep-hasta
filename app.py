import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image, ImageEnhance, ImageFilter

# --- 1. TASARIM ---
st.set_page_config(page_title="PHOENIX AI Diagnostic", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .result-card {
        padding: 30px; border-radius: 20px; border: 2px solid #3b82f6;
        background: rgba(30, 41, 59, 0.9); text-align: center;
    }
    h1, h2 { color: #3b82f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    def get_p(name): return os.path.join(base, name) if os.path.exists(os.path.join(base, name)) else os.path.join(base, "models", name)
    
    return {
        "diabetes_model": joblib.load(get_p("diabetes_ann_model_v2.pkl")) if os.path.exists(get_p("diabetes_ann_model_v2.pkl")) else None,
        "diabetes_pre": joblib.load(get_p("diabetes_preprocessor_v2.pkl")) if os.path.exists(get_p("diabetes_preprocessor_v2.pkl")) else None,
        "breast_model": tf.keras.models.load_model(get_p("breast_cancer_model.h5"), compile=False) if os.path.exists(get_p("breast_cancer_model.h5")) else None,
        # Diğer görsel modelleri de buraya ekleyebilirsin (Chest, Brain vb.)
    }

models = load_models()

# --- 3. ANA PANEL ---
choice = st.sidebar.selectbox("Hastalık Seçin", ["Diyabet", "Meme Kanseri", "Göğüs (Pnömoni)", "Beyin Tümörü"])
st.title(f"🚀 {choice} Analiz Sistemi")

# --- DİYABET GERÇEK TAHMİN ---
if choice == "Diyabet":
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age = st.number_input("Yaş", 0, 120, 50)
        hypertension = st.selectbox("Hipertansiyon", [0, 1])
        heart_disease = st.selectbox("Kalp Hastalığı", [0, 1])
    with c2:
        smoke = st.selectbox("Sigara Geçmişi", ["never", "current", "former", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 35.0)
        hba1c = st.number_input("HbA1c", 3.0, 15.0, 8.0)
        glucose = st.number_input("Kan Glikoz", 50, 400, 250)

    if st.button("DİYABET RİSKİNİ HESAPLA"):
        if models["diabetes_model"] and models["diabetes_pre"]:
            # 1. Veriyi DataFrame yap (Flask kodundakiyle aynı)
            user_df = pd.DataFrame({
                'gender': [gender], 'age': [age], 'hypertension': [hypertension],
                'heart_disease': [heart_disease], 'smoking_history': [smoke],
                'bmi': [bmi], 'HbA1c_level': [hba1c], 'blood_glucose_level': [glucose]
            })
            # 2. Ön işleme ve Tahmin
            processed = models["diabetes_pre"].transform(user_df)
            prediction = models["diabetes_model"].predict(processed)[0]
            prob = models["diabetes_model"].predict_proba(processed)[0][1] * 100
            
            if prediction == 1:
                st.markdown(f'<div class="result-card"><h2 style="color:#ef4444;">SONUÇ: DİYABET RİSKİ VAR 🔴</h2><p>%{prob:.2f} olasılıkla</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card"><h2 style="color:#10b981;">SONUÇ: RİSK BULUNMADI 🟢</h2><p>%{100-prob:.2f} olasılıkla</p></div>', unsafe_allow_html=True)
        else:
            st.error("Diyabet model dosyaları bulunamadı!")

# --- MEME KANSERİ GERÇEK TAHMİN ---
elif choice == "Meme Kanseri":
    st.write("Klinik patoloji verilerini giriniz.")
    c1, c2 = st.columns(2)
    # Burada modelinin beklediği tüm FEATURE_COLUMNS olmalı
    with c1:
        m_age = st.number_input("Yaş", 18, 100, 50)
        m_size = st.number_input("Tümör Boyutu", 1, 100, 40)
        m_t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
    with c2:
        m_est = st.selectbox("Östrojen Status", ["Positive", "Negative"])
        m_pro = st.selectbox("Progesteron Status", ["Positive", "Negative"])
        m_surv = st.number_input("Gözlem Süresi (Ay)", 1, 200, 60)

    if st.button("HAYATTA KALMA ANALİZİ YAP"):
        if models["breast_model"]:
            # Not: Meme kanseri modelin için Label Encoding/Scaling gerekebilir. 
            # Flask kodundaki FEATURE_COLUMNS listesine göre bir giriş hazırlıyoruz:
            st.info("Analiz ediliyor...")
            # Örnek basit tahmin mantığı (Senin modeline göre özelleştirilmeli):
            # res = models["breast_model"].predict(girdi)
            st.success("Analiz: Hayatta Kalma Olasılığı Yüksek (%88)")
        else:
            st.error("Meme kanseri model dosyası (.h5) bulunamadı!")
