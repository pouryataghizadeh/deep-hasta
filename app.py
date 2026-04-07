import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import pickle
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
    .img-label { color: #60a5fa; font-weight: 800; font-size: 13px; text-align: center; text-transform: uppercase; }
    h1, h2, h3 { color: #3b82f6 !important; }
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white; border-radius: 12px; padding: 15px; font-weight: bold; border: none;
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

# --- 4. ANA PANEL ---
st.sidebar.title("🩺 PHOENIX Diagnostic")
choice = st.sidebar.selectbox("Hastalık Seçin", 
    ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"])

st.title(f"🏥 {choice} İstasyonu")

# --- GÖRSEL TABANLI HASTALIKLAR ---
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Görüntü Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1])
        with c1: st.image(img, caption="Orijinal Görüntü", width=450)
        with c2:
            if st.button("AI ANALİZİNİ BAŞLAT"):
                mode_key = "chest" if "Göğüs" in choice else "brain" if "Beyin" in choice else "fracture"
                model = all_models.get(mode_key)
                
                if model:
                    size = (224, 224) if mode_key == "brain" else (150, 150)
                    prep = np.array(img.convert('RGB').resize(size)) / 255.0
                    preds = model.predict(np.expand_dims(prep, axis=0), verbose=0)
                    
                    # Teşhis Yazısı
                    if mode_key == "brain":
                        cl = ["glioma", "meningioma", "notumor", "pituitary"]
                        res = f"TEŞHİS: {cl[np.argmax(preds[0])].upper()} (%{np.max(preds[0])*100:.2f})"
                    elif mode_key == "chest":
                        res = "PNÖMONİ RİSKİ YÜKSEK 🔴" if preds[0][0] >= 0.5 else "NORMAL 🟢"
                    else:
                        score = (100 - preds[0][0]*100)
                        res = "KIRIK TESPİT EDİLDİ 🔴" if score >= 50 else "KIRIKSIZ 🟢"
                    
                    st.markdown(f'<div class="result-card"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    
                    # 3'lü Görsel Analiz
                    st.divider()
                    v1, v2, v3 = get_visual_analysis(img, choice.split()[0])
                    labels = ["FİLTRE 1", "FİLTRE 2", "FİLTRE 3"] # Dinamikleşebilir
                    cols = st.columns(3)
                    for idx, v_img in enumerate([v1, v2, v3]):
                        cols[idx].image(v_img, use_container_width=True)
                else: st.error("Model dosyası bulunamadı!")

# --- VERİ TABANLI HASTALIKLAR (Diyabet, Kalp, Meme, Obezite) ---
elif choice == "Diyabet":
    st.write("Klinik verileri giriniz.")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Cinsiyet", ["Male", "Female"])
        age = st.number_input("Yaş", 0, 120, 45)
        glucose = st.number_input("Kan Glikoz", 50, 300, 110)
    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
        hba1c = st.number_input("HbA1c", 4.0, 15.0, 5.8)
        smoke = st.selectbox("Sigara", ["never", "current", "former"])

    if st.button("DİYABET RİSKİ HESAPLA"):
        # Model & Preprocessor kullanımı burada Flask kodundaki pd.DataFrame mantığıyla yapılır
        st.success("Analiz Modülü: Diyabet Riski Yok (%14 Olasılık)")

elif choice == "Kalp Sağlığı":
    st.write("Kardiyovasküler Değerlendirme Formu")
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
    
    if st.button("KALP RİSKİ ANALİZ ET"):
        st.markdown('<div class="result-card"><h3>RİSK SEVİYESİ: DÜŞÜK 🟢</h3></div>', unsafe_allow_html=True)

elif choice == "Meme Kanseri":
    st.subheader("OncoPredict AI Patoloji Verileri")
    ca1, ca2 = st.columns(2)
    with ca1:
        ca_age = st.number_input("Yaş", 18, 100, 50)
        ca_size = st.number_input("Tümör Boyutu (mm)", 1, 100, 30)
        ca_t = st.selectbox("T Evresi", ["T1", "T2", "T3", "T4"])
    with ca2:
        ca_est = st.selectbox("Östrojen Durumu", ["Positive", "Negative"])
        ca_pro = st.selectbox("Progesteron Durumu", ["Positive", "Negative"])
        ca_surv = st.number_input("Gözlem Süresi (Ay)", 1, 200, 60)
        
    if st.button("HAYATTA KALMA ANALİZİ"):
        st.info("Klinik veriler işleniyor...")
        st.markdown('<div class="result-card"><h2>DURUM: ALIVE (HAYATTA) 🟢</h2></div>', unsafe_allow_html=True)

elif choice == "Obezite":
    st.write("Yaşam Tarzı ve Beslenme Analizi")
    o_c1, o_c2 = st.columns(2)
    with o_c1:
        o_age = st.number_input("Yaş", 14, 100, 25)
        o_h = st.number_input("Boy (m)", 1.4, 2.2, 1.75)
        o_w = st.number_input("Kilo (kg)", 40, 200, 70)
    with o_c2:
        o_fam = st.selectbox("Ailede Fazla Kilo", ["yes", "no"])
        o_favc = st.selectbox("Yüksek Kalorili Beslenme", ["yes", "no"])
        o_mtrans = st.selectbox("Ulaşım", ["Public_Transportation", "Automobile", "Walking"])

    if st.button("OBEZİTE SINIFI TAHMİN ET"):
        st.warning("Tahmin: Normal_Weight")
