import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import joblib
from PIL import Image, ImageEnhance, ImageFilter

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
    # Modeller (.h5 ve .keras) - Kalp modeli çıkarıldı
    m_files = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "breast": "breast_cancer_model.h5",
        "obesity": "obesity_model.h5"
    }

    for k, v in m_files.items():
        p = get_p(v)
        if p: assets[k] = tf.keras.models.load_model(p, compile=False)
    # Scaler ve Encoderlar - Kalp scaler çıkarıldı

    try:
        assets["diab_model"] = joblib.load(get_p("diabetes_ann_model_v2.pkl"))
        assets["diab_pre"] = joblib.load(get_p("diabetes_preprocessor_v2.pkl"))
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
    else: # Kemik
        o1 = cv2.equalizeHist(gray)
        o2 = cv2.Canny(gray, 50, 150)
        o3 = cv2.morphologyEx(o1, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))
        return o1, o2, o3
# --- 4. ANA PANEL VE SEÇİMLER ---

# Kalp Sağlığı menüden çıkarıldı

choice = st.sidebar.selectbox("Teşhis Protokolü", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Meme Kanseri", "Obezite"])
st.title(f"🏥 {choice} Analiz İstasyonu")
# --- GÖRSEL TABANLI (GÖĞÜS, BEYİN, KEMİK) ---
if choice in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    up = st.file_uploader("Görüntü Dosyasını Yükleyin", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns([1, 1])
        with c1: st.image(img, caption="Orijinal Görüntü", use_container_width=True)
        with c2:
            if st.button(f"{choice.upper()} ANALİZİNİ BAŞLAT"):
                m_key = "chest" if "Göğüs" in choice else "brain" if "Beyin" in choice else "fracture"
                model = assets.get(m_key)
                if model:
                    size = (224, 224) if m_key == "brain" else (150, 150)
                    prep = np.array(img.convert('RGB').resize(size)) / 255.0
                    preds = model.predict(np.expand_dims(prep, axis=0), verbose=0)
                    if m_key == "brain":

                        classes = ["Glioma (Tümör) 🔴", "Meningioma (Tümör) 🔴", "Normal (Tümör Yok) 🟢", "Pituitary (Tümör) 🔴"]
                        idx = np.argmax(preds[0])
                        res = f"TEŞHİS: {classes[idx]}"
                        color = "#ef4444" if idx != 2 else "#10b981"
                    elif m_key == "fracture":
                        score = preds[0][0]
                        non_fractured_probability = score * 100 
                        fractured_probability = 100 - non_fractured_probability
                        if fractured_probability >= 50:
                            res = f"KIRIK TESPİT EDİLDİ 🔴 (Olasılık: %{fractured_probability:.1f})"
                            color = "#ef4444"
                        else:
                            res = f"DURUM NORMAL 🟢 (Sağlamlık: %{non_fractured_probability:.1f})"
                            color = "#10b981"
                    else: 
                        score = preds[0][0]
                        res = "RİSK TESPİT EDİLDİ 🔴" if score > 0.4 else "DURUM NORMAL 🟢"
                        color = "#ef4444" if score > 0.5 else "#10b981"
                    st.markdown(f'<div class="result-card" style="border-color:{color}"><h2>{res}</h2></div>', unsafe_allow_html=True)
                    st.divider()
                    v1, v2, v3 = apply_filters(img, choice.split()[0])
                    vcols = st.columns(3)
                    vcols[0].image(v1, caption="Filtre 1", use_container_width=True)
                    vcols[1].image(v2, caption="Filtre 2", use_container_width=True)
                    vcols[2].image(v3, caption="Filtre 3", use_container_width=True)
                else: st.error("Model dosyası bulunamadı!")

# --- DİYABET ---

elif choice == "Diyabet":
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        age = st.number_input("Yaş", 0, 120, 50)
        hyp = st.selectbox("Hipertansiyon (0/1)", [0, 1])
    with c2:
        heart = st.selectbox("Kalp Hastalığı (0/1)", [0, 1])
        smoke = st.selectbox("Sigara Geçmişi", ["never", "current", "former", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
    hba = st.number_input("HbA1c Seviyesi", 3.0, 15.0, 5.5)
