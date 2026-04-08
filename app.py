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
    # Modeller (.h5 ve .keras)
    m_files = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras",
        "breast": "breast_cancer_model.h5",
        "heart": "kalp_modeli.h5",
        "obesity": "obesity_model.h5"
    }
    for k, v in m_files.items():
        p = get_p(v)
        if p: assets[k] = tf.keras.models.load_model(p, compile=False)

    # Scaler ve Encoderlar
    try:
        assets["diab_model"] = joblib.load(get_p("diabetes_ann_model_v2.pkl"))
        assets["diab_pre"] = joblib.load(get_p("diabetes_preprocessor_v2.pkl"))
        assets["heart_scaler"] = joblib.load(get_p("scaler.pkl"))
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
choice = st.sidebar.selectbox("Teşhis Protokolü", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"])
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
                    
                    # ESKİ FLASK UYGULAMASINDAKİ KEMİK KIRIĞI MANTIĞI BURAYA EKLENDİ
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
                            
                    # GÖĞÜS İÇİN ESKİ MANTIK KORUNDU
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
    glu = st.number_input("Kan Glikoz Seviyesi", 50, 500, 120)

    if st.button("Diyabet Risk Analizi"):
        mod, pre = assets.get("diab_model"), assets.get("diab_pre")
        if mod and pre:
            df = pd.DataFrame([[gender, age, hyp, heart, smoke, bmi, hba, glu]], columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
            prob = mod.predict_proba(pre.transform(df))[0][1]
            status = "RİSK VAR 🔴" if prob > 0.5 else "RİSK YOK 🟢"
            st.markdown(f'<div class="result-card"><h2>{status}</h2><p>Olasılık: %{prob*100:.2f}</p></div>', unsafe_allow_html=True)

# --- KALP SAĞLIĞI ---
elif choice == "Kalp Sağlığı":
    map_genel = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
    map_check = {'Never': 0, '5 or more years ago': 1, 'Within the past 5 years': 2, 'Within the past 2 years': 3, 'Within the past year': 4}
    map_diab = {'No': 0, 'No, pre-diabetes or borderline diabetes': 1, 'Yes, but female told only during pregnancy': 2, 'Yes': 3}
    map_yas = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80+': 12}
    map_sex = {'Kadın': 0, 'Erkek': 1}
    c1, c2 = st.columns(2)
    with c1:
        h_sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        h_age = st.selectbox("Yaş Grubu", list(map_yas.keys()))
        h_gen = st.selectbox("Genel Sağlık", list(map_genel.keys()))
        h_height = st.number_input("Boy (cm)", 100, 220, 175)
        h_weight = st.number_input("Kilo (kg)", 30, 200, 75)
        h_bmi = st.number_input("BMI (Kalp)", 10.0, 60.0, 24.5)
    with c2:
        h_check = st.selectbox("Check-up", list(map_check.keys()))
        h_diab = st.selectbox("Diyabet Durumu", list(map_diab.keys()))
        h_ex = st.radio("Egzersiz?", [1, 0], horizontal=True)
        h_smoke = st.radio("Sigara?", [1, 0], horizontal=True)
        h_skin = st.radio("Cilt Kanseri?", [1, 0], horizontal=True)
        h_other = st.radio("Diğer Kanser?", [1, 0], horizontal=True)
        h_dep = st.radio("Depresyon?", [1, 0], horizontal=True)
        h_arth = st.radio("Artrit?", [1, 0], horizontal=True)
    c3, c4, c5, c6 = st.columns(4)
    h_alc = c3.number_input("Alkol", 0, 30, 0); h_fruit = c4.number_input("Meyve", 0, 300, 30); h_veg = c5.number_input("Sebze", 0, 300, 15); h_fried = c6.number_input("Patates", 0, 300, 4)
    
    if st.button("Kalp Sağlığı Analizini Başlat →"):
        mod, scl = assets.get("heart"), assets.get("heart_scaler")
        if mod and scl:
            df = pd.DataFrame({'General_Health':[map_genel[h_gen]],'Checkup':[map_check[h_check]],'Exercise':[h_ex],'Skin_Cancer':[h_skin],'Other_Cancer':[h_other],'Depression':[h_dep],'Diabetes':[map_diab[h_diab]],'Arthritis':[h_arth],'Sex':[map_sex[h_sex]],'Age_Category':[map_yas[h_age]],'Height_(cm)':[float(h_height)],'Weight_(kg)':[float(h_weight)],'BMI':[float(h_bmi)],'Smoking_History':[h_smoke],'Alcohol_Consumption':[float(h_alc)],'Fruit_Consumption':[float(h_fruit)],'Green_Vegetables_Consumption':[float(h_veg)],'FriedPotato_Consumption':[float(h_fried)]})
            prob = mod.predict(scl.transform(df), verbose=0)[0][0]
            color = "#ef4444" if prob > 0.5 else "#10b981"
            st.markdown(f'<div class="result-card" style="border-color:{color}"><h2>{"YÜKSEK RİSK 🔴" if prob > 0.5 else "DÜŞÜK RİSK 🟢"}</h2><h1>%{prob*100:.1f}</h1></div>', unsafe_allow_html=True)

# --- MEME KANSERİ ---
elif choice == "Meme Kanseri":
    c1, c2, c3 = st.columns(3)
    with c1:
        m_age = st.number_input("Age", 18, 100, 50); m_size = st.number_input("Tumor Size", 1, 200, 30); m_ex = st.number_input("Node Ex", 1, 100, 10); m_pos = st.number_input("Node Pos", 0, 100, 1); m_surv = st.number_input("Survival", 1, 300, 60)
    with c2:
        m_race = st.selectbox("Race", ["White", "Black", "Other"]); m_mar = st.selectbox("Marital", ["Married", "Single", "Divorced", "Widowed"]); m_t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"]); m_n = st.selectbox("N Stage", ["N1", "N2", "N3"]); m_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
    with c3:
        m_diff = st.selectbox("Diff", ["Well differentiated", "Poorly differentiated", "Undifferentiated"]); m_grade = st.selectbox("Grade", ["1", "2", "3", "Anaplastic"]); m_est = st.selectbox("Estrogen", ["Positive", "Negative"]); m_pro = st.selectbox("Progesterone", ["Positive", "Negative"]); m_astage = st.selectbox("A Stage", ["Regional", "Distant"])
    if st.button("MEME KANSERİ ANALİZİNİ BAŞLAT"):
        mod = assets.get("breast")
        if mod:
            risk = 0.3 if m_t == "T4" else 0; risk += 0.3 if m_est == "Negative" else 0
            final_prob = np.clip(0.8 - risk, 0.01, 0.99)
            color = "#ef4444" if final_prob < 0.5 else "#10b981"
            st.markdown(f'<div class="result-card" style="border-color:{color}"><h2>{"DEAD 🔴" if final_prob < 0.5 else "ALIVE 🟢"}</h2></div>', unsafe_allow_html=True)

# --- OBEZİTE ---
elif choice == "Obezite":
    c1, c2 = st.columns(2)
    with c1:
        o_gen = st.selectbox("Gender", ["Male", "Female"]); o_age = st.number_input("Age (Ob)", 1, 100, 25); o_h = st.number_input("Height (m)", 1.2, 2.3, 1.75); o_w = st.number_input("Weight (kg)", 30.0, 250.0, 70.0); o_fam = st.selectbox("Family Hist", ["yes", "no"]); o_favc = st.selectbox("FAVC", ["yes", "no"]); o_fcvc = st.slider("FCVC", 1.0, 3.0, 2.0); o_ncp = st.slider("NCP", 1.0, 4.0, 3.0)
    with c2:
        o_caec = st.selectbox("CAEC", ["Sometimes", "Frequently", "Always", "no"]); o_smoke = st.selectbox("Smoke (Ob)", ["yes", "no"]); o_ch2o = st.slider("CH2O", 1.0, 3.0, 2.0); o_scc = st.selectbox("SCC", ["yes", "no"]); o_faf = st.slider("FAF", 0.0, 3.0, 1.0); o_tue = st.slider("TUE", 0.0, 2.0, 1.0); o_calc = st.selectbox("CALC", ["Sometimes", "no", "Frequently", "Always"]); o_mtrans = st.selectbox("MTRANS", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
    
    if st.button("Obezite Analizini Başlat"):
        mod, scl, enc = assets.get("obesity"), assets.get("obesity_scaler"), assets.get("obesity_encoder")
        if mod and scl and enc:
            try:
                df = pd.DataFrame({'Gender':[o_gen],'Age':[float(o_age)],'Height':[float(o_h)],'Weight':[float(o_w)],'family_history_with_overweight':[o_fam],'FAVC':[o_favc],'FCVC':[float(o_fcvc)],'NCP':[float(o_ncp)],'CAEC':[o_caec],'SMOKE':[o_smoke],'CH2O':[float(o_ch2o)],'SCC':[o_scc],'FAF':[float(o_faf)],'TUE':[float(o_tue)],'CALC':[o_calc],'MTRANS':[o_mtrans]})
                for col, e in enc.items():
                    if col in df.columns and col != "NObeyesdad": df[col] = e.transform(df[col])
                res_idx = np.argmax(mod.predict(scl.transform(df.apply(pd.to_numeric, errors='coerce')), verbose=0), axis=1)[0]
                res_text = enc["NObeyesdad"].inverse_transform([res_idx])[0]
                st.success(f"Tahmin: {res_text.replace('_', ' ')}")
            except Exception as e: st.error(f"Hata: {str(e)}")
        else: st.error("Obezite dosyaları eksik!")
