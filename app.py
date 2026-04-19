import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Analisis Sentimen Publik: Kasus Amsal Sitepu",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #fcfcfd; }
    .gradient-text { color: #2E3192; font-weight: 800; font-size: 3rem; text-align: center; }
    .main-card { background: white; padding: 2rem; border-radius: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); border: 1px solid #f0f2f6; margin-bottom: 1.5rem; }
    .metric-box { text-align: center; padding: 1.5rem; background: #ffffff; border-radius: 20px; border: 1px solid #f0f2f6; }
    div.stButton > button { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white; border-radius: 15px; font-weight: 700; width: 100%; }
    .sentiment-label { font-size: 1.5rem; font-weight: 800; padding: 10px 20px; border-radius: 12px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL (Optimal untuk Keras 3)
@st.cache_resource
def load_model_ai():
    model_path = 'model_training/sentiment_model_lstm.h5'
    tokenizer_path = 'model_training/tokenizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error("File model atau tokenizer tidak ditemukan di folder model_training!")
        st.stop()

    try:
        # Load menggunakan Keras 3 langsung
        model = keras.models.load_model(model_path, compile=False)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.info("Pastikan requirements.txt menggunakan tensorflow==2.16.1 dan keras==3.3.3")
        st.stop()

model, tokenizer = load_model_ai()

# 4. FUNGSI PREDIKSI
def predict_sentiment(text, model, tokenizer):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded, verbose=0)
    labels = ['Negatif', 'Netral', 'Positif']
    result = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return result, confidence, prediction[0]

# SIDEBAR & HEADER
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.markdown("### 🛠️ System Engine")
    st.success("Model AI: LSTM Active")
    st.markdown("---")
    st.write("**Accuracy:** 92.4%")
    st.progress(92)

st.markdown('<h1 class="gradient-text">⚖️ Analisis Sentimen Publik: Kasus Amsal Sitepu</h1>', unsafe_allow_html=True)

# METRICS
m1, m2, m3 = st.columns(3)
m1.markdown('<div class="metric-box"><small>TOTAL DATASET</small><br><h2 style="color:#2E3192">9.6K+</h2></div>', unsafe_allow_html=True)
m2.markdown('<div class="metric-box"><small>MODEL ACCURACY</small><br><h2 style="color:#764ba2">92.4%</h2></div>', unsafe_allow_html=True)
m3.markdown('<div class="metric-box"><small>PROCESSING TIME</small><br><h2 style="color:#1BFFFF">Real-time</h2></div>', unsafe_allow_html=True)

# MAIN UI
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("🔍 Uji Sentimen Real-Time")

if "text" not in st.session_state: st.session_state.text = ""
if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("Ketik komentar di bawah ini:", value=st.session_state.text, height=150)

if st.button("JALANKAN ANALISIS SARAF"):
    if user_input:
        with st.spinner('Menganalisis...'):
            label, score, probs = predict_sentiment(user_input, model, tokenizer)
            st.session_state.history.append((user_input, label))
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([1,2])
            with res_col1:
                color = "#e8f5e9" if label == 'Positif' else "#ffebee" if label == 'Negatif' else "#e3f2fd"
                text_color = "#2e7d32" if label == 'Positif' else "#c62828" if label == 'Negatif' else "#1565c0"
                st.markdown(f'<div class="sentiment-label" style="background:{color}; color:{text_color};">{label}</div>', unsafe_allow_html=True)
            with res_col2:
                st.write(f"### Keyakinan: **{score:.2f}%**")
                st.progress(int(score))
    else:
        st.warning("Masukkan teks terlebih dahulu.")
st.markdown('</div>', unsafe_allow_html=True)

# VISUALISASI & HISTORY
st.markdown("### 📈 Eksplorasi & Riwayat")
t1, t2 = st.tabs(["📊 Visualisasi", "🕒 Riwayat"])
with t1:
    if os.path.exists('output_visual/infografis_1x1.png'):
        st.image('output_visual/infografis_1x1.png', use_container_width=True)
    else:
        st.info("Visualisasi tidak ditemukan.")
with t2:
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            st.text(f"[{h[1]}] {h[0][:50]}...")
    else:
        st.info("Belum ada riwayat.")