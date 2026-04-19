import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import tensorflow as tf
from tensorflow import keras
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

# 3. LOAD MODEL (Kompatibel dengan Keras 3.3.3)
@st.cache_resource
def load_model_ai():
    model_path = 'model_training/sentiment_model_lstm.h5'
    tokenizer_path = 'model_training/tokenizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error("❌ File model atau tokenizer tidak ditemukan di folder model_training!")
        st.stop()

    try:
        # Mendefinisikan ulang arsitektur model untuk menghindari bug "quantization_config" pada Keras 3
        model = keras.models.Sequential([
            keras.layers.Input(shape=(100,)),
            keras.layers.Embedding(5000, 128),
            keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        # Hanya memuat bobot (weights) saja dari file model
        model.load_weights(model_path)
        
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        st.success("✅ Model berhasil dimuat!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        st.info("💡 Solusi:")
        st.write("1. Pastikan TensorFlow==2.16.1 dan Keras==3.3.3 sudah terinstall")
        st.write("2. Jalankan: `pip install --upgrade tensorflow==2.16.1 keras==3.3.3`")
        st.write("3. Restart aplikasi")
        st.stop()

model, tokenizer = load_model_ai()

# 4. FUNGSI PREDIKSI
def predict_sentiment(text, model, tokenizer):
    """
    Prediksi sentimen dari teks input
    """
    try:
        # Preprocessing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenisasi dan padding
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=100)
        
        # Prediksi
        prediction = model.predict(padded, verbose=0)
        
        labels = ['Negatif', 'Netral', 'Positif']
        result = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        return result, confidence, prediction[0]
    
    except Exception as e:
        st.error(f"Error pada prediksi: {e}")
        return None, None, None

# SIDEBAR & HEADER
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.markdown("### 🛠️ System Engine")
    st.success("✅ Model AI: LSTM Active")
    st.markdown("---")
    st.write("**Accuracy:** 92.4%")
    st.progress(92)
    st.markdown("---")
    st.write("**Framework:**")
    st.write(f"- TensorFlow: {tf.__version__}")
    st.write(f"- Keras: {keras.__version__}")

st.markdown('<h1 class="gradient-text">⚖️ Sentimen AI: Kasus Amsal</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem; margin-top:-10px; margin-bottom:2rem;'>Pantau opini publik secara presisi menggunakan model Deep Learning LSTM.</p>", unsafe_allow_html=True)

# METRICS
m1, m2, m3 = st.columns(3)
m1.markdown('<div class="metric-box"><small>TOTAL DATASET</small><br><h2 style="color:#2E3192">9.6K+</h2></div>', unsafe_allow_html=True)
m2.markdown('<div class="metric-box"><small>MODEL ACCURACY</small><br><h2 style="color:#764ba2">92.4%</h2></div>', unsafe_allow_html=True)
m3.markdown('<div class="metric-box"><small>PROCESSING TIME</small><br><h2 style="color:#1BFFFF">Real-time</h2></div>', unsafe_allow_html=True)

# MAIN UI
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("🔍 Uji Sentimen Real-Time")
st.markdown("<p>Analisis emosi dan polaritas teks dalam hitungan detik.</p>", unsafe_allow_html=True)

if "text" not in st.session_state: 
    st.session_state.text = ""
if "history" not in st.session_state: 
    st.session_state.history = []

st.markdown("**COBA CEPAT:**")
q1, q2, q3, _ = st.columns([1,1,1,2])

if q1.button("🌟 POSITIF"):
    st.session_state.text = "Keputusan yang sangat adil dan transparan!"
if q2.button("⚠️ NEGATIF"):
    st.session_state.text = "Hukum tajam ke bawah, ini tidak adil bagi rakyat."
if q3.button("⚖️ NETRAL"):
    st.session_state.text = "Jaksa penuntut umum menghadirkan bukti baru dalam persidangan."

user_input = st.text_area("Ketik komentar di bawah ini:", value=st.session_state.text, placeholder="Contoh: Saya rasa keputusan jaksa sudah tepat...", height=150)

if st.button("JALANKAN ANALISIS SARAF"):
    if user_input.strip():
        with st.spinner('⏳ Menganalisis sentimen...'):
            label, score, probs = predict_sentiment(user_input, model, tokenizer)
            
            if label and score:
                st.session_state.history.append((user_input, label))
                
                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    if label == 'Positif':
                        color = "#e8f5e9"
                        text_color = "#2e7d32"
                    elif label == 'Negatif':
                        color = "#ffebee"
                        text_color = "#c62828"
                    else:  # Netral
                        color = "#e3f2fd"
                        text_color = "#1565c0"
                    
                    st.markdown(
                        f'<div class="sentiment-label" style="background:{color}; color:{text_color};">{label}</div>',
                        unsafe_allow_html=True
                    )
                
                with res_col2:
                    st.write(f"### Keyakinan: **{score:.2f}%**")
                    st.progress(int(score) / 100)
    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu.")

st.markdown('</div>', unsafe_allow_html=True)

# VISUALISASI & HISTORY
st.markdown("### 📈 Eksplorasi & Riwayat")
t1, t2 = st.tabs(["📊 Visualisasi", "🕒 Riwayat"])

with t1:
    if os.path.exists('output_visual/infografis_1x1.png'):
        st.image('output_visual/infografis_1x1.png', use_container_width=True)
    else:
        st.info("📌 Visualisasi tidak ditemukan di folder output_visual/")

with t2:
    if st.session_state.history:
        st.write(f"**Total analisis: {len(st.session_state.history)}**")
        st.markdown("---")
        for i, h in enumerate(reversed(st.session_state.history[-5:]), 1):
            sentiment_emoji = "😊" if h[1] == "Positif" else "😟" if h[1] == "Negatif" else "😐"
            st.text(f"{sentiment_emoji} [{h[1]}] {h[0][:80]}...")
    else:
        st.info("🕐 Belum ada riwayat analisis.")