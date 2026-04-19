import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================================
# 1. SETUP FOLDER
# ==========================================================
for folder in ['model_training', 'output_visual']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ==========================================================
# 2. LOAD DATA & PENYEIMBANGAN
# ==========================================================
print("Memuat data...")
df = pd.read_csv('data_collection/komentar_amsal_labeled.csv')

df_netral = df[df['sentimen'] == 'Netral']
df_negatif = df[df['sentimen'] == 'Negatif']
df_positif = df[df['sentimen'] == 'Positif']

# Penyeimbangan data (Downsampling) agar model tidak bias
n_samples = min(len(df_netral), len(df_negatif))
df_netral_down = resample(df_netral, replace=False, n_samples=n_samples, random_state=42)
df_negatif_down = resample(df_negatif, replace=False, n_samples=n_samples, random_state=42)

df_balanced = pd.concat([df_netral_down, df_negatif_down, df_positif])
df_balanced['komentar_clean'] = df_balanced['komentar_clean'].fillna('')

print("\n" + "="*35)
print("Distribusi data seimbang:")
print(df_balanced['sentimen'].value_counts())
print("="*35 + "\n")

# ==========================================================
# 3. TOKENIZATION & PREPARASI
# ==========================================================
X = df_balanced['komentar_clean'].astype(str).values
y_dummies = pd.get_dummies(df_balanced['sentimen'])
label_names = y_dummies.columns.tolist() 
y = y_dummies.values 

max_words, max_len = 5000, 100 
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(X)
X_pad = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# ==========================================================
# 4. MODEL & TRAINING
# ==========================================================
model = Sequential([
    Embedding(max_words, 128),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Memulai Training Model LSTM...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Simpan Model & Tokenizer
model.save('model_training/sentiment_model_lstm.h5')
with open('model_training/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ==========================================================
# 5. EVALUASI MODEL
# ==========================================================
print("\n" + "="*35)
print("HASIL EVALUASI MODEL LENGKAP")
print("="*35)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=label_names))
cm = confusion_matrix(y_true_classes, y_pred_classes)

# ==========================================================
# 6. GENERATE SEMUA OUTPUT VISUAL (LAYOUT 4 BARIS)
# ==========================================================
def generate_all_visuals():
    print("\nMenghasilkan output visual yang rapi...")
    
    # Persiapan WordCloud
    def get_text(s): return " ".join(df_balanced[df_balanced['sentimen'] == s].komentar_clean.astype(str))
    
    wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(get_text('Positif'))
    wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(get_text('Negatif'))
    wc_neu = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(get_text('Netral'))

    # Ukuran Fig diperbesar (Tinggi 22) agar tidak tumpuk
    fig = plt.figure(figsize=(18, 22))
    plt.suptitle('LAPORAN KOMPREHENSIF ANALISIS SENTIMEN\nKASUS AMSAL SITEPU - DEEP LEARNING LSTM', 
                 fontsize=28, fontweight='bold', y=0.97)

    # --- BARIS 1: METRIK TEKNIS ---
    # Akurasi (Atas Kiri)
    ax1 = plt.subplot2grid((4, 2), (0, 0))
    ax1.plot(history.history['accuracy'], label='Train Acc', linewidth=3, marker='o')
    ax1.plot(history.history['val_accuracy'], label='Val Acc', linewidth=3, marker='s')
    ax1.set_title('Progress Akurasi Model', fontsize=18, pad=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confusion Matrix (Atas Kanan)
    ax2 = plt.subplot2grid((4, 2), (0, 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=label_names, yticklabels=label_names, ax=ax2, cbar=False)
    ax2.set_title('Confusion Matrix (Evaluasi Prediksi)', fontsize=18, pad=15)

    # --- BARIS 2: DISTRIBUSI SENTIMEN (Horizontal Bar) ---
    ax3 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
    counts = df_balanced['sentimen'].value_counts()
    colors = ['gray', 'red', 'green'] # Warna bar
    counts.plot(kind='barh', color=colors, ax=ax3)
    ax3.set_title('Distribusi Sentimen Data Latih (Seimbang)', fontsize=18, pad=15)
    # Tambah angka di ujung bar
    for i, v in enumerate(counts):
        ax3.text(v + 10, i, str(v), color='black', fontweight='bold', va='center')

    # --- BARIS 3: WORDCLOUDS UTAMA ---
    # Positif
    ax4 = plt.subplot2grid((4, 2), (2, 0))
    ax4.imshow(wc_pos)
    ax4.set_title('Topik Sentimen: POSITIF', color='green', fontsize=16, pad=10)
    ax4.axis('off')

    # Negatif
    ax5 = plt.subplot2grid((4, 2), (2, 1))
    ax5.imshow(wc_neg)
    ax5.set_title('Topik Sentimen: NEGATIF', color='red', fontsize=16, pad=10)
    ax5.axis('off')

    # --- BARIS 4: WORDCLOUD NETRAL (FULL WIDTH) ---
    ax6 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
    ax6.imshow(wc_neu)
    ax6.set_title('Topik Sentimen: NETRAL', color='blue', fontsize=16, pad=10)
    ax6.axis('off')

    # Pengaturan Spasi agar teks tidak bertubrukan
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    # Simpan Gambar Final
    plt.savefig('output_visual/infografis_1x1.png', dpi=300)
    
    # Simpan CM Terpisah
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix Model LSTM')
    plt.savefig('output_visual/confusion_matrix.png')
    plt.close()

    print("✅ File 'infografis_1x1.png' dan 'confusion_matrix.png' telah diperbarui di 'output_visual'.")
    plt.show()

generate_all_visuals()

print("\n" + "="*35)
print("PROSES TRAINING & EVALUASI SELESAI!")
print("="*35)