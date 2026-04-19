import pandas as pd
import re
import string
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. Konfigurasi Lokasi File
file_input = 'data_collection/komentar_amsal_gabungan.csv'
file_output = 'data_collection/komentar_amsal_clean.csv'

# Cek apakah file input ada
if not os.path.exists(file_input):
    print(f"Error: File {file_input} tidak ditemukan!")
    exit()

print("Memulai proses preprocessing untuk 9.663 data...")
df = pd.read_csv(file_input)

# Inisialisasi Sastrawi (Stopword dan Stemmer)
stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def clean_text(text):
    # a. Case Folding (Kecilkan semua huruf)
    text = str(text).lower()
    
    # b. Menghapus URL/Link
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # c. Menghapus Mention (@) dan Hashtag (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # d. Menghapus Angka dan Tanda Baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # e. Menghapus spasi berlebih
    text = " ".join(text.split())
    
    # f. Stopword Removal (Menghapus kata tidak penting: 'di', 'yang', 'itu')
    text = stopword_remover.remove(text)
    
    # g. Stemming (Mengubah ke kata dasar: 'dihukum' -> 'hukum')
    # Bagian ini yang paling memakan waktu
    text = stemmer.stem(text)
    
    return text

# 2. Proses Pembersihan
print("Sedang memproses... Harap tunggu, ini mungkin memakan waktu 10-20 menit karena data cukup besar.")

# Gunakan progres bar sederhana di terminal
df['komentar_clean'] = df['komentar'].apply(clean_text)

# 3. Finalisasi
# Hapus baris yang kosong atau hanya spasi setelah dibersihkan
df = df.dropna(subset=['komentar_clean'])
df = df[df['komentar_clean'] != '']

# Simpan hasil ke CSV baru
df.to_csv(file_output, index=False, encoding='utf-8')

print("\n" + "="*35)
print("PREPROCESSING SELESAI!")
print(f"Jumlah data awal: {len(pd.read_csv(file_input))}")
print(f"Jumlah data bersih: {len(df)}")
print(f"File disimpan di: {file_output}")
print("="*35)