import pandas as pd
import os

# 1. Load Data Bersih
file_input = 'data_collection/komentar_amsal_clean.csv'
file_output = 'data_collection/komentar_amsal_labeled.csv'

if not os.path.exists(file_input):
    print("File clean tidak ditemukan!")
    exit()

df = pd.read_csv(file_input)

# 2. Daftar Kata Kunci (Lexicon Sederhana)
# Kamu bisa menambah kata kunci di sini sesuai konteks kasus Amsal Sitepu
kata_positif = [
    'bebas', 'adil', 'semangat', 'dukung', 'benar', 'jujur', 'alhamdulillah', 
    'puji', 'tuhan', 'keadilan', 'hebat', 'berani', 'setuju', 'hidup', 'amsal'
]

kata_negatif = [
    'korupsi', 'salah', 'penjara', 'hukum', 'jahat', 'bohong', 'dzalim', 
    'sogok', 'uang', 'rakyat', 'tangkap', 'maling', 'rugi', 'kecewa', 'parah'
]

def tentukan_sentimen(teks):
    teks = str(teks).lower()
    skor = 0
    
    for kata in kata_positif:
        if kata in teks:
            skor += 1
    for kata in kata_negatif:
        if kata in teks:
            skor -= 1
    
    # Beri label berdasarkan skor
    if skor > 0:
        return 'Positif'
    elif skor < 0:
        return 'Negatif'
    else:
        return 'Netral'

print("Sedang melakukan labeling otomatis...")
df['sentimen'] = df['komentar_clean'].apply(tentukan_sentimen)

# 3. Simpan Hasil
# Kita fokuskan pada Positif dan Negatif saja untuk LSTM (hapus Netral jika perlu)
# df = df[df['sentimen'] != 'Netral'] 

df.to_csv(file_output, index=False, encoding='utf-8')

print("\n" + "="*35)
print("LABELING SELESAI!")
print(f"Total Data: {len(df)}")
print(df['sentimen'].value_counts())
print(f"File disimpan di: {file_output}")
print("="*35)