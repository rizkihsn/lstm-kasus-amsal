import time
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# ==========================================================
# 1. DAFTAR URL VIDEO (Sudah diperbarui dengan video terbaru)
# ==========================================================
video_urls = [
    "https://youtu.be/z98kQ-bKFzc?si=x_4weD4ybUoSpMgH", # Denny Sumargo
    "https://youtu.be/h6qV29Chhoc?si=Y3qtmCjeoqPRbIsk", # Ferry Irwandi
    "https://youtu.be/fpWUQxwZ6IA?si=L_lKNZ1b80efmzSV", # Ferry Irwandi
    "https://youtu.be/1_R_z3D7jCo?si=ycMH-T939QSe4J6r", # Kompas TV
    "https://youtu.be/6FCYMZPVawc?si=Xf-0OtwPZBRKUPGo", # Metro TV
    "https://youtu.be/1Fp343iOvTg?si=kfDNgW5Tjaw7VCiS", # Nusantara TV
    "https://www.youtube.com/live/Zpe4lmCaA0Q?si=qJYc3tnhmsAKZsYl", # TVR PARLEMEN
    "https://youtu.be/1mSaAKAP1y8?si=pkT66APLiM9-s5XD", # Tribun Medan TV
    "https://youtu.be/4q2dLDgT2Gs?si=BQXBWc3PkxXS-6qO", # tvOneNews
    "https://youtu.be/_BftSS72eSc?si=IA21gWM81G81FkSH", # CNN Indonesia
    "https://youtu.be/-pnehnLYM3o?si=IA2gFknyqdD0Kins", # Kamar JERI
    "https://youtu.be/GYrl_a6KwkQ?si=DAASgsjJysKn38sx", # VIVA.CO.ID
    "https://youtu.be/_hVZtWHEbas?si=hmCmTFAr2j4l_LF-", # Tribun Sumsel
    "https://youtu.be/Z4mYTowB5es?si=LVdeYM80fm_3_Iyv", # Tribun Bengkulu
    "https://youtu.be/qKI3OjmpEcg?si=_d-JKxBKtNPjDTKE", # Hops Indonesia
    "https://youtu.be/nIwA5wXxLZg?si=Udgeax-UGcdg1YCD", # SINDONews
    "https://youtu.be/Kn2mmEcj4EE?si=f2Ndbk2J_tt51fR2", # SINDONews
    "https://youtu.be/yCSAUrIcjMg?si=AkBCJCHd19RodY2U", # SINDONews
    "https://youtu.be/ny1MccOk1tE?si=gZqeI4yEPfZ7KZmW", # SINDONews
    "https://youtu.be/2-YDEwDgJXk?si=TaqoVVU62uNoS4a_", # SINDONews
    "https://youtu.be/j0IL2mcvapI?si=SG8hPQDDlsnqtWOT", # Tribun Bengkulu
    "https://youtu.be/lt6_w4WoUdk?si=kmirXTpMRldnApm-", # Tribun Bengkulu
    "https://youtu.be/W1YEqUWW2T0?si=4PXLdqeHF9Z3xVrv", # Tribun Bengkulu
    "https://youtu.be/qdR_-fXUGZU?si=PcIJckARbnXDnoH3", # Tribun Bengkulu
    "https://youtu.be/-VeU1JOk1h8?si=izLRjzACiSvVaUUB", # Tribun Bengkulu
    "https://youtu.be/9CHyGMallHY?si=hxeOWNxdoUC2niiX", # Helmy Yahya Bicara
    "https://youtu.be/yyhRLIP9TOM?si=UpygqpxrNfDEyz5l", # Detective Aldo
]
# ==========================================================
# 2. INISIALISASI BROWSER
# ==========================================================
options = webdriver.ChromeOptions()
# options.add_argument("--headless") # Aktifkan jika ingin berjalan di latar belakang
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Folder tempat menyimpan hasil (disesuaikan dengan folder kamu)
folder_name = 'data_collection'
output_filename = os.path.join(folder_name, 'komentar_amsal_gabungan.csv')

# Pastikan folder ada
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

try:
    for url in video_urls:
        print(f"\n--- Membuka Video: {url} ---")
        driver.get(url)
        time.sleep(6)  # Tunggu loading awal lebih lama sedikit

        # Scroll ke bawah untuk memicu elemen komentar
        driver.execute_script("window.scrollTo(0, 600);")
        time.sleep(3)

        # Loop Scrolling
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        scroll_count = 0
        max_scrolls = 300 # Ditingkatkan agar dapat lebih banyak data (target 10.000)

        print("Sedang melakukan scrolling untuk mengambil komentar...")
        while scroll_count < max_scrolls:
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(2) # Jeda aman agar tidak dianggap bot
            
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                # Coba scroll sekali lagi jika berhenti
                driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_UP)
                time.sleep(1)
                driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
                if new_height == last_height: break
            
            last_height = new_height
            scroll_count += 1
            if scroll_count % 20 == 0:
                print(f"Sudah scroll sebanyak {scroll_count} kali...")

        # Ekstrak teks komentar
        print("Mengekstrak teks komentar...")
        comment_elements = driver.find_elements(By.ID, 'content-text')
        
        current_video_data = []
        for el in comment_elements:
            current_video_data.append({
                'platform': 'YouTube',
                'komentar': el.text.replace('\n', ' '), # Menghilangkan enter di dalam komentar
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df_new = pd.DataFrame(current_video_data)
        
        # Simpan bertahap (Mode Append agar data tidak hilang jika error di tengah jalan)
        if not os.path.isfile(output_filename):
            df_new.to_csv(output_filename, index=False, encoding='utf-8')
        else:
            df_new.to_csv(output_filename, mode='a', index=False, header=False, encoding='utf-8')
        
        print(f"Berhasil menambahkan {len(df_new)} komentar dari video ini.")

    # ==========================================================
    # 3. FINALISASI & PEMBERSIHAN DUPLIKAT
    # ==========================================================
    df_final = pd.read_csv(output_filename)
    total_awal = len(df_final)
    df_final = df_final.drop_duplicates(subset=['komentar'])
    df_final.to_csv(output_filename, index=False, encoding='utf-8')
    
    print("\n" + "="*35)
    print(f"PROSES SELESAI!")
    print(f"Total komentar (sebelum filter duplikat): {total_awal}")
    print(f"Total komentar unik terkumpul: {len(df_final)}")
    print(f"File tersimpan di: {output_filename}")
    print("="*35)

except Exception as e:
    print(f"Terjadi kesalahan: {e}")

finally:
    driver.quit()