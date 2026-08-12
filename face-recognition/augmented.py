import cv2
import numpy as np
import os
import sys

# =========================
# KONFIGURASI
# =========================
# Pemakaian:
#   python augmented.py <path_foto_asli> <folder_output>
# Contoh:
#   python augmented.py "dataset_wajah/Pegawai/Nama Pegawai/orig.jpg" "dataset_wajah/Pegawai/Nama Pegawai"
# Jika argumen tidak diberikan, gunakan nilai default berikut.
INPUT_IMAGE = sys.argv[1] if len(sys.argv) > 1 else "dataset_wajah/Pegawai/Contoh/orig.jpg"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "augmented_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_IMAGES = 12  # Target total
count = 0

img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise ValueError("Gambar tidak ditemukan. Pastikan path benar!")

h, w = img.shape[:2]

# =========================
# HELPER FUNCTIONS
# =========================
def save(img_to_save, name):
    global count
    if count < MAX_IMAGES:
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), img_to_save)
        count += 1
        print(f"[{count}] Saved: {name}")

def rotate(img_src, angle):
    # Menggunakan BORDER_REFLECT agar pinggiran hitam rotasi tidak mengganggu deteksi wajah
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_src, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# =========================
# PROSES AUGMENTASI
# =========================

# 1. Gambar Asli
save(img, "orig.jpg")

# 2. Horizontal Flip (Sangat Penting agar lolos Similarity Check)
flip_img = cv2.flip(img, 1)
save(flip_img, "flip_horizontal.jpg")

# 3. Brightness & Contrast (Dari gambar asli)
save(cv2.convertScaleAbs(img, alpha=1.2, beta=20), "bright_plus.jpg")
save(cv2.convertScaleAbs(img, alpha=0.8, beta=-20), "bright_minus.jpg")

# 4. Rotasi Kecil (Mengubah koordinat landmark wajah)
save(rotate(img, 7), "rot_plus_7.jpg")
save(rotate(img, -7), "rot_minus_7.jpg")

# 5. Noise & Blur (Simulasi kualitas kamera rendah)
noise = np.random.normal(0, 10, img.shape).astype(np.int16)
noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
save(noisy_img, "noise_heavy.jpg")
save(cv2.GaussianBlur(img, (5, 5), 0), "blur_5.jpg")

# 6. Kombinasi Paksa (Agar benar-benar unik)
# Flip + Rotasi (Ini akan menghasilkan encoding yang sangat berbeda dari gambar asli)
save(rotate(flip_img, 5), "flip_rot_5.jpg")

# Flip + Brightness
save(cv2.convertScaleAbs(flip_img, alpha=1.1, beta=10), "flip_bright.jpg")

# Zoom In (Crop 10% lalu resize kembali)
startY, startX = int(h*0.1), int(w*0.1)
endY, endX = int(h*0.9), int(w*0.9)
cropped = img[startY:endY, startX:endX]
save(cv2.resize(cropped, (w, h)), "zoom_in.jpg")

# Gamma Correction (Simulasi cahaya ruangan lampu neon)
invGamma = 1.0 / 1.5
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
save(cv2.LUT(img, table), "gamma_correction.jpg")

print("-" * 30)
print(f"✅ Selesai! {count} variasi unik siap untuk training.")