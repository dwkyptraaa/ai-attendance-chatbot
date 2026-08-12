# Struktur Dataset Wajah

Folder ini berisi **struktur contoh** dataset wajah. Foto asli tidak disertakan
di repository publik untuk menjaga **privasi subjek** (pegawai & mahasiswa).

## Struktur

```
dataset_wajah/
├── Mahasiswa/
│   └── <Nama Lengkap>/
│       ├── orig.jpg            # foto asli wajah
│       ├── flip_horizontal.jpg # 12 variasi augmentasi
│       ├── bright_plus.jpg
│       ├── bright_minus.jpg
│       ├── rot_plus_7.jpg
│       ├── rot_minus_7.jpg
│       ├── noise_heavy.jpg
│       ├── blur_5.jpg
│       ├── flip_rot_5.jpg
│       ├── flip_bright.jpg
│       ├── zoom_in.jpg
│       └── gamma_correction.jpg
└── Pegawai/
    └── <Nama Lengkap>/
        └── (variasi yang sama)
```

## Cara membuat dataset

1. Siapkan 1 foto wajah asli per orang (resolusi wajah ≥ 80px).
2. Jalankan augmentasi untuk membuat 12 variasi unik:

   ```bash
   python augmented.py "dataset_wajah/Pegawai/Nama Pegawai/orig.jpg" "dataset_wajah/Pegawai/Nama Pegawai"
   ```

3. Ulangi untuk setiap orang, lalu latih model:

   ```bash
   python trainedFace.py
   ```

## Catatan privasi

- Simpan dataset asli di luar repository (misal: Google Drive pribadi / repo
  private terpisah).
- Jika ingin menampilkan contoh di publik, samarkan wajahnya (blur/mosaic).
