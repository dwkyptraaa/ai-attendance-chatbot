<div align="center">

# 🤖 AI Attendance Chatbot — Face Recognition + n8n

**Implementasi AI Agent berbasis Chatbot untuk Sistem Kehadiran Pegawai dan Mahasiswa dengan Integrasi Face Recognition dan N8N**

Skripsi — Muh. Dwicky P. Sanjaya (60900122041) · Sistem Informasi · UIN Alauddin Makassar · 2026

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![n8n](https://img.shields.io/badge/n8n-Workflow%20Automation-EA4B71?logo=n8n&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-LLM-8E75B2?logo=google&logoColor=white)
![WhatsApp](https://img.shields.io/badge/WAHA-WhatsApp%20API-25D366?logo=whatsapp&logoColor=white)

</div>

---

## 📌 Ringkasan

Sistem kehadiran otomatis yang menggabungkan **face recognition real-time dari CCTV** dengan **AI agent chatbot WhatsApp**. Wajah yang terdeteksi diidentifikasi secara otomatis, dicatat ke Google Sheets, lalu informasi kehadiran disebarkan ke pengguna melalui chatbot yang digerakkan oleh **n8n** + **Gemini 2.5 Flash** dan **WAHA** (WhatsApp HTTP API).

**Masalah yang dijawab:** mahasiswa sering menunggu lama untuk memastikan kehadiran pegawai/dosen (misal untuk bimbingan skripsi) karena masih bergantung pada respon manual via pesan singkat.

| Metrik | Nilai |
|---|---|
| **Akurasi** | 86,16% |
| **Presisi** | 81,13% |
| **Recall** | 69,35% |
| **F1-Score** | 74,78% |
| **Rata-rata total latency** | 15,9 detik |
| **Respon otomatisasi chatbot** | 9,3 detik |

*(Diuji terhadap 419 sampel: TP=86, FP=20, FN=38, TN=275)*

---

## 🏗️ Arsitektur Sistem

```
┌─────────────┐   RTSP    ┌──────────────────────────────────────────────┐
│ Kamera CCTV │──────────▶│  face-recognition/ (Python)                  │
│  (Tenda)    │           │  fc_recog.py                                 │
└─────────────┘           │  1. MediaPipe  → deteksi wajah               │
                          │  2. InsightFace → embedding wajah (512D)     │
                          │  3. Euclidean distance + threshold 1.09      │
                          │  4. Label: Known / Unknown                   │
                          └───────────────┬──────────────────────────────┘
                                          │ upload foto + metadata
                                          ▼
                          ┌──────────────────────────────────────────────┐
                          │  Google Drive (foto bukti)                   │
                          │  Google Sheets (data kehadiran)              │
                          └───────────────┬──────────────────────────────┘
                                          │ Google Sheets Trigger (polling)
                                          ▼
                          ┌──────────────────────────────────────────────┐
                          │  n8n (Docker) — pusat workflow automation    │
                          │  ├─ Trigger: baris baru di Sheets            │
                          │  ├─ AI Agent + Gemini 2.5 Flash              │
                          │  └─ HTTP Request → WAHA API                  │
                          └───────────────┬──────────────────────────────┘
                                          │ kirim pesan via API
                                          ▼
                          ┌──────────────────────────────────────────────┐
                          │  WAHA (Docker) — WhatsApp HTTP API           │
                          │  (sinkron WhatsApp Web via QR)               │
                          └───────────────┬──────────────────────────────┘
                                          ▼
                          ┌──────────────────────────────────────────────┐
                          │  📱 Pengguna (WhatsApp)                      │
                          │  • Cek kehadiran hari ini                    │
                          │  • Cek bukti foto kehadiran                  │
                          │  • Request notifikasi (save antrian)         │
                          └──────────────────────────────────────────────┘
```

**Alur lengkap:**

1. **Akuisisi** — Kamera CCTV (Tenda RP3-393A) mengalirkan video via **RTSP** dengan transport TCP (stabil di WiFi).
2. **Deteksi** — MediaPipe `face detection` melokalisasi wajah per frame, dilacak antar-frame dengan **centroid tracker** (threading multi-wajah).
3. **Identifikasi** — Crop wajah diekstraksi menjadi embedding 512-dimensi oleh **InsightFace (Buffalo_L)** di CPU, lalu dibandingkan dengan dataset terlatih memakai **Euclidean distance**:
   - `distance < 1.09` → **Known** (identitas terverifikasi)
   - `distance ≥ 1.20` → **Unknown** (wajah tidak terdaftar)
4. **Pencatatan** — Hasil (nama, status, waktu, link foto bukti) otomatis ditulis ke **Google Sheets** via Service Account; foto diunggah ke **Google Drive** (dengan retry & rebuild koneksi otomatis).
5. **Orkestrasi** — **n8n** memonitor Google Sheets (polling trigger), AI Agent memformat data kehadiran menjadi pesan informatif, lalu mengirim via **WAHA** ke WhatsApp pengguna.
6. **Interaksi** — Pengguna bertanya "cek kehadiran", "apakah Pak Rahman sudah hadir?", atau "kabari saya kalau Budi terdeteksi" — semuanya ditangani AI Agent dengan pemetaan nama fuzzy (typo/abbreviation).

---

## 🛠️ Teknologi

| Komponen | Teknologi | Peran |
|---|---|---|
| Face detection | **MediaPipe** | Deteksi & lokalisasi wajah cepat |
| Face embedding | **InsightFace** (Buffalo_L, ArcFace 512D) | Ekstraksi vektor fitur wajah |
| Pencocokan | **Euclidean distance** + threshold | Verifikasi identitas (Known/Unknown) |
| Pengolahan citra | **OpenCV** + **NumPy** | Augmentasi, frame processing, RTSP |
| Pelacakan | **Centroid tracker** (scipy) | Tracking multi-wajah antar frame |
| Cloud storage | **Google Sheets** + **Google Drive** API | Database kehadiran & foto bukti |
| Workflow automation | **n8n** (self-hosted Docker) | Pusat alur/orkestrasi sistem |
| AI Agent / LLM | **Gemini 2.5 Flash** | Logika percakapan & pemformatan pesan |
| WhatsApp gateway | **WAHA** (WhatsApp HTTP API) | Kirim/terima pesan WhatsApp |
| Public webhook | **Ngrok** (static domain) | Menembus localhost ke internet |
| Container | **Docker Desktop** | Menjalankan n8n + WAHA |

---

## 📁 Struktur Repository

```
ai-attendance-chatbot/
├── README.md                  # Dokumentasi ini
├── LICENSE                    # MIT
├── .env.example               # Contoh konfigurasi rahasia
├── requirements.txt           # Dependensi Python
│
├── face-recognition/          # Modul pengenalan wajah (Python)
│   ├── fc_recog.py            # ⭐ Main: deteksi real-time CCTV → Google Sheets
│   ├── trainedFace.py         # Training: dataset → file .pkl embedding
│   ├── augmented.py           # Augmentasi citra (12 variasi per wajah)
│   ├── dataset_wajah/         # Struktur dataset (tanpa foto asli)
│   └── trained_models/        # Hasil training (.pkl) — git-ignored
│
└── chatbot/                   # Sisi chatbot WhatsApp
    ├── docker-compose.yml     # n8n + WAHA (kredensial via .env)
    ├── workflows/             # JSON workflow n8n (import ke n8n)
    │   ├── WAHA workflow 1.json
    │   └── WAHA workflow 2 Google Spreadsheet Trigger.json
    └── prompts/
        └── ai-agent-prompt.txt  # System prompt AI Agent Gemini
```

---

## 🚀 Instalasi & Setup

### 0. Prasyarat

| Kebutuhan | Keterangan |
|---|---|
| Docker Desktop | Untuk menjalankan n8n + WAHA |
| Python 3.10 | Untuk modul face recognition |
| Akun Google | Google Cloud Console (Service Account) + Sheets + Drive |
| Akun Ngrok | Mendapat static domain gratis |
| Nomor WhatsApp aktif | Dihubungkan ke WAHA |
| Kamera CCTV | Mendukung RTSP (mis. Tenda RP3-393A) |

### 1. Google Cloud (Service Account)

1. Buat project di [Google Cloud Console](https://console.cloud.google.com).
2. Aktifkan **Google Sheets API** dan **Google Drive API**.
3. Buat **Service Account** → tab *Keys* → *Add Key* → *JSON* → simpan sebagai `service_account.json`.
4. **Share** Google Sheets (tempat data kehadiran) dan folder Google Drive (tempat foto) ke email service account dengan akses **Editor**.

### 2. Jalankan n8n + WAHA (Docker)

```bash
cd chatbot
cp ../.env.example .env      # isi WAHA_API_KEY, WAHA_DASHBOARD_PASSWORD, dst.
docker compose up -d
```

- **n8n** → `http://localhost:5678` (daftar akun admin saat pertama kali).
- **WAHA** → `http://localhost:3000` → buat *session* → **scan QR** dengan WhatsApp → status `WORKING`.

### 3. Ngrok (akses publik untuk webhook)

```bash
ngrok config add-authtoken <TOKEN_ANDA>
ngrok http --domain=<STATIC_DOMAIN_ANDA> 5678
```

Set `N8N_PUBLIC_URL` di `.env` dengan URL statis ngrok Anda, lalu restart container n8n.

### 4. Siapkan dataset wajah

```bash
# 1) Siapkan 1 foto asli per orang di dataset_wajah/<Kategori>/<Nama>/
# 2) Augmentasi (12 variasi):
python augmented.py "dataset_wajah/Pegawai/Nama Pegawai/orig.jpg" "dataset_wajah/Pegawai/Nama Pegawai"
# 3) Latih model:
python trainedFace.py
```

### 5. Jalankan face recognition

```bash
pip install -r requirements.txt
# isi .env: GOOGLE_SERVICE_ACCOUNT_FILE, SPREADSHEET_ID, DRIVE_FOLDER_ID, RTSP_*
python fc_recog.py
```

> Tekan `q` di jendela video untuk menghentikan program.

### 6. Import workflow n8n

1. Buka n8n → **Workflows** → **Import from File** → pilih JSON di `chatbot/workflows/`.
2. Sesuaikan *credentials* (Google Sheets Trigger, dll.) dengan kredensial Anda.
3. Isi system prompt AI Agent dari `chatbot/prompts/ai-agent-prompt.txt`.
4. Aktifkan workflow. Selesai 🎉

---

## 💬 Fitur Chatbot

| Perintah (contoh) | Respon |
|---|---|
| `cek kehadiran` / `siapa yang sudah hadir hari ini?` | Daftar kehadiran hari ini (Unknown disembunyikan) |
| `apakah Pak Rahman sudah hadir?` | Status kehadiran + jam deteksi |
| `dwiky` (ketik nama) | Bukti foto kehadiran dari Google Drive |
| `kabari saya kalau Budi terdeteksi` | Simpan ke antrian notifikasi (`save_antrian`) — dibalas begitu terdeteksi |
| `ya, beritahu saya` | Aktivasi notifikasi dari konteks percakapan sebelumnya |
| Pesan informal `cek dwiky`, `dwiky ada?` | Tetap dipahami via pemetaan nama fuzzy |
| Pesan ambigu `zxcvbnm` | "Maaf, saya tidak mengerti maksud Anda" |

---

## 🧪 Pengujian & Hasil

### Metrik evaluasi face recognition (419 sampel)

| Metrik | Rumus | Nilai |
|---|---|---|
| Akurasi | (TP+TN)/total | **86,16%** |
| Presisi | TP/(TP+FP) | **81,13%** |
| Recall | TP/(TP+FN) | **69,35%** |
| F1-Score | 2·P·R/(P+R) | **74,78%** |

Nilai TN yang tinggi (275) menunjukkan sistem sangat tangguh menolak wajah tidak terdaftar (Unknown) — menjamin integritas data kehadiran.

### Latency end-to-end

| Tahap | Rata-rata |
|---|---|
| CCTV terdeteksi → tersimpan di Sheets | ~9 detik |
| Bot → pesan sampai ke pengguna | ~6,9 detik |
| **Total** | **15,9 detik** |

### Kendala yang ditemukan

- Wajah menunduk → gagal deteksi (fitur wajah tidak utuh) → berkontribusi pada FN.
- Koneksi internet tidak stabil → lag + motion blur → deteksi terlewat.
- Variasi pencahayaan area CCTV mempengaruhi konsistensi.

---

## ⚠️ Keamanan & Privasi

- ❌ **Jangan commit**: `service_account.json`, `.env`, `Credentials.txt`, API key, password RTSP/WAHA.
- Semua kredensial di repo ini sudah dipindahkan ke environment variables (`.env.example`).
- Foto wajah asli **tidak** disertakan di repo publik — baca `face-recognition/dataset_wajah/README.md`.
- Model `.pkl` berisi data biometrik → simpan di penyimpanan pribadi / repo private.

---

## 📚 Referensi & Dokumentasi

- Skripsi lengkap (145 hal.) menyertai project ini: **BUNDEL_DWIKY.pdf** — berisi arsitektur, diagram UML (use case, activity, sequence), konfigurasi Docker/Ngrok/WAHA, hasil pengujian, dan lampiran.
- [n8n Documentation](https://docs.n8n.io) · [WAHA (WhatsApp HTTP API)](https://waha.devlike.pro) · [InsightFace](https://github.com/deepinsight/insightface) · [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) · [Ngrok](https://ngrok.com)

---

## 📄 Lisensi

[MIT](LICENSE) © 2026 Muh. Dwicky P. Sanjaya
