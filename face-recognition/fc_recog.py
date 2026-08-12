import cv2
import threading
import mediapipe as mp
from insightface.app import FaceAnalysis
import pickle
import numpy as np
import time
import os
from collections import OrderedDict
from scipy.spatial import distance as dist
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pyautogui
import socket

# Muat file .env jika ada (opsional, butuh paket python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================================
# KONFIGURASI UTAMA
# Semua nilai rahasia (kredensial Google, ID Spreadsheet/Drive,
# kredensial RTSP) diambil dari environment variables / file .env
# agar aman untuk dipublikasikan. Salin .env.example menjadi .env
# lalu isi nilai sesuai milikmu.
# SESSION_TIMEOUT: jeda minimal (detik) sebelum nama yang sama
# boleh disimpan ulang dalam satu sesi (default 20 menit).
# ==========================================================
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")
SHEET_RANGE = os.getenv("SHEET_RANGE", "Sheet1!A:D")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "detected")

SESSION_TIMEOUT = 1200
saved_names_session = {}

pyautogui.FAILSAFE = False

# ==========================================================
# KONFIGURASI RTSP KAMERA
# Isi melalui environment variables / file .env, contoh:
#   RTSP_USERNAME=admin
#   RTSP_PASSWORD=password_kamera
#   RTSP_IP=192.168.1.100
#   RTSP_PATH=/ch=1?subtype=0
# Cara cek: buka app Tenda Security -> pilih kamera -> Settings
# (icon hexagon) -> cari menu "RTSP" / "Advanced Settings" / "LAN".
# Path stream paling umum untuk Tenda adalah /stream1 (substream
# kualitas lebih rendah, lebih ringan) atau /stream2 (HD/main stream).
# Kalau /stream1 gagal connect, coba /av0_0 atau /av0_1 sebagai alternatif.
# ==========================================================
RTSP_USERNAME = os.getenv("RTSP_USERNAME", "admin")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", "")
RTSP_IP       = os.getenv("RTSP_IP", "")
RTSP_PORT     = int(os.getenv("RTSP_PORT", "554"))
RTSP_PATH     = os.getenv("RTSP_PATH", "/ch=1?subtype=0")

RTSP_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}{RTSP_PATH}"

# Paksa OpenCV/FFMPEG pakai TCP untuk transport RTSP, bukan UDP.
# TCP lebih tahan packet loss di WiFi yang kurang stabil (UDP akan
# drop paket diam-diam tanpa retry, ini salah satu penyebab umum
# "freeze" yang terlihat acak/random pada implementasi RTSP naif).
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt|"
    "flags;low_delay|"
    "loglevel;error"          # <-- menekan warning POC agar tidak muncul di terminal
)

# ==========================================================
# PARAMETER OPTIMASI PERFORMA & THREADING
# FRAME_SKIP     : proses hanya setiap N frame (1 = semua frame)
# RESIZE_WIDTH/HEIGHT : resolusi kerja setelah capture dari RTSP
# FACE_DETECT_CONFIDENCE : ambang batas minimum deteksi wajah MediaPipe
# ACCEPT_THRESHOLD : jarak embedding ≤ nilai ini → dianggap dikenal
# REJECT_THRESHOLD : jarak embedding ≥ nilai ini → dianggap Unknown
# MAX_FACES_TO_PROCESS : maksimal wajah yang diproses per frame
# MAX_THREADS    : jumlah maksimal thread pengenalan berjalan bersamaan
# ==========================================================
FRAME_SKIP = 1
RESIZE_WIDTH = 960
RESIZE_HEIGHT = 540
FACE_DETECT_CONFIDENCE = 0.3
ACCEPT_THRESHOLD = 1.09
REJECT_THRESHOLD = 1.20
MAX_FACES_TO_PROCESS = 3
MAX_THREADS = 2

# ==========================================================
# PENYIMPANAN HASIL DARI BACKGROUND THREAD
# recognition_results : dict {objectID: (name, status, distance)}
#                       diisi oleh thread pengenalan, dibaca oleh main loop
# processing_ids      : set ID wajah yang sedang diproses thread
# results_lock        : lock untuk mengamankan akses ke kedua struktur di atas
# save_lock           : lock khusus untuk operasi tulis file gambar
# ==========================================================
recognition_results = {} 
processing_ids = set()
results_lock = threading.Lock() 
save_lock = threading.Lock()

# ==========================================================
# KONFIGURASI PELACAKAN WAJAH UNKNOWN
# unknown_candidate_start : mencatat waktu pertama kali wajah Unknown
#                           terdeteksi stabil per objectID
# UNKNOWN_STABLE_TIME     : durasi (detik) wajah Unknown harus stabil
#                           sebelum fotonya disimpan
# last_unknown_save_time  : timestamp terakhir penyimpanan Unknown,
#                           digunakan untuk global cooldown
# UNKNOWN_SAVE_COOLDOWN   : jeda minimal (detik) antar penyimpanan Unknown
#                           berlaku secara global untuk semua wajah Unknown
# MAX_RESULT_AGE          : umur maksimal (detik) hasil pengenalan di cache
#                           sebelum dihapus jika wajah sudah tidak terlacak
# ==========================================================
unknown_candidate_start = {}
UNKNOWN_STABLE_TIME = 1

last_unknown_save_time = 0
UNKNOWN_SAVE_COOLDOWN = 60  # 1 menit global cooldown untuk semua unknown

MAX_RESULT_AGE = 10
result_timestamps = {}

# ==========================================================
# INISIALISASI MODEL & LAYANAN GOOGLE
# Memuat model InsightFace Buffalo_L untuk ekstraksi embedding wajah.
# Menggunakan CPUExecutionProvider (tidak butuh GPU).
# Sekaligus membangun koneksi ke Google Sheets dan Google Drive
# menggunakan kredensial Service Account.
# ==========================================================
print("🔄 Memuat InsightFace (Buffalo_L)...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(480, 480))
print("✅ InsightFace siap")

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
sheet_service = build("sheets", "v4", credentials=creds)
drive_service = build("drive", "v3", credentials=creds)

def rebuild_services():
    # Membangun ulang koneksi ke Google Sheets dan Drive.
    # Dipanggil otomatis saat terjadi error koneksi SSL/timeout
    # pada proses upload atau penulisan ke spreadsheet.
    global sheet_service, drive_service
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    sheet_service = build("sheets", "v4", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

# ==========================================================
# MEMUAT DATA WAJAH TERLATIH
# Membaca file .pkl hasil training yang berisi:
#   - known_names     : daftar nama per wajah
#   - known_encodings : array embedding wajah (vektor fitur 512-dim)
#   - known_statuses  : status tiap wajah (misal: Mahasiswa, Pegawai)
# Jika file tidak ditemukan, sistem tetap berjalan tanpa data referensi.
# ==========================================================
try:
    with open("trained_faces_insightface480l.pkl", "rb") as f:
        data = pickle.load(f)
    known_names = data["names"]
    known_encodings = np.array(data["encodings"])
    known_statuses = data.get("statuses", ["Unknown"] * len(known_names))
    print(f"✅ Loaded {len(known_names)} wajah terlatih")
except Exception as e:
    print(f"❌ Error loading pkl: {e}")
    known_names, known_encodings, known_statuses = [], [], []

# ==========================================================
# FUNGSI UTILITAS: KONEKSI, DRIVE, SHEETS, DAN PENYIMPANAN CLOUD
# ==========================================================
def is_connected():
    # Mengecek koneksi internet dengan mencoba terhubung ke DNS Google (8.8.8.8).
    # Mengembalikan True jika berhasil, False jika gagal dalam 3 detik.
    try:
        socket.setdefaulttimeout(3)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
        s.close()
        return True
    except:
        return False
    
def upload_to_drive(image_path, filename):
    # Mengunggah file gambar ke folder Google Drive yang ditentukan.
    # Setelah upload, file diberi izin publik (anyone can view).
    # Mengembalikan URL langsung ke file jika berhasil.
    # Melakukan hingga 4 kali percobaan dengan jeda eksponensial (2s, 4s, 8s).
    # Jika percobaan ke-2 dst gagal, service Drive di-rebuild sebelum retry.
    for i in range(4):
        try:
            file_metadata = {'name': filename, 'parents': [DRIVE_FOLDER_ID]}
            media = MediaFileUpload(image_path, mimetype='image/jpeg')
            file = drive_service.files().create(
                body=file_metadata, media_body=media, fields='id', supportsAllDrives=True
            ).execute()
            file_id = file.get("id")
            drive_service.permissions().create(
                fileId=file_id, body={'type': 'anyone', 'role': 'reader'}, supportsAllDrives=True
            ).execute()
            print(f"✅ Upload Drive sukses: {filename}")
            return f"https://drive.google.com/uc?export=view&id={file_id}"

        except Exception as e:
            if i == 3:
                print(f"❌ Upload gagal setelah 4x retry: {filename}")
                return "Upload Failed"
            
            wait = 2 ** (i + 1)  # 2s, 4s, 8s
            print(f"⚠ Upload retry {i+1}/4, tunggu {wait}s: {str(e)[:80]}")
            if i >= 1:
                try:
                    rebuild_services()
                    print("🔄 Drive service di-rebuild")
                except Exception as re:
                    print(f"❌ Rebuild gagal: {re}")
            time.sleep(wait)

    print(f"❌ Upload gagal setelah 4x retry: {filename}")
    return "Upload Failed"

def save_to_sheet(name, status, link):
    # Menambahkan baris baru ke Google Spreadsheet dengan format:
    # [Nama, Status, Waktu (YYYY-MM-DD HH:MM:SS), Link Foto Drive]
    # Melakukan hingga 4 kali retry khusus untuk error jaringan/SSL.
    # Error non-jaringan (misal: permission) langsung dihentikan tanpa retry.
    for attempt in range(4):
        try:
            waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = {"values": [[name, status, waktu, link]]}
            sheet_service.spreadsheets().values().append(
                spreadsheetId=SPREADSHEET_ID, range=SHEET_RANGE,
                valueInputOption="RAW", body=body
            ).execute()
            print(f"📊 Sheet updated: {name}")
            return

        except Exception as e:
            err_msg = str(e).lower()

            if any(x in err_msg for x in ["eof", "ssl", "timeout", "connection", "reset", "broken"]):
                
                # ✅ Jangan sleep di attempt terakhir, langsung keluar
                if attempt == 3:
                    print(f"❌ Sheet gagal setelah 4x retry: {name}")
                    return

                wait = 2 ** (attempt + 1)  # ✅ 2s, 4s, 8s (lebih masuk akal)
                print(f"⚠ Sheet koneksi error (attempt {attempt+1}/4), retry {wait}s...")

                if attempt >= 1:  # ✅ Rebuild lebih awal, dari attempt ke-2
                    try:
                        rebuild_services()
                        print("🔄 Sheet service di-rebuild")
                    except Exception as re:
                        print(f"❌ Rebuild gagal: {re}")

                time.sleep(wait)

            else:
                print(f"❌ Sheet error (non-SSL): {str(e)[:80]}")
                return

def save_detected_face(name, status, frame):
    # Menyimpan frame wajah yang terdeteksi ke folder lokal sementara,
    # lalu mengunggahnya ke Drive dan mencatat ke Spreadsheet secara asinkron
    # melalui thread terpisah agar tidak memblokir loop utama.
    # Nama file format: NamaWajah_YYYYMMDD_HHMMSS_microsecond.jpg
    # File lokal dihapus otomatis setelah upload berhasil.
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{name}_{timestamp}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        # Lock hanya untuk bagian tulis file saja agar thread-safe
        with save_lock:
            try:
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception as e:
                print("❌ Error save image:", e)
                return
            
        def upload_thread():
            # Menunggu koneksi internet tersedia (maks 10x cek, jeda 3 detik).
            # Jika tidak ada koneksi dalam batas waktu, upload dibatalkan.
            connected = False
            for _ in range(10):
                if is_connected():
                    connected = True
                    break
                print("⏳ Menunggu koneksi internet...")
                time.sleep(3)
            
            if not connected:
                print(f"❌ Tidak ada koneksi, skip: {filename}")
                return

            link = upload_to_drive(save_path, filename)
            save_to_sheet(name, status, link)
            try: os.remove(save_path)
            except: pass

        threading.Thread(target=upload_thread, daemon=True).start()

    except Exception as e:
        print(f"❌ Save face error: {e}")

# ==========================================================
# LOGIKA PENCOCOKAN EMBEDDING WAJAH
# Membandingkan embedding wajah hasil InsightFace dengan semua
# data terlatih menggunakan jarak Euclidean (L2 norm).
# Embedding dinormalisasi terlebih dahulu sebelum dihitung jaraknya.
# Keputusan berdasarkan dua threshold:
#   < ACCEPT_THRESHOLD → wajah dikenal, kembalikan nama + status
#   > REJECT_THRESHOLD → Unknown
#   Di antara keduanya   → diperlakukan sebagai Unknown (zona abu-abu)
# ==========================================================
def match_encoding(embedding):
    embedding = embedding.astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    dists = np.linalg.norm(known_encodings - embedding, axis=1)
    best_idx = np.argmin(dists)
    best = float(dists[best_idx])
    print(f"DEBUG: Best Distance = {best:.4f}", flush=True)
    if best < ACCEPT_THRESHOLD:
        return known_names[best_idx], known_statuses[best_idx], best
    elif best > REJECT_THRESHOLD:
        return "Unknown", "Unknown", best
    else:
        return "Unknown", "Unknown", best

# ==========================================================
# THREAD WORKER UNTUK PENGENALAN WAJAH
# Fungsi ini dijalankan di thread terpisah untuk setiap wajah
# yang perlu diidentifikasi, agar tidak memblokir main loop.
# Alur kerja:
#   1. Konversi crop wajah ke RGB, deteksi ulang dengan InsightFace
#   2. Filter wajah berkualitas buruk (skor deteksi < 0.6),
#      misalnya wajah menyamping atau membelakangi kamera
#   3. Panggil match_encoding() untuk mencocokkan embedding
#   4. Simpan hasil ke recognition_results[objectID]
#   5. Hapus objectID dari processing_ids agar slot thread bebas
# ==========================================================
def recognize_worker(crop, objectID):
    global processing_ids, recognition_results
    try:
        faces = face_app.get(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if len(faces) == 0:
            res = ("NoFace", "NoFace", 999.0)
        else:
            face = faces[0]
            
            # det_score adalah confidence score deteksi wajah dari InsightFace (0.0 - 1.0)
            # Wajah dengan skor di bawah 0.6 dianggap tidak cukup jelas untuk dikenali
            det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.0
            
            if det_score < 0.6:
                # Wajah terdeteksi tapi kualitas buruk = menyamping/membelakangi
                res = ("NoFace", "NoFace", 999.0)
            else:
                res = match_encoding(face.embedding)
        with results_lock:
            recognition_results[objectID] = res
            result_timestamps[objectID] = time.time()

    except Exception as e:
        print(f"❌ Recognition error: {e}")
        with results_lock:
            recognition_results[objectID] = ("Unknown", "Unknown", 999.0)
            result_timestamps[objectID] = time.time()

    finally:
        with results_lock:
            processing_ids.discard(objectID)

# ==========================================================
# KELAS RTSPCamera — CAPTURE LANGSUNG DARI KAMERA IP VIA RTSP
# Menggantikan ScrcpyCamera. Tidak lagi bergantung pada app Tenda,
# HP Android, atau scrcpy — koneksi langsung kamera ke Python lewat
# protokol RTSP, selama kamera dan laptop berada di jaringan lokal
# yang sama (LAN/WiFi yang sama, satu router/satu subnet).
#
# Fitur:
#   - Auto-reconnect jika frame gagal dibaca / koneksi putus
#   - Buffer minimal + grab-drain agar selalu ambil frame TERBARU
#     (bukan numpuk delay seperti masalah yang dialami sebelumnya)
#   - Berjalan di thread terpisah (non-blocking terhadap main loop)
#   - Hitung FPS & bitrate aktual untuk monitoring (lihat get_stats)
# ==========================================================
class RTSPCamera:
    def __init__(self, rtsp_url, reconnect_delay=2, max_consecutive_fail=10):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_consecutive_fail = max_consecutive_fail

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        # Statistik untuk monitoring (opsional, dipakai get_stats())
        self._frame_count_window = 0
        self._byte_count_window = 0
        self._window_start = time.time()
        self._last_fps = 0.0
        self._last_kbps = 0.0
        self._connected = False

        self.cap = None
        self._open_stream()

        threading.Thread(target=self.update, daemon=True).start()

    def _open_stream(self):
        """Membuka/membuka-ulang koneksi RTSP."""
        if self.cap is not None:
            self.cap.release()

        # FFMPEG backend biasanya paling stabil untuk RTSP di OpenCV
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # Buffer size 1 = selalu ambil frame TERBARU, bukan antrian lama
        # Ini penting untuk live face recognition (hindari delay menumpuk)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._connected = self.cap.isOpened()
        if self._connected:
            print(f"✅ RTSP terkoneksi: {self.rtsp_url}")
        else:
            print(f"❌ RTSP gagal dibuka: {self.rtsp_url}")

    def update(self):
        """Loop background: terus membaca frame, auto-reconnect jika gagal."""
        fail_count = 0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print("🔄 Mencoba membuka ulang koneksi RTSP...")
                self._open_stream()
                time.sleep(self.reconnect_delay)
                continue

            # "Grab-drain" pattern: baca beberapa frame cepat tanpa decode
            # penuh untuk membuang frame lama yang nyangkut di buffer
            # internal FFMPEG (CAP_PROP_BUFFERSIZE=1 tidak selalu 100%
            # dihormati oleh semua backend/kamera). grab() jauh lebih
            # murah daripada read() karena tidak decode gambar.
            ret, frame = self.cap.read()

            if not ret or frame is None:
                fail_count += 1
                print(f"⚠ Frame gagal dibaca ({fail_count}/{self.max_consecutive_fail})")

                if fail_count >= self.max_consecutive_fail:
                    print("🔌 Koneksi tampak putus, reconnecting...")
                    self._connected = False
                    self._open_stream()
                    fail_count = 0
                    time.sleep(self.reconnect_delay)

                continue

            # Frame berhasil dibaca, reset fail counter
            fail_count = 0
            self._connected = True

            # Resize ke resolusi kerja, sama seperti perilaku ScrcpyCamera dulu
            try:
                frame_resized = cv2.resize(
                    frame, (RESIZE_WIDTH, RESIZE_HEIGHT),
                    interpolation=cv2.INTER_LANCZOS4
                )
            except Exception:
                continue

            with self.lock:
                self.frame = frame_resized

            # --- Hitung statistik bitrate/fps (untuk monitoring) ---
            self._frame_count_window += 1
            self._byte_count_window += frame.nbytes
            elapsed = time.time() - self._window_start
            if elapsed >= 1.0:
                self._last_fps = self._frame_count_window / elapsed
                self._last_kbps = (self._byte_count_window / 1024) / elapsed
                self._frame_count_window = 0
                self._byte_count_window = 0
                self._window_start = time.time()

    def read(self):
        """Mengambil frame terbaru. Return (ret, frame) — sama seperti ScrcpyCamera."""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
        return False, None

    def get_stats(self):
        """Return dict berisi fps & kbps aktual, untuk ditampilkan/log."""
        return {
            "connected": self._connected,
            "fps": round(self._last_fps, 1),
            "kbps": round(self._last_kbps, 1),
        }

    def is_connected(self):
        return self._connected

    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

# ==========================================================
# KELAS Tracker — PELACAKAN WAJAH LINTAS FRAME (CENTROID TRACKING)
# Setiap wajah yang terdeteksi diberi ID unik (objectID) dan dilacak
# antar frame berdasarkan jarak Euclidean antar centroid bounding box.
# Jika wajah tidak terdeteksi selama maxDisappeared frame berturut-turut,
# ID tersebut dihapus dari pelacakan.
# Atribut tambahan per wajah yang dilacak:
#   last_emb_time      : waktu terakhir embedding dikirim ke thread
#   candidate_names    : nama kandidat saat ini (untuk stabilitas sebelum simpan)
#   candidate_start_time: waktu mulai wajah menunjukkan nama yang konsisten
# ==========================================================
class Tracker:
    def __init__(self, maxDisappeared=60):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.last_emb_time = {}
        self.candidate_names = {}
        self.candidate_start_time = {}

    def register(self, bbox):
        self.objects[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.last_emb_time[self.nextObjectID] = 0
        self.candidate_names[self.nextObjectID] = None
        self.candidate_start_time[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        for attr in [self.objects, self.disappeared, self.last_emb_time, self.candidate_names, self.candidate_start_time]:
            if objectID in attr: del attr[objectID]
        with results_lock:
            recognition_results.pop(objectID, None)
            result_timestamps.pop(objectID, None)
        # Hapus timer unknown tracking jika wajah tidak lagi terlacak
        if objectID in unknown_candidate_start:
            del unknown_candidate_start[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            inputCentroids[i] = (int((startX + endX) / 2.0), int((startY + endY) / 2.0))
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)): self.register(rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array([[int((b[0]+b[2])/2), int((b[1]+b[3])/2)] for b in self.objects.values()])
            D = dist.cdist(objectCentroids, inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row); usedCols.add(col)
            for row in set(range(len(objectIDs))) - usedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            for col in set(range(len(inputCentroids))) - usedCols: self.register(rects[col])
        return self.objects

# ==========================================================
# FUNGSI MAIN — LOOP UTAMA PROGRAM
# ==========================================================
def main():
    last_wake_up = time.time()

    print(f"🔌 Menghubungkan ke kamera RTSP: {RTSP_URL}")
    cam = RTSPCamera(RTSP_URL)
    time.sleep(2)

    tracker = Tracker()
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=FACE_DETECT_CONFIDENCE)
    frame_count = 0

    cv2.namedWindow("Face Recognition CCTV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition CCTV", 960, 540)

    while True:
        try:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.1); continue
            
            # --------------------------------------------------
            # TAHAP 1: DETEKSI WAJAH DENGAN MEDIAPIPE
            # Frame di-skip jika tidak sesuai FRAME_SKIP.
            # MediaPipe mendeteksi lokasi wajah secara cepat,
            # lalu bounding box di-padding 30% untuk memastikan
            # seluruh wajah masuk crop. Wajah terlalu kecil (< 45px)
            # diabaikan untuk menghindari noise dari kejauhan.
            # --------------------------------------------------
            frame_count += 1
            if frame_count % FRAME_SKIP != 0: continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)
            rects = []
            if results.detections:
                ih, iw, _ = frame.shape
                for detection in results.detections:
                    if detection.score[0] < 0.3:
                        continue

                    bboxC = detection.location_data.relative_bounding_box

                    x  = int(bboxC.xmin * iw)
                    y  = int(bboxC.ymin * ih)
                    bw = int(bboxC.width * iw)
                    bh = int(bboxC.height * ih)

                    pad = int(0.3 * bw)  # Padding 30% lebar wajah di semua sisi

                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(iw, x + bw + pad)
                    y2 = min(ih, y + bh + pad)

                    if (x2 - x1) < 45 or (y2 - y1) < 45:
                        continue

                    rects.append((x1, y1, x2, y2))

                    if len(rects) >= MAX_FACES_TO_PROCESS:
                        break

            # --------------------------------------------------
            # TAHAP 2: UPDATE TRACKER & KIRIM KE THREAD PENGENALAN
            # Tracker memperbarui posisi setiap wajah berdasarkan
            # bounding box baru. Untuk wajah yang belum diproses
            # dalam 0.6 detik, crop dikirim ke thread recognize_worker
            # selama slot thread masih tersedia (< MAX_THREADS).
            # Crop di-resize ke 160x160 sebelum diproses InsightFace.
            # --------------------------------------------------
            objects = tracker.update(rects)
            current_time = time.time()

            for objectID, bbox in objects.items():
                if current_time - tracker.last_emb_time.get(objectID, 0) > 0.6:
                    x1, y1, x2, y2 = bbox
                    try:
                        crop = frame[y1:y2, x1:x2].copy()
                    except:
                        continue

                    if crop.size > 0 and crop.shape[0] > 40 and crop.shape[1] > 40:
                        crop = cv2.resize(crop, (160, 160))

                        with results_lock:
                            can_process = (objectID not in processing_ids and
                                        len(processing_ids) < MAX_THREADS)
                            if can_process:
                                processing_ids.add(objectID)

                        if can_process:
                            tracker.last_emb_time[objectID] = current_time
                            threading.Thread(
                                target=recognize_worker,
                                args=(crop, objectID),
                                daemon=True
                            ).start()

            # --------------------------------------------------
            # TAHAP 3: RENDER TAMPILAN & LOGIKA PENYIMPANAN OTOMATIS
            # Setiap wajah yang terlacak ditampilkan dengan:
            #   - Kotak HIJAU jika dikenal, MERAH jika Unknown,
            #     ABU-ABU tipis jika wajah tidak terdeteksi InsightFace
            #   - Label nama, status, dan jarak embedding
            # Penyimpanan otomatis wajah dikenal terjadi jika:
            #   - Nama konsisten selama ≥ 0.2 detik (stabilitas kandidat)
            #   - Nama belum disimpan dalam SESSION_TIMEOUT detik terakhir
            # --------------------------------------------------
            display = frame.copy()
            for oid, (x1, y1, x2, y2) in objects.items():
                # Ambil hasil pengenalan dari cache thread (default Unknown jika belum selesai)
                with results_lock:
                    name, status, best_dist = recognition_results.get(oid, ("Unknown", "Unknown", 999.0))
                    
                if name == "NoFace":
                    color = (128, 128, 128)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)  # kotak tipis abu-abu
                    continue  # tidak perlu tampilkan nama/status
                
                if name == "Unknown": color = (0, 0, 255) # MERAH
                else: color = (0, 255, 0) # HIJAU

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{name}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display, f"{status} ({best_dist:.2f})", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Logika Simpan Otomatis untuk wajah yang dikenal
                if name != "Unknown" and best_dist < 1.09:
                    if tracker.candidate_names.get(oid) == name:
                        # Cek durasi stabilitas wajah (0.2 detik)
                        if current_time - tracker.candidate_start_time.get(oid, current_time) >= 0.2:
                            if current_time - saved_names_session.get(name, 0) > SESSION_TIMEOUT:
                                print(f"[SCREENSHOT] Wajah Stabil: {name} (Dist: {best_dist:.2f}). Mengirim ke Drive...", flush=True)
                                save_detected_face(name, status, frame.copy())
                                saved_names_session[name] = current_time
                    else:
                        tracker.candidate_names[oid] = name
                        tracker.candidate_start_time[oid] = current_time
                        
                # --------------------------------------------------
                # LOGIKA PENYIMPANAN WAJAH UNKNOWN YANG STABIL
                # Wajah Unknown hanya disimpan jika:
                #   1. Jarak embedding < 1.3 (ada wajah nyata, bukan noise)
                #   2. Wajah stabil terdeteksi selama UNKNOWN_STABLE_TIME detik
                #   3. Sudah melewati UNKNOWN_SAVE_COOLDOWN sejak simpan terakhir
                # Cooldown bersifat global (bukan per-ID) untuk mencegah
                # spam penyimpanan saat banyak Unknown terdeteksi bersamaan.
                # Timer per-ID direset setelah tersimpan agar tidak loop terus.
                # --------------------------------------------------
                if name == "Unknown":
                    if best_dist < 1.3:
                        if oid not in unknown_candidate_start:
                            unknown_candidate_start[oid] = current_time
                        else:
                            elapsed = current_time - unknown_candidate_start[oid]
                            if elapsed >= UNKNOWN_STABLE_TIME:
                                # Cek global cooldown, bukan per-ID
                                global last_unknown_save_time
                                if current_time - last_unknown_save_time >= UNKNOWN_SAVE_COOLDOWN:
                                    print(f"[UNKNOWN] Wajah stabil {UNKNOWN_STABLE_TIME}s -> Simpan", flush=True)
                                    save_detected_face("Unknown", "Unknown", frame.copy())
                                    last_unknown_save_time = current_time  # update global timer

                                # Reset timer oid ini agar tidak terus trigger
                                unknown_candidate_start[oid] = current_time + UNKNOWN_SAVE_COOLDOWN
                    else:
                        if oid in unknown_candidate_start:
                            del unknown_candidate_start[oid]
                else:
                    if oid in unknown_candidate_start:
                        del unknown_candidate_start[oid]

            # --------------------------------------------------
            # KEEP AWAKE: Mencegah Windows masuk mode sleep/screensaver
            # Setiap 120 detik, pyautogui menekan tombol Shift secara virtual
            # agar layar tidak mati saat sistem berjalan tanpa interaksi user.
            # --------------------------------------------------
            if current_time - last_wake_up > 120:
                pyautogui.press('shift'); last_wake_up = current_time
                # Log kecil di terminal untuk memastikan fitur jalan
                print("☕ Sinyal 'Keep-Awake' terkirim ke Windows...")

            cv2.imshow("Face Recognition CCTV", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # --------------------------------------------------
            # PEMBERSIHAN CACHE HASIL PENGENALAN
            # Hasil pengenalan wajah yang sudah tidak terlacak dan
            # lebih lama dari MAX_RESULT_AGE detik dihapus dari cache
            # untuk mencegah memory leak pada sesi yang panjang.
            # --------------------------------------------------
            with results_lock:
                for oid in list(recognition_results.keys()):
                    if oid not in objects and \
                    time.time() - result_timestamps.get(oid, 0) > MAX_RESULT_AGE:
                        recognition_results.pop(oid, None)
                        result_timestamps.pop(oid, None)
        
        except Exception as e:
            print("⚠ Main loop error:", e)
            time.sleep(0.1)

    cam.release(); cv2.destroyAllWindows()
if __name__ == "__main__":
    main()