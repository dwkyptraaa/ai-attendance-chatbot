import os
import cv2
import pickle
import numpy as np
from datetime import datetime
from collections import Counter
from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings("ignore")

# ================= KONFIGURASI =================
# Struktur dataset wajah (relatif terhadap folder ini):
#   dataset_wajah/
#   ├── Mahasiswa/
#   │   └── <nama>/*.jpg
#   └── Pegawai/
#       └── <nama>/*.jpg
# Ganti lewat environment variable DATASET_DIR bila perlu.
DATASET_DIR = os.getenv("DATASET_DIR", "dataset_wajah")
OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", "trained_models/trained_faces_insightface480l.pkl")

MAX_SAMPLES_PER_PERSON = 12
MIN_FACE_SIZE = 80

# ================= INIT MODEL ==================
print("🚀 Initializing InsightFace (CPU)...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(320, 320))
print("✅ InsightFace siap")

# ================= LOAD DATASET ================
all_encodings = []
all_names = []
all_statuses = []
errors = []
stats = []

start_time = datetime.now()

for status in ["Mahasiswa", "Pegawai"]:
    status_path = os.path.join(DATASET_DIR, status)
    if not os.path.exists(status_path):
        continue

    for person_name in os.listdir(status_path):
        person_dir = os.path.join(status_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n🔹 Processing {person_name} ({status})")
        added = 0

        images = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ][:MAX_SAMPLES_PER_PERSON]

        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                errors.append(f"{person_name}/{img_name}: read error")
                continue

            faces = app.get(img)
            if len(faces) == 0:
                errors.append(f"{person_name}/{img_name}: no face")
                continue

            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )

            x1, y1, x2, y2 = map(int, face.bbox)
            if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                continue

            emb = face.embedding.astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            all_encodings.append(emb)
            all_names.append(person_name)
            all_statuses.append(status)
            added += 1

        stats.append(f"{person_name}: {added} embedding")
        print(f"   ✓ {added} embedding")

# ================= ANALISIS ====================
print("\n📊 ANALISIS DATASET")
name_counts = Counter(all_names)
for name, cnt in name_counts.items():
    print(f"   {name}: {cnt}")

# ================= SIMPAN MODEL =================
if len(all_encodings) == 0:
    print("❌ Tidak ada data wajah")
    exit()

data = {
    "encodings": np.array(all_encodings, dtype=np.float32),
    "names": all_names,
    "statuses": all_statuses,
    "meta": {
        "model": "InsightFace ArcFace (512D)",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_people": len(set(all_names)),
        "total_embeddings": len(all_encodings)
    }
}

os.makedirs(os.path.dirname(OUTPUT_MODEL) or ".", exist_ok=True)
with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump(data, f)

print("\n" + "=" * 60)
print("✅ TRAINING INSIGHTFACE SELESAI")
print(f"📁 File: {OUTPUT_MODEL}")
print(f"👥 Orang: {len(set(all_names))}")
print(f"🧠 Embedding: {len(all_encodings)}")
print("=" * 60)
