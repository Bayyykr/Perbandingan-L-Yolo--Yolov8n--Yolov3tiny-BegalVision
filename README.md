# YOLO Multi-Model ONNX REST API & Video Testing Dashboard (Flask)

API berbasis **Flask** dan **ONNX Runtime** untuk menjalankan inferensi dan membandingkan **3 model YOLO** berformat `.onnx` (`lyolo.onnx`, `yolov3tiny.onnx`, dan `yolov8nano.onnx`) pada **Video** (di folder `videos/`), **Gambar**, maupun **Webcam**.

---

## Struktur Proyek

```text
api yol/
├── app.py              # Server utama Flask REST API + Dashboard Web UI Multi-Model & Video
├── yolo_onnx.py        # Engine inferensi YOLO ONNX (Letterbox, NMS, Metadata)
├── webcam_test.py      # Script pengujian Webcam / Video via OpenCV (tekan 'M' untuk ganti model)
├── test_api.py         # Script pengujian endpoint API
├── requirements.txt    # Daftar dependensi Python
├── models/             # Folder tempat menyimpan 3 model .onnx
│   ├── lyolo.onnx      # Model 1: lYOLO (Light YOLO - 640x640)
│   ├── yolov3tiny.onnx # Model 2: YOLOv3-Tiny (640x640)
│   └── yolov8nano.onnx # Model 3: YOLOv8-Nano (512x512)
└── videos/             # Folder tempat menyimpan video pengujian
    └── Asset_Video.mp4 # Video pengujian yang siap diuji langsung di Website
```

---

## Cara Penggunaan

### 1. Letakkan File Model `.onnx`
Salin file model YOLOv8 Anda (misalnya `best.onnx` atau `model.onnx`) ke dalam folder ini (`api yol/`) atau ke dalam subfolder `models/`.
- Server akan **otomatis mendeteksi** file `.onnx` yang ada di folder tanpa perlu mengubah kode.
- Nama-nama class (`names`) akan dibaca **otomatis** dari metadata model ONNX hasil export Ultralytics. Jika Anda ingin menggunakan daftar nama class kustom, buat file `classes.txt` (satu nama class per baris) di folder ini.

### 2. Jalankan Server Flask
Semua library yang dibutuhkan (`flask`, `flask-cors`, `onnxruntime`, `opencv-python`, `numpy`) sudah terinstal di environment Anda. Jalankan perintah berikut:

```powershell
python app.py
```

Server akan berjalan di `http://127.0.0.1:5000`.
Anda juga dapat membuka `http://127.0.0.1:5000` langsung di **Browser** untuk menggunakan **Dashboard Pengujian Visual** (upload gambar, atur threshold, lihat gambar hasil bounding box, atau upload file `.onnx`).

---

## Dokumentasi Endpoint API

### 1. `GET /health`
Mengecek status server, model `.onnx` yang sedang aktif, resolusi input, dan daftar class.

```bash
curl http://127.0.0.1:5000/health
```

### 2. `POST /predict` (atau `POST /detect`)
Melakukan deteksi objek pada gambar dan mengembalikan koordinat bounding box dalam format JSON.

**Parameter (Form-Data / Query / JSON):**
- `image` atau `file`: File gambar (`JPG`, `PNG`, `WEBP`) atau string Base64 jika mengirim body JSON.
- `conf` *(opsional, default: `0.25`)*: Confidence threshold (`0.01` - `1.0`).
- `iou` *(opsional, default: `0.45`)*: IoU threshold untuk Non-Maximum Suppression (`0.01` - `1.0`).
- `return_image` *(opsional, default: `false`)*: Isi `true` jika ingin menyertakan gambar hasil anotasi dalam format Base64 (`annotated_image_base64`) di dalam response JSON.

**Contoh cURL:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@contoh_gambar.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

**Contoh Python (`requests`):**
```python
import requests

url = "http://127.0.0.1:5000/predict"
with open("contoh_gambar.jpg", "rb") as f:
    files = {"image": f}
    data = {"conf": 0.25, "iou": 0.45}
    response = requests.post(url, files=files, data=data)

print(response.json())
```

**Contoh Response JSON:**
```json
{
  "success": true,
  "model": "best.onnx",
  "parameters": {
    "conf": 0.25,
    "iou": 0.45
  },
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "inference_time_ms": 38.42,
  "count": 1,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.9142,
      "bbox": {
        "x1": 210.5,
        "y1": 140.0,
        "x2": 480.25,
        "y2": 650.75,
        "width": 269.75,
        "height": 510.75
      },
      "bbox_normalized": {
        "x_center": 0.269824,
        "y_center": 0.549132,
        "width": 0.210742,
        "height": 0.709375
      }
    }
  ]
}
```

### 3. `POST /predict/image`
Melakukan deteksi objek dan langsung mengembalikan file gambar JPEG (`image/jpeg`) yang sudah digambar kotak bounding box beserta label dan nilai confidence-nya.

```bash
curl -X POST http://127.0.0.1:5000/predict/image \
  -F "image=@contoh_gambar.jpg" \
  --output hasil_deteksi.jpg
```

### 4. `POST /reload`
Memuat ulang model `.onnx` dari folder tanpa perlu mematikan server.
```bash
curl -X POST http://127.0.0.1:5000/reload
```
