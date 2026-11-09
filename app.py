import base64
import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from yolo_onnx import YOLOv8ONNX

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
MODELS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)

# Konfigurasi default
DEFAULT_CONF = float(os.getenv("YOLO_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("YOLO_IOU", "0.45"))

# Label deskriptif untuk model yang tersedia
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "lyolo.onnx": "lYOLO",
    "yolov8nano.onnx": "YOLOv8 Nano",
    "yolov3tiny.onnx": "YOLOv3 Tiny",
    "yolov26.onnx": "YOLOv26",
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Cache & state global untuk multi-model ONNX
model_lock = threading.Lock()
detectors_cache: Dict[str, YOLOv8ONNX] = {}
detector: Optional[YOLOv8ONNX] = None
detector_error: Optional[str] = None
video_meta_cache: Dict[Tuple[str, float], Dict[str, Any]] = {}


def find_onnx_models() -> List[Path]:
    """Mencari semua file .onnx di folder models/ maupun folder utama."""
    candidates: List[Path] = []
    priority_names = [
        "lyolo.onnx",
        "yolov8nano.onnx",
        "yolov3tiny.onnx",
        "yolov26.onnx",
        "best.onnx",
        "model.onnx",
        "yolov8.onnx",
        "yolov8n.onnx",
    ]

    # 1. Cek environment variable MODEL_PATH terlebih dahulu
    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).is_file():
        candidates.append(Path(env_path).resolve())

    # 2. Cari di folder models/ dan folder utama
    search_dirs = [MODELS_DIR, BASE_DIR]
    found_files: List[Path] = []
    for d in search_dirs:
        if d.exists():
            found_files.extend(sorted(d.glob("*.onnx")))

    # Urutkan berdasarkan prioritas nama terlebih dahulu
    for pname in priority_names:
        for f in found_files:
            if f.name.lower() == pname and f.resolve() not in candidates:
                candidates.append(f.resolve())

    for f in found_files:
        if f.resolve() not in candidates:
            candidates.append(f.resolve())

    return candidates


def find_classes_file() -> Optional[str]:
    """Mencari file classes.txt atau labels.txt jika disediakan pengguna."""
    for name in ["classes.txt", "labels.txt", "class_names.txt"]:
        for folder in [BASE_DIR, MODELS_DIR]:
            candidate = folder / name
            if candidate.is_file():
                return str(candidate)
    return None


def resolve_model_path(specific_path: Optional[str]) -> Optional[Path]:
    """Mencari Path absolut dari nama atau path file model .onnx."""
    if not specific_path:
        models = find_onnx_models()
        return models[0] if models else None

    target = Path(specific_path)
    if target.is_file():
        return target.resolve()

    for folder in [MODELS_DIR, BASE_DIR]:
        candidate = folder / specific_path
        if candidate.is_file():
            return candidate.resolve()
        # Cocokkan case-insensitive berdasarkan nama file
        if folder.exists():
            for f in folder.glob("*.onnx"):
                if f.name.lower() == Path(specific_path).name.lower():
                    return f.resolve()

    return None


def load_or_reload_model(
    specific_path: Optional[str] = None, force_reload: bool = False
) -> Tuple[bool, str]:
    """Memuat atau mengganti model ONNX aktif (menggunakan cache memori agar instan)."""
    global detector, detector_error

    with model_lock:
        try:
            target = resolve_model_path(specific_path)
            if target is None:
                if specific_path:
                    detector_error = f"File model '{specific_path}' tidak ditemukan."
                    return False, detector_error
                detector = None
                detector_error = (
                    "Belum ada file model .onnx di folder ini. "
                    "Silakan letakkan file model di subfolder 'models/'."
                )
                return False, detector_error

            cache_key = str(target)
            if force_reload or cache_key not in detectors_cache:
                classes_path = find_classes_file()
                detectors_cache[cache_key] = YOLOv8ONNX(
                    model_path=cache_key,
                    classes_path=classes_path,
                    conf_thres=DEFAULT_CONF,
                    iou_thres=DEFAULT_IOU,
                )

            detector = detectors_cache[cache_key]
            detector_error = None
            return True, f"Model '{target.name}' berhasil diaktifkan."
        except Exception as e:
            detector_error = f"Gagal memuat model ONNX: {str(e)}"
            return False, detector_error


def preload_all_models() -> None:
    """Memuat seluruh model .onnx di folder ke dalam cache agar pergantian model instan."""
    models = find_onnx_models()
    classes_path = find_classes_file()
    for m_path in models:
        cache_key = str(m_path)
        if cache_key not in detectors_cache:
            try:
                detectors_cache[cache_key] = YOLOv8ONNX(
                    model_path=cache_key,
                    classes_path=classes_path,
                    conf_thres=DEFAULT_CONF,
                    iou_thres=DEFAULT_IOU,
                )
            except Exception as e:
                print(f"[WARN] Gagal preload model {m_path.name}: {e}")


def ensure_model_loaded(requested_model: Optional[str] = None) -> Optional[YOLOv8ONNX]:
    """
    Memastikan model sudah dimuat.
    Jika requested_model diberikan pada request /predict, ambil dari cache tanpa
    menimpa detector aktif global jika terjadi request paralel.
    """
    global detector
    if not detectors_cache:
        preload_all_models()
    if detector is None:
        load_or_reload_model()
    if requested_model:
        target = resolve_model_path(requested_model)
        if target is not None:
            cache_key = str(target)
            if cache_key not in detectors_cache:
                load_or_reload_model(requested_model)
            return detectors_cache.get(cache_key, detector)
    return detector


def get_models_catalog() -> List[Dict[str, Any]]:
    """Mengembalikan daftar lengkap semua model .onnx beserta detailnya."""
    preload_all_models()
    active_path = str(Path(detector.model_path).resolve()) if detector else None
    catalog: List[Dict[str, Any]] = []

    for p in find_onnx_models():
        full_key = str(p.resolve())
        size_mb = round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else 0.0
        cached_inst = detectors_cache.get(full_key)
        info = cached_inst.get_model_info() if cached_inst else None

        catalog.append(
            {
                "name": p.name,
                "display_name": MODEL_DISPLAY_NAMES.get(p.name.lower(), p.stem),
                "size_mb": size_mb,
                "active": full_key == active_path,
                "loaded_in_memory": cached_inst is not None,
                "input_resolution": info["input_resolution"] if info else None,
                "description": info.get("description", "") if info else "",
                "num_classes": info["num_classes"] if info else None,
            }
        )
    return catalog


def find_video_files() -> List[Path]:
    """Mencari semua file video di dalam folder videos/."""
    if not VIDEOS_DIR.exists():
        return []
    files: List[Path] = []
    for p in sorted(VIDEOS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(p.resolve())
    return files


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Membaca metadata video (resolusi, FPS, durasi, ukuran file) dengan caching."""
    mtime = video_path.stat().st_mtime if video_path.exists() else 0.0
    cache_key = (str(video_path), mtime)
    if cache_key in video_meta_cache:
        return video_meta_cache[cache_key]

    size_mb = round(video_path.stat().st_size / (1024 * 1024), 2) if video_path.exists() else 0.0
    width, height, fps, frame_count, duration_sec = 0, 0, 0.0, 0, 0.0

    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and frame_count > 0:
            duration_sec = round(frame_count / fps, 2)
        cap.release()

    meta = {
        "filename": video_path.name,
        "url": f"/videos/{video_path.name}",
        "size_mb": size_mb,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }
    video_meta_cache[cache_key] = meta
    return meta


def get_videos_catalog() -> List[Dict[str, Any]]:
    """Mengembalikan daftar semua video di folder videos/ beserta metadatanya."""
    return [get_video_metadata(p) for p in find_video_files()]


def read_image_from_request() -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Membaca gambar dari request:
    - multipart/form-data dengan key 'image' atau 'file'
    - JSON dengan key 'image' atau 'image_base64' (berisi string Base64)
    """
    img_bytes: Optional[bytes] = None

    # 1. Cek multipart/form-data
    if "image" in request.files:
        file_obj = request.files["image"]
        if file_obj.filename == "":
            return None, "Nama file gambar kosong."
        img_bytes = file_obj.read()
    elif "file" in request.files:
        file_obj = request.files["file"]
        if file_obj.filename == "":
            return None, "Nama file gambar kosong."
        img_bytes = file_obj.read()

    # 2. Cek JSON body (Base64)
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        b64_str = data.get("image") or data.get("image_base64")
        if b64_str:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64_str)
            except Exception as e:
                return None, f"Format Base64 tidak valid: {str(e)}"

    if not img_bytes:
        return (
            None,
            "Gambar tidak ditemukan pada request. Kirimkan melalui form-data key 'image' (atau 'file') atau JSON key 'image' (base64).",
        )

    # Decode bytes ke array gambar OpenCV (BGR)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, "Gagal membaca file gambar. Pastikan format gambar valid (JPG, PNG, WEBP, BMP)."

    return img_bgr, None


def get_inference_params() -> Tuple[float, float, bool, Optional[str]]:
    """Mengambil parameter conf, iou, return_image, dan model dari query, form, atau JSON."""
    data = request.get_json(silent=True) if request.is_json else {}
    if not isinstance(data, dict):
        data = {}

    def _get_val(key: str, default: str = "") -> str:
        return str(
            request.args.get(key)
            or request.form.get(key)
            or data.get(key)
            or default
        )

    try:
        conf = float(_get_val("conf", str(DEFAULT_CONF)))
        conf = max(0.01, min(1.0, conf))
    except ValueError:
        conf = DEFAULT_CONF

    try:
        iou = float(_get_val("iou", str(DEFAULT_IOU)))
        iou = max(0.01, min(1.0, iou))
    except ValueError:
        iou = DEFAULT_IOU

    ret_img_raw = _get_val("return_image", "false").lower()
    return_image = ret_img_raw in ("true", "1", "yes", "base64")

    req_model = _get_val("model", "").strip() or None
    return conf, iou, return_image, req_model


def encode_image_base64(image_bgr: np.ndarray) -> str:
    """Encode gambar BGR OpenCV ke string Base64 JPEG."""
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# =====================================================================
# HTML UI Bawaan untuk Pengujian Multi-Model, Video, Gambar & Webcam
# =====================================================================
INDEX_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Deteksi Objek</title>
    <style>
        :root {
            --bg: #f8f7fc;
            --bg-soft: #f2f0fa;
            --card: #ffffff;
            --purple: #6e62e5;
            --purple-dark: #584cd1;
            --purple-light: #eeecfc;
            --pink: #e959b5;
            --pink-hover: #d845a3;
            --text: #1e1b4b;
            --text-secondary: #4b5563;
            --muted: #7c7a96;
            --border: #e5e3f3;
            --danger: #e11d48;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            min-height: 100vh;
            padding-bottom: 3rem;
        }

        .page-wrap {
            max-width: 1040px;
            margin: 0 auto;
            padding: 0 1.25rem;
        }

        /* Navbar Flat Minimalis */
        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.15rem 0;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
            gap: 1rem;
        }
        .brand-logo {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text);
        }
        .brand-logo span {
            color: var(--purple);
        }
        .nav-links {
            display: flex;
            gap: 1.5rem;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .nav-links a {
            color: inherit;
            text-decoration: none;
            cursor: pointer;
        }
        .nav-links a:hover { color: var(--purple); }
        .active-model-text {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--muted);
        }
        .active-model-text b {
            color: var(--purple);
        }

        /* Judul Utama Flat & Sederhana */
        .hero-header {
            margin-bottom: 1.25rem;
        }
        .hero-title {
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
        }
        .hero-title span {
            color: var(--purple);
        }
        .hero-desc {
            color: var(--muted);
            font-size: 0.92rem;
        }

        /* Blok Ungu Flat (Pemilih 3 Model) */
        .purple-stage {
            background: var(--purple);
            border-radius: 1.15rem;
            padding: 1.35rem 1.5rem;
            color: #ffffff;
            margin-bottom: 1.5rem;
        }
        .stage-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .stage-title {
            font-size: 1rem;
            font-weight: 700;
            color: #ffffff;
        }
        .stage-sub {
            font-size: 0.82rem;
            color: rgba(255, 255, 255, 0.82);
        }

        /* 4 Kartu Model Flat */
        .model-cards {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        @media (max-width: 900px) {
            .model-cards { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 540px) {
            .model-cards { grid-template-columns: 1fr; }
        }
        .model-card {
            background: rgba(255, 255, 255, 0.12);
            border: 1.5px solid rgba(255, 255, 255, 0.22);
            border-radius: 0.85rem;
            padding: 0.95rem 1.1rem;
            cursor: pointer;
            transition: background 0.15s ease, border-color 0.15s ease;
            color: #ffffff;
        }
        .model-card:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .model-card.active {
            background: #ffffff;
            color: var(--text);
            border-color: #ffffff;
        }
        .model-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.2rem;
        }
        .model-card-title {
            font-weight: 700;
            font-size: 0.96rem;
        }
        .model-status-label {
            font-size: 0.75rem;
            font-weight: 700;
            opacity: 0.85;
        }
        .model-card.active .model-status-label {
            color: var(--pink);
            opacity: 1;
        }
        .model-card-meta {
            font-size: 0.8rem;
            opacity: 0.82;
        }
        .model-card.active .model-card-meta {
            color: var(--muted);
            opacity: 1;
        }

        /* Baris Pengaturan di Dalam Blok Ungu */
        .stage-controls {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            align-items: flex-end;
            padding-top: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.18);
        }
        .stage-controls > div { flex: 1; min-width: 140px; }
        .stage-controls label {
            color: rgba(255, 255, 255, 0.92);
            font-size: 0.76rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
            display: block;
        }
        .stage-controls select, .stage-controls input {
            background: #ffffff;
            border: 1px solid transparent;
            border-radius: 0.6rem;
            padding: 0.55rem 0.75rem;
            font-size: 0.86rem;
            color: var(--text);
            font-weight: 600;
            width: 100%;
        }

        /* Kontainer Putih Flat */
        .white-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 1.15rem;
            padding: 1.5rem;
        }
        .workspace-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        .section-heading {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text);
        }

        /* Tab Flat */
        .tabs {
            display: inline-flex;
            gap: 0.3rem;
            background: var(--bg-soft);
            padding: 0.3rem;
            border-radius: 0.75rem;
            flex-wrap: wrap;
        }
        .tab-btn {
            background: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.55rem;
            font-weight: 600;
            font-size: 0.84rem;
            cursor: pointer;
            width: auto;
        }
        .tab-btn:hover:not(.active) {
            color: var(--purple);
        }
        .tab-btn.active {
            background: var(--purple);
            color: #ffffff;
        }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }

        /* Form & Tombol Flat */
        .form-group { margin-bottom: 0.95rem; }
        label {
            display: block;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.35rem;
        }
        input[type="file"], input[type="number"], select {
            width: 100%;
            padding: 0.6rem 0.8rem;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 0.65rem;
            color: var(--text);
            font-size: 0.88rem;
            outline: none;
        }
        input[type="file"]:focus, input[type="number"]:focus, select:focus {
            border-color: var(--purple);
            background: #ffffff;
        }
        input[type="range"] {
            width: 100%;
            accent-color: var(--purple);
            cursor: pointer;
        }
        .row { display: flex; gap: 0.75rem; flex-wrap: wrap; }
        .row > div { flex: 1; min-width: 135px; }

        button {
            padding: 0.62rem 1.1rem;
            border-radius: 0.65rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            font-size: 0.86rem;
            border: none;
            transition: background 0.15s ease;
        }
        button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }
        .btn-purple {
            background: var(--purple);
            color: #ffffff;
        }
        .btn-purple:hover:not(:disabled) {
            background: var(--purple-dark);
        }
        .btn-pink {
            background: var(--pink);
            color: #ffffff;
        }
        .btn-pink:hover:not(:disabled) {
            background: var(--pink-hover);
        }
        .btn-soft {
            background: var(--bg-soft);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .btn-soft:hover:not(:disabled) {
            background: var(--purple-light);
            color: var(--purple-dark);
        }
        .btn-danger-soft {
            background: #fff1f2;
            color: var(--danger);
            border: 1px solid #fecdd3;
        }
        .btn-danger-soft:hover:not(:disabled) {
            background: #ffe4e6;
        }

        /* Area Tampilan Video / Gambar Flat */
        .preview-box {
            margin-top: 0.75rem;
            text-align: center;
            background: var(--bg);
            border-radius: 0.9rem;
            padding: 0.9rem;
            border: 1px solid var(--border);
            min-height: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .preview-box img, .preview-box canvas {
            max-width: 100%;
            max-height: 490px;
            border-radius: 0.6rem;
        }

        .status-banner {
            width: 100%;
            padding: 0.55rem 0.9rem;
            border-radius: 0.6rem;
            font-weight: 600;
            font-size: 0.84rem;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            background: #ffffff;
            border: 1px solid var(--border);
        }
        .status-banner.safe {
            background: #ecfdf5;
            border-color: #a7f3d0;
            color: #047857;
        }
        .status-banner.danger {
            background: #fff1f2;
            border-color: #fecdd3;
            color: #be123c;
        }
        .status-banner.neutral {
            background: #ffffff;
            border-color: var(--border);
            color: var(--text-secondary);
        }

        .stats-bar {
            width: 100%;
            margin-top: 0.75rem;
            padding: 0.55rem 0.9rem;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 0.6rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        pre {
            background: #1e1b4b;
            padding: 1rem;
            border-radius: 0.75rem;
            overflow-x: auto;
            font-size: 0.8rem;
            color: #e0e7ff;
            max-height: 360px;
            text-align: left;
        }
    </style>
</head>
<body>
    <div class="page-wrap">
        <!-- NAVBAR FLAT -->
        <nav class="navbar">
            <div class="brand-logo">Deteksi<span>Objek</span></div>
            <div class="nav-links">
                <a onclick="document.querySelector('[data-tab=tabVideo]').click()">Video</a>
                <a onclick="document.querySelector('[data-tab=tabImage]').click()">Gambar</a>
                <a onclick="document.querySelector('[data-tab=tabWebcam]').click()">Kamera</a>
                <a onclick="document.querySelector('[data-tab=tabUpload]').click()">Tambah File</a>
            </div>
            <div class="active-model-text" id="statusBadge">Memuat model...</div>
        </nav>

        <!-- JUDUL UTAMA SEDERHANA (TANPA ISTILAH ONNX) -->
        <div class="hero-header">
            <h1 class="hero-title">Pengujian <span>Deteksi Objek</span></h1>
            <p class="hero-desc">
                Pilih salah satu model di bawah untuk menguji dan membandingkan hasil deteksi pada video maupun gambar.
            </p>
        </div>

        <!-- BLOK UNGU FLAT (PEMILIH 3 MODEL) -->
        <section class="purple-stage">
            <div class="stage-header">
                <div>
                    <div class="stage-title">Pilih Model Deteksi</div>
                    <div class="stage-sub">Klik kartu model untuk mengganti model (video otomatis berhenti saat berganti model)</div>
                </div>
                <div id="switchingIndicator" style="font-size: 0.82rem; font-weight: 600; color: #fde047;"></div>
            </div>

            <div id="modelCardsContainer" class="model-cards">
                <!-- Diisi otomatis -->
            </div>

            <div class="stage-controls">
                <div style="flex: 2.2;">
                    <label>Daftar Model</label>
                    <select id="modelSelect"></select>
                </div>
                <div style="flex: 1;">
                    <label>Batas Akurasi (0.01 - 1.0)</label>
                    <input type="number" id="confInput" value="0.25" min="0.01" max="1.0" step="0.05">
                </div>
                <div style="flex: 1;">
                    <label>Batas Tumpuk Kotak (0.01 - 1.0)</label>
                    <input type="number" id="iouInput" value="0.45" min="0.01" max="1.0" step="0.05">
                </div>
                <div style="flex: 1;">
                    <button type="button" id="reloadModelBtn" class="btn-pink">Terapkan Model</button>
                </div>
            </div>
        </section>

        <!-- AREA PENGUJIAN FLAT -->
        <div class="white-card">
            <div class="workspace-top">
                <h2 class="section-heading">Area Pengujian</h2>
                <div class="tabs">
                    <button type="button" class="tab-btn active" data-tab="tabVideo">Video</button>
                    <button type="button" class="tab-btn" data-tab="tabImage">Gambar</button>
                    <button type="button" class="tab-btn" data-tab="tabWebcam">Kamera</button>
                    <button type="button" class="tab-btn" data-tab="tabUpload">Tambah File</button>
                    <button type="button" class="tab-btn" data-tab="tabJson">Data Hasil</button>
                </div>
            </div>

            <!-- TAB 1: UJI VIDEO -->
            <div id="tabVideo" class="tab-panel active">
                <div class="form-group">
                    <label>Pilih Video</label>
                    <div class="row">
                        <div style="flex: 3.5;">
                            <select id="videoSelect">
                                <option value="">Memuat daftar video...</option>
                            </select>
                        </div>
                        <div style="flex: 1;">
                            <button type="button" id="refreshVideosBtn" class="btn-soft">Muat Ulang Daftar</button>
                        </div>
                    </div>
                </div>

                <div class="row form-group">
                    <div style="flex: 1.4;">
                        <button type="button" id="playVideoBtn" class="btn-pink">Mulai Deteksi Video</button>
                    </div>
                    <div>
                        <button type="button" id="pauseVideoBtn" class="btn-soft" disabled>Jeda</button>
                    </div>
                    <div>
                        <button type="button" id="stopVideoBtn" class="btn-danger-soft" disabled>Berhenti</button>
                    </div>
                    <div>
                        <button type="button" id="detectFrameBtn" class="btn-purple">Deteksi Gambar Saat Ini</button>
                    </div>
                </div>

                <div class="row form-group" style="align-items: center; background: var(--bg); padding: 0.7rem 0.9rem; border-radius: 0.75rem; border: 1px solid var(--border);">
                    <div style="flex: 2.5;">
                        <label style="margin-bottom: 0.15rem;">
                            Posisi Waktu: <span id="videoTimeLabel" style="color: var(--purple); font-weight: 700;">00:00 / 00:00</span>
                        </label>
                        <input type="range" id="videoSeekSlider" min="0" max="100" step="0.05" value="0">
                    </div>
                    <div style="flex: 1; min-width: 115px;">
                        <label style="margin-bottom: 0.15rem;">Kecepatan Putar</label>
                        <select id="videoSpeedSelect" style="padding: 0.42rem 0.65rem; background: #fff;">
                            <option value="1">1.0x (Normal)</option>
                            <option value="0.5">0.5x (Lambat)</option>
                            <option value="0.25">0.25x (Sangat Lambat)</option>
                        </select>
                    </div>
                    <div style="flex: 0.8; min-width: 95px; display: flex; align-items: center; gap: 0.45rem; padding-top: 1rem;">
                        <input type="checkbox" id="videoLoopCheck" checked style="width: 16px; height: 16px; cursor: pointer; accent-color: var(--purple);">
                        <label for="videoLoopCheck" style="margin: 0; cursor: pointer;">Ulangi Otomatis</label>
                    </div>
                </div>

                <div class="preview-box" id="videoPreviewBox">
                    <div id="videoStatusBanner" class="status-banner neutral">
                        <span id="videoStatusText">Siap menguji video. Klik "Mulai Deteksi Video" untuk memulai.</span>
                        <span id="videoActiveModelBadge" style="font-weight: 700; color: var(--purple);">Model: -</span>
                    </div>
                    <video id="testVideo" playsinline muted crossorigin="anonymous" style="display: none;"></video>
                    <canvas id="videoCanvas"></canvas>
                    <div class="stats-bar" id="videoStatsBar">
                        <span id="videoFpsStat">Kecepatan: 0 ms</span>
                        <span id="videoCountStat">Objek Terdeteksi: 0</span>
                    </div>
                </div>
            </div>

            <!-- TAB 2: UJI GAMBAR -->
            <div id="tabImage" class="tab-panel">
                <form id="predictForm">
                    <div class="form-group">
                        <label>Pilih File Gambar (JPG / PNG / WEBP)</label>
                        <input type="file" id="imageInput" accept="image/*" required>
                    </div>
                    <button type="submit" id="submitBtn" class="btn-pink">Deteksi Gambar</button>
                </form>

                <div class="preview-box" id="imagePreview">
                    <span style="color: var(--muted); font-size: 0.88rem;">Hasil deteksi gambar akan tampil di sini</span>
                </div>
            </div>

            <!-- TAB 3: KAMERA LANGSUNG -->
            <div id="tabWebcam" class="tab-panel">
                <div class="form-group">
                    <label>Pilih Kamera</label>
                    <div class="row">
                        <div style="flex: 3;">
                            <select id="cameraSelect">
                                <option value="">-- Klik "Cari Kamera" atau "Mulai Kamera" --</option>
                            </select>
                        </div>
                        <div style="flex: 1;">
                            <button type="button" id="detectCamBtn" class="btn-soft">Cari Kamera</button>
                        </div>
                    </div>
                </div>
                <div class="row form-group">
                    <div>
                        <button type="button" id="startCamBtn" class="btn-pink">Mulai Kamera</button>
                    </div>
                    <div>
                        <button type="button" id="stopCamBtn" class="btn-danger-soft" disabled>Hentikan Kamera</button>
                    </div>
                </div>
                <div class="preview-box" id="webcamBox" style="display: none;">
                    <video id="webcamVideo" autoplay playsinline muted style="display: none;"></video>
                    <canvas id="webcamCanvas"></canvas>
                    <div class="stats-bar" id="webcamStats">Menunggu kamera...</div>
                </div>
            </div>

            <!-- TAB 4: TAMBAH FILE MODEL / VIDEO -->
            <div id="tabUpload" class="tab-panel">
                <div class="row">
                    <div style="flex: 1; background: var(--bg); padding: 1.1rem; border-radius: 0.8rem; border: 1px solid var(--border);">
                        <form id="modelForm">
                            <div class="form-group">
                                <label>Tambah File Model Baru</label>
                                <input type="file" id="onnxInput" accept=".onnx" required>
                            </div>
                            <button type="submit" id="uploadModelBtn" class="btn-purple">Unggah & Gunakan Model</button>
                        </form>
                    </div>
                    <div style="flex: 1; background: var(--bg); padding: 1.1rem; border-radius: 0.8rem; border: 1px solid var(--border);">
                        <form id="videoUploadForm">
                            <div class="form-group">
                                <label>Tambah File Video Baru</label>
                                <input type="file" id="videoFileInput" accept="video/*" required>
                            </div>
                            <button type="submit" id="uploadVideoBtn" class="btn-pink">Unggah Video</button>
                        </form>
                    </div>
                </div>
            </div>

            <!-- TAB 5: DATA HASIL -->
            <div id="tabJson" class="tab-panel">
                <pre id="jsonOutput">Memuat informasi...</pre>
            </div>
        </div>
    </div>

    <script>
        const boxColors = ['#ef4444', '#f97316', '#06b6d4', '#dc2626', '#22c55e', '#eab308'];
        let currentActiveModel = '';
        let availableModelsList = [];

        function cleanModelLabel(name) {
            if (!name) return '-';
            return name.replace(/\.onnx$/i, '');
        }

        // Navigasi Tab
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                const target = document.getElementById(btn.dataset.tab);
                if (target) target.classList.add('active');
            });
        });

        function renderModelCards(modelsDetail, activeModelName) {
            const container = document.getElementById('modelCardsContainer');
            const selectEl = document.getElementById('modelSelect');
            if (!container || !selectEl) return;
            container.innerHTML = '';
            selectEl.innerHTML = '';

            if (!modelsDetail || !modelsDetail.length) {
                container.innerHTML = '<div style="color: #fde047;">Belum ada model tersedia</div>';
                return;
            }

            const targetActive = (activeModelName || '').toLowerCase();

            modelsDetail.forEach(m => {
                const isActive = targetActive
                    ? (m.name.toLowerCase() === targetActive)
                    : Boolean(m.active);
                const resText = m.input_resolution
                    ? `${m.input_resolution.width}×${m.input_resolution.height}`
                    : 'Auto';
                const cleanTitle = m.display_name || cleanModelLabel(m.name);

                const card = document.createElement('div');
                card.className = 'model-card' + (isActive ? ' active' : '');
                card.innerHTML = `
                    <div class="model-card-header">
                        <span class="model-card-title">${cleanTitle}</span>
                        <span class="model-status-label">${isActive ? 'Digunakan' : 'Pilih'}</span>
                    </div>
                    <div class="model-card-meta">
                        Resolusi ${resText} • ${m.size_mb} MB
                    </div>
                `;
                card.addEventListener('click', () => {
                    switchModel(m.name);
                });
                container.appendChild(card);

                const opt = document.createElement('option');
                opt.value = m.name;
                opt.textContent = `${cleanTitle} (${resText}, ${m.size_mb} MB)`;
                if (isActive) opt.selected = true;
                selectEl.appendChild(opt);
            });
        }

        function updateActiveModelUI(modelName, modelsDetail) {
            if (!modelName) return;
            currentActiveModel = modelName;
            const friendlyName = cleanModelLabel(currentActiveModel);
            const badge = document.getElementById('statusBadge');
            if (badge) {
                badge.innerHTML = `Model Aktif: <b>${friendlyName}</b>`;
            }
            const videoBadge = document.getElementById('videoActiveModelBadge');
            if (videoBadge) {
                videoBadge.textContent = 'Model: ' + friendlyName;
            }
            if (modelsDetail && modelsDetail.length) {
                availableModelsList = modelsDetail;
            }
            renderModelCards(availableModelsList, currentActiveModel);
        }

        async function checkHealth(updateJson = true) {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                const badge = document.getElementById('statusBadge');

                if (data.model_loaded && data.model_info) {
                    updateActiveModelUI(data.model_info.model_path, data.models_detail || []);
                } else {
                    if (badge) {
                        badge.textContent = 'Belum ada model';
                    }
                    availableModelsList = data.models_detail || [];
                    renderModelCards(availableModelsList, '');
                }

                if (updateJson) {
                    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
                }
            } catch (err) {
                document.getElementById('jsonOutput').textContent = 'Gagal menghubungi server: ' + err;
            }
        }

        async function switchModel(modelName) {
            if (!modelName) return;
            const indicator = document.getElementById('switchingIndicator');
            const friendlyName = cleanModelLabel(modelName);

            // 1. Hentikan video secara otomatis saat ganti model
            stopVideoForModelSwitch(friendlyName);

            // 2. Hentikan kamera jika sedang berjalan
            if (typeof camRunning !== 'undefined' && camRunning) {
                document.getElementById('stopCamBtn')?.click();
            }

            // 3. Perbarui tampilan kartu & dropdown secara langsung
            updateActiveModelUI(modelName, availableModelsList);
            if (indicator) {
                indicator.textContent = `Mengganti ke ${friendlyName}...`;
            }

            try {
                const res = await fetch('/reload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: modelName })
                });
                const data = await res.json();
                if (data.success && data.model_info) {
                    const newActive = data.active_model || data.model_info.model_path;
                    updateActiveModelUI(newActive, data.models_detail || availableModelsList);
                    if (indicator) {
                        indicator.textContent = `Model diganti ke ${cleanModelLabel(newActive)}`;
                        setTimeout(() => { indicator.textContent = ''; }, 3000);
                    }
                    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
                } else {
                    if (indicator) {
                        indicator.textContent = `Gagal mengganti model`;
                    }
                }
            } catch (err) {
                if (indicator) {
                    indicator.textContent = 'Gagal menghubungi server';
                }
            }
        }

        document.getElementById('modelSelect').addEventListener('change', (e) => {
            switchModel(e.target.value);
        });

        document.getElementById('reloadModelBtn').addEventListener('click', () => {
            const sel = document.getElementById('modelSelect').value;
            switchModel(sel);
        });

        function updateHeroTelemetry(fps, ms, count) {
            // Flat UI: tidak menggunakan kartu telemetri tambahan
        }

        function drawBoxesOnCanvas(ctx, sourceEl, width, height, detections) {
            ctx.drawImage(sourceEl, 0, 0, width, height);
            for (const det of detections) {
                if (!det.bbox) continue;
                const { x1, y1, width: bw, height: bh } = det.bbox;
                const color = boxColors[det.class_id % boxColors.length] || '#6e62e5';
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(x1, y1, bw, bh);

                const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
                ctx.font = 'bold 14px sans-serif';
                const tw = ctx.measureText(label).width;
                const ty = Math.max(y1, 22);
                ctx.fillStyle = color;
                ctx.fillRect(x1, ty - 22, tw + 10, 22);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, x1 + 5, ty - 6);
            }
        }

        function setBannerMessage(type, statusMessage, modelName) {
            const bannerEl = document.getElementById('videoStatusBanner');
            const textEl = document.getElementById('videoStatusText');
            const modelEl = document.getElementById('videoActiveModelBadge');
            if (bannerEl) bannerEl.className = 'status-banner ' + type;
            if (textEl) textEl.textContent = statusMessage;
            if (modelEl && modelName) modelEl.textContent = 'Model: ' + cleanModelLabel(modelName);
        }

        function updateSceneStatusBanner(bannerEl, detections, modelName) {
            const names = detections.map(d => d.class_name);
            const hasDanger = names.includes('status_warning') || names.includes('begal') || names.includes('senjata');
            const hasSafe = names.includes('status_safe');

            const counts = {};
            detections.forEach(d => { counts[d.class_name] = (counts[d.class_name] || 0) + 1; });
            const summaryStr = Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(', ') || 'Tidak ada objek';

            if (hasDanger) {
                setBannerMessage('danger', `Peringatan Bahaya (${summaryStr})`, modelName);
            } else if (hasSafe) {
                setBannerMessage('safe', `Status Aman (${summaryStr})`, modelName);
            } else {
                setBannerMessage('neutral', `Hasil Deteksi: ${summaryStr}`, modelName);
            }
        }

        // =====================================================================
        // 1. LOGIKA PENGUJIAN VIDEO (FOLDER videos/)
        // =====================================================================
        const videoSelect = document.getElementById('videoSelect');
        const testVideo = document.getElementById('testVideo');
        const videoCanvas = document.getElementById('videoCanvas');
        const videoCtx = videoCanvas.getContext('2d');
        const videoCaptureCanvas = document.createElement('canvas');
        const videoCaptureCtx = videoCaptureCanvas.getContext('2d');
        const videoSeekSlider = document.getElementById('videoSeekSlider');
        const videoTimeLabel = document.getElementById('videoTimeLabel');
        const videoStatusBanner = document.getElementById('videoStatusBanner');

        let videoPlaying = false;
        let videoLoadedReady = false;
        let videoInferringSingle = false;
        let videoAbortController = null;

        function formatSec(sec) {
            if (!isFinite(sec) || sec < 0) return '00:00';
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        }

        async function loadVideosList(autoSelectFirst = true) {
            try {
                const res = await fetch('/videos');
                const data = await res.json();
                const videos = data.videos || [];
                videoSelect.innerHTML = '';

                if (!videos.length) {
                    videoSelect.innerHTML = '<option value="">-- Belum ada video di folder videos/ --</option>';
                    return;
                }

                videos.forEach((v, idx) => {
                    const opt = document.createElement('option');
                    opt.value = v.url;
                    opt.dataset.filename = v.filename;
                    opt.textContent = `${v.filename} (${v.width}×${v.height}, ${v.fps} FPS, ${v.duration_sec}s, ${v.size_mb} MB)`;
                    if (idx === 0 && autoSelectFirst) opt.selected = true;
                    videoSelect.appendChild(opt);
                });

                if (autoSelectFirst && videos[0]) {
                    selectVideoSource(videos[0].url);
                }
            } catch (err) {
                videoSelect.innerHTML = '<option value="">Gagal memuat daftar video</option>';
            }
        }

        function selectVideoSource(videoUrl) {
            if (!videoUrl) return;
            stopVideoCompletely('Video siap diuji. Klik "Mulai Deteksi Video (Play)" untuk memulai.');
            videoLoadedReady = false;
            testVideo.src = videoUrl;
            testVideo.playbackRate = parseFloat(document.getElementById('videoSpeedSelect').value || '1');
            testVideo.load();
        }

        testVideo.addEventListener('loadedmetadata', () => {
            const w = testVideo.videoWidth || 848;
            const h = testVideo.videoHeight || 478;
            videoCanvas.width = w;
            videoCanvas.height = h;
            videoCaptureCanvas.width = w;
            videoCaptureCanvas.height = h;
            videoSeekSlider.max = testVideo.duration || 100;
            videoSeekSlider.value = 0;
            videoTimeLabel.textContent = `00:00 / ${formatSec(testVideo.duration)}`;
        });

        testVideo.addEventListener('loadeddata', () => {
            videoLoadedReady = true;
            videoCtx.drawImage(testVideo, 0, 0, videoCanvas.width, videoCanvas.height);
        });

        testVideo.addEventListener('seeked', () => {
            if (!videoPlaying && videoLoadedReady && !videoInferringSingle) {
                videoCtx.drawImage(testVideo, 0, 0, videoCanvas.width, videoCanvas.height);
            }
        });

        testVideo.addEventListener('timeupdate', () => {
            if (!videoSeekSlider.matches(':active')) {
                videoSeekSlider.value = testVideo.currentTime;
            }
            videoTimeLabel.textContent = `${formatSec(testVideo.currentTime)} / ${formatSec(testVideo.duration)}`;
        });

        testVideo.addEventListener('ended', () => {
            if (document.getElementById('videoLoopCheck').checked && videoPlaying) {
                testVideo.currentTime = 0;
                testVideo.play();
            } else {
                pauseVideoLoop();
            }
        });

        videoSelect.addEventListener('change', (e) => {
            selectVideoSource(e.target.value);
        });

        document.getElementById('refreshVideosBtn').addEventListener('click', () => loadVideosList(true));

        document.getElementById('videoSpeedSelect').addEventListener('change', (e) => {
            testVideo.playbackRate = parseFloat(e.target.value || '1');
        });

        videoSeekSlider.addEventListener('input', (e) => {
            const targetTime = parseFloat(e.target.value || '0');
            testVideo.currentTime = targetTime;
            videoTimeLabel.textContent = `${formatSec(targetTime)} / ${formatSec(testVideo.duration)}`;
        });

        videoSeekSlider.addEventListener('change', () => {
            if (!videoPlaying && videoLoadedReady) {
                setTimeout(() => detectCurrentVideoFrame(true), 80);
            }
        });

        async function detectCurrentVideoFrame(allowWhenPaused = false) {
            if (!videoLoadedReady || videoInferringSingle) return;
            if (!videoPlaying && !allowWhenPaused) return;

            videoInferringSingle = true;
            const activeModelAtStart = currentActiveModel;
            const t0 = performance.now();
            videoCaptureCtx.drawImage(testVideo, 0, 0, videoCaptureCanvas.width, videoCaptureCanvas.height);

            const blob = await new Promise(r => videoCaptureCanvas.toBlob(r, 'image/jpeg', 0.85));
            if (!blob || (!videoPlaying && !allowWhenPaused)) {
                videoInferringSingle = false;
                return;
            }

            const formData = new FormData();
            formData.append('image', blob, 'video_frame.jpg');
            formData.append('conf', document.getElementById('confInput').value);
            formData.append('iou', document.getElementById('iouInput').value);
            if (activeModelAtStart) {
                formData.append('model', activeModelAtStart);
            }

            videoAbortController = new AbortController();

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                    signal: videoAbortController.signal
                });
                const data = await res.json();

                if ((!videoPlaying && !allowWhenPaused) || (activeModelAtStart && currentActiveModel !== activeModelAtStart)) {
                    return;
                }

                const detections = data.detections || [];
                drawBoxesOnCanvas(videoCtx, videoCaptureCanvas, videoCanvas.width, videoCanvas.height, detections);

                const activeMod = cleanModelLabel(data.model || currentActiveModel || '-');
                updateSceneStatusBanner(videoStatusBanner, detections, activeMod);

                const infMs = data.inference_time_ms || 0;
                document.getElementById('videoFpsStat').textContent =
                    `Model: ${activeMod} | Kecepatan: ${infMs} ms`;
                document.getElementById('videoCountStat').textContent =
                    `Objek Terdeteksi: ${detections.length}`;
                document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                if (err.name !== 'AbortError') {
                    document.getElementById('videoFpsStat').textContent = 'Gagal memproses video: ' + err;
                }
            } finally {
                videoInferringSingle = false;
                videoAbortController = null;
            }
        }

        async function loopVideoDetection() {
            while (videoPlaying) {
                if (testVideo.paused || testVideo.ended) {
                    if (document.getElementById('videoLoopCheck').checked && testVideo.ended && videoPlaying) {
                        testVideo.currentTime = 0;
                        await testVideo.play();
                    } else {
                        break;
                    }
                }
                await detectCurrentVideoFrame(false);
                await new Promise(r => setTimeout(r, 15));
            }
        }

        async function startVideoLoop() {
            if (!videoSelect.value) {
                alert('Pilih video terlebih dahulu!');
                return;
            }
            try {
                document.querySelector('[data-tab="tabVideo"]')?.click();
                testVideo.playbackRate = parseFloat(document.getElementById('videoSpeedSelect').value || '1');
                videoPlaying = true;
                document.getElementById('playVideoBtn').disabled = true;
                document.getElementById('pauseVideoBtn').disabled = false;
                document.getElementById('stopVideoBtn').disabled = false;
                await testVideo.play();
                loopVideoDetection();
            } catch (err) {
                videoPlaying = false;
                alert('Gagal memutar video: ' + err);
            }
        }

        function pauseVideoLoop() {
            videoPlaying = false;
            if (videoAbortController) {
                videoAbortController.abort();
                videoAbortController = null;
            }
            testVideo.pause();
            document.getElementById('playVideoBtn').disabled = false;
            document.getElementById('pauseVideoBtn').disabled = true;
        }

        function stopVideoCompletely(customMessage) {
            videoPlaying = false;
            videoInferringSingle = false;
            if (videoAbortController) {
                videoAbortController.abort();
                videoAbortController = null;
            }
            testVideo.pause();
            try {
                testVideo.currentTime = 0;
            } catch (_) {}
            videoSeekSlider.value = 0;
            videoTimeLabel.textContent = `00:00 / ${formatSec(testVideo.duration)}`;

            document.getElementById('playVideoBtn').disabled = false;
            document.getElementById('pauseVideoBtn').disabled = true;
            document.getElementById('stopVideoBtn').disabled = true;

            if (videoLoadedReady) {
                videoCtx.drawImage(testVideo, 0, 0, videoCanvas.width, videoCanvas.height);
            }
            document.getElementById('videoFpsStat').textContent = 'Kecepatan: 0 ms';
            document.getElementById('videoCountStat').textContent = 'Objek Terdeteksi: 0';
            setBannerMessage(
                'neutral',
                customMessage || 'Video dihentikan. Klik "Mulai Deteksi Video" untuk memulai.',
                currentActiveModel || '-'
            );
        }

        function stopVideoForModelSwitch(newModelName) {
            stopVideoCompletely(
                `Video dihentikan karena ganti model ke ${newModelName}. Klik "Mulai Deteksi Video" untuk menguji.`
            );
        }

        document.getElementById('playVideoBtn').addEventListener('click', startVideoLoop);
        document.getElementById('pauseVideoBtn').addEventListener('click', pauseVideoLoop);
        document.getElementById('stopVideoBtn').addEventListener('click', () => {
            stopVideoCompletely('Video dihentikan.');
        });
        document.getElementById('detectFrameBtn').addEventListener('click', () => {
            detectCurrentVideoFrame(true);
        });

        // =====================================================================
        // 2. LOGIKA PENGUJIAN GAMBAR
        // =====================================================================
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('imageInput');
            if (!fileInput.files.length) return;

            const btn = document.getElementById('submitBtn');
            btn.disabled = true;
            btn.textContent = 'Memproses...';

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('conf', document.getElementById('confInput').value);
            formData.append('iou', document.getElementById('iouInput').value);
            if (currentActiveModel) formData.append('model', currentActiveModel);
            formData.append('return_image', 'true');

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.annotated_image_base64) {
                    document.getElementById('imagePreview').innerHTML =
                        `<img src="${data.annotated_image_base64}" alt="Hasil Deteksi">`;
                    updateHeroTelemetry('-', data.inference_time_ms || 0, data.count || 0);
                    const displayData = { ...data };
                    displayData.annotated_image_base64 = '[Base64 Image Displayed]';
                    document.getElementById('jsonOutput').textContent = JSON.stringify(displayData, null, 2);
                } else {
                    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
                }
            } catch (err) {
                document.getElementById('jsonOutput').textContent = 'Error: ' + err;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Jalankan Deteksi Gambar';
            }
        });

        // =====================================================================
        // 3. LOGIKA LIVE WEBCAM
        // =====================================================================
        let camStream = null;
        let camRunning = false;
        const videoEl = document.getElementById('webcamVideo');
        const canvasEl = document.getElementById('webcamCanvas');
        const ctx = canvasEl.getContext('2d');
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        const cameraSelect = document.getElementById('cameraSelect');

        async function loadCameraDevices() {
            try {
                const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                tempStream.getTracks().forEach(t => t.stop());

                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(d => d.kind === 'videoinput');
                cameraSelect.innerHTML = '';

                let externalIdx = -1;
                videoDevices.forEach((dev, idx) => {
                    const opt = document.createElement('option');
                    opt.value = dev.deviceId;
                    const label = dev.label || `Kamera ${idx} (Index ${idx})`;
                    opt.textContent = `[Kamera ${idx}] ${label}`;
                    cameraSelect.appendChild(opt);

                    const low = label.toLowerCase();
                    if (externalIdx === -1 && (low.includes('usb') || low.includes('logi') || low.includes('fhd') || low.includes('external') || idx > 0)) {
                        externalIdx = idx;
                    }
                });

                if (externalIdx >= 0 && videoDevices[externalIdx]) {
                    cameraSelect.selectedIndex = externalIdx;
                }
            } catch (err) {
                console.warn('Tidak dapat membaca daftar kamera:', err);
            }
        }

        document.getElementById('detectCamBtn').addEventListener('click', loadCameraDevices);

        async function startSelectedCamera() {
            try {
                if (!cameraSelect.options.length || !cameraSelect.value) {
                    await loadCameraDevices();
                }
                if (camStream) {
                    camStream.getTracks().forEach(t => t.stop());
                }
                const selectedId = cameraSelect.value;
                const videoConstraint = selectedId
                    ? { deviceId: { exact: selectedId }, width: { ideal: 640 }, height: { ideal: 480 } }
                    : { width: { ideal: 640 }, height: { ideal: 480 } };

                camStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: false });
                videoEl.srcObject = camStream;
                await videoEl.play();
                canvasEl.width = videoEl.videoWidth || 640;
                canvasEl.height = videoEl.videoHeight || 480;
                captureCanvas.width = canvasEl.width;
                captureCanvas.height = canvasEl.height;

                document.getElementById('webcamBox').style.display = 'flex';
                document.getElementById('startCamBtn').disabled = true;
                document.getElementById('stopCamBtn').disabled = false;
                if (!camRunning) {
                    camRunning = true;
                    loopWebcam();
                }
            } catch (err) {
                alert('Gagal mengakses kamera: ' + err);
            }
        }

        document.getElementById('startCamBtn').addEventListener('click', startSelectedCamera);

        cameraSelect.addEventListener('change', () => {
            if (camRunning) {
                startSelectedCamera();
            }
        });

        document.getElementById('stopCamBtn').addEventListener('click', () => {
            camRunning = false;
            if (camStream) {
                camStream.getTracks().forEach(t => t.stop());
                camStream = null;
            }
            document.getElementById('startCamBtn').disabled = false;
            document.getElementById('stopCamBtn').disabled = true;
            document.getElementById('webcamStats').textContent = 'Webcam dihentikan.';
        });

        async function loopWebcam() {
            while (camRunning) {
                const t0 = performance.now();
                captureCtx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);

                const blob = await new Promise(r => captureCanvas.toBlob(r, 'image/jpeg', 0.8));
                const formData = new FormData();
                formData.append('image', blob, 'webcam.jpg');
                formData.append('conf', document.getElementById('confInput').value);
                formData.append('iou', document.getElementById('iouInput').value);
                if (currentActiveModel) formData.append('model', currentActiveModel);

                try {
                    const res = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (!camRunning) break;

                    const detections = data.detections || [];
                    drawBoxesOnCanvas(ctx, captureCanvas, canvasEl.width, canvasEl.height, detections);

                    const infMs = data.inference_time_ms || 0;
                    document.getElementById('webcamStats').textContent =
                        `Model: ${cleanModelLabel(data.model || currentActiveModel)} | Kecepatan: ${infMs} ms | Objek: ${detections.length}`;
                    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    document.getElementById('webcamStats').textContent = 'Gagal memproses kamera: ' + err;
                    await new Promise(r => setTimeout(r, 500));
                }
            }
        }

        // =====================================================================
        // 4. LOGIKA UPLOAD MODEL & VIDEO BARU
        // =====================================================================
        document.getElementById('modelForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const onnxInput = document.getElementById('onnxInput');
            if (!onnxInput.files.length) return;

            const btn = document.getElementById('uploadModelBtn');
            btn.disabled = true;
            btn.textContent = 'Mengunggah model...';

            const formData = new FormData();
            formData.append('model', onnxInput.files[0]);

            try {
                const res = await fetch('/upload-model', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
                await checkHealth(false);
            } catch (err) {
                document.getElementById('jsonOutput').textContent = 'Error upload model: ' + err;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Unggah & Gunakan Model';
            }
        });

        document.getElementById('videoUploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const vInput = document.getElementById('videoFileInput');
            if (!vInput.files.length) return;

            const btn = document.getElementById('uploadVideoBtn');
            btn.disabled = true;
            btn.textContent = 'Mengunggah video...';

            const formData = new FormData();
            formData.append('video', vInput.files[0]);

            try {
                const res = await fetch('/upload-video', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
                if (data.success) {
                    await loadVideosList(true);
                    document.querySelector('[data-tab="tabVideo"]').click();
                }
            } catch (err) {
                document.getElementById('jsonOutput').textContent = 'Error upload video: ' + err;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Unggah Video';
            }
        });

        // Inisialisasi awal
        checkHealth();
        loadVideosList(true);
    </script>
</body>
</html>
"""


@app.after_request
def add_no_cache_headers(response):
    """Mencegah browser melakukan cache pada halaman utama dan endpoint status model."""
    if request.path in ("/", "/health", "/info", "/models", "/reload", "/models/select", "/videos"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.route("/", methods=["GET"])
def index():
    """Halaman utama: menampilkan UI Web atau info JSON jika diminta via Accept header."""
    ensure_model_loaded()
    if request.accept_mimetypes.best == "application/json" and not request.accept_mimetypes.accept_html:
        return health()
    return render_template_string(INDEX_HTML)


@app.route("/health", methods=["GET"])
@app.route("/info", methods=["GET"])
def health():
    """Endpoint untuk mengecek kesehatan server, daftar model ONNX, dan daftar video."""
    model = ensure_model_loaded()
    available_models = [p.name for p in find_onnx_models()]
    models_detail = get_models_catalog()
    available_videos = [v["filename"] for v in get_videos_catalog()]

    if model is not None:
        return jsonify(
            {
                "success": True,
                "status": "ready",
                "model_loaded": True,
                "active_model": Path(model.model_path).name,
                "available_models": available_models,
                "models_detail": models_detail,
                "available_videos": available_videos,
                "model_info": model.get_model_info(),
            }
        )
    return (
        jsonify(
            {
                "success": False,
                "status": "waiting_for_model",
                "model_loaded": False,
                "active_model": None,
                "available_models": available_models,
                "models_detail": models_detail,
                "available_videos": available_videos,
                "message": detector_error,
                "instructions": "Letakkan file model YOLO (.onnx) Anda di subfolder 'models/' atau unggah melalui endpoint POST /upload-model.",
            }
        ),
        200,
    )


@app.route("/models", methods=["GET"])
def list_models():
    """Endpoint untuk melihat seluruh model .onnx yang tersedia beserta status aktifnya."""
    ensure_model_loaded()
    catalog = get_models_catalog()
    return jsonify(
        {
            "success": True,
            "count": len(catalog),
            "active_model": Path(detector.model_path).name if detector else None,
            "models": catalog,
        }
    )


@app.route("/reload", methods=["POST"])
@app.route("/models/select", methods=["POST"])
def reload_model():
    """Endpoint untuk mengganti atau memuat ulang model .onnx tanpa restart server."""
    data = request.get_json(silent=True) or {}
    model_name = request.args.get("model") or request.form.get("model") or data.get("model")
    force = str(request.args.get("force") or data.get("force") or "false").lower() in ("true", "1")
    ok, msg = load_or_reload_model(model_name, force_reload=force)
    status_code = 200 if ok else 400
    return (
        jsonify(
            {
                "success": ok,
                "message": msg,
                "active_model": Path(detector.model_path).name if detector else None,
                "available_models": [p.name for p in find_onnx_models()],
                "models_detail": get_models_catalog(),
                "model_info": detector.get_model_info() if detector else None,
            }
        ),
        status_code,
    )


@app.route("/upload-model", methods=["POST"])
def upload_model():
    """Endpoint untuk mengunggah file .onnx ke folder models/ dan langsung memuatnya."""
    if "model" not in request.files and "file" not in request.files:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "File model tidak ditemukan. Gunakan form-data key 'model' atau 'file'.",
                }
            ),
            400,
        )

    file_obj = request.files.get("model") or request.files.get("file")
    if not file_obj or not file_obj.filename:
        return jsonify({"success": False, "error": "Nama file kosong."}), 400

    filename = secure_filename(file_obj.filename)
    if not filename.lower().endswith(".onnx"):
        return (
            jsonify({"success": False, "error": "File harus berformat .onnx"}),
            400,
        )

    save_path = MODELS_DIR / filename
    file_obj.save(str(save_path))

    ok, msg = load_or_reload_model(str(save_path), force_reload=True)
    if not ok:
        return jsonify({"success": False, "error": msg}), 500

    return jsonify(
        {
            "success": True,
            "message": msg,
            "active_model": Path(detector.model_path).name if detector else None,
            "models_detail": get_models_catalog(),
            "model_info": detector.get_model_info() if detector else None,
        }
    )


@app.route("/videos", methods=["GET"])
def list_videos():
    """Endpoint untuk mendapatkan daftar video di dalam folder videos/."""
    videos = get_videos_catalog()
    return jsonify(
        {
            "success": True,
            "count": len(videos),
            "videos": videos,
        }
    )


@app.route("/videos/<path:filename>", methods=["GET"])
def serve_video(filename: str):
    """Endpoint untuk melakukan streaming file video dari folder videos/."""
    safe_name = Path(filename).name
    target = VIDEOS_DIR / safe_name
    if not target.is_file():
        return jsonify({"success": False, "error": f"Video '{safe_name}' tidak ditemukan di folder videos/."}), 404

    return send_file(str(target.resolve()), conditional=True)


@app.route("/upload-video", methods=["POST"])
def upload_video():
    """Endpoint untuk mengunggah file video baru ke dalam folder videos/."""
    if "video" not in request.files and "file" not in request.files:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "File video tidak ditemukan. Gunakan form-data key 'video' atau 'file'.",
                }
            ),
            400,
        )

    file_obj = request.files.get("video") or request.files.get("file")
    if not file_obj or not file_obj.filename:
        return jsonify({"success": False, "error": "Nama file video kosong."}), 400

    filename = secure_filename(file_obj.filename)
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Format video tidak didukung ({ext}). Gunakan: {', '.join(sorted(VIDEO_EXTENSIONS))}",
                }
            ),
            400,
        )

    save_path = VIDEOS_DIR / filename
    file_obj.save(str(save_path))
    meta = get_video_metadata(save_path)

    return jsonify(
        {
            "success": True,
            "message": f"Video '{filename}' berhasil diunggah ke folder videos/.",
            "video": meta,
            "videos": get_videos_catalog(),
        }
    )


@app.route("/predict", methods=["POST"])
@app.route("/detect", methods=["POST"])
def predict():
    """
    Endpoint utama untuk melakukan deteksi objek dengan model YOLO ONNX.
    Menerima gambar dari multipart/form-data ('image' atau 'file') atau JSON ('image' base64).
    Mendukung parameter opsional 'model' untuk memilih model tertentu.
    """
    conf, iou, return_image, req_model = get_inference_params()
    model = ensure_model_loaded(req_model)
    if model is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": detector_error
                    or "Model ONNX belum tersedia. Letakkan file .onnx di folder models/.",
                }
            ),
            503,
        )

    image_bgr, err = read_image_from_request()
    if image_bgr is None:
        return jsonify({"success": False, "error": err}), 400

    try:
        result = model.predict(image_bgr, conf_thres=conf, iou_thres=iou)
        response_data = {
            "success": True,
            "model": Path(model.model_path).name,
            "parameters": {"conf": conf, "iou": iou},
            "image_size": result["image_size"],
            "inference_time_ms": result["inference_time_ms"],
            "count": result["count"],
            "detections": result["detections"],
        }

        if return_image:
            annotated = model.draw_detections(image_bgr, result["detections"])
            response_data["annotated_image_base64"] = encode_image_base64(annotated)

        return jsonify(response_data), 200
    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Terjadi kesalahan saat inferensi: {str(e)}",
                }
            ),
            500,
        )


@app.route("/predict/image", methods=["POST"])
def predict_image():
    """
    Endpoint untuk melakukan deteksi dan langsung mengembalikan gambar JPEG
    yang sudah digambar bounding box serta label kelasnya.
    """
    conf, iou, _, req_model = get_inference_params()
    model = ensure_model_loaded(req_model)
    if model is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": detector_error
                    or "Model ONNX belum tersedia. Letakkan file .onnx di folder models/.",
                }
            ),
            503,
        )

    image_bgr, err = read_image_from_request()
    if image_bgr is None:
        return jsonify({"success": False, "error": err}), 400

    try:
        result = model.predict(image_bgr, conf_thres=conf, iou_thres=iou)
        annotated = model.draw_detections(image_bgr, result["detections"])
        success, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not success:
            return jsonify({"success": False, "error": "Gagal mengenkode gambar hasil."}), 500

        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="prediction.jpg",
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Terjadi kesalahan saat inferensi: {str(e)}",
                }
            ),
            500,
        )


# =====================================================================
# Modul Continuous Live Streaming (MJPEG) untuk Webcam USB & YOLO AI
# =====================================================================

BEGAL_COLORS = [
    (0, 0, 255),      # 0: begal (Merah)
    (255, 165, 0),    # 1: korban (Biru/Oranye)
    (255, 200, 0),    # 2: motor (Cyan)
    (0, 0, 200),      # 3: senjata (Merah Tua)
    (0, 200, 0),      # 4: status_safe (Hijau)
    (0, 215, 255),    # 5: status_warning (Kuning/Emas)
]


def get_begal_color(class_id: int) -> Tuple[int, int, int]:
    if 0 <= class_id < len(BEGAL_COLORS):
        return BEGAL_COLORS[class_id]
    np.random.seed(class_id + 42)
    return tuple(int(c) for c in np.random.randint(50, 240, size=3))  # type: ignore


def scan_available_cameras(max_index: int = 4) -> List[int]:
    """Mencari index kamera yang terhubung (0 = default/laptop, 1 = webcam USB eksternal)."""
    available: List[int] = []
    for idx in range(max_index):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        except Exception:
            pass
    return available


class VideoCameraStream:
    """
    Background Threaded Video Capture untuk Webcam USB.
    Menghilangkan buffer lag OpenCV sehingga streaming web selalu real-time (< 100ms).
    Mendukung banyak viewer secara simultan tanpa overhead membuka device berkali-kali.
    """

    def __init__(self, source: int | str = 0):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0.0
        self.actual_fps = 0.0
        self.active_viewers = 0
        self.frame_count = 0
        self.width = 640
        self.height = 480
        self.is_connected = False
        self.error_msg: Optional[str] = None
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_threat: Dict[str, Any] = {
            "has_threat": False,
            "primary_label": "normal",
            "max_conf": 0.0,
            "begal_count": 0,
            "senjata_count": 0,
            "total_detections": 0,
            "timestamp": 0.0,
            "detections": [],
        }

    def _open_capture(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        try:
            if isinstance(src, int) and os.name == "nt":
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(src)
            else:
                cap = cv2.VideoCapture(src)

            if cap.isOpened():
                self.cap = cap
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                self.is_connected = True
                self.error_msg = None
                return True
            else:
                self.cap = None
                self.is_connected = False
                self.error_msg = f"Tidak dapat membuka webcam pada source: {self.source}"
                return False
        except Exception as e:
            self.cap = None
            self.is_connected = False
            self.error_msg = str(e)
            return False

    def start(self):
        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()

    def _update_loop(self):
        fps_timer = time.perf_counter()
        fps_frames = 0

        while self.is_running:
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                if not self._open_capture():
                    # Buat placeholder frame saat kamera belum terhubung
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        dummy,
                        f"Webcam USB [{self.source}] Tidak Terdeteksi",
                        (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        dummy,
                        "Pastikan webcam terpasang pada port USB",
                        (50, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        1,
                    )
                    with self.lock:
                        self.latest_frame = dummy
                    time.sleep(1.0)
                    continue

            ret, frame = self.cap.read()
            now = time.perf_counter()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = now
                    self.frame_count += 1
                fps_frames += 1
                if now - fps_timer >= 1.0:
                    self.actual_fps = round(fps_frames / (now - fps_timer), 1)
                    fps_frames = 0
                    fps_timer = now
            else:
                self.is_connected = False
                time.sleep(0.05)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def change_source(self, new_source: int | str):
        with self.lock:
            self.source = new_source
            self.is_connected = False
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False


camera_streamer: Optional[VideoCameraStream] = None
camera_streamer_lock = threading.Lock()


def get_camera_streamer(source: Optional[int | str] = None) -> VideoCameraStream:
    """Mengambil atau menginisialisasi singleton VideoCameraStream."""
    global camera_streamer
    with camera_streamer_lock:
        if camera_streamer is None:
            default_source = os.getenv("CAMERA_SOURCE", "0")
            if default_source.isdigit():
                default_source = int(default_source)
            initial_source = source if source is not None else default_source
            camera_streamer = VideoCameraStream(source=initial_source)
            camera_streamer.start()
        elif source is not None and str(camera_streamer.source) != str(source):
            camera_streamer.change_source(source)
        return camera_streamer


def draw_begal_annotations(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    model_name: str = "",
    fps: float = 0.0,
    infer_ms: float = 0.0,
    show_hud: bool = True,
) -> np.ndarray:
    """Menggambar bounding box BegalVision dan HUD bar di atas frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Gambar bounding box dan label deteksi
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1 = int(round(bbox["x1"])), int(round(bbox["y1"]))
        x2, y2 = int(round(bbox["x2"])), int(round(bbox["y2"]))
        cls_id = int(det.get("class_id", 0))
        cls_name = det.get("class_name", f"class_{cls_id}")
        conf = float(det.get("confidence", 0.0))

        color = get_begal_color(cls_id)
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_y = max(y1, th + 8)
        cv2.rectangle(annotated, (x1, text_y - th - 6), (x1 + tw + 6, text_y + 2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 2. Gambar HUD Header (Status Bar CCTV BegalVision) jika diaktifkan
    if show_hud:
        cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
        cv2.circle(annotated, (18, 18), 6, (0, 0, 255), -1)  # Titik merah LIVE
        hud_text = (
            f"LIVE CCTV | BegalVision AI | Model: {model_name or 'YOLO'} | "
            f"FPS: {fps:.1f} | Latensi: {infer_ms:.0f}ms | Objek: {len(detections)}"
        )
        cv2.putText(
            annotated,
            hud_text,
            (32, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    return annotated


def generate_mjpeg_stream(
    with_ai: bool = True,
    model_name: Optional[str] = None,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    source: Optional[int | str] = None,
    quality: int = 80,
    max_fps: int = 30,
):
    """Generator stream MJPEG multipart/x-mixed-replace."""
    streamer = get_camera_streamer(source)
    with streamer.lock:
        streamer.active_viewers += 1

    selected_model = ensure_model_loaded(model_name)
    display_model_name = (
        Path(selected_model.model_path).name if selected_model else "No Model"
    )

    frame_interval = 1.0 / max_fps if max_fps > 0 else 0.033
    last_yield_time = 0.0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - last_yield_time
            if elapsed < frame_interval:
                time.sleep(max(0.001, frame_interval - elapsed))

            frame = streamer.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            last_yield_time = time.perf_counter()

            if with_ai and selected_model is not None and streamer.is_connected:
                t0 = time.perf_counter()
                try:
                    result = selected_model.predict(frame, conf_thres=conf, iou_thres=iou)
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    detections = result.get("detections", [])

                    # Ekstrak ancaman Begal/Senjata untuk disinkronkan ke Web & Fuzzy Logic
                    for d in detections:
                        c_name = str(d.get("class_name") or d.get("class") or "").strip()
                        d["class_name"] = c_name
                        d["class"] = c_name  # Sinkronkan agar kompatibel dengan key 'class' dan 'class_name'

                    begal_dets = [
                        d for d in detections 
                        if "begal" in d.get("class_name", "").lower()
                    ]
                    senjata_dets = [
                        d for d in detections 
                        if any(k in d.get("class_name", "").lower() for k in ["senjata", "clurit", "pisau", "weapon", "knife", "celurit", "parang", "pedang"])
                    ]
                    korban_dets = [
                        d for d in detections 
                        if any(k in d.get("class_name", "").lower() for k in ["korban", "person", "orang", "mencurigakan"])
                    ]

                    # OPSI B (Proteksi Ketat):
                    # BAHAYA (Sirine Buzzer ON) HANYA jika terdeteksi KEDUANYA: Begal DAN Senjata!
                    # Jika hanya salah satu (hanya begal tanpa sajam, atau hanya sajam), masuk status MENCURIGAKAN (Waspada, Buzzer OFF).
                    has_begal = len(begal_dets) > 0
                    has_senjata = len(senjata_dets) > 0
                    has_both = (has_begal and has_senjata)
                    primary_label = "normal"
                    max_conf = 0.0

                    if has_both:
                        primary_label = "begal_bersenjata"
                        max_conf = max(
                            max(d.get("confidence", 0.0) for d in begal_dets),
                            max(d.get("confidence", 0.0) for d in senjata_dets)
                        )
                    elif has_begal:
                        primary_label = "begal"  # Hanya begal tanpa senjata -> Waspada
                        max_conf = max(d.get("confidence", 0.0) for d in begal_dets)
                    elif has_senjata:
                        primary_label = "senjata"  # Hanya senjata tanpa begal -> Waspada
                        max_conf = max(d.get("confidence", 0.0) for d in senjata_dets)
                    elif len(korban_dets) > 0:
                        primary_label = "mencurigakan" if len(korban_dets) > 1 else "normal"
                        max_conf = max(d.get("confidence", 0.0) for d in korban_dets)

                    now_ts = time.time()
                    if has_both:
                        streamer.last_threat_ts = now_ts
                        streamer.persisted_threat = {
                            "has_threat": True,
                            "has_both": True,
                            "has_begal": True,
                            "has_senjata": True,
                            "primary_label": primary_label,
                            "max_conf": round(float(max_conf), 3),
                            "begal_count": len(begal_dets),
                            "senjata_count": len(senjata_dets),
                            "total_detections": len(detections),
                            "timestamp": now_ts,
                            "detections": detections,
                        }
                        streamer.latest_threat = streamer.persisted_threat
                    else:
                        # Pertahankan status bahaya selama 2.5 detik untuk mitigasi jitter frame kamera
                        last_ts = getattr(streamer, "last_threat_ts", 0.0)
                        if (now_ts - last_ts) < 2.5 and getattr(streamer, "persisted_threat", None):
                            streamer.latest_threat = streamer.persisted_threat
                        else:
                            streamer.latest_threat = {
                                "has_threat": False,
                                "has_both": False,
                                "has_begal": has_begal,
                                "has_senjata": has_senjata,
                                "primary_label": primary_label,
                                "max_conf": round(float(max_conf), 3),
                                "begal_count": len(begal_dets),
                                "senjata_count": len(senjata_dets),
                                "total_detections": len(detections),
                                "timestamp": now_ts,
                                "detections": detections,
                            }

                    streamer.latest_detections = detections

                    output_frame = draw_begal_annotations(
                        frame,
                        detections,
                        model_name=display_model_name,
                        fps=streamer.actual_fps,
                        infer_ms=infer_ms,
                    )
                except Exception as e:
                    output_frame = frame
            else:
                output_frame = frame

            ret, buffer = cv2.imencode(
                ".jpg", output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                + frame_bytes + b"\r\n"
            )
    finally:
        with streamer.lock:
            streamer.active_viewers = max(0, streamer.active_viewers - 1)


@app.route("/video_feed", methods=["GET"])
def video_feed():
    """
    Endpoint Continuous Live Stream MJPEG dengan anotasi YOLO AI.
    Dapat disematkan langsung di Website Laravel menggunakan:
    <img src="http://flask-host:5000/video_feed" />
    
    Query Params:
    - model: nama file .onnx (opsional)
    - conf: confidence threshold (default: 0.25)
    - iou: iou threshold (default: 0.45)
    - source: index kamera (default: 0 atau env CAMERA_SOURCE)
    - quality: kualitas JPEG 30-100 (default: 80)
    - fps: batasan FPS maksimal (default: 30)
    """
    conf, iou, _, req_model = get_inference_params()
    source_arg = request.args.get("source")
    quality = int(request.args.get("quality", 80))
    quality = max(30, min(100, quality))
    max_fps = int(request.args.get("fps", 30))
    max_fps = max(5, min(60, max_fps))

    source_val = None
    if source_arg is not None:
        source_val = int(source_arg) if source_arg.isdigit() else source_arg

    return Response(
        generate_mjpeg_stream(
            with_ai=True,
            model_name=req_model,
            conf=conf,
            iou=iou,
            source=source_val,
            quality=quality,
            max_fps=max_fps,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/video_feed_raw", methods=["GET"])
def video_feed_raw():
    """
    Endpoint Continuous Live Stream MJPEG Webcam USB Murni (tanpa inferensi AI).
    Sangat hemat bandwidth & CPU.
    """
    source_arg = request.args.get("source")
    quality = int(request.args.get("quality", 80))
    max_fps = int(request.args.get("fps", 30))
    source_val = int(source_arg) if (source_arg and source_arg.isdigit()) else source_arg

    return Response(
        generate_mjpeg_stream(
            with_ai=False,
            source=source_val,
            quality=quality,
            max_fps=max_fps,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/camera/status", methods=["GET"])
def camera_status():
    """Mengembalikan status koneksi webcam USB, resolusi, FPS, dan kamera yang tersedia beserta deteksi ancaman AI."""
    streamer = get_camera_streamer()
    available = scan_available_cameras(max_index=4)
    latest_threat = getattr(streamer, "latest_threat", {
        "has_threat": False,
        "primary_label": "normal",
        "max_conf": 0.0,
        "detections": [],
    })
    return jsonify({
        "success": True,
        "is_connected": streamer.is_connected,
        "current_source": streamer.source,
        "resolution": {"width": streamer.width, "height": streamer.height},
        "fps": streamer.actual_fps,
        "active_viewers": streamer.active_viewers,
        "available_cameras": available,
        "latest_threat": latest_threat,
        "latest_detections": getattr(streamer, "latest_detections", []),
        "error": streamer.error_msg,
    })


@app.route("/camera/detections", methods=["GET"])
def camera_detections():
    """Endpoint real-time untuk mengambil deteksi BegalVision AI terkini."""
    streamer = get_camera_streamer()
    latest_threat = getattr(streamer, "latest_threat", {
        "has_threat": False,
        "primary_label": "normal",
        "max_conf": 0.0,
        "detections": [],
    })
    return jsonify({
        "success": True,
        "threat": latest_threat,
        "detections": getattr(streamer, "latest_detections", []),
        "timestamp": time.time(),
    })


@app.route("/camera/source", methods=["POST"])
def change_camera_source():
    """Mengubah index webcam USB aktif secara dinamis."""
    data = request.get_json(silent=True) or request.form.to_dict()
    new_src = data.get("source")
    if new_src is None:
        return jsonify({"success": False, "error": "Parameter 'source' wajib diisi."}), 400

    if str(new_src).isdigit():
        new_src = int(new_src)

    streamer = get_camera_streamer()
    streamer.change_source(new_src)
    return jsonify({
        "success": True,
        "message": f"Kamera berhasil dialihkan ke source: {new_src}",
        "current_source": streamer.source,
    })


if __name__ == "__main__":
    preload_all_models()
    ok, msg = load_or_reload_model()
    available_models = [p.name for p in find_onnx_models()]
    available_videos = [p.name for p in find_video_files()]
    print("=" * 65)
    print(" YOLO Multi-Model ONNX Flask API Server")
    print("=" * 65)
    print(f" Status Model   : {msg}")
    print(f" Daftar Model   : {', '.join(available_models) if available_models else '-'}")
    print(f" Daftar Video   : {', '.join(available_videos) if available_videos else '-'}")
    port = int(os.getenv("PORT", "5000"))
    print(f" URL Website UI : http://127.0.0.1:{port}")
    print(f" Live Stream AI : GET  http://127.0.0.1:{port}/video_feed")
    print(f" Live Stream Raw: GET  http://127.0.0.1:{port}/video_feed_raw")
    print(f" Status Kamera  : GET  http://127.0.0.1:{port}/camera/status")
    print(f" Endpoint API   : POST http://127.0.0.1:{port}/predict")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=True)
