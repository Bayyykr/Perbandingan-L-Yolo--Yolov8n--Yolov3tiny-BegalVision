import argparse
import time
import cv2
import numpy as np
import requests


# Warna berbeda untuk tiap class_id (BGR)
COLORS = [
    (0, 0, 255),      # 0: begal (Merah)
    (255, 165, 0),    # 1: korban (Biru/Oranye)
    (255, 200, 0),    # 2: motor (Cyan)
    (0, 0, 200),      # 3: senjata (Merah Tua)
    (0, 200, 0),      # 4: status_safe (Hijau)
    (0, 215, 255),    # 5: status_warning (Kuning/Emas)
]


def get_color(class_id: int):
    if 0 <= class_id < len(COLORS):
        return COLORS[class_id]
    np.random.seed(class_id + 42)
    return tuple(int(c) for c in np.random.randint(50, 240, size=3))


def scan_available_cameras(max_index: int = 4) -> list[int]:
    """Mencari index kamera yang terhubung (0 = biasanya laptop, 1 = webcam USB eksternal)."""
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def open_camera(source: str | int):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
        return cap
    return cv2.VideoCapture(source)


def draw_boxes(
    frame: np.ndarray,
    detections: list,
    fps: float,
    api_ms: float,
    cam_label: str,
    model_name: str = "-",
) -> np.ndarray:
    annotated = frame.copy()

    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue

        x1, y1 = int(round(bbox["x1"])), int(round(bbox["y1"]))
        x2, y2 = int(round(bbox["x2"])), int(round(bbox["y2"]))
        cls_id = int(det.get("class_id", 0))
        cls_name = det.get("class_name", f"class_{cls_id}")
        conf = float(det.get("confidence", 0.0))

        color = get_color(cls_id)
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, th + 10)
        cv2.rectangle(annotated, (x1, text_y - th - 8), (x1 + tw + 8, text_y), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Panel info di pojok kiri atas
    info_text = f"[{cam_label}] Model: {model_name} | FPS: {fps:.1f} | API: {api_ms:.0f}ms | Objek: {len(detections)}"
    help_text = "Tekan 'M' = Ganti Model | 'C' = Ganti Kamera | 'Q' = Keluar"
    cv2.rectangle(annotated, (8, 8), (680, 66), (15, 23, 42), -1)
    cv2.putText(
        annotated,
        info_text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (56, 189, 248),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        help_text,
        (16, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (203, 213, 225),
        1,
        cv2.LINE_AA,
    )
    return annotated


def run_webcam_simulation(
    api_url: str = "http://127.0.0.1:5000/predict",
    source: str | int | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    model: str | None = None,
):
    available_cams = scan_available_cameras()
    model_candidates = ["lyolo.onnx", "yolov3tiny.onnx", "yolov8nano.onnx"]
    current_model = model

    # Jika user tidak menentukan --source, prioritaskan index 1 (Webcam USB Eksternal) jika ada
    if source is None:
        if 1 in available_cams:
            current_source: str | int = 1
        elif available_cams:
            current_source = available_cams[0]
        else:
            current_source = 1
    else:
        current_source = source

    print("=" * 65)
    print(" Simulasi Webcam / Video YOLO Multi-Model ONNX via Flask API")
    print("=" * 65)
    print(f" Kamera Terdeteksi : {available_cams if available_cams else 'Cek manual (0, 1, 2)'}")
    print(f" Source Aktif      : {current_source}")
    print(f" Endpoint API      : {api_url}")
    print(f" Confidence / IoU  : {conf} / {iou}")
    print(" Kontrol Keyboard  : Tekan 'M' ganti model | 'C' pindah kamera | 'Q' keluar")
    print("=" * 65)

    cap = open_camera(current_source)
    if not cap.isOpened() and isinstance(current_source, int) and current_source != 0:
        print(f"[INFO] Kamera index {current_source} tidak terbuka, mencoba index 0...")
        current_source = 0
        cap = open_camera(current_source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka kamera/video pada source: {current_source}")
        return

    session = requests.Session()
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(current_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                print("Stream kamera/video selesai atau terputus.")
                break

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue

        detections = []
        api_ms = 0.0
        active_model_str = current_model or "auto"

        try:
            files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            data = {"conf": str(conf), "iou": str(iou)}
            if current_model:
                data["model"] = current_model
            resp = session.post(api_url, files=files, data=data, timeout=5)
            if resp.status_code == 200:
                res_json = resp.json()
                detections = res_json.get("detections", [])
                api_ms = float(res_json.get("inference_time_ms", 0.0))
                active_model_str = res_json.get("model", active_model_str)
            else:
                cv2.putText(
                    frame,
                    f"API Error {resp.status_code}",
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
        except requests.exceptions.ConnectionError:
            cv2.putText(
                frame,
                "Server Flask belum nyala! Jalankan: python app.py",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
        except Exception as e:
            cv2.putText(
                frame,
                f"Error: {str(e)[:50]}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cam_label = f"Cam {current_source}" if isinstance(current_source, int) else "Video"
        annotated_frame = draw_boxes(frame, detections, fps, api_ms, cam_label, active_model_str)
        cv2.imshow("Simulasi Webcam / Video - YOLO Multi-Model ONNX API", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("m"), ord("M")):
            curr_idx = model_candidates.index(active_model_str) if active_model_str in model_candidates else -1
            current_model = model_candidates[(curr_idx + 1) % len(model_candidates)]
            print(f"[INFO] Mengganti model ke: {current_model}")
        elif key in (ord("c"), ord("C")) and isinstance(current_source, int):
            # Ganti ke index kamera berikutnya
            candidates = available_cams if len(available_cams) > 1 else [0, 1, 2]
            next_idx = (candidates.index(current_source) + 1) % len(candidates) if current_source in candidates else 0
            new_cam = candidates[next_idx]
            print(f"[INFO] Mengganti kamera dari Index {current_source} -> Index {new_cam} ...")
            cap.release()
            new_cap = open_camera(new_cam)
            if new_cap.isOpened():
                cap = new_cap
                current_source = new_cam
            else:
                print(f"[WARN] Kamera Index {new_cam} tidak tersedia, kembali ke Index {current_source}.")
                cap = open_camera(current_source)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulasi Webcam / Video untuk menguji Flask API YOLO ONNX")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:5000/predict",
        help="URL endpoint Flask API (default: http://127.0.0.1:5000/predict)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Index kamera: 1 untuk Webcam USB Eksternal, 0 untuk Webcam Laptop, atau path video misal videos/Asset_Video.mp4",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nama model ONNX (lyolo.onnx, yolov3tiny.onnx, yolov8nano.onnx)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU NMS threshold (default: 0.45)",
    )
    args = parser.parse_args()

    if args.source is None:
        src = None
    elif args.source.isdigit():
        src = int(args.source)
    else:
        src = args.source

    run_webcam_simulation(api_url=args.url, source=src, conf=args.conf, iou=args.iou, model=args.model)
