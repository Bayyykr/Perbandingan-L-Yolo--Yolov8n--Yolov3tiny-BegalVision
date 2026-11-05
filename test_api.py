import json
import sys
import cv2
import numpy as np
import requests


def create_dummy_image() -> bytes:
    """Membuat gambar dummy 640x640 untuk pengujian endpoint API."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    cv2.rectangle(img, (120, 120), (420, 480), (50, 120, 220), -1)
    cv2.putText(
        img,
        "Test Image",
        (160, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_server(base_url: str = "http://127.0.0.1:5000", image_path: str | None = None):
    print(f"Menguji server di {base_url} ...\n")

    # 1. Cek /health
    res_health = requests.get(f"{base_url}/health", timeout=10)
    print(f"[GET /health] Status Code: {res_health.status_code}")
    print(json.dumps(res_health.json(), indent=2, ensure_ascii=False))
    print("-" * 50)

    # 2. Uji POST /predict
    if image_path:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        filename = image_path
    else:
        img_bytes = create_dummy_image()
        filename = "dummy_test.jpg"

    files = {"image": (filename, img_bytes, "image/jpeg")}
    data = {"conf": "0.25", "iou": "0.45"}

    res_pred = requests.post(f"{base_url}/predict", files=files, data=data, timeout=30)
    print(f"[POST /predict] Status Code: {res_pred.status_code}")
    print(json.dumps(res_pred.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_server(image_path=img_arg)
