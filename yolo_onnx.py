import ast
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8ONNX:
    """
    Wrapper inference untuk model YOLOv8 format ONNX menggunakan ONNX Runtime.
    Mendukung preprocessing letterbox, pembacaan metadata class names otomatis,
    Non-Maximum Suppression (NMS), dan visualisasi bounding box.
    """

    def __init__(
        self,
        model_path: str,
        classes_path: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"File model ONNX tidak ditemukan: {self.model_path}")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Pilih Execution Provider terbaik yang tersedia (CUDA jika ada, fallback ke CPU)
        available_providers = ort.get_available_providers()
        preferred_providers = [
            p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers
        ]
        if not preferred_providers:
            preferred_providers = available_providers

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path, sess_options=sess_options, providers=preferred_providers
        )
        self.providers = self.session.get_providers()

        # Ambil detail input model
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_type = model_inputs[0].type

        # Ambil detail output model
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]
        self.output_shapes = [out.shape for out in model_outputs]

        # Baca metadata dari model ONNX (Ultralytics menyimpan 'names', 'stride', 'imgsz', 'task')
        self.metadata: Dict[str, Any] = {}
        try:
            meta = self.session.get_modelmeta()
            if meta and meta.custom_metadata_map:
                self.metadata = dict(meta.custom_metadata_map)
        except Exception:
            self.metadata = {}

        # Tentukan ukuran input (height, width)
        self.input_height, self.input_width = self._resolve_input_size()

        # Muat daftar nama kelas (class names)
        self.class_names: Dict[int, str] = self._load_class_names(classes_path)

    def _resolve_input_size(self) -> Tuple[int, int]:
        """Menentukan resolusi input (height, width) dari shape tensor atau metadata."""
        h, w = 640, 640
        if len(self.input_shape) == 4:
            raw_h, raw_w = self.input_shape[2], self.input_shape[3]
            if isinstance(raw_h, int) and raw_h > 0 and isinstance(raw_w, int) and raw_w > 0:
                return int(raw_h), int(raw_w)

        if "imgsz" in self.metadata:
            try:
                imgsz = ast.literal_eval(self.metadata["imgsz"])
                if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
                    return int(imgsz[0]), int(imgsz[1])
                elif isinstance(imgsz, int):
                    return int(imgsz), int(imgsz)
            except Exception:
                pass
        return h, w

    def _load_class_names(self, classes_path: Optional[str] = None) -> Dict[int, str]:
        """Memuat nama-nama class dari file eksternal atau metadata model ONNX."""
        # 1. Jika ada file classes.txt / labels.txt
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                return {i: name for i, name in enumerate(lines)}

        # 2. Cek metadata 'names' bawaan hasil export Ultralytics YOLOv8
        if "names" in self.metadata:
            try:
                names_parsed = ast.literal_eval(self.metadata["names"])
                if isinstance(names_parsed, dict):
                    return {int(k): str(v) for k, v in names_parsed.items()}
                elif isinstance(names_parsed, (list, tuple)):
                    return {i: str(v) for i, v in enumerate(names_parsed)}
            except Exception:
                pass

        return {}

    def get_class_name(self, class_id: int) -> str:
        """Mengembalikan nama class berdasarkan ID."""
        return self.class_names.get(int(class_id), f"class_{int(class_id)}")

    def letterbox(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize gambar dengan menjaga rasio aspek menggunakan padding (standar YOLOv8).
        Mengembalikan: (resized_padded_img, ratio, (pad_w, pad_h))
        """
        shape = image.shape[:2]  # [height, width] saat ini
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Rasio skala (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Ukuran baru sebelum padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # bagi padding ke 2 sisi
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return image, r, (dw, dh)

    def preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Melakukan preprocessing gambar BGR menjadi input tensor ONNX."""
        img_padded, ratio, pad = self.letterbox(
            image_bgr, new_shape=(self.input_height, self.input_width)
        )
        # BGR ke RGB
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        # HWC ke CHW
        img_chw = img_rgb.transpose((2, 0, 1))
        img_chw = np.ascontiguousarray(img_chw)

        if "float16" in self.input_type:
            blob = img_chw.astype(np.float16) / 255.0
        else:
            blob = img_chw.astype(np.float32) / 255.0

        # Tambahkan dimensi batch: (1, 3, H, W)
        blob = np.expand_dims(blob, axis=0)
        return blob, ratio, pad

    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
        conf_thres: float,
        iou_thres: float,
    ) -> List[Dict[str, Any]]:
        """
        Memproses output tensor YOLOv8 menjadi daftar deteksi objek.
        Mendukung output standar YOLOv8 (1, 4+nc, N) maupun output dengan built-in NMS (1, N, 6).
        """
        pred = outputs[0]
        orig_h, orig_w = orig_shape[:2]
        pad_w, pad_h = pad

        # Jika model klasifikasi (1, num_classes)
        if pred.ndim == 2:
            scores = pred[0]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < conf_thres:
                return []
            return [
                {
                    "class_id": class_id,
                    "class_name": self.get_class_name(class_id),
                    "confidence": round(conf, 4),
                    "bbox": None,
                }
            ]

        # Hilangkan dimensi batch -> (D1, D2)
        predictions = pred[0]

        # Cek apakah model diekspor dengan nms=True: shape (max_det, 6) -> [x1, y1, x2, y2, conf, cls]
        is_end2end_nms = (
            predictions.shape[-1] == 6
            and predictions.shape[0] <= 1000
            and len(outputs) == 1
        )

        detections: List[Dict[str, Any]] = []

        if is_end2end_nms:
            for row in predictions:
                x1, y1, x2, y2, score, cls_id = row
                if score < conf_thres:
                    continue
                # Kembalikan ke skala gambar asli
                x1 = (x1 - pad_w) / ratio
                y1 = (y1 - pad_h) / ratio
                x2 = (x2 - pad_w) / ratio
                y2 = (y2 - pad_h) / ratio

                x1 = float(np.clip(x1, 0, orig_w))
                y1 = float(np.clip(y1, 0, orig_h))
                x2 = float(np.clip(x2, 0, orig_w))
                y2 = float(np.clip(y2, 0, orig_h))

                detections.append(
                    self._format_detection(
                        x1, y1, x2, y2, float(score), int(cls_id), orig_w, orig_h
                    )
                )
            return detections

        # Output standar YOLOv8: shape (4 + nc [+ nm], num_anchors), misal (84, 8400)
        # Transpose ke (num_anchors, 4 + nc [+ nm])
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        # Jika model segmentasi (memiliki 2 output: deteksi + proto mask)
        num_masks = 0
        if len(outputs) > 1 and outputs[1].ndim == 4:
            num_masks = int(outputs[1].shape[1])

        num_classes = predictions.shape[1] - 4 - num_masks
        if num_classes <= 0:
            return []

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4 : 4 + num_classes]

        # Ambil class dengan skor tertinggi untuk setiap anchor
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter berdasarkan confidence threshold
        mask = confidences >= conf_thres
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # Konversi [x_center, y_center, width, height] ke [x1, y1, x2, y2] pada koordinat input
        x_center = boxes_xywh[:, 0]
        y_center = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # Kembalikan ke skala koordinat gambar asli
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Siapkan box [x, y, width, height] dengan class offset untuk class-aware NMS
        max_wh = max(orig_w, orig_h) * 2
        nms_boxes = []
        for i in range(len(x1)):
            offset = int(class_ids[i]) * max_wh
            bw = float(max(0.0, x2[i] - x1[i]))
            bh = float(max(0.0, y2[i] - y1[i]))
            nms_boxes.append([float(x1[i]) + offset, float(y1[i]) + offset, bw, bh])

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=confidences.tolist(),
            score_threshold=float(conf_thres),
            nms_threshold=float(iou_thres),
        )

        if len(indices) == 0:
            return []

        for idx in np.array(indices).flatten():
            detections.append(
                self._format_detection(
                    float(x1[idx]),
                    float(y1[idx]),
                    float(x2[idx]),
                    float(y2[idx]),
                    float(confidences[idx]),
                    int(class_ids[idx]),
                    orig_w,
                    orig_h,
                )
            )

        return detections

    def _format_detection(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        score: float,
        class_id: int,
        orig_w: int,
        orig_h: int,
    ) -> Dict[str, Any]:
        """Format satu hasil deteksi ke dictionary JSON yang rapi."""
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0

        return {
            "class_id": class_id,
            "class_name": self.get_class_name(class_id),
            "confidence": round(score, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(bw, 2),
                "height": round(bh, 2),
            },
            "bbox_normalized": {
                "x_center": round(xc / orig_w, 6) if orig_w else 0.0,
                "y_center": round(yc / orig_h, 6) if orig_h else 0.0,
                "width": round(bw / orig_w, 6) if orig_w else 0.0,
                "height": round(bh / orig_h, 6) if orig_h else 0.0,
            },
        }

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Menjalankan pipeline penuh: preprocess -> inference -> postprocess."""
        conf = float(conf_thres if conf_thres is not None else self.conf_thres)
        iou = float(iou_thres if iou_thres is not None else self.iou_thres)

        t0 = time.perf_counter()
        blob, ratio, pad = self.preprocess(image_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        detections = self.postprocess(
            outputs, image_bgr.shape[:2], ratio, pad, conf, iou
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "detections": detections,
            "count": len(detections),
            "inference_time_ms": round(elapsed_ms, 2),
            "image_size": {
                "width": int(image_bgr.shape[1]),
                "height": int(image_bgr.shape[0]),
            },
        }

    # Warna BGR konsisten untuk tiap class (0: begal, 1: korban, 2: motor, 3: senjata, 4: status_safe, 5: status_warning)
    CLASS_COLORS_BGR = [
        (68, 68, 239),    # 0: begal (Merah #ef4444)
        (22, 115, 249),   # 1: korban (Oranye #f97316)
        (212, 182, 6),    # 2: motor (Cyan #06b6d4)
        (38, 38, 220),    # 3: senjata (Merah Tua #dc2626)
        (94, 197, 34),    # 4: status_safe (Hijau #22c55e)
        (8, 179, 234),    # 5: status_warning (Kuning #eab308)
    ]

    def draw_detections(
        self, image_bgr: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Menggambar bounding box dan label pada gambar."""
        annotated = image_bgr.copy()
        for det in detections:
            bbox = det.get("bbox")
            if not bbox:
                continue

            x1, y1 = int(round(bbox["x1"])), int(round(bbox["y1"]))
            x2, y2 = int(round(bbox["x2"])), int(round(bbox["y2"]))
            cls_id = int(det["class_id"])
            label = f"{det['class_name']} {det['confidence']:.2f}"

            # Warna konsisten per class_id
            if 0 <= cls_id < len(self.CLASS_COLORS_BGR):
                color = self.CLASS_COLORS_BGR[cls_id]
            else:
                np.random.seed(cls_id + 42)
                color = tuple(int(c) for c in np.random.randint(40, 235, size=3))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(y1, th + 8)
            cv2.rectangle(
                annotated,
                (x1, text_y - th - 8),
                (x1 + tw + 6, text_y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, text_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def get_model_info(self) -> Dict[str, Any]:
        """Mengembalikan informasi lengkap mengenai model ONNX yang dimuat."""
        size_mb = (
            round(os.path.getsize(self.model_path) / (1024 * 1024), 2)
            if os.path.exists(self.model_path)
            else 0.0
        )
        return {
            "model_path": Path(self.model_path).name,
            "full_path": os.path.abspath(self.model_path),
            "file_size_mb": size_mb,
            "description": self.metadata.get("description", ""),
            "providers": self.providers,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "input_resolution": {"width": self.input_width, "height": self.input_height},
            "output_names": self.output_names,
            "output_shapes": self.output_shapes,
            "num_classes": len(self.class_names) if self.class_names else None,
            "class_names": self.class_names,
        }
