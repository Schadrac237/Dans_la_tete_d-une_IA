"""
yolo_processor.py
─────────────────
Pipeline de détection d'objets en temps réel avec YOLOv8n.

IMPORTANT — Architecture des imports pour éviter le crash libgomp :
  Ce module importe torch EN PREMIER (aucun cv2 ici).
  cv2 est importé UNIQUEMENT dans webrtc_handler.py, APRÈS
  que ce module a été chargé. Cela garantit que torch initialise
  son runtime OMP avant qu'OpenCV ne charge le sien.
  (bug: "double free in tcache 2" sur Linux/WSL)
"""

import logging
import os
from dataclasses import dataclass, field
from threading import Lock

# ── Fix OMP : définir AVANT tout import C-extension ──────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# torch EN PREMIER → initialise le runtime OMP/libgomp en autorité
import torch                    # noqa: E402
import numpy as np              # noqa: E402
from ultralytics import YOLO    # noqa: E402
# ─── PAS d'import cv2 ici ! ──────────────────────────────────────────────────
# cv2 est importé dans webrtc_handler.py après ce module.

logger = logging.getLogger(__name__)


# ─── Configuration runtime (thread-safe) ─────────────────────────────────────
@dataclass
class YOLOConfig:
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    device: str = "cpu"
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def set_confidence(self, value: float) -> None:
        with self._lock:
            self.confidence_threshold = max(0.0, min(1.0, value))
        logger.debug("Confidence threshold → %.2f", self.confidence_threshold)

    @property
    def conf(self) -> float:
        with self._lock:
            return self.confidence_threshold


# ─── Processeur YOLO (singleton) ─────────────────────────────────────────────
class YOLOProcessor:
    """
    Charge YOLOv8n une seule fois et expose process_frame().

    Dessin des bounding boxes : délégué à results[0].plot() d'ultralytics
    (évite d'importer cv2 dans ce module et donc le conflit libgomp).
    """

    def __init__(self):
        self.config = YOLOConfig()
        self._model: YOLO | None = None
        self._class_names: dict = {}
        self._loaded = False

    def load(self) -> None:
        """Charge le modèle ONNX (sans dépendance runtime PyTorch)."""
        if self._loaded:
            return
        logger.info("Chargement de YOLOv8n (ONNX)…")
        # En utilisant .onnx, ultralytics utilise onnxruntime au lieu de PyTorch.
        # Cela empêche le conflit de mémoire (double free) avec PyAV/aiortc.
        self._model = YOLO("yolov8n.onnx", task="detect")
        self._class_names = self._model.names   # dict {id: str}
        self._loaded = True
        logger.info("YOLOv8n ONNX chargé — %d classes", len(self._class_names))

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Exécute la détection YOLO et retourne la frame annotée.

        Le dessin est fait par ultralytics (results[0].plot()) qui gère
        cv2 en interne — aucun appel cv2 direct ici.

        Returns:
            annotated_bgr : frame BGR avec bounding boxes et labels
            detections    : liste de dicts {label, confidence, bbox}
        """
        if not self._loaded or self._model is None:
            return frame_bgr, []

        conf_thresh = self.config.conf

        results = self._model.predict(
            source=frame_bgr,
            conf=conf_thresh,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            device=self.config.device,
            verbose=False,
            stream=False,
        )

        # ── Annotation via ultralytics (pas de cv2 direct) ────────────────────
        # plot() retourne un ndarray BGR avec boxes, labels et confiances
        annotated: np.ndarray = results[0].plot()

        # ── Extraction des détections pour l'API ──────────────────────────────
        detections: list[dict] = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self._class_names.get(cls_id, f"cls_{cls_id}")
                detections.append({
                    "label": label,
                    "confidence": conf_val,
                    "bbox": [x1, y1, x2, y2],
                })

        return annotated, detections


# ─── Instance globale (singleton) ─────────────────────────────────────────────
yolo_processor = YOLOProcessor()
