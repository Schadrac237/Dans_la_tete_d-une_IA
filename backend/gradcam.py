"""
gradcam.py
──────────
Implémentation Grad-CAM sur YOLOv8n.pt avec hooks PyTorch.

⚠️  ORDRE D'IMPORT CRITIQUE (bug libgomp / OMP double-free sur Linux) :
    torch DOIT être importé avant cv2.
    Ne jamais importer cv2 au niveau module dans ce fichier.
    cv2 est utilisé uniquement dans les fonctions, APRÈS que torch soit chargé.

Architecture :
  - Singleton `GradCAMProcessor` : charge yolov8n.pt une seule fois
    (distinct du yolov8n.onnx utilisé pour le temps réel WebRTC)
  - Forward hook sur la couche demandée → capture feature_map
  - Backward hook → capture gradients
  - Génère une heatmap colorisée superposée sur l'image originale
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from typing import Optional

# ── ORDRE CRITIQUE : torch avant tout import C-extension ─────────────────────
import torch                    # noqa: E402
import torch.nn.functional as F # noqa: E402
import numpy as np              # noqa: E402
from PIL import Image           # noqa: E402
from ultralytics import YOLO    # noqa: E402

logger = logging.getLogger(__name__)

# ─── Constantes ───────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"

# Couches de détection YOLOv8n disponibles pour Grad-CAM
# model.model[21] = dernière couche C2f avant la tête (recommandé pour Grad-CAM)
AVAILABLE_LAYERS = {
    "model.model[21]": "Dernière couche C2f (recommandé)",
    "model.model[18]": "Neck — C2f 18",
    "model.model[-4]": "Neck — C2f 4",
    "model.model[9]":  "Backbone — C2f 9",
    "model.model[6]":  "Backbone — C2f 6",
}


# ─── Singleton Grad-CAM ───────────────────────────────────────────────────────
class GradCAMProcessor:
    """
    Charge yolov8n.pt et expose `compute(image_b64, target_class, layer_name)`.
    Thread-safe via un Lock (le serveur peut recevoir plusieurs requêtes en parallèle).
    """

    def __init__(self) -> None:
        self._model: Optional[YOLO] = None
        self._torch_model = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        """Chargement paresseux du modèle .pt (une seule fois)."""
        if self._loaded:
            return
        logger.info("Chargement de YOLOv8n.pt pour Grad-CAM…")
        self._model = YOLO(MODEL_PATH, task="detect")
        # Accès au modèle PyTorch natif
        self._torch_model = self._model.model
        self._torch_model.eval()
        logger.info("YOLOv8n.pt chargé pour Grad-CAM ✓")
        self._loaded = True

    def _decode_image(self, image_b64: str) -> tuple[np.ndarray, Image.Image]:
        """Décode une image base64 → numpy BGR + PIL Image originale."""
        raw = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        # Resize à 640x640 (taille d'entrée YOLO)
        pil_resized = pil_img.resize((640, 640))
        np_rgb = np.array(pil_resized, dtype=np.float32) / 255.0
        return np_rgb, pil_img

    def _get_layer(self, layer_expr: str):
        """Résout une expression de couche (ex: 'model.model[-1]') → module PyTorch."""
        try:
            # Sécurisé : on n'évalue que des chemins simples connus
            parts = layer_expr.split(".")
            obj = self._torch_model
            for part in parts[1:]:   # skip "model" racine
                if part.endswith("]"):
                    attr, idx_str = part[:-1].split("[")
                    obj = getattr(obj, attr)[int(idx_str)]
                else:
                    obj = getattr(obj, part)
            return obj
        except Exception as exc:
            raise ValueError(
                f"Couche '{layer_expr}' introuvable dans le modèle : {exc}"
            ) from exc

    def compute(
        self,
        image_b64: str,
        target_class: Optional[int],
        layer_name: str,
    ) -> dict:
        """
        Calcule la carte Grad-CAM et retourne le résultat encodé en base64.

        Returns:
            dict avec : status, heatmap_b64, target_class, class_label, message
        """
        import cv2  # importé ICI seulement (après torch) pour éviter le conflit OMP

        with self._lock:
            self._load()

        np_rgb, pil_orig = self._decode_image(image_b64)

        # ── Tensor d'entrée ───────────────────────────────────────────────────
        device = next(self._torch_model.parameters()).device
        tensor_in = torch.from_numpy(np_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]

        # ── Hooks pour capturer feature_map et gradients ──────────────────────
        feature_maps: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        target_layer = self._get_layer(layer_name)

        def fwd_hook(module, input, output):
            feature_maps.append(output)

        def bwd_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

        try:
            # ── Forward pass ──────────────────────────────────────────────────
            tensor_in.requires_grad_(True)
            with torch.enable_grad():
                outputs = self._torch_model(tensor_in)

            # outputs est un tuple/tensor selon la version ; on prend le tenseur de prédictions
            # YOLOv8 retourne un tuple (preds, ...) ou directement un tensor
            if isinstance(outputs, (tuple, list)):
                pred_tensor = outputs[0]
            else:
                pred_tensor = outputs

            # Aplatir les prédictions : shape [1, num_classes+4+1, N_anchors]
            # On prend le score de confiance de la classe cible (ou max)
            if pred_tensor.dim() == 3:
                # Format [batch, 4+classes, anchors] — YOLOv8 style
                scores = pred_tensor[0, 4:, :]          # [classes, anchors]
                if target_class is None:
                    # Score max sur toutes classes et anchors
                    score = scores.max()
                    target_class_used = int(scores.max(dim=0).values.argmax().item())
                else:
                    score = scores[target_class, :].max()
                    target_class_used = target_class
            else:
                score = pred_tensor.sum()
                target_class_used = target_class or 0

            # ── Backward ──────────────────────────────────────────────────────
            self._torch_model.zero_grad()
            score.backward(retain_graph=False)

            if not feature_maps or not gradients:
                return {
                    "status": "error",
                    "heatmap_b64": None,
                    "target_class": target_class,
                    "class_label": None,
                    "message": "Hooks n'ont pas capturé de données — essayez une autre couche.",
                }

            # ── Calcul Grad-CAM ───────────────────────────────────────────────
            # Extraction des tenseurs (parfois enveloppés dans des tuples selon la couche)
            fmap = feature_maps[0][0].detach() if isinstance(feature_maps[0], tuple) else feature_maps[0].detach()
            grad = gradients[0][0].detach() if isinstance(gradients[0], tuple) else gradients[0].detach()

            if fmap.dim() != 4 or grad.dim() != 4:
                return {
                    "status": "error",
                    "heatmap_b64": None,
                    "target_class": target_class,
                    "class_label": None,
                    "message": f"La couche choisie produit un tenseur {fmap.dim()}D au lieu de 4D. Veuillez sélectionner une couche 'C2f' pour générer une carte spatiale.",
                }

            # Pondération globale des canaux par les gradients moyens
            weights = grad.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
            cam = (weights * fmap).sum(dim=1, keepdim=True)  # [1, 1, H, W]
            cam = F.relu(cam)

            # Normalisation [0, 1]
            cam = cam.squeeze()                  # [H, W]
            cam_min = cam.min()
            cam_max = cam.max()
            if (cam_max - cam_min).item() > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            # Resize vers taille d'entrée (640x640)
            cam_np = cam.numpy()
            cam_resized = cv2.resize(cam_np, (640, 640))

            # ── Superposition heatmap + image ─────────────────────────────────
            orig_bgr = cv2.cvtColor(np.array(pil_orig.resize((640, 640))), cv2.COLOR_RGB2BGR)
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)

            # ── Encode en base64 ──────────────────────────────────────────────
            _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            heatmap_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            # Nom de la classe YOLO
            class_names = self._model.names if self._model else {}
            class_label = class_names.get(target_class_used, f"cls_{target_class_used}")

            logger.info(
                "Grad-CAM OK — couche: %s  classe: %s (%d)",
                layer_name, class_label, target_class_used,
            )

            return {
                "status": "ok",
                "heatmap_b64": heatmap_b64,
                "target_class": target_class_used,
                "class_label": class_label,
                "message": (
                    f"Grad-CAM généré sur la couche '{layer_name}' "
                    f"pour la classe '{class_label}' (id={target_class_used})."
                ),
            }

        except Exception as exc:
            logger.error("Erreur Grad-CAM : %s", exc, exc_info=True)
            return {
                "status": "error",
                "heatmap_b64": None,
                "target_class": target_class,
                "class_label": None,
                "message": f"Erreur lors du calcul Grad-CAM : {exc}",
            }

        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    def compute_live(
        self,
        img_bgr: np.ndarray,
        target_class: Optional[int],
        layer_name: str,
    ) -> np.ndarray:
        """
        Version temps réel pour WebRTC.
        Prend un tableau BGR numpy (image d'entrée) et retourne l'image
        composée (Heatmap sur BGR) sans conversion Base64.
        """
        import cv2

        with self._lock:
            self._load()

        # On s'assure que le tenseur d'entrée est à 640x640, format [1, 3, H, W] RGB
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (640, 640)), cv2.COLOR_BGR2RGB)
        np_rgb = np.array(img_rgb, dtype=np.float32) / 255.0
        device = next(self._torch_model.parameters()).device
        tensor_in = torch.from_numpy(np_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        feature_maps: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        target_layer = self._get_layer(layer_name)

        def fwd_hook(module, input, output):
            feature_maps.append(output)

        def bwd_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

        try:
            tensor_in.requires_grad_(True)
            with torch.enable_grad():
                outputs = self._torch_model(tensor_in)

            if isinstance(outputs, (tuple, list)):
                pred_tensor = outputs[0]
            else:
                pred_tensor = outputs

            if pred_tensor.dim() == 3:
                scores = pred_tensor[0, 4:, :]
                if target_class is None:
                    score = scores.max()
                else:
                    score = scores[target_class, :].max()
            else:
                score = pred_tensor.sum()

            self._torch_model.zero_grad()
            score.backward(retain_graph=False)

            if not feature_maps or not gradients:
                return img_bgr  # fallback silicieux

            fmap = feature_maps[0][0].detach() if isinstance(feature_maps[0], tuple) else feature_maps[0].detach()
            grad = gradients[0][0].detach() if isinstance(gradients[0], tuple) else gradients[0].detach()

            if fmap.dim() != 4 or grad.dim() != 4:
                return img_bgr

            weights = grad.mean(dim=[2, 3], keepdim=True)
            cam = (weights * fmap).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = cam.squeeze()
            cam_min = cam.min()
            cam_max = cam.max()
            if (cam_max - cam_min).item() > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            # Il faut repasser sur CPU pour NumPy si on utilise CUDA
            cam_np = cam.cpu().numpy()
            cam_resized = cv2.resize(cam_np, (img_bgr.shape[1], img_bgr.shape[0]))

            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

            return overlay

        except Exception as exc:
            logger.error("Erreur Live Grad-CAM : %s", exc)
            return img_bgr  # Fallback
        finally:
            fwd_handle.remove()
            bwd_handle.remove()


# ─── Instance globale ─────────────────────────────────────────────────────────
gradcam_processor = GradCAMProcessor()
