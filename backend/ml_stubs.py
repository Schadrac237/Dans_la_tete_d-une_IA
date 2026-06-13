"""
ml_stubs.py
────────────
Stubs propres pour les fonctionnalités ML avancées à venir :
  - Grad-CAM (visualisation des features maps)
  - Transfer Learning / Entraînement interactif

Ces endpoints retournent des réponses structurées avec statut
"not_implemented" pour ne pas casser le frontend.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Schémas Pydantic ─────────────────────────────────────────────────────────

class GradCAMRequest(BaseModel):
    """Requête Grad-CAM : une image base64 + classe cible optionnelle."""
    image_b64: str = Field(..., description="Image encodée en base64 (JPEG ou PNG)")
    target_class: int | None = Field(
        None, description="Indice de classe COCO ciblé (None = classe la plus probable)"
    )
    layer_name: str = Field(
        "model.22",
        description="Nom de la couche YOLO à visualiser (ex: model.22 pour la tête de détection)"
    )


class GradCAMResponse(BaseModel):
    status: str
    heatmap_b64: str | None = None
    target_class: int | None = None
    class_label: str | None = None
    message: str


class TrainRequest(BaseModel):
    """Requête d'entraînement interactif."""
    dataset_path: str = Field(..., description="Chemin vers le dataset YOLO (format ultralytics)")
    epochs: int = Field(10, ge=1, le=100, description="Nombre d'époques")
    base_model: str = Field("yolov8n.pt", description="Modèle de base pour le Transfer Learning")
    learning_rate: float = Field(0.001, ge=1e-6, le=0.1)
    freeze_backbone: bool = Field(True, description="Geler le backbone (Transfer Learning standard)")


class TrainResponse(BaseModel):
    status: str
    job_id: str | None = None
    message: str
    estimated_duration_minutes: float | None = None


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    progress_percent: float
    current_epoch: int
    total_epochs: int
    metrics: dict[str, Any]
    message: str


# ─── Logique des stubs ────────────────────────────────────────────────────────

async def run_gradcam(request: GradCAMRequest) -> GradCAMResponse:
    """
    TODO: Implémenter Grad-CAM avec PyTorch hooks sur YOLOv8n.

    Approche recommandée :
      1. Décoder l'image base64 → tensor PyTorch
      2. Enregistrer un forward hook sur `request.layer_name`
      3. Enregistrer un backward hook pour capturer les gradients
      4. Forward pass → backprop sur le score de `target_class`
      5. Pondérer la feature map par les gradients moyennés
      6. Appliquer ReLU + normalisation + resize → heatmap
      7. Superposer avec l'image originale via cv2.applyColorMap
      8. Encoder le résultat en base64 et retourner
    """
    logger.info("Grad-CAM demandé — stub actif (non implémenté)")
    return GradCAMResponse(
        status="not_implemented",
        heatmap_b64=None,
        target_class=request.target_class,
        class_label=None,
        message=(
            "Grad-CAM n'est pas encore implémenté. "
            "Cette route est un placeholder structuré pour la prochaine itération. "
            "Voir ml_stubs.py::run_gradcam() pour le guide d'implémentation."
        ),
    )


async def start_training(request: TrainRequest) -> TrainResponse:
    """
    TODO: Implémenter l'entraînement interactif avec Ultralytics.

    Approche recommandée :
      1. Valider le dataset (vérifier le fichier data.yaml)
      2. Créer un job_id unique (uuid4)
      3. Lancer model.train() dans un ProcessPoolExecutor
         pour ne pas bloquer le serveur FastAPI
      4. Stocker l'état dans un dict {job_id: TrainState}
      5. Utiliser les callbacks Ultralytics (on_epoch_end) pour
         mettre à jour la progression en temps réel
      6. Retourner le job_id pour polling via GET /api/train/{job_id}
    """
    logger.info("Entraînement demandé — stub actif (non implémenté)")
    return TrainResponse(
        status="not_implemented",
        job_id=None,
        message=(
            "L'entraînement interactif n'est pas encore implémenté. "
            "Voir ml_stubs.py::start_training() pour le guide d'implémentation."
        ),
        estimated_duration_minutes=None,
    )


async def get_training_status(job_id: str) -> TrainStatusResponse:
    """
    TODO: Retourner l'état en temps réel d'un job d'entraînement.
    """
    logger.info("Statut entraînement %s demandé — stub actif", job_id)
    return TrainStatusResponse(
        job_id=job_id,
        status="not_found",
        progress_percent=0.0,
        current_epoch=0,
        total_epochs=0,
        metrics={},
        message=f"Job '{job_id}' introuvable — entraînement non implémenté.",
    )
