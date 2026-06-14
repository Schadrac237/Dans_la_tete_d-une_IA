"""
ml_stubs.py
────────────
Schémas Pydantic + wiring vers les implémentations réelles :
  - Grad-CAM  → gradcam.py
  - Training  → train_manager.py (ResNet18/50 + CIFAR-10)
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Schémas Pydantic — Grad-CAM ─────────────────────────────────────────────

class GradCAMRequest(BaseModel):
    """Requête Grad-CAM : une image base64 + classe cible optionnelle."""
    image_b64: str = Field(..., description="Image encodée en base64 (JPEG ou PNG)")
    target_class: Optional[int] = Field(
        None, description="Indice de classe COCO ciblé (None = classe la plus probable)"
    )
    layer_name: str = Field(
        "model.model[21]",
        description=(
            "Couche YOLO à visualiser. "
            "Options : model.model[21] (recommandé), model.model[18], "
            "model.model[9] (backbone-C2f9), model.model[6] (backbone-C2f6)"
        ),
    )


class GradCAMResponse(BaseModel):
    status: str
    heatmap_b64: Optional[str] = None
    target_class: Optional[int] = None
    class_label: Optional[str] = None
    message: str


# ─── Schémas Pydantic — Transfer Learning (CIFAR-10 + ResNet) ────────────────

class TrainRequest(BaseModel):
    """Requête d'entraînement — ResNet fine-tuné sur CIFAR-10 (download auto)."""
    base_model: Literal["resnet18", "resnet50"] = Field(
        "resnet18",
        description="Architecture de base : resnet18 (rapide) ou resnet50 (plus précis)",
    )
    epochs: int = Field(
        5, ge=1, le=50,
        description="Nombre d'époques (1–50). Recommandé : 5–10 en mode freeze.",
    )
    learning_rate: float = Field(
        0.001, ge=1e-6, le=0.1,
        description="Taux d'apprentissage SGD",
    )
    freeze_backbone: bool = Field(
        True,
        description=(
            "True = geler le backbone, entraîner seulement la tête FC "
            "(Transfer Learning classique, rapide). "
            "False = fine-tuning complet (lent mais plus précis)."
        ),
    )
    batch_size: int = Field(
        64, ge=8, le=256,
        description="Taille des mini-batches",
    )


class TrainResponse(BaseModel):
    status: str
    job_id: Optional[str] = None
    message: str
    estimated_duration_minutes: Optional[float] = None


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str          # queued | running | completed | failed
    progress_percent: float
    current_epoch: int
    total_epochs: int
    metrics: dict[str, Any]
    message: str
    error: Optional[str] = None


# ─── Handlers ────────────────────────────────────────────────────────────────

async def run_gradcam(request: GradCAMRequest) -> GradCAMResponse:
    """Délègue au GradCAMProcessor (gradcam.py)."""
    from gradcam import gradcam_processor

    logger.info("Grad-CAM demandé — couche: %s  classe: %s",
                request.layer_name, request.target_class)
    result = gradcam_processor.compute(
        image_b64=request.image_b64,
        target_class=request.target_class,
        layer_name=request.layer_name,
    )
    return GradCAMResponse(**result)


async def start_training(request: TrainRequest) -> TrainResponse:
    """Démarre un job d'entraînement ResNet+CIFAR-10 en arrière-plan."""
    from train_manager import start_training_job

    logger.info(
        "Entraînement demandé — modèle: %s  epochs: %d  lr: %.6f  freeze: %s",
        request.base_model, request.epochs, request.learning_rate, request.freeze_backbone,
    )

    # Estimation grossière du temps (CPU) : ~2 min/epoch resnet18 freeze, ~8 min resnet50 full
    if request.freeze_backbone:
        est_min = request.epochs * (1.5 if request.base_model == "resnet18" else 3.0)
    else:
        est_min = request.epochs * (4.0 if request.base_model == "resnet18" else 8.0)

    job_id = await start_training_job(
        base_model=request.base_model,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        freeze_backbone=request.freeze_backbone,
        batch_size=request.batch_size,
    )

    return TrainResponse(
        status="started",
        job_id=job_id,
        message=(
            f"Job d'entraînement démarré — {request.base_model} sur CIFAR-10 "
            f"({request.epochs} epochs). Poll GET /api/train/{job_id} pour suivre."
        ),
        estimated_duration_minutes=round(est_min, 1),
    )


async def get_training_status(job_id: str) -> TrainStatusResponse:
    """Retourne le statut en temps réel d'un job."""
    from train_manager import get_job_status

    job = get_job_status(job_id)

    if job is None:
        return TrainStatusResponse(
            job_id=job_id,
            status="not_found",
            progress_percent=0.0,
            current_epoch=0,
            total_epochs=0,
            metrics={},
            message=f"Job '{job_id}' introuvable.",
            error=None,
        )

    return TrainStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_percent=job.progress_percent,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        metrics=job.metrics,
        message=job.message,
        error=job.error,
    )
