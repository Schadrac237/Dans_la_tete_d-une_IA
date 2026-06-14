"""
train_manager.py
────────────────
Gestionnaire de jobs d'entraînement par Transfer Learning.

Modèle  : ResNet18 ou ResNet50 (pretrained ImageNet via torchvision)
Dataset : CIFAR-10 (téléchargé automatiquement, ~170 Mo)
Classes : 10 — avion, auto, oiseau, chat, cerf, chien, grenouille,
                cheval, bateau, camion

Architecture asynchrone :
  - Le serveur FastAPI reçoit la requête et retourne immédiatement un job_id
  - L'entraînement PyTorch tourne dans un ProcessPoolExecutor séparé
  - L'état du job est stocké dans un fichier JSON partagé (évite les
    problèmes de mémoire partagée entre processus)
  - Le client poll GET /api/train/{job_id} pour suivre la progression
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Répertoire de travail pour les jobs ──────────────────────────────────────
# On utilise /tmp/ pour ne pas déclencher le --reload de Uvicorn à chaque sauvegarde !
import tempfile
JOBS_DIR = Path(tempfile.gettempdir()) / "ia_train_jobs"
JOBS_DIR.mkdir(exist_ok=True)

CIFAR10_DATA_DIR = Path.home() / "cifar10_data"
CIFAR10_CLASSES = [
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
]

# ─── Executor partagé (1 seul training à la fois pour éviter OOM) ─────────────
_executor: Optional[ProcessPoolExecutor] = None
_executor_lock = threading.Lock()


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            # Requis pour PyTorch avec CUDA : utiliser 'spawn' au lieu de 'fork'
            ctx = multiprocessing.get_context("spawn")
            _executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    return _executor


# ─── Modèle de données d'un job ───────────────────────────────────────────────
@dataclass
class TrainJob:
    job_id: str
    status: str = "queued"           # queued | running | completed | failed
    progress_percent: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = "En attente de démarrage…"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def save(self) -> None:
        """Persiste l'état du job sur disque (thread-safe pour inter-process)."""
        self.updated_at = time.time()
        path = JOBS_DIR / f"{self.job_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, job_id: str) -> Optional["TrainJob"]:
        """Charge un job depuis le disque. Retourne None si introuvable."""
        path = JOBS_DIR / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except Exception:
            return None


# ─── Fonction d'entraînement (exécutée dans un sous-processus) ───────────────
def _run_training_subprocess(
    job_id: str,
    base_model: str,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
    batch_size: int,
) -> None:
    """
    Exécutée dans un ProcessPoolExecutor.
    Pas d'accès au logger asyncio — utilise print + job.save() pour la communication.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

    job = TrainJob.load(job_id)
    if job is None:
        return

    try:
        job.status = "running"
        job.total_epochs = epochs
        job.message = "Initialisation de l'entraînement…"
        job.save()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Dataset CIFAR-10 ─────────────────────────────────────────────────
        job.message = "Téléchargement/vérification de CIFAR-10…"
        job.save()

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=str(CIFAR10_DATA_DIR), train=True, download=True,
            transform=transform_train,
        )
        val_set = torchvision.datasets.CIFAR10(
            root=str(CIFAR10_DATA_DIR), train=False, download=True,
            transform=transform_val,
        )

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=0,
        )

        # ── Modèle ──────────────────────────────────────────────────────────
        job.message = f"Chargement de {base_model} (pretrained ImageNet)…"
        job.save()

        if base_model == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features  # 2048
        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features  # 512

        # Remplacer la tête FC pour 10 classes CIFAR-10
        model.fc = nn.Linear(num_features, 10)

        if freeze_backbone:
            # Geler tout sauf la couche fc
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        model = model.to(device)

        # ── Optimiseur & critère ─────────────────────────────────────────────
        params_to_train = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            params_to_train,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # ── Boucle d'entraînement ────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            # — Phase train —
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                # Update progression intra-epoch
                if batch_idx % 50 == 0:
                    epoch_progress = (batch_idx + 1) / len(train_loader)
                    total_progress = ((epoch - 1) + epoch_progress) / epochs * 100
                    job2 = TrainJob.load(job_id)
                    if job2:
                        job2.progress_percent = round(total_progress, 1)
                        job2.current_epoch = epoch
                        job2.message = (
                            f"Epoch {epoch}/{epochs} — batch {batch_idx+1}/{len(train_loader)}"
                        )
                        job2.save()

            train_loss = running_loss / total_train
            train_acc = 100.0 * correct_train / total_train

            # — Phase validation —
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

            val_acc = 100.0 * correct_val / total_val

            scheduler.step()

            # Mise à jour état du job
            job = TrainJob.load(job_id)
            if job:
                job.current_epoch = epoch
                job.progress_percent = round(epoch / epochs * 100, 1)
                job.metrics = {
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 2),
                    "val_acc": round(val_acc, 2),
                    "learning_rate": round(scheduler.get_last_lr()[0], 6),
                }
                job.message = (
                    f"Epoch {epoch}/{epochs} — "
                    f"loss: {train_loss:.4f}  train_acc: {train_acc:.1f}%  val_acc: {val_acc:.1f}%"
                )
                job.save()

        # ── Sauvegarde du modèle ─────────────────────────────────────────────
        model_path = JOBS_DIR / f"{job_id}_model.pth"
        torch.save(model.state_dict(), str(model_path))

        job = TrainJob.load(job_id)
        if job:
            job.status = "completed"
            job.progress_percent = 100.0
            job.message = (
                f"Entraînement terminé ! "
                f"Val acc finale : {job.metrics.get('val_acc', 0):.1f}% "
                f"— Modèle sauvegardé : {model_path.name}"
            )
            job.save()

    except Exception as exc:
        job = TrainJob.load(job_id)
        if job:
            job.status = "failed"
            job.error = str(exc)
            job.message = f"Erreur d'entraînement : {exc}"
            job.save()
        raise


# ─── API publique ─────────────────────────────────────────────────────────────
async def start_training_job(
    base_model: str,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
    batch_size: int = 64,
) -> str:
    """
    Démarre un job d'entraînement dans un ProcessPoolExecutor.
    Retourne le job_id immédiatement.
    """
    job_id = str(uuid.uuid4())
    job = TrainJob(
        job_id=job_id,
        total_epochs=epochs,
        message="Job créé, en attente de démarrage…",
    )
    job.save()

    executor = _get_executor()
    executor.submit(
        _run_training_subprocess,
        job_id,
        base_model,
        epochs,
        learning_rate,
        freeze_backbone,
        batch_size,
    )

    logger.info("Job d'entraînement démarré — id: %s  modèle: %s  epochs: %d",
                job_id, base_model, epochs)
    return job_id


def get_job_status(job_id: str) -> Optional[TrainJob]:
    """
    Charge le statut du job depuis le disque.
    Retourne None si le job n'existe pas.
    """
    return TrainJob.load(job_id)
