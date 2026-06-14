import logging
import cv2
import numpy as np
import torch
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from torchvision import transforms

from train_manager import get_latest_completed_job, CIFAR10_CLASSES

logger = logging.getLogger(__name__)

class CustomModelProcessor:
    def __init__(self):
        self._model = None
        self._loaded_job_id = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def _load_latest_model(self):
        job = get_latest_completed_job()
        if not job:
            return False
            
        if self._loaded_job_id == job.job_id:
            return True # Déjà chargé
            
        logger.info(f"Chargement du nouveau modèle personnalisé: {job.job_id}")
        import os
        from train_manager import JOBS_DIR
        model_path = JOBS_DIR / f"{job.job_id}_model.pth"
        
        if not os.path.exists(model_path):
            return False
            
        try:
            # On cherche à savoir si c'est un resnet18 ou 50 à partir des logs du job, 
            # mais par défaut, supposons resnet18 ou regardons les paramètres du job si on les avait sauvegardés.
            # Pour l'instant, on lit le state_dict pour deviner.
            state_dict = torch.load(model_path, map_location=self._device)
            # resnet18 a 'layer4.1.conv2.weight', resnet50 a 'layer4.2.conv3.weight'
            is_resnet50 = any("layer4.2.conv3" in k for k in state_dict.keys())
            
            if is_resnet50:
                self._model = resnet50(weights=None)
                self._model.fc = nn.Linear(self._model.fc.in_features, 10)
            else:
                self._model = resnet18(weights=None)
                self._model.fc = nn.Linear(self._model.fc.in_features, 10)
                
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            self._loaded_job_id = job.job_id
            return True
        except Exception as exc:
            logger.error(f"Erreur lors du chargement du modèle custom : {exc}")
            return False

    def process_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        if not self._load_latest_model():
            # Fallback si aucun modèle entraîné
            cv2.putText(img_bgr, "Aucun modele personnalise trouve", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img_bgr
            
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self._transform(img_rgb).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                outputs = self._model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                score, predicted = torch.max(probabilities, 0)
                
            class_name = CIFAR10_CLASSES[predicted.item()]
            conf_percent = score.item() * 100
            
            # Affichage en gros sur l'image
            text = f"Custom: {class_name} ({conf_percent:.1f}%)"
            
            # Rectangle semi-transparent de fond pour lisibilité
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (10, 65), (340, 115), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)
            
            # Texte
            color = (0, 255, 0) if conf_percent > 70 else (0, 165, 255)
            cv2.putText(img_bgr, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return img_bgr
            
        except Exception as exc:
            logger.error(f"Erreur inference custom : {exc}")
            return img_bgr

custom_model_processor = CustomModelProcessor()
