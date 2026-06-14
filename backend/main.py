"""
main.py
────────
Point d'entrée du backend « Dans la tête d'une IA ».

Routes :
  POST   /api/offer              WebRTC SDP offer/answer
  WS     /ws/control             Contrôle temps réel (seuil de confiance)
  POST   /api/gradcam            Stub Grad-CAM
  POST   /api/train              Stub Transfer Learning
  GET    /api/train/{job_id}     Stub statut entraînement
  GET    /api/health             Health check
"""

from __future__ import annotations

import asyncio
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from webrtc_handler import handle_offer, close_all_connections, live_gradcam_config, custom_model_config
from yolo_processor import yolo_processor
from ml_stubs import (
    GradCAMRequest,
    GradCAMResponse,
    TrainRequest,
    TrainResponse,
    TrainStatusResponse,
    run_gradcam,
    start_training,
    get_training_status,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Gestionnaire de durée de vie ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles au démarrage, libère les ressources à l'arrêt."""
    logger.info("🚀 Démarrage du serveur — chargement de YOLOv8n…")
    await asyncio.get_event_loop().run_in_executor(None, yolo_processor.load)
    logger.info("✅ YOLOv8n prêt.")
    yield
    logger.info("🛑 Arrêt du serveur — fermeture des connexions WebRTC…")
    await close_all_connections()


# ─── Application FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="Dans la tête d'une IA — Backend",
    version="1.0.0",
    description="API WebRTC + YOLO pour le démonstrateur interactif temps réel",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schémas ─────────────────────────────────────────────────────────────────
class OfferRequest(BaseModel):
    sdp: str
    type: str


class OfferResponse(BaseModel):
    sdp: str
    type: str


# ─── Routes HTTP ─────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["Système"])
async def health_check():
    """Vérifie que le serveur et YOLO sont opérationnels."""
    return {
        "status": "ok",
        "yolo_loaded": yolo_processor._loaded,
        "confidence_threshold": yolo_processor.config.conf,
    }


@app.post("/api/offer", response_model=OfferResponse, tags=["WebRTC"])
async def webrtc_offer(request: OfferRequest):
    """
    Reçoit une offre SDP WebRTC du client React et retourne la réponse SDP.
    C'est le point d'entrée de la session vidéo temps réel.
    """
    logger.info("Offre SDP reçue (type=%s)", request.type)
    answer = await handle_offer(sdp=request.sdp, sdp_type=request.type)
    return OfferResponse(**answer)


# ─── WebSocket : contrôle temps réel ─────────────────────────────────────────

@app.websocket("/ws/control")
async def websocket_control(websocket: WebSocket):
    """
    Canal WebSocket bidirectionnel pour le contrôle en temps réel.

    Messages attendus du client (JSON) :
      {"type": "set_confidence", "value": 0.65}

    Messages envoyés au client (JSON) :
      {"type": "ack", "confidence": 0.65}
      {"type": "error", "message": "..."}
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "inconnu"
    logger.info("WebSocket connecté — client: %s", client_host)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "set_confidence":
                    value = float(msg.get("value", 0.5))
                    yolo_processor.config.set_confidence(value)
                    await websocket.send_json({
                        "type": "ack",
                        "confidence": yolo_processor.config.conf,
                    })
                    logger.info("Seuil mis à jour → %.2f", yolo_processor.config.conf)

                elif msg_type == "set_live_gradcam":
                    live_gradcam_config.enabled = bool(msg.get("enabled", False))
                    if live_gradcam_config.enabled:
                        custom_model_config.enabled = False  # Exclusion mutuelle
                    
                    tc = msg.get("targetClass")
                    live_gradcam_config.target_class = int(tc) if tc not in (None, "") else None
                    
                    ln = msg.get("layerName")
                    if ln:
                        live_gradcam_config.layer_name = ln
                    
                    await websocket.send_json({
                        "type": "ack_live_gradcam",
                        "enabled": live_gradcam_config.enabled
                    })
                    logger.info("Live Grad-CAM config mis à jour: %s", live_gradcam_config)

                elif msg_type == "set_custom_model_mode":
                    custom_model_config.enabled = bool(msg.get("enabled", False))
                    if custom_model_config.enabled:
                        live_gradcam_config.enabled = False  # Exclusion mutuelle
                    await websocket.send_json({
                        "type": "ack_custom_model",
                        "enabled": custom_model_config.enabled
                    })
                    logger.info("Custom Model config mis à jour: %s", custom_model_config)

                elif msg_type == "get_status":
                    await websocket.send_json({
                        "type": "status",
                        "confidence": yolo_processor.config.conf,
                        "yolo_loaded": yolo_processor._loaded,
                        "live_gradcam": {
                            "enabled": live_gradcam_config.enabled,
                            "layer": live_gradcam_config.layer_name,
                            "targetClass": live_gradcam_config.target_class,
                        }
                    })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Type de message inconnu : '{msg_type}'",
                    })

            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Message WebSocket invalide : %s — %s", raw, exc)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Format de message invalide : {exc}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket déconnecté — client: %s", client_host)
    except Exception as exc:
        logger.error("Erreur WebSocket inattendue : %s", exc, exc_info=True)


# ─── Routes ML (stubs) ────────────────────────────────────────────────────────

@app.post(
    "/api/gradcam",
    response_model=GradCAMResponse,
    tags=["ML — Visualisation"],
    summary="Génère une carte Grad-CAM (stub)",
)
async def gradcam_endpoint(request: GradCAMRequest):
    """
    [STUB] Visualisation Grad-CAM des features maps de YOLOv8n.
    Retourne actuellement un statut 'not_implemented'.
    Voir ml_stubs.py pour le guide d'implémentation.
    """
    return await run_gradcam(request)


@app.post(
    "/api/train",
    response_model=TrainResponse,
    tags=["ML — Entraînement"],
    summary="Lance un job d'entraînement (stub)",
)
async def train_endpoint(request: TrainRequest):
    """
    [STUB] Démarre un entraînement par Transfer Learning sur YOLOv8n.
    Retourne actuellement un statut 'not_implemented'.
    Voir ml_stubs.py pour le guide d'implémentation.
    """
    return await start_training(request)


@app.get(
    "/api/train/{job_id}",
    response_model=TrainStatusResponse,
    tags=["ML — Entraînement"],
    summary="Statut d'un job d'entraînement (stub)",
)
async def train_status_endpoint(job_id: str):
    """
    [STUB] Retourne le statut en temps réel d'un job d'entraînement.
    """
    return await get_training_status(job_id)
