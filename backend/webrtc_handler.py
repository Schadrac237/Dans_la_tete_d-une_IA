"""
webrtc_handler.py
──────────────────
Gestion de la connexion WebRTC via aiortc.

Flux :
  1. Le client envoie une offre SDP (POST /api/offer).
  2. On crée un RTCPeerConnection côté serveur.
  3. On intercepte la VideoTrack entrante et on lui substitue
     un VideoTransformTrack qui exécute YOLO sur chaque frame.
  4. On renvoie la réponse SDP au client.
"""

import asyncio
import logging
import time
from fractions import Fraction

import av
from dataclasses import dataclass
from typing import Optional
import numpy as np

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

# ── ORDRE CRITIQUE : yolo_processor charge torch avant cv2 ───────────────────
# torch doit initialiser son runtime OMP AVANT qu'OpenCV charge le sien.
# Violation de cet ordre → "double free in tcache 2" sur Linux/WSL.
from yolo_processor import yolo_processor   # 1. charge torch via OMP fix interne

import cv2                                  # 2. cv2 après torch : safe
cv2.setNumThreads(0)                        # 3. désactiver threading OMP d'OpenCV


logger = logging.getLogger(__name__)

# ─── Configuration Live Grad-CAM ─────────────────────────────────────────────
@dataclass
class LiveGradCAMConfig:
    enabled: bool = False
    target_class: Optional[int] = None
    layer_name: str = "model.model[21]"

live_gradcam_config = LiveGradCAMConfig()

@dataclass
class CustomModelConfig:
    enabled: bool = False

custom_model_config = CustomModelConfig()

# ─── Ensemble des connexions actives (pour cleanup) ───────────────────────────
active_connections: set[RTCPeerConnection] = set()


# ─── Track transformée : YOLO sur chaque frame ───────────────────────────────
class VideoTransformTrack(MediaStreamTrack):
    """
    Prend un MediaStreamTrack vidéo entrant et produit un flux annoté par YOLO.

    Optimisation anti-latence : On utilise une tâche en arrière-plan pour
    consommer le flux vidéo entrant aussi vite que possible. Si l'inférence YOLO
    est plus lente que la webcam, on écrase les vieilles frames pour toujours
    traiter la frame la plus récente. C'est le vrai "frame skipping".
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self._track = track
        self._frame_count = 0
        self._fps_count = 0
        self._last_fps_time = time.monotonic()
        self._fps = 0.0
        
        # File d'attente pour la dernière frame (maxsize=1)
        self._queue = asyncio.Queue(maxsize=1)
        self._consume_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        """Vide le buffer WebRTC entrant en continu pour éviter la latence."""
        try:
            while True:
                frame = await self._track.recv()
                # S'il y a déjà une frame non traitée, on la jette (frame drop)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                self._queue.put_nowait(frame)
        except Exception:
            pass # Piste fermée ou erreur réseau

    async def recv(self) -> av.VideoFrame:
        # Bloque jusqu'à avoir une frame fraîche
        frame: av.VideoFrame = await self._queue.get()

        # ── Calcul FPS ────────────────────────────────────────────────────────
        self._fps_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._last_fps_time = now

        try:
            # av.VideoFrame → numpy BGR
            img_bgr: np.ndarray = frame.to_ndarray(format="bgr24")

            # Inférence (YOLO classique ou Live Grad-CAM)
            loop = asyncio.get_running_loop()
            
            if custom_model_config.enabled:
                from custom_model_processor import custom_model_processor
                annotated_bgr = await loop.run_in_executor(
                    None,
                    custom_model_processor.process_frame,
                    img_bgr
                )
            elif live_gradcam_config.enabled:
                # ── Mode Live Grad-CAM ──
                from gradcam import gradcam_processor
                annotated_bgr = await loop.run_in_executor(
                    None,
                    gradcam_processor.compute_live,
                    img_bgr,
                    live_gradcam_config.target_class,
                    live_gradcam_config.layer_name
                )
            else:
                # ── Mode YOLO Détection standard ──
                annotated_bgr, _ = await loop.run_in_executor(
                    None, yolo_processor.process_frame, img_bgr
                )

            # HUD FPS
            cv2.putText(
                annotated_bgr,
                f"FPS: {self._fps:.1f}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            # numpy BGR → av.VideoFrame
            new_frame = av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

            # On copie les timestamps de la frame source
            if frame.pts is not None:
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
            else:
                self._frame_count += 1
                new_frame.pts = self._frame_count
                new_frame.time_base = Fraction(1, 90000)

        except Exception as exc:
            logger.error("Erreur VideoTransformTrack.recv() : %s", exc, exc_info=True)
            new_frame = frame   # Fallback : frame brute sans annotation

        return new_frame

    def stop(self):
        super().stop()
        if self._consume_task:
            self._consume_task.cancel()


# ─── Gestionnaire d'offre SDP ────────────────────────────────────────────────
async def handle_offer(sdp: str, sdp_type: str) -> dict:
    """
    Crée une RTCPeerConnection, branche le VideoTransformTrack,
    et retourne la réponse SDP.
    """
    offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
    pc = RTCPeerConnection()
    active_connections.add(pc)
    logger.info("Nouvelle connexion WebRTC — total: %d", len(active_connections))

    audio_sink = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_state_change():
        state = pc.connectionState
        logger.info("État WebRTC : %s", state)
        if state in ("failed", "closed", "disconnected"):
            await _cleanup(pc)

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        logger.info("Track reçu : kind=%s  id=%s", track.kind, track.id)
        if track.kind == "video":
            transformed = VideoTransformTrack(track)
            pc.addTrack(transformed)
            logger.info("VideoTransformTrack branché ✓")
        elif track.kind == "audio":
            audio_sink.addTrack(track)
            logger.info("Audio redirigé vers MediaBlackhole ✓")

    # Négociation SDP
    await pc.setRemoteDescription(offer)
    await audio_sink.start()
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("SDP answer prêt — type=%s", pc.localDescription.type)
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


async def _cleanup(pc: RTCPeerConnection) -> None:
    """Ferme proprement une connexion et la retire du registre."""
    try:
        await pc.close()
    except Exception:
        pass
    active_connections.discard(pc)
    logger.info("Connexion WebRTC fermée — actives: %d", len(active_connections))


async def close_all_connections() -> None:
    """Appelé au shutdown du serveur pour fermer toutes les connexions."""
    for pc in list(active_connections):
        await _cleanup(pc)
