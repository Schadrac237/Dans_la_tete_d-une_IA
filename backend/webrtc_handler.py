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

# ─── Ensemble des connexions actives (pour cleanup) ───────────────────────────
active_connections: set[RTCPeerConnection] = set()


# ─── Track transformée : YOLO sur chaque frame ───────────────────────────────
class VideoTransformTrack(MediaStreamTrack):
    """
    Prend un MediaStreamTrack vidéo entrant et produit un flux annoté par YOLO.

    Optimisation anti-latence : si une inférence YOLO est déjà en cours,
    la frame courante est renvoyée brute (frame skipping) plutôt que
    d'accumuler un retard en file d'attente.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self._track = track
        self._frame_count = 0
        self._fps_count = 0
        self._last_fps_time = time.monotonic()
        self._fps = 0.0
        self._processing = False   # Verrou logique anti-accumulation de latence

    async def recv(self) -> av.VideoFrame:
        # Recevoir la prochaine frame depuis le client
        frame: av.VideoFrame = await self._track.recv()

        # ── Calcul FPS ────────────────────────────────────────────────────────
        self._fps_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._last_fps_time = now

        # ── Frame skipping : évite l'accumulation de latence ──────────────────
        # Si YOLO tourne déjà (inférence ~30-80ms sur CPU), on renvoie la frame
        # brute immédiatement plutôt que de créer une file d'attente.
        if self._processing:
            return frame

        self._processing = True
        try:
            # av.VideoFrame → numpy BGR
            img_bgr: np.ndarray = frame.to_ndarray(format="bgr24")

            # Inférence YOLO dans le thread pool (non-bloquant pour asyncio)
            loop = asyncio.get_running_loop()   # ← fix bug 4 (get_event_loop déprécié)
            annotated_bgr, _ = await loop.run_in_executor(
                None, yolo_processor.process_frame, img_bgr
            )

            # HUD FPS (OpenCV importé au niveau module, pas ici)
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

            # ── Fix bug 5 : pts peut être None (crash encodeur) ────────────────
            # On copie les timestamps de la frame source si disponibles,
            # sinon on utilise un compteur avec une time_base fixe à 90kHz
            # (standard RTP video).
            if frame.pts is not None:
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
            else:
                self._frame_count += 1
                new_frame.pts = self._frame_count
                new_frame.time_base = Fraction(1, 90000)

        except Exception as exc:
            # Log complet (pas juste warning) pour déboguer
            logger.error(
                "Erreur VideoTransformTrack.recv() : %s", exc, exc_info=True
            )
            new_frame = frame   # Fallback : frame brute sans annotation

        finally:
            self._processing = False

        return new_frame


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
