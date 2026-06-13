/**
 * useWebRTC.js
 * ─────────────
 * Hook React gérant l'intégralité du cycle de vie WebRTC :
 *   - Capture de la webcam locale (getUserMedia)
 *   - Création de la RTCPeerConnection
 *   - Négociation SDP offre/réponse avec le backend FastAPI
 *   - Gestion des ICE candidates
 *   - Réception du stream vidéo transformé (YOLO)
 *   - Nettoyage propre à la déconnexion
 */

import { useRef, useState, useCallback, useEffect } from 'react'

const ICE_SERVERS = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
]

/**
 * @typedef {'idle' | 'requesting_camera' | 'connecting' | 'connected' | 'error' | 'closed'} RTCStatus
 */

/**
 * @returns {{
 *   localVideoRef: React.RefObject,
 *   remoteVideoRef: React.RefObject,
 *   status: RTCStatus,
 *   errorMessage: string | null,
 *   startSession: () => Promise<void>,
 *   stopSession: () => void,
 * }}
 */
export function useWebRTC() {
  const localVideoRef  = useRef(null)
  const remoteVideoRef = useRef(null)
  const pcRef          = useRef(null)   // RTCPeerConnection
  const streamRef      = useRef(null)   // MediaStream local

  const [status, setStatus]           = useState('idle')
  const [errorMessage, setErrorMessage] = useState(null)

  // ── Nettoyage ────────────────────────────────────────────────────────────────
  const stopSession = useCallback(() => {
    // Arrêter la webcam locale
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }

    // Fermer la PeerConnection
    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }

    // Vider les éléments vidéo
    if (localVideoRef.current)  localVideoRef.current.srcObject  = null
    if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null

    setStatus('closed')
  }, [])

  // ── Démarrage de session ──────────────────────────────────────────────────────
  const startSession = useCallback(async () => {
    if (status === 'connecting' || status === 'connected') return

    setErrorMessage(null)
    setStatus('requesting_camera')

    let localStream
    try {
      localStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      })
    } catch (err) {
      const msg = err.name === 'NotAllowedError'
        ? "Accès caméra refusé. Autorisez l'accès dans les paramètres du navigateur."
        : err.name === 'NotFoundError'
        ? 'Aucune caméra détectée sur cet appareil.'
        : `Erreur caméra : ${err.message}`
      setErrorMessage(msg)
      setStatus('error')
      return
    }

    streamRef.current = localStream
    if (localVideoRef.current) {
      localVideoRef.current.srcObject = localStream
    }

    setStatus('connecting')

    // Création RTCPeerConnection
    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS })
    pcRef.current = pc

    // Ajout des tracks locaux
    localStream.getTracks().forEach(track => pc.addTrack(track, localStream))

    // Réception du stream distant (YOLO-processed)
    pc.ontrack = (event) => {
      if (remoteVideoRef.current && event.streams[0]) {
        remoteVideoRef.current.srcObject = event.streams[0]
      }
    }

    // Changement d'état de connexion
    pc.onconnectionstatechange = () => {
      const state = pc.connectionState
      console.info('[WebRTC] connectionState:', state)
      switch (state) {
        case 'connected':
          setStatus('connected')
          break
        case 'disconnected':
        case 'failed':
          setErrorMessage(`Connexion WebRTC perdue (${state}). Relancez la session.`)
          setStatus('error')
          stopSession()
          break
        case 'closed':
          setStatus('closed')
          break
      }
    }

    pc.onicecandidateerror = (e) => {
      console.warn('[WebRTC] ICE candidate error:', e.errorText)
    }

    // Création et envoi de l'offre SDP
    try {
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)

      // Attendre la fin de la collecte ICE avant d'envoyer
      await waitForIceGathering(pc)

      const response = await fetch('/api/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp:  pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      })

      if (!response.ok) {
        throw new Error(`Erreur serveur : ${response.status} ${response.statusText}`)
      }

      const answer = await response.json()
      await pc.setRemoteDescription(new RTCSessionDescription(answer))

    } catch (err) {
      console.error('[WebRTC] Erreur de négociation :', err)
      setErrorMessage(`Impossible de se connecter au serveur : ${err.message}`)
      setStatus('error')
      stopSession()
    }
  }, [status, stopSession])

  // Nettoyage au démontage du composant
  useEffect(() => {
    return () => { stopSession() }
  }, [stopSession])

  return { localVideoRef, remoteVideoRef, status, errorMessage, startSession, stopSession }
}

// ── Utilitaire : attendre la fin de la collecte ICE ───────────────────────────
function waitForIceGathering(pc) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') {
      resolve()
      return
    }
    const timeout = setTimeout(resolve, 3000) // timeout 3s
    pc.addEventListener('icegatheringstatechange', function handler() {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(timeout)
        pc.removeEventListener('icegatheringstatechange', handler)
        resolve()
      }
    })
  })
}
