/**
 * useWebSocket.js
 * ────────────────
 * Hook React gérant la connexion WebSocket avec le backend FastAPI.
 * Utilisé pour envoyer le seuil de confiance YOLO en temps réel.
 *
 * Fonctionnalités :
 *   - Connexion automatique à /ws/control
 *   - Reconnexion exponentielle (max 5 tentatives)
 *   - Envoi de messages JSON typés
 *   - Accusé de réception du backend
 */

import { useRef, useState, useCallback, useEffect } from 'react'

const MAX_RETRIES     = 5
const RETRY_BASE_MS   = 1000   // délai initial de reconnexion

/**
 * @typedef {'disconnected' | 'connecting' | 'connected' | 'error'} WSStatus
 */

/**
 * @param {boolean} enabled - Active la connexion WebSocket
 * @returns {{
 *   wsStatus: WSStatus,
 *   sendConfidence: (value: number) => void,
 *   confirmedConfidence: number | null,
 *   sendLiveGradCAMConfig: (enabled: boolean, layer: string, targetClass: number|null) => void,
 *   liveGradCAMStatus: { enabled: boolean, layer: string, targetClass: number|null }
 * }}
 */
export function useWebSocket(enabled = false) {
  const wsRef    = useRef(null)
  const retryRef = useRef(0)
  const timerRef = useRef(null)

  const [wsStatus, setWsStatus]                   = useState('disconnected')
  const [confirmedConfidence, setConfirmedConf]   = useState(null)
  const [liveGradCAMStatus, setLiveGradCAMStatus] = useState({
    enabled: false,
    layer: 'model.model[21]',
    targetClass: null
  })
  const [customModelEnabled, setCustomModelEnabled] = useState(false)

  // ── Connexion ────────────────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    retryRef.current = 0    // ← reset au début de chaque connexion intentionnelle
    setWsStatus('connecting')
    const ws = new WebSocket(`ws://${window.location.host}/ws/control`)
    wsRef.current = ws

    ws.onopen = () => {
      console.info('[WS] Connecté à /ws/control')
      setWsStatus('connected')
      retryRef.current = 0
      // Demander le statut initial
      ws.send(JSON.stringify({ type: 'get_status' }))
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'ack' || msg.type === 'status') {
          if (msg.confidence !== undefined) setConfirmedConf(msg.confidence)
          if (msg.gradcam !== undefined) setLiveGradCAMStatus(msg.gradcam)
          if (msg.custom_model !== undefined) setCustomModelEnabled(msg.custom_model)
        } else if (msg.type === 'error') {
          console.warn('[WS] Erreur backend :', msg.message)
        }
      } catch (e) {
        console.warn('[WS] Message non-JSON :', event.data)
      }
    }

    ws.onerror = (err) => {
      console.error('[WS] Erreur :', err)
      setWsStatus('error')
    }

    ws.onclose = (event) => {
      console.info('[WS] Fermé (code=%d)', event.code)
      setWsStatus('disconnected')
      wsRef.current = null

      // Reconnexion exponentielle si le composant est toujours monté
      if (retryRef.current < MAX_RETRIES) {
        const delay = RETRY_BASE_MS * Math.pow(2, retryRef.current)
        retryRef.current++
        console.info('[WS] Reconnexion dans %dms (tentative %d)', delay, retryRef.current)
        timerRef.current = setTimeout(connect, delay)
      } else {
        setWsStatus('error')
      }
    }
  }, [])

  // ── Déconnexion propre ───────────────────────────────────────────────────────
  const disconnect = useCallback(() => {
    clearTimeout(timerRef.current)
    retryRef.current = MAX_RETRIES // empêche la reconnexion auto
    wsRef.current?.close()
    wsRef.current = null
    setWsStatus('disconnected')
  }, [])

  // ── Connexion/déconnexion selon `enabled` ────────────────────────────────────
  useEffect(() => {
    if (enabled) {
      connect()
    } else {
      disconnect()
    }
    return () => {
      clearTimeout(timerRef.current)
    }
  }, [enabled, connect, disconnect])

  // ── Envoi du seuil de confiance ──────────────────────────────────────────────
  const sendConfidence = useCallback((value) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type:  'set_confidence',
        value: value,
      }))
    } else {
      console.warn('[WS] Impossible d\'envoyer : WebSocket non connecté')
    }
  }, [])

  // ── Envoi de la config Grad-CAM ──────────────────────────────────────────────
  const sendLiveGradCAMConfig = useCallback((enabled, layerName = 'model.model[21]', targetClass = null) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'set_live_gradcam',
        enabled,
        layerName,
        targetClass
      }))
      setLiveGradCAMStatus({ enabled, layer: layerName, targetClass })
    } else {
      console.warn('[WS] Impossible d\'envoyer : WebSocket non connecté')
    }
  }, [])

  // ── Envoi du mode modèle personnalisé ────────────────────────────────────────
  const sendCustomModelMode = useCallback((enabled) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'set_custom_model_mode',
        enabled
      }))
      setCustomModelEnabled(enabled)
    } else {
      console.warn('[WS] Impossible d\'envoyer : WebSocket non connecté')
    }
  }, [])

  return { 
    wsStatus, 
    sendConfidence, 
    confirmedConfidence, 
    sendLiveGradCAMConfig, 
    liveGradCAMStatus,
    sendCustomModelMode,
    customModelEnabled
  }
}
