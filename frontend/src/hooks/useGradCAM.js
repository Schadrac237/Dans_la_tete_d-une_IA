/**
 * useGradCAM.js
 * ─────────────
 * Capture une frame depuis la vidéo locale et envoie une requête Grad-CAM
 * au backend (POST /api/gradcam). Retourne la heatmap base64 résultante.
 */

import { useState, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

export function useGradCAM(localVideoRef) {
  const [heatmapB64, setHeatmapB64]   = useState(null)
  const [classLabel, setClassLabel]   = useState(null)
  const [targetClass, setTargetClass] = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [status, setStatus]           = useState(null)

  /**
   * Capture une frame de la vidéo locale et l'encode en base64 JPEG.
   * Retourne null si la vidéo n'est pas disponible.
   */
  const captureFrame = useCallback(() => {
    const video = localVideoRef?.current
    if (!video || video.readyState < 2) return null

    const canvas = document.createElement('canvas')
    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Retirer le préfixe "data:image/jpeg;base64,"
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92)
    return dataUrl.split(',')[1]
  }, [localVideoRef])

  /**
   * Lance le calcul Grad-CAM.
   * @param {Object} options
   * @param {string}      options.layerName    - Couche YOLO à visualiser
   * @param {number|null} options.targetClass  - Classe cible (null = auto)
   * @param {string|null} options.imageB64     - Image en base64 (null = capture webcam)
   */
  const runGradCAM = useCallback(async ({
    layerName    = 'model.model[21]',
    targetClass  = null,
    imageB64     = null,
  } = {}) => {
    setLoading(true)
    setError(null)
    setHeatmapB64(null)
    setClassLabel(null)

    try {
      const b64 = imageB64 ?? captureFrame()
      if (!b64) {
        throw new Error('Impossible de capturer la frame — vérifiez que la session WebRTC est active.')
      }

      const response = await fetch(`${API_BASE}/api/gradcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64:    b64,
          target_class: targetClass,
          layer_name:   layerName,
        }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Erreur inconnue' }))
        throw new Error(err.detail || `HTTP ${response.status}`)
      }

      const data = await response.json()
      setStatus(data.status)
      setHeatmapB64(data.heatmap_b64 ?? null)
      setClassLabel(data.class_label ?? null)
      setTargetClass(data.target_class ?? null)

      if (data.status !== 'ok') {
        setError(data.message)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [captureFrame])

  const reset = useCallback(() => {
    setHeatmapB64(null)
    setClassLabel(null)
    setTargetClass(null)
    setError(null)
    setStatus(null)
  }, [])

  return {
    heatmapB64,
    classLabel,
    targetClass,
    loading,
    error,
    status,
    runGradCAM,
    captureFrame,
    reset,
  }
}
