/**
 * useTraining.js
 * ──────────────
 * Gestion des jobs d'entraînement ResNet + CIFAR-10.
 * - Démarre un job via POST /api/train
 * - Poll GET /api/train/{job_id} toutes les 2s pendant l'exécution
 */

import { useState, useCallback, useRef, useEffect } from 'react'

const API_BASE      = 'http://localhost:8000'
const POLL_INTERVAL = 2000   // ms

export function useTraining() {
  const [jobId, setJobId]             = useState(null)
  const [jobStatus, setJobStatus]     = useState(null)   // queued|running|completed|failed
  const [progress, setProgress]       = useState(0)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs]   = useState(0)
  const [metrics, setMetrics]         = useState({})
  const [history, setHistory]         = useState([])
  const [message, setMessage]         = useState('')
  const [error, setError]             = useState(null)
  const [loading, setLoading]         = useState(false)
  const [estimatedMinutes, setEstimatedMinutes] = useState(null)

  const pollTimerRef = useRef(null)
  const currentJobId = useRef(null)

  // ── Arrêt du polling ──────────────────────────────────────────────────────
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  // ── Polling du statut ─────────────────────────────────────────────────────
  const pollStatus = useCallback(async (jid) => {
    try {
      const res = await fetch(`${API_BASE}/api/train/${jid}`)
      if (!res.ok) return

      const data = await res.json()
      setJobStatus(data.status)
      setProgress(data.progress_percent)
      setCurrentEpoch(data.current_epoch)
      setTotalEpochs(data.total_epochs)
      setMetrics(data.metrics ?? {})
      setHistory(data.history ?? [])
      setMessage(data.message ?? '')

      if (data.status === 'failed') {
        setError(data.error || data.message)
        stopPolling()
      } else if (data.status === 'completed') {
        stopPolling()
      }
    } catch (err) {
      // Erreur réseau → on continue de poller
      console.warn('[useTraining] Erreur poll :', err.message)
    }
  }, [stopPolling])

  // ── Démarrage de l'entraînement ───────────────────────────────────────────
  const startTraining = useCallback(async (config) => {
    stopPolling()
    setLoading(true)
    setError(null)
    setJobId(null)
    setJobStatus(null)
    setProgress(0)
    setCurrentEpoch(0)
    setMetrics({})
    setHistory([])
    setMessage('')

    try {
      const res = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Erreur inconnue' }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      const jid  = data.job_id

      setJobId(jid)
      setJobStatus(data.status)
      setMessage(data.message ?? '')
      setEstimatedMinutes(data.estimated_duration_minutes ?? null)
      currentJobId.current = jid

      // Démarrer le polling
      await pollStatus(jid)
      pollTimerRef.current = setInterval(() => pollStatus(jid), POLL_INTERVAL)

    } catch (err) {
      setError(err.message)
      setJobStatus('failed')
    } finally {
      setLoading(false)
    }
  }, [stopPolling, pollStatus])

  // ── Reset ─────────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    stopPolling()
    setJobId(null)
    setJobStatus(null)
    setProgress(0)
    setCurrentEpoch(0)
    setTotalEpochs(0)
    setMetrics({})
    setHistory([])
    setMessage('')
    setError(null)
    setEstimatedMinutes(null)
    currentJobId.current = null
  }, [stopPolling])

  // Nettoyage au démontage
  useEffect(() => () => stopPolling(), [stopPolling])

  const isRunning = jobStatus === 'queued' || jobStatus === 'running'

  return {
    jobId,
    jobStatus,
    progress,
    currentEpoch,
    totalEpochs,
    metrics,
    history,
    message,
    error,
    loading,
    isRunning,
    estimatedMinutes,
    startTraining,
    reset,
  }
}
