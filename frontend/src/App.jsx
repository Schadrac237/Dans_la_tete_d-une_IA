/**
 * App.jsx
 * ────────
 * Composant racine — assemble VideoPanel + ControlPanel
 * et orchestre les hooks useWebRTC et useWebSocket.
 */

import React from 'react'
import { useWebRTC }    from './hooks/useWebRTC'
import { useWebSocket } from './hooks/useWebSocket'
import { VideoPanel }   from './components/VideoPanel'
import { ControlPanel } from './components/ControlPanel'

const STATUS_DISPLAY = {
  idle:              { label: 'Prêt',             dot: '' },
  requesting_camera: { label: 'Accès caméra…',    dot: 'connecting' },
  connecting:        { label: 'Connexion…',        dot: 'connecting' },
  connected:         { label: 'Session active',    dot: 'connected' },
  error:             { label: 'Erreur',            dot: 'error' },
  closed:            { label: 'Déconnecté',        dot: '' },
}

export default function App() {
  const {
    localVideoRef,
    remoteVideoRef,
    status,
    errorMessage,
    startSession,
    stopSession,
  } = useWebRTC()

  const isSessionActive = status === 'connected'

  const { wsStatus, sendConfidence, confirmedConfidence } = useWebSocket(isSessionActive)

  const statusInfo = STATUS_DISPLAY[status] || STATUS_DISPLAY.idle

  return (
    <div className="app-wrapper">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="app-header" role="banner">
        <div className="header-logo">
          <div className="header-logo-icon" aria-hidden="true">🧠</div>
          <div>
            <div className="header-title">Dans la tête d'une IA</div>
            <div className="header-subtitle">YOLOv8n · WebRTC · FastAPI · Real-Time</div>
          </div>
        </div>

        <div className="header-status" role="status" aria-live="polite" aria-label="État de la session">
          <span className={`status-dot ${statusInfo.dot}`} aria-hidden="true" />
          {statusInfo.label}
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────────────── */}
      <main className="app-main" id="main-content">

        {/* Titre de section */}
        <div className="section-header">
          <div>
            <h1 className="section-title">
              Détection d'objets <span>en temps réel</span>
            </h1>
            <p className="section-desc">
              Votre webcam est analysée frame par frame par YOLOv8n.
              Les objets détectés sont annotés et renvoyés en live via WebRTC (&lt; 300 ms).
            </p>
          </div>
        </div>

        {/* Panneau vidéo */}
        <VideoPanel
          localVideoRef={localVideoRef}
          remoteVideoRef={remoteVideoRef}
          status={status}
          errorMessage={errorMessage}
          onStart={startSession}
          onStop={stopSession}
        />

        {/* Panneau de contrôle */}
        <ControlPanel
          wsStatus={wsStatus}
          sendConfidence={sendConfidence}
          confirmedConfidence={confirmedConfidence}
          isSessionActive={isSessionActive}
        />

      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="app-footer" role="contentinfo">
        Dans la tête d'une IA · MVP v1.0 · YOLOv8n + aiortc + FastAPI + React
      </footer>

    </div>
  )
}
