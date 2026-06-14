/**
 * App.jsx
 * ────────
 * Composant racine — assemble VideoPanel + ControlPanel + GradCAMPanel + TrainingPanel
 * et orchestre les hooks useWebRTC et useWebSocket.
 */

import React from 'react'
import { useWebRTC }    from './hooks/useWebRTC'
import { useWebSocket } from './hooks/useWebSocket'
import { useTraining }  from './hooks/useTraining'
import { VideoPanel }   from './components/VideoPanel'
import { ControlPanel } from './components/ControlPanel'
import { TrainingPanel } from './components/TrainingPanel'

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

  const { startTraining, trainingStatus } = useTraining()

  const isSessionActive = status === 'connected'

  const { wsStatus, sendConfidence, confirmedConfidence, sendLiveGradCAMConfig, liveGradCAMStatus, sendCustomModelMode, customModelEnabled } = useWebSocket(isSessionActive)

  const statusInfo = STATUS_DISPLAY[status] || STATUS_DISPLAY.idle

  return (
    <div className="app-wrapper">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="app-header" role="banner">
        <div className="header-logo">
          <div className="header-logo-icon" aria-hidden="true">🧠</div>
          <div>
            <div className="header-title">Dans la tête d'une IA</div>
            <div className="header-subtitle">YOLOv8n · ResNet · WebRTC · FastAPI · Real-Time</div>
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

        {/* Colonne gauche (Video + Training) */}
        <div className="left-column">
          {/* Panneau vidéo */}
          <VideoPanel
            localVideoRef={localVideoRef}
            remoteVideoRef={remoteVideoRef}
            status={status}
            errorMessage={errorMessage}
            onStart={startSession}
            onStop={stopSession}
          />

          {/* ── Séparateur ML ────────────────────────────────────────────────── */}
          <div className="ml-section-header" style={{ gridColumn: 'unset' }}>
            <div className="ml-section-line" />
            <span className="ml-section-label">🔬 Outils d'analyse IA</span>
            <div className="ml-section-line" />
          </div>

          {/* Panneau Transfer Learning */}
          <TrainingPanel
            onStartTraining={startTraining}
            trainingStatus={trainingStatus}
          />
        </div>

        {/* Panneau de contrôle */}
        <ControlPanel
          wsStatus={wsStatus}
          sendConfidence={sendConfidence}
          confirmedConfidence={confirmedConfidence}
          sendLiveGradCAMConfig={sendLiveGradCAMConfig}
          liveGradCAMStatus={liveGradCAMStatus}
          sendCustomModelMode={sendCustomModelMode}
          customModelEnabled={customModelEnabled}
          isSessionActive={isSessionActive}
        />

      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="app-footer" role="contentinfo">
        Dans la tête d'une IA · v2.0 · YOLOv8n + ResNet18 + CIFAR-10 + aiortc + FastAPI + React
      </footer>

    </div>
  )
}
