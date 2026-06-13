/**
 * VideoPanel.jsx
 * ───────────────
 * Affiche la webcam locale (brute) et le stream distant (annoté par YOLO).
 * Gère les états : idle, demande caméra, connexion, erreur.
 */

import React from 'react'

const STATUS_LABELS = {
  idle:              'Prêt',
  requesting_camera: 'Accès caméra…',
  connecting:        'Connexion WebRTC…',
  connected:         'Connecté',
  error:             'Erreur',
  closed:            'Déconnecté',
}

export function VideoPanel({ localVideoRef, remoteVideoRef, status, errorMessage, onStart, onStop }) {
  const isActive     = status === 'connected' || status === 'connecting'
  const isConnecting = status === 'requesting_camera' || status === 'connecting'
  const isError      = status === 'error'

  return (
    <section className="video-panel" aria-label="Panneau vidéo">

      {/* Grille double vidéo */}
      <div className="video-grid">

        {/* Webcam locale */}
        <div className="card video-card">
          <span className="video-label">📷 Webcam locale</span>
          <video
            ref={localVideoRef}
            autoPlay
            playsInline
            muted
            aria-label="Flux webcam local"
          />
          {!isActive && !isError && (
            <div className="video-placeholder">
              <span className="placeholder-icon">🎥</span>
              <span>Flux local</span>
              <span style={{ color: 'var(--clr-text-dim)', fontSize: '0.72rem' }}>
                Lancez la session pour activer la caméra
              </span>
            </div>
          )}
          {isError && (
            <div className="video-error-overlay">
              ⚠️ {errorMessage || 'Erreur caméra'}
            </div>
          )}
        </div>

        {/* Stream traité par YOLO */}
        <div className="card video-card">
          <span className="video-label">🤖 Vue IA — YOLOv8n</span>
          <video
            ref={remoteVideoRef}
            autoPlay
            playsInline
            aria-label="Flux traité par l'IA"
          />
          {status !== 'connected' && (
            <div className="video-placeholder">
              <span className="placeholder-icon">
                {isConnecting ? '⏳' : '🧠'}
              </span>
              <span>
                {isConnecting
                  ? STATUS_LABELS[status]
                  : 'Flux IA (YOLOv8n)'}
              </span>
              {isConnecting && (
                <LoadingDots />
              )}
              {!isConnecting && !isError && (
                <span style={{ color: 'var(--clr-text-dim)', fontSize: '0.72rem' }}>
                  Le flux annoté apparaîtra ici
                </span>
              )}
            </div>
          )}
        </div>

      </div>

      {/* Bouton principal */}
      <div style={{ display: 'flex', justifyContent: 'center', padding: '0 0 4px' }}>
        {!isActive ? (
          <button
            id="btn-start-session"
            className="btn-start"
            onClick={onStart}
            disabled={isConnecting}
            aria-label="Démarrer la session WebRTC"
          >
            {isConnecting ? (
              <>⏳ {STATUS_LABELS[status]}…</>
            ) : (
              <>▶ Démarrer la session</>
            )}
          </button>
        ) : (
          <button
            id="btn-stop-session"
            className="btn-start btn-stop"
            onClick={onStop}
            aria-label="Arrêter la session WebRTC"
          >
            ⏹ Arrêter la session
          </button>
        )}
      </div>

    </section>
  )
}

// ── Indicateur de chargement animé ───────────────────────────────────────────
function LoadingDots() {
  return (
    <span style={{ display: 'flex', gap: '4px', marginTop: '4px' }}>
      {[0, 1, 2].map(i => (
        <span
          key={i}
          style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: 'var(--clr-accent)',
            animation: `pulse-warning 1.2s ${i * 0.2}s infinite`,
            display: 'inline-block',
          }}
        />
      ))}
    </span>
  )
}
