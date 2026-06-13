/**
 * ControlPanel.jsx
 * ─────────────────
 * Panneau de contrôle droit :
 *   - Slider de seuil de confiance YOLO (envoyé en temps réel via WebSocket)
 *   - Métriques en direct (latence estimée, objets détectés, état WS)
 *   - Liste des dernières détections
 *   - Sections stubs pour les fonctionnalités futures
 */

import React, { useCallback, useRef } from 'react'

const DEBOUNCE_MS = 80   // évite de spammer le WS à chaque pixel de slider

export function ControlPanel({
  wsStatus,
  sendConfidence,
  confirmedConfidence,
  isSessionActive,
}) {
  const confidenceRef   = useRef(50)       // valeur locale non-réactive (perf)
  const debounceTimer   = useRef(null)
  const [localConf, setLocalConf] = React.useState(50)

  // ── Envoi debounced ────────────────────────────────────────────────────────
  const handleSliderChange = useCallback((e) => {
    const pct = Number(e.target.value)       // 0 – 100
    setLocalConf(pct)
    confidenceRef.current = pct

    clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      sendConfidence(pct / 100)              // backend attend 0.0 – 1.0
    }, DEBOUNCE_MS)
  }, [sendConfidence])

  // ── Couleur de la barre du slider (CSS custom property trick) ─────────────
  const sliderBg = `linear-gradient(
    to right,
    var(--clr-accent) 0%,
    var(--clr-accent-2) ${localConf}%,
    var(--clr-surface-2) ${localConf}%
  )`

  const wsBadgeClass = `ws-badge ${
    wsStatus === 'connected'    ? 'ws-connected'    :
    wsStatus === 'disconnected' ? 'ws-disconnected' : ''
  }`

  return (
    <aside className="control-panel" aria-label="Panneau de contrôle">

      {/* ── Seuil de confiance ──────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <span className="icon">🎯</span> Seuil de confiance
          </span>
          <span className={wsBadgeClass}>
            <span style={{
              width: '6px', height: '6px', borderRadius: '50%', display: 'inline-block',
              background: wsStatus === 'connected' ? 'var(--clr-success)' :
                          wsStatus === 'disconnected' ? 'var(--clr-danger)' : 'var(--clr-warning)',
              flexShrink: 0,
            }} />
            WS
          </span>
        </div>
        <div className="control-section">
          <div className="slider-label-row">
            <span className="slider-label">Confiance min.</span>
            <span className="slider-value">{localConf}%</span>
          </div>
          <div className="slider-wrapper">
            <input
              id="slider-confidence"
              type="range"
              min={0}
              max={100}
              step={1}
              value={localConf}
              onChange={handleSliderChange}
              disabled={!isSessionActive}
              style={{ background: sliderBg }}
              aria-label="Seuil de confiance YOLO"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={localConf}
            />
          </div>
          <div className="slider-ticks">
            {['0%', '25%', '50%', '75%', '100%'].map(t => (
              <span key={t} className="slider-tick">{t}</span>
            ))}
          </div>
          {confirmedConfidence !== null && (
            <p style={{
              fontSize: '0.7rem',
              color: 'var(--clr-success)',
              fontFamily: 'var(--font-mono)',
              marginTop: '8px',
              textAlign: 'right',
            }}>
              ✓ Serveur : {(confirmedConfidence * 100).toFixed(0)}%
            </p>
          )}
          {!isSessionActive && (
            <p style={{
              fontSize: '0.7rem',
              color: 'var(--clr-text-dim)',
              fontFamily: 'var(--font-mono)',
              marginTop: '8px',
            }}>
              Lancez une session pour activer le contrôle.
            </p>
          )}
        </div>
      </div>

      {/* ── Fonctionnalités à venir ─────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <span className="icon">🔬</span> Prochainement
          </span>
        </div>
        <div className="stub-section">
          <StubBadge
            icon="🌡️"
            label="Grad-CAM"
            desc="Visualisation des zones d'attention du réseau de neurones sur l'image"
            endpoint="POST /api/gradcam"
          />
          <StubBadge
            icon="🏋️"
            label="Transfer Learning"
            desc="Entraîner YOLOv8n sur vos propres données directement dans l'interface"
            endpoint="POST /api/train"
          />
        </div>
      </div>

    </aside>
  )
}

// ── Composant stub badge ──────────────────────────────────────────────────────
function StubBadge({ icon, label, desc, endpoint }) {
  return (
    <div className="stub-badge" title={`Route disponible : ${endpoint}`}>
      <span className="stub-icon">{icon}</span>
      <div>
        <div className="stub-label">{label}</div>
        <div className="stub-desc">{desc}</div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.62rem',
          color: 'var(--clr-text-dim)',
          marginTop: '3px',
        }}>
          {endpoint} · stub
        </div>
      </div>
    </div>
  )
}
