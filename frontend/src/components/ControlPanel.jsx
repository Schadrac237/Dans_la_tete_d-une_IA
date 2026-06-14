/**
 * ControlPanel.jsx
 * ─────────────────
 * Panneau de contrôle droit :
 *   - Slider de seuil de confiance YOLO (envoyé en temps réel via WebSocket)
 *   - Statut WebSocket
 */

import React, { useCallback, useRef } from 'react'

const DEBOUNCE_MS = 80   // évite de spammer le WS à chaque pixel de slider

const YOLO_LAYERS = [
  { value: 'model.model[21]', label: 'Dernière couche C2f' },
  { value: 'model.model[18]', label: 'Neck — C2f 18' },
]

const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
]

export function ControlPanel({
  wsStatus,
  sendConfidence,
  confirmedConfidence,
  sendLiveGradCAMConfig,
  liveGradCAMStatus,
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

      {/* ── Mode Live Grad-CAM ──────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <span className="icon">🔥</span> Live Grad-CAM
          </span>
        </div>
        <div className="control-section">
          <div className="toggle-row" style={{ marginBottom: 'var(--gap-sm)', background: 'transparent', padding: 0, border: 'none' }}>
            <label className="toggle-label" htmlFor="live-gradcam-toggle" style={{ width: '100%' }}>
              <span>
                <strong>Activer le rendu en direct</strong>
                <span className="toggle-desc">Remplace la détection standard par la Heatmap Grad-CAM via GPU.</span>
              </span>
              <div className="toggle-switch-wrapper">
                <input
                  id="live-gradcam-toggle"
                  type="checkbox"
                  className="toggle-input"
                  checked={liveGradCAMStatus?.enabled || false}
                  onChange={e => sendLiveGradCAMConfig(e.target.checked, liveGradCAMStatus?.layer || 'model.model[21]', liveGradCAMStatus?.targetClass)}
                  disabled={!isSessionActive}
                />
                <div className="toggle-switch" />
              </div>
            </label>
          </div>

          {liveGradCAMStatus?.enabled && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--gap-xs)', marginTop: 'var(--gap-md)' }}>
              <div className="select-group">
                <label className="select-label" style={{ fontSize: '0.65rem' }}>Couche</label>
                <select
                  className="styled-select"
                  style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                  value={liveGradCAMStatus.layer}
                  onChange={e => sendLiveGradCAMConfig(true, e.target.value, liveGradCAMStatus.targetClass)}
                >
                  {YOLO_LAYERS.map(l => (
                    <option key={l.value} value={l.value}>{l.label}</option>
                  ))}
                </select>
              </div>

              <div className="select-group">
                <label className="select-label" style={{ fontSize: '0.65rem' }}>Classe cible</label>
                <select
                  className="styled-select"
                  style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                  value={liveGradCAMStatus.targetClass === null ? '' : liveGradCAMStatus.targetClass}
                  onChange={e => sendLiveGradCAMConfig(true, liveGradCAMStatus.layer, e.target.value === '' ? null : e.target.value)}
                >
                  <option value="">🔍 Auto (max)</option>
                  {COCO_CLASSES.map((c, i) => (
                    <option key={i} value={i}>{c}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Infos modèle ────────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <span className="icon">🤖</span> Modèles actifs
          </span>
        </div>
        <div className="models-info-section">
          <div className="model-info-item">
            <span className="model-info-icon">⚡</span>
            <div>
              <div className="model-info-name">YOLOv8n ONNX</div>
              <div className="model-info-desc">Détection temps réel · WebRTC · CPU</div>
            </div>
            <span className="model-info-badge live">live</span>
          </div>
          <div className="model-info-item">
            <span className="model-info-icon">🌡️</span>
            <div>
              <div className="model-info-name">YOLOv8n PyTorch</div>
              <div className="model-info-desc">Grad-CAM · chargement à la demande</div>
            </div>
            <span className="model-info-badge lazy">lazy</span>
          </div>
          <div className="model-info-item">
            <span className="model-info-icon">🏋️</span>
            <div>
              <div className="model-info-name">ResNet-18/50</div>
              <div className="model-info-desc">Transfer Learning · CIFAR-10</div>
            </div>
            <span className="model-info-badge train">train</span>
          </div>
        </div>
      </div>

    </aside>
  )
}
