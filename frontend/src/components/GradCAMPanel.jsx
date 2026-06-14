/**
 * GradCAMPanel.jsx
 * ─────────────────
 * Panneau de visualisation Grad-CAM.
 * Permet de capturer une frame de la webcam et d'afficher
 * la carte d'attention du réseau YOLOv8n sur cette image.
 */

import React, { useState } from 'react'
import { useGradCAM } from '../hooks/useGradCAM'

// 80 classes COCO (numérotées 0–79)
const COCO_CLASSES = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck',
  'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
  'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
  'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
  'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
  'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
  'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
  'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
  'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
  'toaster','sink','refrigerator','book','clock','vase','scissors',
  'teddy bear','hair drier','toothbrush',
]

const YOLO_LAYERS = [
  { value: 'model.model[21]', label: 'Dernière couche C2f (recommandé)' },
  { value: 'model.model[18]', label: 'Neck — C2f 18' },
  { value: 'model.model[-4]', label: 'Neck — C2f 4' },
  { value: 'model.model[9]',  label: 'Backbone — C2f 9' },
  { value: 'model.model[6]',  label: 'Backbone — C2f 6' },
]

export function GradCAMPanel({ localVideoRef }) {
  const [layerName, setLayerName]       = useState('model.model[21]')
  const [targetClass, setTargetClass]   = useState('')   // '' = auto

  const {
    heatmapB64,
    classLabel,
    targetClass: detectedClass,
    loading,
    error,
    status,
    runGradCAM,
    reset,
  } = useGradCAM(localVideoRef)

  const handleCapture = () => {
    runGradCAM({
      layerName,
      targetClass: targetClass === '' ? null : parseInt(targetClass, 10),
    })
  }

  return (
    <section className="ml-panel gradcam-panel" aria-label="Panneau Grad-CAM">
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <span className="icon">🌡️</span> Grad-CAM — Zones d'attention
          </span>
          <span className="ml-badge">YOLOv8n · PyTorch</span>
        </div>

        {/* ── Contrôles ───────────────────────────────────────────────────── */}
        <div className="gradcam-controls">

          {/* Sélecteur de couche */}
          <div className="select-group">
            <label htmlFor="select-layer" className="select-label">Couche visualisée</label>
            <select
              id="select-layer"
              className="styled-select"
              value={layerName}
              onChange={e => setLayerName(e.target.value)}
              disabled={loading}
            >
              {YOLO_LAYERS.map(l => (
                <option key={l.value} value={l.value}>{l.label}</option>
              ))}
            </select>
          </div>

          {/* Sélecteur de classe */}
          <div className="select-group">
            <label htmlFor="select-class" className="select-label">Classe cible</label>
            <select
              id="select-class"
              className="styled-select"
              value={targetClass}
              onChange={e => setTargetClass(e.target.value)}
              disabled={loading}
            >
              <option value="">🔍 Auto (classe max)</option>
              {COCO_CLASSES.map((name, i) => (
                <option key={i} value={i}>{i} — {name}</option>
              ))}
            </select>
          </div>

          {/* Bouton capture */}
          <button
            id="btn-gradcam-capture"
            className="capture-btn"
            onClick={handleCapture}
            disabled={loading}
            aria-label="Capturer une frame et calculer Grad-CAM"
          >
            {loading ? (
              <><span className="spin-icon">⚙️</span> Calcul en cours…</>
            ) : (
              <><span>📸</span> Capturer &amp; Analyser</>
            )}
          </button>
        </div>

        {/* ── Affichage résultat ───────────────────────────────────────────── */}
        <div className="gradcam-output">
          {!heatmapB64 && !loading && !error && (
            <div className="gradcam-placeholder">
              <span className="placeholder-icon" style={{ fontSize: '2.5rem', opacity: 0.3 }}>🌡️</span>
              <span style={{ color: 'var(--clr-text-dim)', fontSize: '0.82rem', fontFamily: 'var(--font-mono)' }}>
                Lancez une session WebRTC puis cliquez « Capturer »
              </span>
            </div>
          )}

          {loading && (
            <div className="gradcam-placeholder">
              <div className="gradcam-loading-ring" />
              <span style={{ color: 'var(--clr-accent)', fontSize: '0.82rem', fontFamily: 'var(--font-mono)' }}>
                Chargement du modèle + calcul des gradients…
              </span>
            </div>
          )}

          {error && !loading && (
            <div className="gradcam-error">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          {heatmapB64 && !loading && (
            <div className="heatmap-container">
              <img
                src={`data:image/jpeg;base64,${heatmapB64}`}
                alt="Carte Grad-CAM"
                className="heatmap-img"
              />
              {/* Légende */}
              <div className="heatmap-legend">
                <div className="legend-gradient" />
                <div className="legend-labels">
                  <span>Faible activation</span>
                  <span>Forte activation</span>
                </div>
                {classLabel && (
                  <div className="heatmap-class-badge">
                    <span>Classe : </span>
                    <strong>{classLabel}</strong>
                    {detectedClass !== null && (
                      <span style={{ color: 'var(--clr-text-dim)', marginLeft: '4px' }}>
                        (id {detectedClass})
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Bouton reset */}
              <button
                className="heatmap-reset-btn"
                onClick={reset}
                aria-label="Effacer la heatmap"
              >
                ✕ Effacer
              </button>
            </div>
          )}
        </div>

        {/* ── Info bas de carte ────────────────────────────────────────────── */}
        <div className="gradcam-footer">
          <span className="gradcam-info">
            🔬 Grad-CAM (Gradient-weighted Class Activation Mapping) visualise
            les régions de l'image qui ont le plus influencé la décision du réseau.
          </span>
        </div>
      </div>
    </section>
  )
}
