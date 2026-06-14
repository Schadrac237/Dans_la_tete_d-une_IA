/**
 * TrainingPanel.jsx
 * ──────────────────
 * Panneau de Transfer Learning ResNet + CIFAR-10.
 * Permet de configurer et lancer un entraînement,
 * puis de suivre la progression en temps réel.
 */

import React, { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useTraining } from '../hooks/useTraining'

const CIFAR10_CLASSES = [
  'avion','auto','oiseau','chat','cerf',
  'chien','grenouille','cheval','bateau','camion',
]

const STATUS_CONFIG = {
  queued:    { label: 'En attente',   color: 'var(--clr-warning)',  icon: '⏳' },
  running:   { label: 'En cours',     color: 'var(--clr-accent)',   icon: '⚡' },
  completed: { label: 'Terminé ✓',    color: 'var(--clr-success)',  icon: '🎯' },
  failed:    { label: 'Échec',        color: 'var(--clr-danger)',   icon: '✕'  },
  not_found: { label: 'Introuvable',  color: 'var(--clr-text-dim)', icon: '?' },
}

export function TrainingPanel() {
  const [config, setConfig] = useState({
    base_model:      'resnet18',
    epochs:          5,
    learning_rate:   0.001,
    freeze_backbone: true,
    batch_size:      64,
  })

  const {
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
  } = useTraining()

  const handleChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    startTraining({
      ...config,
      epochs:       parseInt(config.epochs, 10),
      batch_size:   parseInt(config.batch_size, 10),
      learning_rate: parseFloat(config.learning_rate),
    })
  }

  const statusInfo = STATUS_CONFIG[jobStatus] || null

  return (
    <section className="ml-panel training-panel" aria-label="Panneau Transfer Learning">
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <span className="icon">🏋️</span> Transfer Learning — CIFAR-10
          </span>
          <span className="ml-badge">ResNet · PyTorch</span>
        </div>

        {/* ── Infos dataset ────────────────────────────────────────────────── */}
        <div className="dataset-info">
          <div className="dataset-badge">
            <span className="dataset-icon">📦</span>
            <div>
              <div className="dataset-name">CIFAR-10</div>
              <div className="dataset-desc">60 000 images · 10 classes · téléchargement auto</div>
            </div>
          </div>
          <div className="cifar-classes">
            {CIFAR10_CLASSES.map((cls, i) => (
              <span key={i} className="cifar-class-chip">{cls}</span>
            ))}
          </div>
        </div>

        {/* ── Formulaire ──────────────────────────────────────────────────── */}
        <form className="training-form" onSubmit={handleSubmit}>
          <div className="form-grid">

            {/* Modèle */}
            <div className="form-field">
              <label htmlFor="train-model" className="form-label">Architecture</label>
              <select
                id="train-model"
                className="styled-select"
                value={config.base_model}
                onChange={e => handleChange('base_model', e.target.value)}
                disabled={isRunning}
              >
                <option value="resnet18">ResNet-18 (rapide, ~11M params)</option>
                <option value="resnet50">ResNet-50 (précis, ~25M params)</option>
              </select>
            </div>

            {/* Epochs */}
            <div className="form-field">
              <label htmlFor="train-epochs" className="form-label">
                Epochs
                <span className="form-value">{config.epochs}</span>
              </label>
              <input
                id="train-epochs"
                type="range"
                min={1} max={50} step={1}
                value={config.epochs}
                onChange={e => handleChange('epochs', e.target.value)}
                disabled={isRunning}
                className="styled-range"
              />
              <div className="range-ticks">
                {[1, 10, 20, 30, 50].map(v => (
                  <span key={v}>{v}</span>
                ))}
              </div>
            </div>

            {/* Learning rate */}
            <div className="form-field">
              <label htmlFor="train-lr" className="form-label">
                Learning Rate
                <span className="form-value" style={{ fontFamily: 'var(--font-mono)' }}>
                  {parseFloat(config.learning_rate).toExponential(0)}
                </span>
              </label>
              <select
                id="train-lr"
                className="styled-select"
                value={config.learning_rate}
                onChange={e => handleChange('learning_rate', e.target.value)}
                disabled={isRunning}
              >
                <option value="0.01">0.01 (agressif)</option>
                <option value="0.001">0.001 (recommandé)</option>
                <option value="0.0001">0.0001 (fin)</option>
                <option value="0.00001">0.00001 (très fin)</option>
              </select>
            </div>

            {/* Batch size */}
            <div className="form-field">
              <label htmlFor="train-batch" className="form-label">Batch Size</label>
              <select
                id="train-batch"
                className="styled-select"
                value={config.batch_size}
                onChange={e => handleChange('batch_size', e.target.value)}
                disabled={isRunning}
              >
                <option value="32">32</option>
                <option value="64">64 (recommandé)</option>
                <option value="128">128</option>
              </select>
            </div>

          </div>

          {/* Toggle freeze backbone */}
          <div className="toggle-row">
            <label className="toggle-label" htmlFor="train-freeze">
              <span>
                <strong>Freeze backbone</strong>
                <span className="toggle-desc">
                  {config.freeze_backbone
                    ? 'Seule la tête FC est entraînée (Transfer Learning classique, rapide)'
                    : 'Tout le réseau est fine-tuné (plus lent, potentiellement plus précis)'}
                </span>
              </span>
              <div className="toggle-switch-wrapper">
                <input
                  id="train-freeze"
                  type="checkbox"
                  className="toggle-input"
                  checked={config.freeze_backbone}
                  onChange={e => handleChange('freeze_backbone', e.target.checked)}
                  disabled={isRunning}
                />
                <div className="toggle-switch" />
              </div>
            </label>
          </div>

          {/* Estimation durée */}
          {!jobStatus && (
            <p className="duration-estimate">
              ⏱ Durée estimée (CPU) :{' '}
              <strong>
                {config.freeze_backbone
                  ? `~${Math.round(config.epochs * (config.base_model === 'resnet18' ? 1.5 : 3))} min`
                  : `~${Math.round(config.epochs * (config.base_model === 'resnet18' ? 4 : 8))} min`}
              </strong>
            </p>
          )}

          {/* Bouton lancer */}
          {!jobStatus && (
            <button
              id="btn-start-training"
              type="submit"
              className="btn-start-training"
              disabled={isRunning || loading}
            >
              {loading ? '⏳ Démarrage…' : '🚀 Lancer l\'entraînement sur CIFAR-10'}
            </button>
          )}
        </form>

        {/* ── Statut & Progression ─────────────────────────────────────────── */}
        {jobStatus && (
          <div className="training-progress-section">

            {/* Badge statut */}
            <div className="training-status-bar">
              {statusInfo && (
                <span
                  className="training-status-badge"
                  style={{ borderColor: statusInfo.color, color: statusInfo.color }}
                >
                  {statusInfo.icon} {statusInfo.label}
                </span>
              )}
              {jobId && (
                <span className="job-id-label">
                  job: {jobId.slice(0, 8)}…
                </span>
              )}
              {estimatedMinutes && isRunning && (
                <span className="duration-label">~{estimatedMinutes} min estimées</span>
              )}
            </div>

            {/* Barre de progression */}
            <div className="progress-bar-wrapper">
              <div
                className="progress-bar-fill"
                style={{
                  width: `${progress}%`,
                  background: jobStatus === 'completed'
                    ? 'linear-gradient(90deg, var(--clr-success), #00ff99)'
                    : jobStatus === 'failed'
                    ? 'var(--clr-danger)'
                    : 'linear-gradient(90deg, var(--clr-accent), var(--clr-accent-2))',
                }}
              />
            </div>
            <div className="progress-labels">
              <span className="progress-pct">{progress.toFixed(0)}%</span>
              {totalEpochs > 0 && (
                <span className="progress-epochs">
                  Epoch {currentEpoch} / {totalEpochs}
                </span>
              )}
            </div>

            {/* Message */}
            {message && (
              <p className="training-message">{message}</p>
            )}

            {/* Erreur */}
            {error && (
              <div className="training-error">⚠️ {error}</div>
            )}

            {/* Métriques */}
            {Object.keys(metrics).length > 0 && (
              <div className="metrics-grid training-metrics">
                {metrics.train_loss !== undefined && (
                  <div className="metric-card">
                    <div className="metric-value">{metrics.train_loss}</div>
                    <div className="metric-label">Train Loss</div>
                  </div>
                )}
                {metrics.train_acc !== undefined && (
                  <div className="metric-card">
                    <div className="metric-value" style={{ color: 'var(--clr-accent-2)' }}>
                      {metrics.train_acc}%
                    </div>
                    <div className="metric-label">Train Acc</div>
                  </div>
                )}
                {metrics.val_acc !== undefined && (
                  <div className="metric-card">
                    <div className="metric-value" style={{ color: 'var(--clr-success)' }}>
                      {metrics.val_acc}%
                    </div>
                    <div className="metric-label">Val Acc</div>
                  </div>
                )}
                {metrics.learning_rate !== undefined && (
                  <div className="metric-card">
                    <div className="metric-value" style={{ fontSize: '0.9rem' }}>
                      {metrics.learning_rate}
                    </div>
                    <div className="metric-label">LR actuel</div>
                  </div>
                )}
              </div>
            )}

            {/* Historique - Graphique */}
            {jobStatus === 'completed' && history && history.length > 0 && (
              <div className="training-graph" style={{ marginTop: '20px', width: '100%', height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--clr-border)" />
                    <XAxis dataKey="epoch" stroke="var(--clr-text-dim)" />
                    <YAxis yAxisId="left" stroke="var(--clr-text-dim)" />
                    <YAxis yAxisId="right" orientation="right" stroke="var(--clr-text-dim)" />
                    <Tooltip contentStyle={{ backgroundColor: 'var(--clr-bg)', borderColor: 'var(--clr-border)', color: 'var(--clr-text)' }} />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="train_loss" stroke="var(--clr-warning)" name="Loss" dot={false} strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="val_acc" stroke="var(--clr-success)" name="Validation Acc (%)" dot={false} strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="train_acc" stroke="var(--clr-accent)" name="Train Acc (%)" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Bouton nouveau training */}
            {(jobStatus === 'completed' || jobStatus === 'failed') && (
              <button
                className="btn-reset-training"
                onClick={reset}
              >
                ↺ Nouveau training
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
