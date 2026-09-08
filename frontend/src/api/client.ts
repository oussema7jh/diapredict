// Thin typed client for the DiaPredict FastAPI backend.

import { API_BASE } from '../lib/config'
import type {
  DashboardStats,
  FeatureImportance,
  ModelsStatusResponse,
  ModelPerformance,
  PatientInput,
  PredictionResponse,
} from '../types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  const text = await res.text()
  let body: unknown = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = text
  }
  if (!res.ok) {
    const msg =
      (body as { detail?: string })?.detail ||
      (typeof body === 'string' ? body : `Request failed (${res.status})`)
    throw new Error(msg)
  }
  return body as T
}

export const api = {
  health: () => request<{ status: string; models_loaded: number }>('/health'),

  predict: (input: PatientInput) =>
    request<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(input),
    }),

  modelsPerformance: () =>
    request<ModelPerformance[]>('/models/performance'),

  modelsStatus: () => request<ModelsStatusResponse>('/models/status'),

  featureImportance: () =>
    request<{ importance: FeatureImportance[] }>('/models/feature-importance'),

  stats: () => request<DashboardStats>('/models/stats'),

  curves: () => request<{ curves: Curves[] }>('/models/curves'),

  dlTraining: () => request<{ history: DlTraining | null }>('/models/dl-training'),
}

export interface Curves {
  model_id: string
  model_name: string
  roc: { fpr: number; tpr: number }[]
  pr: { recall: number; precision: number }[]
}

export interface DlTraining {
  train_loss: number[]
  val_loss: number[]
  val_auc: number[]
}

export default api
