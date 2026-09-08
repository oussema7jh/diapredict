// Shared domain types mirroring the FastAPI Pydantic schemas.

export interface ModelPrediction {
  model_id: string
  model_name: string
  probability: number
  risk_percent: number
  status: string
}

export interface ShapBar {
  feature: string
  label: string
  value: number
  contribution: number
  direction: 'increases' | 'decreases' | 'neutral'
}

export interface Explainability {
  method: string
  model: string
  base_value: number
  patient_value: number
  top_features: ShapBar[]
  summary_english: string
}

export interface PredictionResponse {
  assessment_id: number
  timestamp: string
  input: Record<string, number>
  input_pretty: { feature: string; label: string; value: string; unit: string }[]
  model_predictions: ModelPrediction[]
  ensemble: ModelPrediction
  risk_percent: number
  risk_category: string
  risk_category_explanation: string
  explanation: Explainability | null
  disclaimer: string
}

export interface PatientInput {
  Pregnancies: number
  Glucose: number
  BloodPressure: number
  SkinThickness: number
  Insulin: number
  BMI: number
  DiabetesPedigreeFunction: number
  Age: number
}

export interface ModelPerformance {
  model_id: string
  model_name: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
  trained_at: string | null
  completed: boolean
  message: string | null
}

export interface ModelStatus {
  model_id: string
  model_name: string
  type: 'ml' | 'dl'
  loaded: boolean
  status: string
  file: string
  message: string | null
}

export interface ModelsStatusResponse {
  models: ModelStatus[]
  all_loaded: boolean
  disclaimer: string
}

export interface FeatureImportance {
  feature: string
  label: string
  mean_abs_shap: number
}

export interface DashboardStats {
  total_assessments: number
  avg_risk_percent: number
  higher_risk_count: number
  recent_assessments: {
    id: number
    created_at: string
    risk_percent: number
    risk_category: string
  }[]
}

export interface Health {
  status: string
  version: string
  time: string
  models_loaded: number
  disclaimer: string
}
