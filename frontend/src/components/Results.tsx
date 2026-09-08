import { useState } from 'react'
import { Activity, Sparkles, TrendingUp, Info } from 'lucide-react'
import type { PredictionResponse } from '../types'

function categoryColor(category: string) {
  if (category.startsWith('Lower')) return '#10b981'
  if (category.startsWith('Moderate')) return '#f59e0b'
  return '#ef4444'
}

function categoryBg(category: string) {
  if (category.startsWith('Lower')) return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (category.startsWith('Moderate')) return 'bg-amber-50 text-amber-700 border-amber-200'
  return 'bg-rose-50 text-rose-700 border-rose-200'
}

export default function Results({ result }: { result: PredictionResponse }) {
  const [tab, setTab] = useState<'explain' | 'models'>('explain')
  const risk = result.risk_percent
  const color = categoryColor(result.risk_category)
  const explanation = result.explanation
  const maxAbs = explanation
    ? Math.max(...explanation.top_features.map((b) => Math.abs(b.contribution)), 0.001)
    : 1

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">AI Risk Assessment</h2>
          <p className="text-sm text-slate-500">
            Assessment #{result.assessment_id} · {new Date(result.timestamp).toLocaleString()}
          </p>
        </div>
        <span className={`chip border ${categoryBg(result.risk_category)}`}>
          {result.risk_category}
        </span>
      </div>

      {/* Risk score */}
      <div className="card flex flex-col items-center p-8">
        <div className="relative h-40 w-40">
          <svg viewBox="0 0 36 36" className="h-40 w-40 -rotate-90">
            <circle cx="18" cy="18" r="15.915" fill="none" stroke="#e2e8f0" strokeWidth="3.5" />
            <circle
              cx="18"
              cy="18"
              r="15.915"
              fill="none"
              stroke={color}
              strokeWidth="3.5"
              strokeLinecap="round"
              strokeDasharray={`${(risk / 100) * 100} 100`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-extrabold text-slate-900">{Math.round(risk)}%</span>
            <span className="text-xs text-slate-500">predicted risk</span>
          </div>
        </div>
        <div className="mt-4 rounded-lg bg-slate-50 px-4 py-2 text-center">
          <h3 className="text-base font-bold" style={{ color }}>
            {result.risk_category}
          </h3>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            {result.risk_category_explanation}
          </p>
        </div>
        <p className="mt-3 max-w-md text-center text-[11px] leading-relaxed text-slate-400">
          These are model-defined risk categories, not clinical diagnostic thresholds.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        <button
          className={`btn ${tab === 'explain' ? 'bg-brand-600 text-white' : 'bg-white text-slate-600 border border-slate-300'}`}
          onClick={() => setTab('explain')}
        >
          <Sparkles size={16} /> Why this result?
        </button>
        <button
          className={`btn ${tab === 'models' ? 'bg-brand-600 text-white' : 'bg-white text-slate-600 border border-slate-300'}`}
          onClick={() => setTab('models')}
        >
          <Activity size={16} /> Model results
        </button>
      </div>

      {/* Explanation */}
      {tab === 'explain' && (
        <div className="card p-6 space-y-5">
          <div>
            <h3 className="text-sm font-semibold text-slate-800">
              What influenced the prediction?
            </h3>
            <p className="mt-1 text-sm text-slate-600">
              {explanation?.summary_english ??
                'SHAP explanation is not available for this request.'}
            </p>
          </div>

          <div className="space-y-3">
            {(explanation?.top_features ?? []).map((bar) => {
              const width = (Math.abs(bar.contribution) / maxAbs) * 100
              const up = bar.contribution > 0
              return (
                <div key={bar.feature} className="flex items-center gap-3">
                  <div className="w-44 shrink-0 text-sm text-slate-700">{bar.label}</div>
                  <div className="flex-1">
                    <div className="relative h-5 overflow-hidden rounded bg-slate-100">
                      <div
                        className={`absolute inset-y-0 ${up ? 'left-0 bg-rose-400' : 'left-0 bg-sky-400'}`}
                        style={{ width: `${width}%` }}
                      />
                      {bar.direction === 'decreases' && (
                        <div className="absolute inset-y-0 right-0 bg-sky-100" style={{ width: `${100 - width}%` }} />
                      )}
                    </div>
                  </div>
                  <div className="w-16 shrink-0 text-right text-xs font-medium text-slate-600">
                    {up ? '+' : ''}
                    {(bar.contribution * 100).toFixed(1)}pt
                  </div>
                </div>
              )
            })}
          </div>

          {explanation && (
            <div className="flex items-start gap-2 rounded-lg bg-blue-50 p-3 text-xs text-blue-800">
              <Info size={15} className="mt-0.5 shrink-0" />
              <p>
                Contributions show how much each feature shifted the predicted risk
                relative to the model's average. Red bars pushed risk up; blue bars
                pushed it down. SHAP values are computed with a tree-based explainer
                on the XGBoost model.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Model results */}
      {tab === 'models' && (
        <div className="card p-6 space-y-3">
          <h3 className="text-sm font-semibold text-slate-800">Model breakdown</h3>
          <div className="space-y-2">
            {result.model_predictions.map((m) => (
              <ModelRow key={m.model_id} name={m.model_name} value={m.risk_percent} />
            ))}
            <ModelRow name="Ensemble" value={result.ensemble.risk_percent} highlight />
          </div>
          <p className="text-[11px] text-slate-400">
            Ensemble = weighted blend of the ML models and the neural network
            (see docs/ARCHITECTURE.md).
          </p>
        </div>
      )}

      {/* Disclaimer */}
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-relaxed text-amber-800">
        <strong className="block mb-1">Disclaimer</strong>
        {result.disclaimer}
      </div>
    </div>
  )
}

function ModelRow({ name, value, highlight }: { name: string; value: number; highlight?: boolean }) {
  return (
    <div
      className={`flex items-center justify-between rounded-lg px-4 py-2.5 ${
        highlight ? 'bg-brand-50 border border-brand-100' : 'bg-slate-50'
      }`}
    >
      <div className="flex items-center gap-2">
        {highlight && <TrendingUp size={16} className="text-brand-600" />}
        <span className={`text-sm ${highlight ? 'font-semibold text-brand-800' : 'text-slate-700'}`}>
          {name}
        </span>
      </div>
      <span className={`text-sm ${highlight ? 'font-bold text-brand-700' : 'font-medium text-slate-800'}`}>
        {Math.round(value)}%
      </span>
    </div>
  )
}
