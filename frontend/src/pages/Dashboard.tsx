import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
} from 'recharts'
import {
  ClipboardList,
  Gauge,
  AlertTriangle,
  Target,
  Trophy,
  Loader2,
} from 'lucide-react'
import api from '../api/client'
import StatCard from '../components/StatCard'
import type { DashboardStats, ModelPerformance, FeatureImportance } from '../types'

const COLORS = ['#10b981', '#f59e0b', '#ef4444']

function riskBucket(pct: number) {
  if (pct <= 30) return 'Lower (0-30%)'
  if (pct <= 60) return 'Moderate (31-60%)'
  return 'Higher (61-100%)'
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [perf, setPerf] = useState<ModelPerformance[] | null>(null)
  const [importance, setImportance] = useState<FeatureImportance[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.stats(), api.modelsPerformance(), api.featureImportance()])
      .then(([s, p, f]) => {
        setStats(s)
        setPerf(p)
        setImportance(f.importance)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500">
        <Loader2 className="animate-spin" size={24} /> &nbsp;Loading dashboard…
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-8 text-center text-slate-600">
        <p className="text-sm">Could not load the API. Is the backend running?</p>
        <p className="mt-2 text-xs text-slate-400">{error}</p>
      </div>
    )
  }

  const completed = (perf ?? []).filter((p) => p.completed)
  const bestModel = completed.length
    ? completed.reduce((a, b) => (b.roc_auc > a.roc_auc ? b : a))
    : null
  const bestAccuracy = completed.length
    ? completed.reduce((a, b) => (b.accuracy > a.accuracy ? b : a))
    : null

  const riskDist = [
    { name: 'Lower (0-30%)', value: 0, color: COLORS[0] },
    { name: 'Moderate (31-60%)', value: 0, color: COLORS[1] },
    { name: 'Higher (61-100%)', value: 0, color: COLORS[2] },
  ]
  for (const a of stats?.recent_assessments ?? []) {
    const bucket = riskBucket(a.risk_percent)
    const hit = riskDist.find((r) => r.name === bucket)
    if (hit) hit.value += 1
  }

  const overTime = (stats?.recent_assessments ?? [])
    .slice()
    .reverse()
    .map((a) => ({
      label: new Date(a.created_at).toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }),
      risk: a.risk_percent,
    }))

  const modelCompare = (perf ?? []).map((m) => ({
    name: m.model_name,
    accuracy: m.accuracy * 100,
    recall: m.recall * 100,
    f1: m.f1 * 100,
  }))

  const featureData = (importance ?? []).map((f) => ({
    name: f.label,
    value: f.mean_abs_shap,
  }))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        <p className="mt-1 text-sm text-slate-500">
          Live overview of assessments and model performance.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <StatCard
          label="Total assessments"
          value={String(stats?.total_assessments ?? 0)}
          icon={ClipboardList}
          sub="Stored anonymously"
        />
        <StatCard
          label="Average predicted risk"
          value={`${(stats?.avg_risk_percent ?? 0).toFixed(1)}%`}
          icon={Gauge}
          accent="bg-emerald-50 text-emerald-600"
        />
        <StatCard
          label="Higher-risk predictions"
          value={String(stats?.higher_risk_count ?? 0)}
          icon={AlertTriangle}
          accent="bg-rose-50 text-rose-600"
        />
        <StatCard
          label="Best model (ROC-AUC)"
          value={bestModel ? `${(bestModel.roc_auc * 100).toFixed(1)}%` : '—'}
          sub={bestModel?.model_name ?? 'No trained models'}
          icon={Trophy}
          accent="bg-amber-50 text-amber-600"
        />
        <StatCard
          label="Best model (accuracy)"
          value={bestAccuracy ? `${(bestAccuracy.accuracy * 100).toFixed(1)}%` : '—'}
          sub={bestAccuracy?.model_name ?? 'No trained models'}
          icon={Target}
          accent="bg-sky-50 text-sky-600"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Model comparison */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Model comparison</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modelCompare} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-12} height={50} />
              <YAxis tick={{ fontSize: 11 }} unit="%" />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="accuracy" name="Accuracy" fill="#337eff" radius={[4, 4, 0, 0]} />
              <Bar dataKey="recall" name="Recall" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1" name="F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk distribution */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Risk distribution</h2>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={riskDist}
                dataKey="value"
                nameKey="name"
                innerRadius={60}
                outerRadius={95}
                paddingAngle={2}
              >
                {riskDist.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Pie>
              <Legend />
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Assessments over time */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Assessments over time</h2>
          {overTime.length === 0 ? (
            <div className="flex h-52 items-center justify-center text-sm text-slate-400">
              No assessments yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={overTime}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" domain={[0, 100]} />
                <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Line
                  type="monotone"
                  dataKey="risk"
                  name="Predicted risk"
                  stroke="#337eff"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Feature importance */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Feature importance (SHAP)</h2>
          {featureData.length === 0 ? (
            <div className="flex h-52 items-center justify-center text-sm text-slate-400">
              Not available yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={featureData} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={90} />
                <Tooltip />
                <Bar dataKey="value" name="Mean |SHAP|" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  )
}
