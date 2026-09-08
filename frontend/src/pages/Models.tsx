import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  Cell,
} from 'recharts'
import { Activity, Loader2, Lock, AlertTriangle } from 'lucide-react'
import api, { Curves, DlTraining } from '../api/client'
import type { ModelPerformance, ModelsStatusResponse } from '../types'

export default function Models() {
  const [perf, setPerf] = useState<ModelPerformance[] | null>(null)
  const [status, setStatus] = useState<ModelsStatusResponse | null>(null)
  const [curves, setCurves] = useState<Curves[] | null>(null)
  const [dl, setDl] = useState<DlTraining | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.modelsPerformance(), api.modelsStatus(), api.curves(), api.dlTraining()])
      .then(([p, s, c, d]) => {
        setPerf(p)
        setStatus(s)
        setCurves(c.curves)
        setDl(d.history)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500">
        <Loader2 className="animate-spin" size={24} /> &nbsp;Loading models…
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-8 text-center text-slate-600">
        <p className="text-sm">Could not load model data.</p>
        <p className="mt-2 text-xs text-slate-400">{error}</p>
      </div>
    )
  }

  const allReady = status?.all_loaded ?? false

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Model Performance</h1>
          <p className="mt-1 text-sm text-slate-500">
            Real metrics computed on the held-out test set.
          </p>
        </div>
        <span
          className={`chip border ${
            allReady
              ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
              : 'border-amber-200 bg-amber-50 text-amber-700'
          }`}
        >
          {allReady ? <Activity size={13} /> : <AlertTriangle size={13} />}
          {allReady ? 'All models ready' : 'Some models unavailable'}
        </span>
      </div>

      {/* Status list */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
        {(status?.models ?? []).map((m) => (
          <div key={m.model_id} className="card p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-800">{m.model_name}</span>
              <span className={`h-2.5 w-2.5 rounded-full ${m.loaded ? 'bg-emerald-500' : 'bg-slate-300'}`} />
            </div>
            <p className="mt-1 text-[11px] text-slate-500">
              {m.type === 'ml' ? 'Machine Learning' : 'Deep Learning'}
            </p>
          </div>
        ))}
      </div>

      {/* Comparison table */}
      <div className="card overflow-hidden">
        <div className="border-b border-slate-100 px-5 py-4">
          <h2 className="text-sm font-semibold text-slate-800">Metrics comparison</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <th className="px-5 py-3 font-semibold">Model</th>
                <th className="px-5 py-3 font-semibold">Accuracy</th>
                <th className="px-5 py-3 font-semibold">Precision</th>
                <th className="px-5 py-3 font-semibold">Recall</th>
                <th className="px-5 py-3 font-semibold">F1</th>
                <th className="px-5 py-3 font-semibold">ROC-AUC</th>
                <th className="px-5 py-3 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody>
              {(perf ?? []).map((p, i) => (
                <tr key={p.model_id} className={i % 2 ? 'bg-slate-50/50' : ''}>
                  <td className="px-5 py-3 font-medium text-slate-800">{p.model_name}</td>
                  <td className="px-5 py-3 text-slate-700">{(p.accuracy * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-slate-700">{(p.precision * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-slate-700">{(p.recall * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-slate-700">{(p.f1 * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-slate-700">{(p.roc_auc * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3">
                    {p.completed ? (
                      <span className="chip bg-emerald-50 text-emerald-700">
                        <Lock size={11} /> computed
                      </span>
                    ) : (
                      <span className="chip bg-slate-100 text-slate-500">pending</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Curves */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ROCChart curves={curves ?? []} />
        <PRChart curves={curves ?? []} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Confusion matrices */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Confusion matrices</h2>
          <div className="grid grid-cols-2 gap-4">
            {(perf ?? []).map((p) => (
              <ConfusionMatrix key={p.model_id} name={p.model_name} cm={p.confusion_matrix} />
            ))}
          </div>
        </div>

        {/* DL training curves */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Neural network training</h2>
          {dl ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={dl.train_loss.map((v, i) => ({ epoch: i + 1, train: v, val: dl.val_loss[i] }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="train" name="Train loss" stroke="#337eff" dot={false} />
                <Line type="monotone" dataKey="val" name="Val loss" stroke="#f59e0b" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-52 items-center justify-center text-sm text-slate-400">
              Training history not available.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const METRIC_COLORS = ['#337eff', '#10b981', '#f59e0b', '#8b5cf6']

function ROCChart({ curves }: { curves: Curves[] }) {
  return (
    <div className="card p-5">
      <h2 className="mb-4 text-sm font-semibold text-slate-700">ROC curves</h2>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
          <YAxis dataKey="tpr" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          {curves.map((c, i) => (
            <Line
              key={c.model_id}
              data={c.roc}
              dataKey="tpr"
              name={c.model_name}
              stroke={METRIC_COLORS[i % METRIC_COLORS.length]}
              dot={false}
              strokeWidth={2}
            />
          ))}
          <Line
            data={[{ fpr: 0, tpr: 0, ignored: 1 }, { fpr: 1, tpr: 1, ignored: 1 }]}
            dataKey="tpr"
            name="Random"
            stroke="#94a3b8"
            strokeDasharray="4 4"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function PRChart({ curves }: { curves: Curves[] }) {
  return (
    <div className="card p-5">
      <h2 className="mb-4 text-sm font-semibold text-slate-700">Precision-recall curves</h2>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
          <YAxis dataKey="precision" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          {curves.map((c, i) => (
            <Line
              key={c.model_id}
              data={c.pr}
              dataKey="precision"
              name={c.model_name}
              stroke={METRIC_COLORS[i % METRIC_COLORS.length]}
              dot={false}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function ConfusionMatrix({
  name,
  cm,
}: {
  name: string
  cm: { tn: number; fp: number; fn: number; tp: number }
}) {
  const cells = [
    { label: 'TN', value: cm.tn, cls: 'bg-emerald-100 text-emerald-800' },
    { label: 'FP', value: cm.fp, cls: 'bg-orange-100 text-orange-800' },
    { label: 'FN', value: cm.fn, cls: 'bg-orange-100 text-orange-800' },
    { label: 'TP', value: cm.tp, cls: 'bg-emerald-100 text-emerald-800' },
  ]
  return (
    <div>
      <div className="mb-2 text-xs font-medium text-slate-600">{name}</div>
      <div className="grid grid-cols-2 gap-1.5">
        {cells.map((c) => (
          <div key={c.label} className={`rounded-lg px-3 py-2 text-center ${c.cls}`}>
            <div className="text-lg font-bold">{c.value}</div>
            <div className="text-[10px] uppercase">{c.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
