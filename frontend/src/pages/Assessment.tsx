import { useMemo, useState } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Loader2,
  PlayCircle,
  AlertCircle,
  ShieldAlert,
} from 'lucide-react'
import api from '../api/client'
import Results from '../components/Results'
import type { PatientInput, PredictionResponse } from '../types'

// Field metadata used to render the form and validate inputs.
const FIELDS: {
  key: keyof PatientInput
  label: string
  unit: string
  min: number
  max: number
  step: number | 'any'
  decimal?: boolean
  stepIndex: number
  hint?: string
}[] = [
  { key: 'Age', label: 'Age', unit: 'years', min: 1, max: 120, step: 1, stepIndex: 1 },
  { key: 'Pregnancies', label: 'Pregnancies', unit: 'count', min: 0, max: 20, step: 1, stepIndex: 1, hint: 'Number of pregnancies' },
  { key: 'Glucose', label: 'Glucose', unit: 'mg/dL', min: 40, max: 400, step: 1, stepIndex: 2 },
  { key: 'BloodPressure', label: 'Blood Pressure', unit: 'mmHg', min: 30, max: 250, step: 1, stepIndex: 2 },
  { key: 'SkinThickness', label: 'Skin Thickness', unit: 'mm', min: 5, max: 100, step: 1, stepIndex: 2 },
  { key: 'Insulin', label: 'Insulin', unit: 'mu U/mL', min: 5, max: 900, step: 1, stepIndex: 2 },
  { key: 'BMI', label: 'BMI', unit: 'kg/m²', min: 10, max: 90, step: 0.1, decimal: true, stepIndex: 2 },
  { key: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree Function', unit: 'score', min: 0.02, max: 3, step: 0.01, decimal: true, stepIndex: 2 },
]

const STEPS = [
  { title: 'Patient information', desc: 'Basic demographic details.' },
  { title: 'Clinical measurements', desc: 'Standard health measurements.' },
  { title: 'Review', desc: 'Check the information before running.' },
  { title: 'Run assessment', desc: 'Process with all trained models.' },
  { title: 'Results', desc: 'Your AI risk score and explanation.' },
]

function validateField(f: (typeof FIELDS)[number], value: number): string | null {
  if (Number.isNaN(value)) return 'Required'
  if (value < f.min) return `Must be ≥ ${f.min}`
  if (value > f.max) return `Must be ≤ ${f.max}`
  return null
}

const defaults: PatientInput = {
  Pregnancies: 2,
  Glucose: 145,
  BloodPressure: 85,
  SkinThickness: 32,
  Insulin: 120,
  BMI: 31.2,
  DiabetesPedigreeFunction: 0.52,
  Age: 45,
}

export default function Assessment() {
  const [step, setStep] = useState(0)
  const [form, setForm] = useState<PatientInput>(defaults)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)

  const fieldErrors = useMemo(() => {
    const e: Record<string, string> = {}
    for (const f of FIELDS) {
      const err = validateField(f, form[f.key])
      if (err) e[f.key] = err
    }
    return e
  }, [form])

  const isValid = Object.keys(fieldErrors).length === 0

  function setValue(key: keyof PatientInput, raw: string) {
    const num = parseFloat(raw)
    setForm((prev) => ({ ...prev, [key]: raw === '' ? NaN : num }))
    setErrors((prev) => {
      const next = { ...prev }
      delete next[key]
      return next
    })
  }

  async function runAssessment() {
    setStep(3)
    setRunning(true)
    setRunError(null)
    try {
      const payload = { ...form }
      for (const f of FIELDS) {
        if (f.decimal) payload[f.key] = Number(payload[f.key].toFixed(2))
      }
      const res = await api.predict(payload)
      setResult(res)
      setStep(4)
    } catch (e) {
      setRunError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  function next() {
    if (step === 1 && !isValid) {
      setErrors(fieldErrors)
      return
    }
    setStep((s) => Math.min(s + 1, STEPS.length - 1))
  }

  function back() {
    if (step === 4) setStep(2)
    else setStep((s) => Math.max(s - 1, 0))
  }

  function resetAll() {
    setForm(defaults)
    setResult(null)
    setErrors({})
    setStep(0)
  }

  const stepFields = (index: number) => FIELDS.filter((f) => f.stepIndex === index)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Patient Assessment</h1>
        <p className="mt-1 text-sm text-slate-500">
          Enter anonymised patient measurements to run the AI risk assessment.
        </p>
      </div>

      {/* Stepper */}
      <ol className="flex items-center gap-1 overflow-x-auto whitespace-nowrap">
        {STEPS.map((s, i) => (
          <li key={s.title} className="flex items-center gap-1">
            <div className="flex items-center gap-1.5">
              <span
                className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold ${
                  i < step
                    ? 'bg-brand-600 text-white'
                    : i === step
                      ? 'bg-brand-100 text-brand-700 ring-2 ring-brand-500'
                      : 'bg-slate-100 text-slate-400'
                }`}
              >
                {i < step ? <Check size={14} /> : i + 1}
              </span>
              <span className={`hidden text-xs font-medium md:inline ${i === step ? 'text-slate-900' : 'text-slate-400'}`}>
                {s.title}
              </span>
            </div>
            {i < STEPS.length - 1 && <div className="mx-1 h-px w-4 bg-slate-200" />}
          </li>
        ))}
      </ol>

      {/* Step content */}
      <div className="card p-6">
        {step === 0 && (
          <div className="space-y-4">
            <StepIntro index={0} />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {stepFields(1).map((f) => (
                <FieldInput key={f.key} field={f} value={form[f.key]} errors={errors} onChange={setValue} />
              ))}
            </div>
          </div>
        )}

        {step === 1 && (
          <div className="space-y-4">
            <StepIntro index={1} />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {stepFields(2).map((f) => (
                <FieldInput key={f.key} field={f} value={form[f.key]} errors={errors} onChange={setValue} />
              ))}
            </div>
            {!isValid && (
              <div className="flex items-center gap-2 text-sm text-rose-600">
                <AlertCircle size={16} /> Some values are out of range. Please correct them.
              </div>
            )}
          </div>
        )}

        {step === 2 && <ReviewStep form={form} />}

        {step === 3 && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            {running ? (
              <>
                <Loader2 className="animate-spin text-brand-600" size={36} />
                <p className="mt-4 text-sm font-medium text-slate-700">
                  Running all models…
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Logistic Regression · Random Forest · XGBoost · Neural Network
                </p>
              </>
            ) : (
              <>
                <PlayCircle size={36} className="text-brand-600" />
                <p className="mt-4 text-sm text-slate-600">Ready to start the assessment.</p>
                <button className="btn-primary mt-4" onClick={runAssessment}>
                  Run AI Assessment
                </button>
              </>
            )}
          </div>
        )}

        {step === 4 && result && <Results result={result} />}
        {step === 4 && !result && (
          <div className="py-16 text-center text-sm text-slate-500">
            <ShieldAlert className="mx-auto mb-3 text-slate-400" size={32} />
            {runError ? `Error: ${runError}` : 'No result available. Re-run the assessment.'}
          </div>
        )}
      </div>

      {/* Nav buttons */}
      {step < 3 && (
        <div className="flex justify-between">
          <button className="btn-secondary" onClick={back} disabled={step === 0}>
            <ArrowLeft size={16} /> Back
          </button>
          {step === 2 ? (
            <button className="btn-primary" onClick={runAssessment}>
              Run AI Assessment <ArrowRight size={16} />
            </button>
          ) : (
            <button className="btn-primary" onClick={next}>
              Continue <ArrowRight size={16} />
            </button>
          )}
        </div>
      )}
      {step === 4 && (
        <div className="flex justify-end">
          <button className="btn-secondary" onClick={resetAll}>
            New assessment
          </button>
        </div>
      )}
    </div>
  )
}

function StepIntro({ index }: { index: number }) {
  const meta = STEPS[index]
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-900">{meta.title}</h2>
      <p className="text-sm text-slate-500">{meta.desc}</p>
    </div>
  )
}

function FieldInput({
  field,
  value,
  errors,
  onChange,
}: {
  field: (typeof FIELDS)[number]
  value: number
  errors: Record<string, string>
  onChange: (key: keyof PatientInput, raw: string) => void
}) {
  const err = errors[field.key] ?? (Number.isNaN(value) ? 'Required' : undefined)
  return (
    <div>
      <label className="label">
        {field.label}
        <span className="ml-1 text-xs text-slate-400">({field.unit})</span>
      </label>
      <input
        className="input mt-1"
        type="number"
        step={field.step === 'any' ? 'any' : field.step}
        value={Number.isNaN(value) ? '' : value}
        onChange={(e) => onChange(field.key, e.target.value)}
        min={field.min}
        max={field.max}
      />
      {field.hint && <p className="mt-1 text-[11px] text-slate-400">{field.hint}</p>}
      {err && <p className="mt-1 text-xs text-rose-600">{err}</p>}
    </div>
  )
}

function ReviewStep({ form }: { form: PatientInput }) {
  return (
    <div className="space-y-4">
      <StepIntro index={2} />
      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {FIELDS.map((f) => (
          <div key={f.key} className="rounded-lg border border-slate-200 p-3">
            <dt className="text-xs text-slate-500">{f.label}</dt>
            <dd className="mt-0.5 text-sm font-semibold text-slate-900">
              {Number.isNaN(form[f.key]) ? '—' : `${form[f.key]} ${f.unit}`}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
