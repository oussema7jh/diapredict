import {
  ShieldCheck,
  Activity,
  GitBranch,
  Database,
  Code2,
  FlaskConical,
} from 'lucide-react'

const MODELS = [
  {
    name: 'Logistic Regression',
    type: 'Machine Learning',
    desc: 'A linear baseline model that fits a logistic sigmoid to the features.',
  },
  {
    name: 'Random Forest',
    type: 'Machine Learning',
    desc: 'An ensemble of decision trees; bagging + random feature selection reduces variance.',
  },
  {
    name: 'XGBoost',
    type: 'Machine Learning',
    desc: 'Gradient-boosted trees with regularisation; the SHAP surrogate in this project.',
  },
  {
    name: 'Neural Network (MLP)',
    type: 'Deep Learning',
    desc: 'A PyTorch fully-connected network: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Sigmoid.',
  },
]

const PIPELINE = [
  'Dataset',
  'Cleaning',
  'Missing values',
  'EDA',
  'Feature engineering',
  'Train/test split',
  'Scaling',
  'Training',
  'Cross-validation',
  'Evaluation',
  'Model saving',
  'API inference',
]

export default function About() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">About DiaPredict AI</h1>
        <p className="mt-1 text-sm text-slate-500">
          An explainable diabetes risk prediction system built as a university-level
          applied AI / machine-learning project.
        </p>
      </div>

      <div className="card p-6">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-brand-50 text-brand-700">
            <Activity size={22} />
          </div>
          <div>
            <h2 className="text-base font-semibold text-slate-900">What it does</h2>
            <p className="mt-1 text-sm leading-relaxed text-slate-600">
              DiaPredict AI takes anonymised health measurements and estimates a
              person's risk of diabetes using four trained models: Logistic
              Regression, Random Forest, XGBoost and a PyTorch neural network. It
              combines them into an ensemble score and uses SHAP explainability to
              show <em>why</em> each prediction was produced.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="card p-6">
          <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-800">
            <Code2 size={18} className="text-brand-600" /> Models
          </h2>
          <div className="space-y-3">
            {MODELS.map((m) => (
              <div key={m.name} className="rounded-lg border border-slate-200 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-800">{m.name}</span>
                  <span className="chip bg-slate-100 text-slate-500">{m.type}</span>
                </div>
                <p className="mt-1 text-xs text-slate-500">{m.desc}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="card p-6">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-800">
              <GitBranch size={18} className="text-brand-600" /> ML pipeline
            </h2>
            <div className="flex flex-wrap gap-2">
              {PIPELINE.map((p, i) => (
                <span
                  key={p}
                  className="rounded-full bg-brand-50 px-3 py-1 text-xs font-medium text-brand-700"
                >
                  {i + 1}. {p}
                </span>
              ))}
            </div>
          </div>

          <div className="card p-6">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-800">
              <Database size={18} className="text-brand-600" /> Dataset
            </h2>
            <p className="text-sm leading-relaxed text-slate-600">
              Built on the publicly available Pima Indians Diabetes dataset (768
              samples, 8 features). See{' '}
              <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">data/README.md</code>{' '}
              for full details.
            </p>
          </div>

          <div className="card p-6">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-800">
              <FlaskConical size={18} className="text-brand-600" /> Explainable AI
            </h2>
            <p className="text-sm leading-relaxed text-slate-600">
              SHAP (Shapley Additive exPlanations) attributes each prediction to its
              contributing features, scaled to the probability space so values like
              <em> +0.21</em> are directly readable. Global feature importance is also
              computed across the whole dataset.
            </p>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-amber-200 bg-amber-50 p-5">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-700">
            <ShieldCheck size={22} />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-amber-800">Ethical disclaimer</h2>
            <p className="mt-1 text-sm leading-relaxed text-amber-800">
              DiaPredict AI is an educational and research prototype. It provides
              machine-learning-based risk estimates and is <strong>not</strong> a medical
              diagnosis. The results should not be used to make medical decisions.
              Always consult a qualified healthcare professional for medical advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
