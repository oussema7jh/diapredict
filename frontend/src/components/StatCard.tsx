import { LucideIcon } from 'lucide-react'

interface Props {
  label: string
  value: string
  sub?: string
  icon: LucideIcon
  accent?: string
}

export default function StatCard({ label, value, sub, icon: Icon, accent = 'bg-brand-50 text-brand-700' }: Props) {
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium text-slate-500">{label}</div>
          <div className="mt-1 text-2xl font-bold text-slate-900">{value}</div>
          {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
        </div>
        <div className={`flex h-11 w-11 items-center justify-center rounded-xl ${accent}`}>
          <Icon size={22} />
        </div>
      </div>
    </div>
  )
}
