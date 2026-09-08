import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Activity,
  LayoutDashboard,
  ClipboardList,
  Cpu,
  Info,
} from 'lucide-react'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/assessment', label: 'Assessment', icon: ClipboardList },
  { to: '/models', label: 'Models', icon: Cpu },
  { to: '/about', label: 'About', icon: Info },
]

function Sidebar() {
  const { pathname } = useLocation()
  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="flex items-center gap-2.5 border-b border-slate-200 px-5 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-white">
          <Activity size={20} />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-bold text-slate-900">DiaPredict AI</div>
          <div className="text-[11px] text-slate-500">Explainable Diabetes Risk</div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {NAV.map((item) => {
          const active = pathname === item.to
          const Icon = item.icon
          return (
            <Link
              key={item.to}
              to={item.to}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${
                active
                  ? 'bg-brand-50 text-brand-700'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <Icon size={18} />
              {item.label}
            </Link>
          )
        })}
      </nav>
      <div className="border-t border-slate-200 p-4">
        <div className="rounded-lg bg-amber-50 p-3 text-[11px] leading-relaxed text-amber-800">
          Educational &amp; research prototype. Not a medical diagnosis.
        </div>
      </div>
    </aside>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-6 py-6">{children}</div>
      </main>
    </div>
  )
}
