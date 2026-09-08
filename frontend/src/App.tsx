import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Assessment from './pages/Assessment'
import Models from './pages/Models'
import About from './pages/About'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/assessment" element={<Assessment />} />
        <Route path="/models" element={<Models />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Layout>
  )
}
