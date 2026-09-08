import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The browser-facing app must never call localhost directly (the browser is
// not in the sandbox). Instead it always uses the RELATIVE path /api, and the
// Vite dev server proxies those requests to the backend. This works both for
// the Arena live preview and for a normal `npm run dev` on localhost.
//
// In production (static dist) there is no dev proxy, so set VITE_API_URL to
// the deployed API URL when build-publishing.
const apiUrl = process.env.VITE_API_URL ?? '/api'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // Allow the Arena live-preview host (…e2b.app) and local hosts.
    allowedHosts: ['.e2b.app', 'localhost', '127.0.0.1'],
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  // When VITE_API_URL is an absolute URL (e.g. a deployed API) we pass it
  // through; otherwise the app falls back to the relative /api path.
  define: {
    __API_BASE__: JSON.stringify(apiUrl),
  },
})
