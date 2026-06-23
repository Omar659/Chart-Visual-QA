import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server runs on 5173 and proxies /api/* to the Flask backend (5000),
// so the frontend can call same-origin paths and we avoid CORS in dev.
// The backend port can be overridden with the VITE_BACKEND_PORT env var.
const backendPort = process.env.VITE_BACKEND_PORT || '5000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
})
