import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_DEV_PROXY_TARGET || 'http://localhost:8000'
  const mediaTarget = env.VITE_DEV_MEDIA_PROXY_TARGET || 'http://localhost:9000'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': apiTarget,
        '/internal': apiTarget,
        '/hls': mediaTarget,
        '/live': mediaTarget,
        '/ws': { target: apiTarget.replace(/^http/, 'ws'), ws: true },
      },
    },
  }
})
