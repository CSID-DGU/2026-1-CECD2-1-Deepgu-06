import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://43.201.17.169:8000',
      '/internal': 'http://43.201.17.169:8000',
      '/hls': 'http://3.35.171.137:9000',
      '/ws': { target: 'ws://43.201.17.169:8000', ws: true },
    }
  }
})
