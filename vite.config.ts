import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import eslint from 'vite-plugin-eslint'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://172.16.1.10:81',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://172.16.1.10:81',
        changeOrigin: true,
      },
    },
  },
})