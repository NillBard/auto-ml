import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import eslint from 'vite-plugin-eslint'
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'), // Убедитесь, что путь соответствует вашей структуре
    },
  },
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