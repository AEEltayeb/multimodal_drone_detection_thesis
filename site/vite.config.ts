import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Standalone static site (no backend). Deployed on Vercel with Root Directory = site.
export default defineConfig({
  plugins: [react()],
  base: '/',
})
