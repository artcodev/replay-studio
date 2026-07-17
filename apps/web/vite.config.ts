import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    // Three.js is an intentionally isolated 546 kB vendor chunk (136 kB gzip).
    // Keep the warning threshold tight enough to catch growth in the application chunk.
    chunkSizeWarningLimit: 560,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/three/')) return 'three-vendor'
          if (id.includes('/node_modules/vue/') || id.includes('/node_modules/@vue/')) return 'vue-vendor'
        },
      },
    },
  },
  server: {
    port: 5188,
    strictPort: true,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
