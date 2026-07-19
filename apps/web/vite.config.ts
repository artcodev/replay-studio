import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const usePolling = env.DEV_USE_POLLING === '1'

  return {
    plugins: [vue()],
    build: {
      // Three.js is an intentionally isolated 546 kB vendor chunk (136 kB gzip).
      // Keep the warning threshold tight enough to catch growth in the application chunk.
      chunkSizeWarningLimit: 560,
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes('/node_modules/three/')) return 'three-vendor'
            if (
              id.includes('/node_modules/vue/')
              || id.includes('/node_modules/@vue/')
              || id.includes('/node_modules/vue-router/')
            ) return 'vue-vendor'
          },
        },
      },
    },
    server: {
      port: 5188,
      strictPort: true,
      watch: usePolling ? { usePolling: true, interval: 250 } : undefined,
      proxy: {
        '/api': {
          target: env.API_PROXY_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
