import { defineConfig } from 'vite';

export default defineConfig({
  server: { host: '127.0.0.1' },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  worker: { format: 'es' },
});
