import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The components use JSX inside `.js` files (carried over from the original
// Create-React-App project), so esbuild is told to load `src/*.js` as JSX.
// Dev server runs on 5173 and is reachable on the LAN (host: true) so other
// players can open it during development. In production the Flask backend
// serves the built `dist/` folder instead.
export default defineConfig({
  plugins: [react()],
  esbuild: {
    loader: 'jsx',
    include: /src\/.*\.jsx?$/,
    exclude: [],
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: { '.js': 'jsx' },
    },
  },
  server: {
    port: 5173,
    host: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
});
