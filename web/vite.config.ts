import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { spawn, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function spawnBackend(): Plugin {
  let backend: ChildProcess | null = null;
  const BACKEND_PORT = 8000;

  return {
    name: 'spawn-backend',
    async configureServer(server) {
      // Project root is one level above web/
      const cwd = path.resolve(__dirname, '..');

      backend = spawn(
        'conda',
        ['run', '-n', 'agents', 'python', '-m', 'project.api.run_api'],
        {
          cwd,
          stdio: 'pipe',
          shell: true,
          env: { ...process.env, PYTHONUNBUFFERED: '1' },
        }
      );

      backend.on('error', (err: NodeJS.ErrnoException) => {
        server.config.logger.error(`[backend] Failed to start: ${err.message}`);
      });

      backend.on('exit', (code: number | null, signal: string | null) => {
        if (code !== 0) {
          server.config.logger.warn(`[backend] Process exited with code ${code} signal ${signal}`);
        }
        backend = null;
      });

      // Suppress verbose backend output — only log errors
      backend.stderr?.on('data', (chunk: Buffer) => {
        const text = chunk.toString().trim();
        if (text && !text.includes('INFO')) {
          server.config.logger.warn(`[backend] ${text}`, { timestamp: true });
        }
      });

      // Wait for backend to be ready (health check)
      server.config.logger.info('[backend] Starting API server...', { timestamp: true });
      for (let i = 0; i < 30; i++) {
        try {
          const res = await fetch(`http://localhost:${BACKEND_PORT}/docs`);
          if (res.ok) {
            server.config.logger.info('[backend] Ready — API server is running', { timestamp: true });
            return;
          }
        } catch {
          // not ready yet
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
      server.config.logger.warn('[backend] Did not become ready within 30s — continuing anyway');
    },

    closeBundle() {
      if (backend) {
        backend.kill('SIGTERM');
        setTimeout(() => {
          if (backend) {
            backend.kill('SIGKILL');
            backend = null;
          }
        }, 3000);
        backend = null;
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), spawnBackend()],
  server: {
    port: 5173,
    proxy: {
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/v1': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
});
