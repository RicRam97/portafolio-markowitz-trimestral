import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    server: {
        port: 5173,
        host: true
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                login: resolve(__dirname, 'login.html'),
                privacidad: resolve(__dirname, 'privacidad.html'),
                terminos: resolve(__dirname, 'terminos.html'),
                disclaimer: resolve(__dirname, 'disclaimer.html'),
                cookies: resolve(__dirname, 'cookies.html'),
            }
        }
    }
});
