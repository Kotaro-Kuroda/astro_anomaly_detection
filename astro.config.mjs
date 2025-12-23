// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://Kotaro-Kuroda.github.io',
  base: '/astro_anomaly_detection',
  vite: {
    plugins: [tailwindcss()]
  }
});
