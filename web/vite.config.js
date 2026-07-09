import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" keeps asset URLs relative so the static build works on any host
// (Vercel, Netlify, GitHub Pages subpaths) without extra config.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
