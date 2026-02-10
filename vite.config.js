import { defineConfig } from "vite";

export default defineConfig({
	server: {
		open: true,
	},
	assetsInclude: ["**/*.wasm"],
	build: {
		target: "esnext",
	},
	// Fix for Web Workers with modules
	worker: {
		format: 'es',
		plugins: () => []
	},
	// Ensure workers are treated as modules
	optimizeDeps: {
		exclude: ['imageGenerationWorker.js']
	}
});