import { defineConfig } from "vite";

export default defineConfig({
	server: {
		open: true, // Opens the browser automatically on start
	},
	assetsInclude: ["**/*.wasm"],
	build: {
		target: "esnext",
	},
});
