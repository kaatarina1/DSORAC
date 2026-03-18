const MODE_META = {
    POINTS:     { label: "● Points",     info: "GL points. Fast, 1 vertex/point." },
    DISKS:      { label: "◉ Disks",      info: "Camera-facing circular quads with hard edge." },
    BILLBOARDS: { label: "⬡ Billboards", info: "Normal-facing circular quads with hard edge.." },
    GAUSSIANS:  { label: "✦ Gaussians",  info: "Normal-facing Gaussian alpha splats with blending." },
};

export class RenderingControls {
    constructor({ modes, currentMode, currentPointSize, onModeChange, onSizeChange, useReconstruction = true, onReconstructionChange }) {
        this.modes            = modes;
        this.currentMode      = currentMode;
        this.currentPointSize = currentPointSize;
        this.onModeChange     = onModeChange;
        this.onSizeChange     = onSizeChange;
        this.useReconstruction = useReconstruction;
        this.onReconstructionChange = onReconstructionChange;
        this._modeButtons     = {};
        this._reconstructionCheckbox = null;
    }

    mount(container) {
        this._injectStyles();
        container.appendChild(this._buildPanel());
    }

    _buildPanel() {
        const panel = document.createElement("div");
        panel.id = "rendering-controls";

        const h = document.createElement("h3");
        h.textContent = "Rendering";
        panel.appendChild(h);

        const grid = document.createElement("div");
        grid.className = "rc-mode-grid";
        for (const mode of this.modes) {
            const btn = document.createElement("button");
            btn.className    = "rc-mode-btn" + (mode === this.currentMode ? " active" : "");
            btn.textContent  = (MODE_META[mode] || {}).label || mode;
            btn.addEventListener("click", () => this._selectMode(mode));
            grid.appendChild(btn);
            this._modeButtons[mode] = btn;
        }
        panel.appendChild(grid);

        const sizeRow = document.createElement("div");
        sizeRow.className = "rc-label-row";
        const sizeLabel = document.createElement("span");
        sizeLabel.className   = "rc-label";
        sizeLabel.textContent = "Size";
        this._sizeValueEl = document.createElement("span");
        this._sizeValueEl.className   = "rc-value";
        this._sizeValueEl.textContent = this._fmt(this.currentPointSize);
        sizeRow.append(sizeLabel, this._sizeValueEl);
        panel.appendChild(sizeRow);

        const slider = document.createElement("input");
        slider.type      = "range";
        slider.className = "rc-slider";
        slider.min = "0"; slider.max = "1000"; slider.step = "1";
        slider.value = String(this._sizeToSlider(this.currentPointSize));
        slider.addEventListener("input", () => {
            const sz = this._sliderToSize(Number(slider.value));
            this._sizeValueEl.textContent = this._fmt(sz);
            if (this.onSizeChange) this.onSizeChange(sz);
        });
        panel.appendChild(slider);

        const div = document.createElement("div");
        div.className = "rc-divider";
        panel.appendChild(div);

        const reconstructionRow = document.createElement("div");
        reconstructionRow.className = "rc-label-row";
        this._reconstructionCheckbox = document.createElement("input");
        this._reconstructionCheckbox.type = "checkbox";
        this._reconstructionCheckbox.checked = this.useReconstruction;
        this._reconstructionCheckbox.addEventListener("change", () => {
            // Only allow changes when in POINTS mode
            if (this.currentMode === "POINTS") {
                this.useReconstruction = this._reconstructionCheckbox.checked;
                if (this.onReconstructionChange) this.onReconstructionChange(this.useReconstruction);
            } else {
                // Reset checkbox if somehow changed when disabled
                this._reconstructionCheckbox.checked = false;
            }
        });
        const reconstructionLabel = document.createElement("label");
        reconstructionLabel.className = "rc-checkbox-label";
        reconstructionLabel.style.display = "flex";
        reconstructionLabel.style.alignItems = "center";
        reconstructionLabel.style.gap = "8px";
        reconstructionLabel.style.cursor = "pointer";
        reconstructionLabel.appendChild(this._reconstructionCheckbox);
        reconstructionLabel.appendChild(document.createTextNode("Use Reconstruction"));
        reconstructionRow.appendChild(reconstructionLabel);
        panel.appendChild(reconstructionRow);

        // Initialize checkbox state based on current mode
        this._updateReconstructionCheckbox(this.currentMode);

        const div2 = document.createElement("div");
        div2.className = "rc-divider";
        panel.appendChild(div2);

        this._infoEl = document.createElement("div");
        this._infoEl.className   = "rc-info";
        this._infoEl.textContent = (MODE_META[this.currentMode] || {}).info || "";
        panel.appendChild(this._infoEl);

        return panel;
    }

    _selectMode(mode) {
        this.currentMode = mode;
        for (const [k, btn] of Object.entries(this._modeButtons)) {
            btn.classList.toggle("active", k === mode);
        }
        this._infoEl.textContent = (MODE_META[mode] || {}).info || "";
        
        // Update reconstruction checkbox based on mode
        this._updateReconstructionCheckbox(mode);
        
        if (this.onModeChange) this.onModeChange(mode);
    }

    _updateReconstructionCheckbox(mode) {
        if (!this._reconstructionCheckbox) return;
        
        const isPointsMode = mode === "POINTS";
        this._reconstructionCheckbox.checked = isPointsMode ? this.useReconstruction : false;
        this._reconstructionCheckbox.disabled = !isPointsMode;
        
        // Update label styling to show disabled state
        const label = this._reconstructionCheckbox.parentElement;
        if (isPointsMode) {
            label.style.opacity = "1";
            label.style.pointerEvents = "auto";
        } else {
            label.style.opacity = "0.5";
            label.style.pointerEvents = "none";
        }
    }

    _sliderToSize(v) { return 0.1 + (v / 1000) * 0.9; }
    _sizeToSlider(s) { return Math.round((s - 0.1) / 0.9 * 1000); }
    _fmt(s)          { return s < 0.01 ? s.toExponential(2) : s.toFixed(4); }

    _injectStyles() {
        if (document.getElementById("rc-styles")) return;
        const style = document.createElement("style");
        style.id = "rc-styles";
        style.textContent = `
        #rendering-controls {
            position: fixed; top: 16px; right: 16px; z-index: 1000;
            background: rgba(12,12,18,0.88);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.1); border-radius: 14px;
            padding: 14px 16px 16px; min-width: 220px;
            font-family: system-ui, -apple-system, sans-serif;
            color: #dde; box-shadow: 0 8px 40px rgba(0,0,0,0.55); user-select: none;
        }
        #rendering-controls h3 {
            margin: 0 0 12px; font-size: 10px; letter-spacing: 2px;
            text-transform: uppercase; color: #667; font-weight: 700;
        }
        .rc-mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 14px; }
        .rc-mode-btn {
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px; color: #aab; padding: 8px 4px; font-size: 12px;
            cursor: pointer; text-align: center; line-height: 1.5;
            transition: background 0.12s, border-color 0.12s, color 0.12s;
        }
        .rc-mode-btn:hover { background: rgba(255,255,255,0.1); color: #eef; }
        .rc-mode-btn.active {
            background: rgba(99,179,237,0.18); border-color: rgba(99,179,237,0.55);
            color: #7ec8f4; font-weight: 600;
        }
        .rc-label-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .rc-label { font-size: 12px; color: #889; }
        .rc-value {
            font-size: 11px; font-variant-numeric: tabular-nums; color: #ccd;
            background: rgba(255,255,255,0.07); padding: 2px 7px; border-radius: 5px;
        }
        .rc-slider {
            -webkit-appearance: none; appearance: none; width: 100%; height: 4px;
            border-radius: 2px; background: rgba(255,255,255,0.13);
            outline: none; cursor: pointer; margin-bottom: 14px;
        }
        .rc-slider::-webkit-slider-thumb {
            -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
            background: #7ec8f4; cursor: pointer; box-shadow: 0 0 8px rgba(99,179,237,0.5);
        }
        .rc-slider::-moz-range-thumb {
            width: 14px; height: 14px; border-radius: 50%;
            background: #7ec8f4; cursor: pointer; border: none;
        }
        .rc-divider { height: 1px; background: rgba(255,255,255,0.07); margin: 4px 0 10px; }
        .rc-checkbox-label { font-size: 12px; color: #889; }
        .rc-checkbox-label input[type="checkbox"] {
            width: 16px; height: 16px; cursor: pointer;
            accent-color: #7ec8f4;
        }
        .rc-checkbox-label input[type="checkbox"]:disabled {
            cursor: not-allowed;
            opacity: 0.5;
        }
        .rc-info { font-size: 10.5px; color: #556; line-height: 1.55; min-height: 28px; }
        `;
        document.head.appendChild(style);
    }
}
