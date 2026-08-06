const MODE_META = {
    POINTS:     { label: "● Points",     info: "GL points. Fast, 1 vertex/point." },
    DISKS:      { label: "◉ Disks",      info: "Camera-facing circular quads with hard edge." },
    BILLBOARDS: { label: "⬡ Billboards", info: "Normal-facing circular quads with hard edge.." },
    GAUSSIANS:  { label: "✦ Gaussians",  info: "Normal-facing Gaussian alpha splats with blending." },
};

export class RenderingControls {
    constructor({ modes, currentMode, currentPointSize, onModeChange, onSizeChange, useReconstruction = true, onReconstructionChange, onOrthoChange, onColorModeChange, onMasksLoaded, onDownloadLas, onConfigLoaded }) {
        this.modes            = modes;
        this.currentMode      = currentMode;
        this.currentPointSize = currentPointSize;
        this.onModeChange     = onModeChange;
        this.onSizeChange     = onSizeChange;
        this.useReconstruction = useReconstruction;
        this.onReconstructionChange = onReconstructionChange;
        this.onOrthoChange      = onOrthoChange;
        this.onColorModeChange  = onColorModeChange;
        this.onMasksLoaded      = onMasksLoaded;
        this.onDownloadLas      = onDownloadLas;
        this.onConfigLoaded     = onConfigLoaded;
        this.orthoMode          = false;
        this.colorMode          = 0; // 0=RGB, 1=segmentation classes, 2=LAS classes
        this.votingMode         = 'simple';
        this._modeButtons     = {};
        this._reconstructionCheckbox = null;
        this._orthoBtn        = null;
        this._orthoHint       = null;
        this._classBtn        = null;
        this._lasClassBtn     = null;
        this._segStatusEl     = null;
        this._downloadBtn     = null;
        this._evalStatusEl     = null;
    }

    mount(container) {
        this._injectStyles();
        container.appendChild(this._buildPanel());
        const hint = document.createElement('div');
        hint.id = 'ortho-hint';
        hint.textContent = 'W/S — move up/down  ·  Arrow keys — pan  ·  Scroll — zoom';
        this._orthoHint = hint;
        document.body.appendChild(hint);
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

        const div3 = document.createElement("div");
        div3.className = "rc-divider";
        panel.appendChild(div3);

        const classBtn = document.createElement("button");
        classBtn.className = "rc-ortho-btn";
        classBtn.innerHTML = `<span>◈</span> Segmentation classes`;
        classBtn.addEventListener('click', () => {
            const next = this.colorMode === 1 ? 0 : 1;
            this._setColorMode(next);
        });
        this._classBtn = classBtn;
        panel.appendChild(classBtn);

        const lasClassBtn = document.createElement("button");
        lasClassBtn.className = "rc-ortho-btn";
        lasClassBtn.innerHTML = `<span>◉</span> LAS classes`;
        lasClassBtn.style.marginTop = "4px";
        lasClassBtn.addEventListener('click', () => {
            const next = this.colorMode === 2 ? 0 : 2;
            this._setColorMode(next);
        });
        this._lasClassBtn = lasClassBtn;
        panel.appendChild(lasClassBtn);

        const div4 = document.createElement("div");
        div4.className = "rc-divider";
        panel.appendChild(div4);

        const orthoBtn = document.createElement("button");
        orthoBtn.className = "rc-ortho-btn";
        orthoBtn.innerHTML = `<span>⊡</span> Top-Down`;
        orthoBtn.addEventListener('click', () => {
            this.orthoMode = !this.orthoMode;
            orthoBtn.classList.toggle('active', this.orthoMode);
            orthoBtn.innerHTML = this.orthoMode
                ? `<span>⊡</span> Top-Down <span class="rc-ortho-tag">[ON]</span>`
                : `<span>⊡</span> Top-Down`;
            if (this._orthoHint) this._orthoHint.classList.toggle('visible', this.orthoMode);
            if (this.onOrthoChange) this.onOrthoChange(this.orthoMode);
        });
        this._orthoBtn = orthoBtn;
        panel.appendChild(orthoBtn);

        const divSeg = document.createElement("div");
        divSeg.className = "rc-divider";
        panel.appendChild(divSeg);

        const segLabel = document.createElement("div");
        segLabel.style.cssText = "font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#667;font-weight:700;margin-bottom:8px;";
        segLabel.textContent = "Segmentation";
        panel.appendChild(segLabel);

        const votingLabel = document.createElement("div");
        votingLabel.style.cssText = "font-size:11px;color:#889;margin-bottom:6px;";
        votingLabel.textContent = "Voting mode";
        panel.appendChild(votingLabel);

        const votingGrid = document.createElement("div");
        votingGrid.className = "rc-mode-grid";
        votingGrid.style.marginBottom = "10px";

        const simpleBtn = document.createElement("button");
        simpleBtn.className = "rc-mode-btn" + (this.votingMode === 'simple' ? " active" : "");
        simpleBtn.textContent = "Simple";
        simpleBtn.title = "Per-point argmax";

        const spatialBtn = document.createElement("button");
        spatialBtn.className = "rc-mode-btn" + (this.votingMode === 'spatial' ? " active" : "");
        spatialBtn.textContent = "Spatial";
        spatialBtn.title = "Top-3 + voxel neighbourhood refinement";

        simpleBtn.addEventListener("click", () => {
            this.votingMode = 'simple';
            simpleBtn.classList.add('active');
            spatialBtn.classList.remove('active');
        });
        spatialBtn.addEventListener("click", () => {
            this.votingMode = 'spatial';
            spatialBtn.classList.add('active');
            simpleBtn.classList.remove('active');
        });

        votingGrid.append(simpleBtn, spatialBtn);
        panel.appendChild(votingGrid);

        const fileInput = document.createElement("input");
        fileInput.type = "file"; fileInput.accept = ".zip"; fileInput.style.display = "none";
        fileInput.addEventListener("change", () => {
            if (fileInput.files[0] && this.onMasksLoaded) this.onMasksLoaded(fileInput.files[0], this.votingMode);
        });
        panel.appendChild(fileInput);

        const loadBtn = document.createElement("button");
        loadBtn.className = "rc-ortho-btn";
        loadBtn.innerHTML = `<span>📂</span> Upload masks (ZIP)`;
        loadBtn.addEventListener("click", () => fileInput.click());
        panel.appendChild(loadBtn);

        this._segStatusEl = document.createElement("div");
        this._segStatusEl.style.cssText = "font-size:11px;color:#667;margin-top:8px;min-height:16px;";
        panel.appendChild(this._segStatusEl);

        this._downloadBtn = document.createElement("button");
        this._downloadBtn.className = "rc-ortho-btn";
        this._downloadBtn.innerHTML = `<span>⬇</span> Download LAS`;
        this._downloadBtn.style.display = "none";
        this._downloadBtn.addEventListener("click", () => { if (this.onDownloadLas) this.onDownloadLas(); });
        panel.appendChild(this._downloadBtn);

        const divEval = document.createElement("div");
        divEval.className = "rc-divider";
        panel.appendChild(divEval);

        const evalLabel = document.createElement("div");
        evalLabel.style.cssText = "font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#667;font-weight:700;margin-bottom:8px;";
        evalLabel.textContent = "Evaluation";
        panel.appendChild(evalLabel);

        const jsonInput = document.createElement("input");
        jsonInput.type = "file"; jsonInput.accept = ".json"; jsonInput.style.display = "none";
        jsonInput.addEventListener("change", () => {
            if (jsonInput.files[0] && this.onConfigLoaded) this.onConfigLoaded(jsonInput.files[0]);
        });
        panel.appendChild(jsonInput);

        const evalBtn = document.createElement("button");
        evalBtn.className = "rc-ortho-btn";
        evalBtn.innerHTML = `<span>📝</span> Upload evaluation confih file (JSON)`;
        evalBtn.addEventListener("click", () => jsonInput.click());
        panel.appendChild(evalBtn);

        this._evalStatusEl = document.createElement("div");
        this._evalStatusEl.style.cssText = "font-size:11px;color:#667;margin-top:8px;min-height:16px;";
        panel.appendChild(this._evalStatusEl);

        return panel;
    }

    setSegmentationStatus(text, done = false) {
        if (this._segStatusEl) this._segStatusEl.textContent = text;
        if (this._downloadBtn) this._downloadBtn.style.display = done ? "" : "none";
    }

    enableClassification() {
        this._setColorMode(1);
    }

    _setColorMode(mode) {
        this.colorMode = mode;

        const classActive = mode === 1;
        const lasActive   = mode === 2;

        if (this._classBtn) {
            this._classBtn.classList.toggle('active', classActive);
            this._classBtn.innerHTML = classActive
                ? `<span>◈</span> Segmentation classes <span class="rc-ortho-tag">[ON]</span>`
                : `<span>◈</span> Segmentation classes`;
        }
        if (this._lasClassBtn) {
            this._lasClassBtn.classList.toggle('active', lasActive);
            this._lasClassBtn.innerHTML = lasActive
                ? `<span>◉</span> LAS classes <span class="rc-ortho-tag">[ON]</span>`
                : `<span>◉</span> LAS classes`;
        }

        if (this.onColorModeChange) this.onColorModeChange(mode);
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

        const label = this._reconstructionCheckbox.parentElement;
        label.style.opacity = isPointsMode ? "1" : "0.5";
        label.style.pointerEvents = isPointsMode ? "auto" : "none";

    }

    _sliderToSize(v) { return 0.001 + (v / 1000) * 0.9; }
    _sizeToSlider(s) { return Math.round((s - 0.001) / 0.9 * 1000); }
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
        .rc-ortho-btn {
            width: 100%; display: flex; align-items: center; gap: 7px;
            padding: 8px 10px; margin-top: 2px;
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;
            font-size: 12px; color: #aab; cursor: pointer; user-select: none;
            transition: background 0.12s, border-color 0.12s, color 0.12s; white-space: nowrap;
        }
        .rc-ortho-btn:hover { background: rgba(255,255,255,0.1); color: #eef; }
        .rc-ortho-btn.active {
            background: rgba(99,179,237,0.18); border-color: rgba(99,179,237,0.55);
            color: #7ec8f4; font-weight: 600;
        }
        .rc-ortho-tag { font-size: 10px; opacity: 0.7; }
        #ortho-hint {
            position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
            z-index: 1000; padding: 7px 16px;
            background: rgba(12,12,18,0.82); backdrop-filter: blur(8px);
            border: 1px solid rgba(99,179,237,0.3); border-radius: 8px;
            font-family: system-ui, -apple-system, sans-serif; font-size: 11px;
            color: #7ec8f4; pointer-events: none; opacity: 0; transition: opacity 0.25s;
        }
        #ortho-hint.visible { opacity: 1; }
        `;
        document.head.appendChild(style);
    }
}
