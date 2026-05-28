/**
 * Pathology Active Learning UI
 * Handles Canvas rendering, tool states, backend syncing,
 * and interactive Sandbox Inference mode with correction workflow.
 */

// --- Color palette for per-class rendering in sandbox ---
const CLASS_COLORS = [
    '#34d399', '#60a5fa', '#f472b6', '#fbbf24',
    '#a78bfa', '#fb923c', '#2dd4bf', '#f87171',
    '#818cf8', '#4ade80'
];

// --- State ---
let taxonomy = [];
let currentTile = null; // { id, path }
let boxes = []; // { id, class_label, x_c, y_c, w, h, selected, confidence? }
let boxIdCounter = 0;

let currentMode = 'draw'; // 'draw' | 'move'
let selectedBoxId = null;
let appMode = 'al'; // 'al' | 'sandbox'

// Drawing state
let isDrawing = false;
let drawStartX = 0;
let drawStartY = 0;
let drawCurrentX = 0;
let drawCurrentY = 0;

// Dragging state
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

// Sandbox state
let sandboxPredictions = []; // raw predictions from API
let confThreshold = 0.25;
let sandboxImageLoaded = false;
let sandboxFilenameOnServer = null; // filename returned by backend after upload
let sandboxCorrected = false; // tracks if user made any correction

// --- Elements ---
const canvasContainer = document.getElementById('canvas-container');
const imgCanvas = document.getElementById('image-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const ctxImg = imgCanvas.getContext('2d');
const ctxOverlay = overlayCanvas.getContext('2d');

// AL tools
const btnDraw = document.getElementById('tool-draw');
const btnMove = document.getElementById('tool-move');
const btnClear = document.getElementById('btn-clear');
const btnSave = document.getElementById('btn-save');
const btnDeleteBox = document.getElementById('btn-delete-box');

const controlsEmpty = document.getElementById('controls-empty');
const controlsActive = document.getElementById('controls-active');
const classSelect = document.getElementById('class-select');

const statId = document.getElementById('stat-id');
const statBoxes = document.getElementById('stat-boxes');
const tileStatus = document.getElementById('tile-status');

// Mode switcher
const modeAlBtn = document.getElementById('mode-al');
const modeSandboxBtn = document.getElementById('mode-sandbox');

// Toolbars
const alToolbar = document.getElementById('al-toolbar');
const sandboxToolbar = document.getElementById('sandbox-toolbar');
const sandboxFilenameEl = document.getElementById('sandbox-filename');
const btnClearSandbox = document.getElementById('btn-clear-sandbox');

// Sandbox tools
const sandboxToolDraw = document.getElementById('sandbox-tool-draw');
const sandboxToolMove = document.getElementById('sandbox-tool-move');

// Sidebar panels
const alSidebarPanels = document.getElementById('al-sidebar-panels');
const sandboxSidebarPanels = document.getElementById('sandbox-sidebar-panels');

// Sandbox box editor
const sandboxControlsEmpty = document.getElementById('sandbox-controls-empty');
const sandboxControlsActive = document.getElementById('sandbox-controls-active');
const sandboxClassSelect = document.getElementById('sandbox-class-select');
const btnDeleteSandboxBox = document.getElementById('btn-delete-sandbox-box');

// Actions
const alActions = document.getElementById('al-actions');
const sandboxActions = document.getElementById('sandbox-actions');
const fileUpload = document.getElementById('file-upload');
const btnUpload = document.getElementById('btn-upload');
const btnSaveCorrections = document.getElementById('btn-save-corrections');

// Sandbox results
const sandboxResultsEmpty = document.getElementById('sandbox-results-empty');
const sandboxResultsList = document.getElementById('sandbox-results-list');
const sandboxDetCount = document.getElementById('sandbox-det-count');
const sandboxVisCount = document.getElementById('sandbox-vis-count');
const sandboxBreakdown = document.getElementById('sandbox-breakdown');

// Confidence slider
const confSlider = document.getElementById('conf-slider');
const confValue = document.getElementById('conf-value');

// Dropzone
const dropzone = document.getElementById('dropzone');

// Spinner
const inferenceSpinner = document.getElementById('inference-spinner');

// An Image object to hold the current raw_tile
let currentImg = new Image();

// --- Initialization ---
async function init() {
    await fetchConfig();
    setupEventListeners();
    setupSandboxListeners();
    await fetchNextTile();
}

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        taxonomy = data.taxonomy;
        
        // Populate AL select box
        classSelect.innerHTML = '';
        taxonomy.forEach(cls => {
            const opt = document.createElement('option');
            opt.value = cls;
            opt.textContent = cls;
            classSelect.appendChild(opt);
        });
        
        // Populate Sandbox select box
        sandboxClassSelect.innerHTML = '';
        taxonomy.forEach(cls => {
            const opt = document.createElement('option');
            opt.value = cls;
            opt.textContent = cls;
            sandboxClassSelect.appendChild(opt);
        });
    } catch (e) {
        console.error("Failed to load taxonomy", e);
    }
}

async function fetchNextTile() {
    tileStatus.textContent = "Fetching next tile...";
    tileStatus.style.color = "var(--accent-primary)";
    btnSave.disabled = true;
    boxes = [];
    selectedBoxId = null;
    updateSidebar();

    try {
        const res = await fetch('/api/tile/next');
        const data = await res.json();
        
        if (data.status === 'empty') {
            tileStatus.textContent = "Queue Empty. All caught up!";
            tileStatus.style.color = "var(--box-color-default)";
            ctxImg.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
            ctxOverlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            currentTile = null;
            return;
        }

        currentTile = data.tile;
        statId.textContent = currentTile.id;
        tileStatus.textContent = `Reviewing Tile #${currentTile.id}`;
        tileStatus.style.color = "var(--text-main)";

        // Load image into canvas
        currentImg.onload = () => {
            imgCanvas.width = currentImg.width;
            imgCanvas.height = currentImg.height;
            overlayCanvas.width = currentImg.width;
            overlayCanvas.height = currentImg.height;

            ctxImg.drawImage(currentImg, 0, 0);

            data.annotations.forEach(a => {
                boxes.push({
                    id: boxIdCounter++,
                    class_label: a.class_label,
                    x_c: a.x_center,
                    y_c: a.y_center,
                    w: a.width,
                    h: a.height,
                    selected: false
                });
            });

            btnSave.disabled = false;
            redrawOverlay();
        };

        currentImg.src = data.tile.path;

    } catch (e) {
        console.error(e);
        tileStatus.textContent = "Error loading tile.";
        tileStatus.style.color = "var(--accent-danger)";
    }
}

// --- Mode Switching ---
function switchAppMode(mode) {
    appMode = mode;
    
    modeAlBtn.classList.toggle('active', mode === 'al');
    modeSandboxBtn.classList.toggle('active', mode === 'sandbox');
    
    if (mode === 'al') {
        alToolbar.classList.remove('hidden');
        sandboxToolbar.classList.add('hidden');
        alSidebarPanels.classList.remove('hidden');
        sandboxSidebarPanels.classList.add('hidden');
        alActions.classList.remove('hidden');
        sandboxActions.classList.add('hidden');
        dropzone.classList.add('hidden');
        
        tileStatus.textContent = currentTile 
            ? `Reviewing Tile #${currentTile.id}` 
            : 'Queue Empty. All caught up!';
        tileStatus.style.color = 'var(--text-main)';
        
        overlayCanvas.style.cursor = 'crosshair';
        
    } else {
        alToolbar.classList.add('hidden');
        sandboxToolbar.classList.remove('hidden');
        alSidebarPanels.classList.add('hidden');
        sandboxSidebarPanels.classList.remove('hidden');
        alActions.classList.add('hidden');
        sandboxActions.classList.remove('hidden');
        
        tileStatus.textContent = 'Sandbox — Predict & Correct';
        tileStatus.style.color = 'var(--accent-amber)';
        
        overlayCanvas.style.cursor = 'crosshair';
        
        if (!sandboxImageLoaded) {
            dropzone.classList.remove('hidden');
            ctxImg.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
            ctxOverlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        } else {
            dropzone.classList.add('hidden');
        }
    }
}

// --- Tools & Events ---
function setSandboxMode(mode) {
    currentMode = mode;
    sandboxToolDraw.classList.toggle('active', mode === 'draw');
    sandboxToolMove.classList.toggle('active', mode === 'move');
    if (mode === 'draw') {
        sandboxSelectBox(null);
    }
}

function setMode(mode) {
    currentMode = mode;
    btnDraw.classList.toggle('active', mode === 'draw');
    btnMove.classList.toggle('active', mode === 'move');
    if (mode === 'draw') {
        selectBox(null);
    }
}

function setupEventListeners() {
    btnDraw.addEventListener('click', () => setMode('draw'));
    btnMove.addEventListener('click', () => setMode('move'));
    
    btnClear.addEventListener('click', () => {
        boxes = [];
        selectBox(null);
        redrawOverlay();
    });

    btnDeleteBox.addEventListener('click', () => {
        if (selectedBoxId !== null) {
            boxes = boxes.filter(b => b.id !== selectedBoxId);
            selectBox(null);
            redrawOverlay();
        }
    });

    classSelect.addEventListener('change', (e) => {
        if (selectedBoxId !== null) {
            const b = boxes.find(bx => bx.id === selectedBoxId);
            if (b) b.class_label = e.target.value;
            redrawOverlay();
        }
    });

    btnSave.addEventListener('click', saveAndNext);

    // Mouse events on overlay
    overlayCanvas.addEventListener('mousedown', onMouseDown);
    overlayCanvas.addEventListener('mousemove', onMouseMove);
    overlayCanvas.addEventListener('mouseup', onMouseUp);
    overlayCanvas.addEventListener('mouseleave', onMouseUp);
    
    // Mode switcher
    modeAlBtn.addEventListener('click', () => switchAppMode('al'));
    modeSandboxBtn.addEventListener('click', () => switchAppMode('sandbox'));
}

function setupSandboxListeners() {
    // Upload button triggers hidden file input
    btnUpload.addEventListener('click', () => fileUpload.click());
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSandboxFile(e.target.files[0]);
            fileUpload.value = ''; // reset so same file can be re-uploaded
        }
    });
    
    // Drag and drop on dropzone
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSandboxFile(e.dataTransfer.files[0]);
        }
    });
    
    // Sandbox tools
    sandboxToolDraw.addEventListener('click', () => setSandboxMode('draw'));
    sandboxToolMove.addEventListener('click', () => setSandboxMode('move'));
    
    // Clear sandbox
    btnClearSandbox.addEventListener('click', resetSandbox);
    
    // Sandbox class select
    sandboxClassSelect.addEventListener('change', (e) => {
        if (selectedBoxId !== null) {
            const b = boxes.find(bx => bx.id === selectedBoxId);
            if (b) {
                b.class_label = e.target.value;
                sandboxCorrected = true;
                btnSaveCorrections.disabled = false;
            }
            redrawOverlay();
            updateSandboxBreakdown();
        }
    });
    
    // Delete sandbox box
    btnDeleteSandboxBox.addEventListener('click', () => {
        if (selectedBoxId !== null) {
            boxes = boxes.filter(b => b.id !== selectedBoxId);
            sandboxSelectBox(null);
            sandboxCorrected = true;
            btnSaveCorrections.disabled = false;
            redrawOverlay();
            updateSandboxBreakdown();
        }
    });
    
    // Confidence slider
    confSlider.addEventListener('input', (e) => {
        confThreshold = parseInt(e.target.value) / 100;
        confValue.textContent = confThreshold.toFixed(2);
        filterAndRenderSandbox();
    });
    
    // Save corrections
    btnSaveCorrections.addEventListener('click', saveCorrections);
}

function resetSandbox() {
    sandboxImageLoaded = false;
    sandboxPredictions = [];
    sandboxFilenameOnServer = null;
    sandboxCorrected = false;
    boxes = [];
    selectedBoxId = null;
    sandboxFilenameEl.textContent = 'No image loaded';
    sandboxResultsEmpty.classList.remove('hidden');
    sandboxResultsList.classList.add('hidden');
    sandboxBreakdown.innerHTML = '';
    sandboxSelectBox(null);
    btnSaveCorrections.disabled = true;
    ctxImg.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
    ctxOverlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    dropzone.classList.remove('hidden');
}

// --- Sandbox Box Selection ---
function sandboxSelectBox(id) {
    selectedBoxId = id;
    boxes.forEach(b => b.selected = (b.id === id));
    
    if (id !== null) {
        sandboxControlsEmpty.classList.add('hidden');
        sandboxControlsActive.classList.remove('hidden');
        const b = boxes.find(bx => bx.id === id);
        if (b) sandboxClassSelect.value = b.class_label;
    } else {
        sandboxControlsEmpty.classList.remove('hidden');
        sandboxControlsActive.classList.add('hidden');
    }
    
    redrawOverlay();
}

// --- Sandbox Logic ---
async function handleSandboxFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Invalid file type. Please upload JPG, PNG, BMP, TIFF, or WebP.');
        return;
    }
    
    // Reset state for new image
    sandboxCorrected = false;
    sandboxPredictions = [];
    boxes = [];
    selectedBoxId = null;
    sandboxSelectBox(null);
    btnSaveCorrections.disabled = true;
    
    // Show spinner, hide dropzone
    dropzone.classList.add('hidden');
    inferenceSpinner.classList.remove('hidden');
    sandboxFilenameEl.textContent = file.name;
    tileStatus.textContent = 'Running inference...';
    tileStatus.style.color = 'var(--accent-primary)';
    
    // Load image into canvas for preview while waiting
    const localUrl = URL.createObjectURL(file);
    const previewImg = new Image();
    
    previewImg.onload = () => {
        imgCanvas.width = previewImg.width;
        imgCanvas.height = previewImg.height;
        overlayCanvas.width = previewImg.width;
        overlayCanvas.height = previewImg.height;
        ctxImg.drawImage(previewImg, 0, 0);
        ctxOverlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        URL.revokeObjectURL(localUrl);
    };
    previewImg.src = localUrl;
    
    // Upload to backend
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const res = await fetch('/api/test/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error ${res.status}`);
        }
        
        const data = await res.json();
        sandboxPredictions = data.predictions || [];
        sandboxFilenameOnServer = data.sandbox_filename;
        sandboxImageLoaded = true;
        
        tileStatus.textContent = `Sandbox — ${sandboxPredictions.length} detection(s) — Edit & Save`;
        tileStatus.style.color = 'var(--accent-success)';
        
        // Enable save button immediately (user can accept model predictions as-is)
        btnSaveCorrections.disabled = false;
        
        filterAndRenderSandbox();
        
    } catch (err) {
        console.error('Inference error:', err);
        tileStatus.textContent = `Error: ${err.message}`;
        tileStatus.style.color = 'var(--accent-danger)';
        dropzone.classList.remove('hidden');
    } finally {
        inferenceSpinner.classList.add('hidden');
    }
}

function filterAndRenderSandbox() {
    // Filter predictions by confidence threshold
    const filtered = sandboxPredictions.filter(p => p.confidence >= confThreshold);
    
    // Preserve any user-drawn boxes (those without confidence field)
    const userBoxes = boxes.filter(b => b.confidence === undefined);
    
    // Convert filtered predictions to box objects
    const predBoxes = filtered.map((p, i) => ({
        id: 10000 + i, // high IDs to avoid collision with user-drawn ones
        class_label: p.class_label,
        x_c: p.x_center,
        y_c: p.y_center,
        w: p.width,
        h: p.height,
        confidence: p.confidence,
        selected: false
    }));
    
    boxes = [...predBoxes, ...userBoxes];
    selectedBoxId = null;
    sandboxSelectBox(null);
    
    updateSandboxBreakdown();
    redrawOverlay();
}

function updateSandboxBreakdown() {
    // Update stats
    sandboxDetCount.textContent = sandboxPredictions.length;
    sandboxVisCount.textContent = boxes.length;
    
    if (boxes.length > 0 || sandboxPredictions.length > 0) {
        sandboxResultsEmpty.classList.add('hidden');
        sandboxResultsList.classList.remove('hidden');
        renderBreakdown(boxes);
    } else {
        sandboxResultsEmpty.classList.remove('hidden');
        sandboxResultsList.classList.add('hidden');
    }
}

function renderBreakdown(visibleBoxes) {
    // Aggregate by class
    const counts = {};
    visibleBoxes.forEach(b => {
        counts[b.class_label] = (counts[b.class_label] || 0) + 1;
    });
    
    // Build unique class list for consistent coloring
    const allClasses = [...new Set(visibleBoxes.map(b => b.class_label))];
    
    sandboxBreakdown.innerHTML = '';
    allClasses.forEach((cls, i) => {
        const count = counts[cls] || 0;
        const color = CLASS_COLORS[i % CLASS_COLORS.length];
        
        const row = document.createElement('div');
        row.className = 'det-class-row';
        row.innerHTML = `
            <span class="det-class-swatch" style="background: ${color};"></span>
            <span class="det-class-name">${cls}</span>
            <span class="det-class-count">${count}</span>
        `;
        sandboxBreakdown.appendChild(row);
    });
}

// --- Save Corrections to Replay Buffer ---
async function saveCorrections() {
    if (!sandboxFilenameOnServer || boxes.length === 0) return;
    
    const btn = btnSaveCorrections;
    const btnSpan = btn.querySelector('span');
    btn.disabled = true;
    btnSpan.textContent = 'Saving...';
    
    try {
        const payload = {
            sandbox_filename: sandboxFilenameOnServer,
            annotations: boxes.map(b => ({
                class_label: b.class_label,
                x_center: b.x_c,
                y_center: b.y_c,
                width: b.w,
                height: b.h,
                confidence: 1.0 // all corrections are treated as ground truth
            }))
        };
        
        const res = await fetch('/api/test/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Save failed`);
        }
        
        const data = await res.json();
        
        // Show success toast
        showToast(`✓ ${data.message}`);
        
        tileStatus.textContent = 'Saved! Upload another image to continue.';
        tileStatus.style.color = 'var(--accent-success)';
        
        btnSpan.textContent = 'Saved ✓';
        
        // After 2 seconds, re-enable for another upload
        setTimeout(() => {
            btnSpan.textContent = 'Save Corrections & Learn';
            btn.disabled = true;
        }, 2000);
        
    } catch (err) {
        console.error('Save error:', err);
        showToast(`✗ Error: ${err.message}`);
        btn.disabled = false;
        btnSpan.textContent = 'Save Corrections & Learn';
    }
}

function showToast(message) {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 3000);
}

// --- Canvas Interactions ---
function getActualCoords(e) {
    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function onMouseDown(e) {
    // In sandbox mode, require image to be loaded
    if (appMode === 'sandbox' && !sandboxImageLoaded) return;
    // In AL mode, require tile to be loaded
    if (appMode === 'al' && !currentTile) return;
    
    const {x, y} = getActualCoords(e);

    if (currentMode === 'draw') {
        isDrawing = true;
        drawStartX = x;
        drawStartY = y;
        drawCurrentX = x;
        drawCurrentY = y;
    } else if (currentMode === 'move') {
        const clickedBox = getBoxAtPosition(x, y);
        
        if (appMode === 'sandbox') {
            sandboxSelectBox(clickedBox ? clickedBox.id : null);
        } else {
            selectBox(clickedBox ? clickedBox.id : null);
        }
        
        if (clickedBox) {
            isDragging = true;
            dragStartX = x;
            dragStartY = y;
        }
    }
}

function onMouseMove(e) {
    if (appMode === 'sandbox' && !sandboxImageLoaded) return;
    if (appMode === 'al' && !currentTile) return;
    
    const {x, y} = getActualCoords(e);

    if (isDrawing) {
        drawCurrentX = x;
        drawCurrentY = y;
        redrawOverlay();
        
        // Draw the temporary box
        ctxOverlay.strokeStyle = 'rgba(255,255,255,0.8)';
        ctxOverlay.lineWidth = 2;
        ctxOverlay.setLineDash([5, 5]);
        ctxOverlay.strokeRect(
            drawStartX, 
            drawStartY, 
            drawCurrentX - drawStartX, 
            drawCurrentY - drawStartY
        );
        ctxOverlay.setLineDash([]);
        
    } else if (isDragging && selectedBoxId !== null) {
        const dx = (x - dragStartX) / overlayCanvas.width;
        const dy = (y - dragStartY) / overlayCanvas.height;
        
        const b = boxes.find(bx => bx.id === selectedBoxId);
        if (b) {
            b.x_c += dx;
            b.y_c += dy;
            if (appMode === 'sandbox') {
                sandboxCorrected = true;
                btnSaveCorrections.disabled = false;
            }
        }
        
        dragStartX = x;
        dragStartY = y;
        redrawOverlay();
    } else {
        // Just hover crosshairs
        redrawOverlay();
        drawCrosshairs(x, y);
    }
}

function onMouseUp(e) {
    if (isDrawing) {
        isDrawing = false;
        
        const width = Math.abs(drawCurrentX - drawStartX);
        const height = Math.abs(drawCurrentY - drawStartY);
        
        if (width > 5 && height > 5) {
            const minX = Math.min(drawStartX, drawCurrentX);
            const minY = Math.min(drawStartY, drawCurrentY);
            
            // Normalize
            const w_n = width / overlayCanvas.width;
            const h_n = height / overlayCanvas.height;
            const xc_n = (minX / overlayCanvas.width) + (w_n / 2);
            const yc_n = (minY / overlayCanvas.height) + (h_n / 2);
            
            const newBoxId = boxIdCounter++;
            const defaultClass = taxonomy.length > 0 ? taxonomy[0] : "Benign";

            const newBox = {
                id: newBoxId,
                class_label: defaultClass,
                x_c: xc_n,
                y_c: yc_n,
                w: w_n,
                h: h_n,
                selected: false
                // NOTE: no `confidence` key — marks this as a user-drawn box
            };
            
            boxes.push(newBox);
            
            if (appMode === 'sandbox') {
                sandboxCorrected = true;
                btnSaveCorrections.disabled = false;
                // Auto-select the new box so user can set class
                setSandboxMode('move');
                sandboxSelectBox(newBoxId);
                updateSandboxBreakdown();
            } else {
                updateSidebar();
            }
        }
        redrawOverlay();
    }
    
    isDragging = false;
}

// --- Helper Functions ---
function getBoxAtPosition(px, py) {
    for (let i = boxes.length - 1; i >= 0; i--) {
        const b = boxes[i];
        const bw = b.w * overlayCanvas.width;
        const bh = b.h * overlayCanvas.height;
        const bx = (b.x_c * overlayCanvas.width) - (bw / 2);
        const by = (b.y_c * overlayCanvas.height) - (bh / 2);
        
        if (px >= bx && px <= bx + bw && py >= by && py <= by + bh) {
            return b;
        }
    }
    return null;
}

function selectBox(id) {
    selectedBoxId = id;
    boxes.forEach(b => b.selected = (b.id === id));
    updateSidebar();
    redrawOverlay();
}

function updateSidebar() {
    statBoxes.textContent = boxes.length;
    
    if (selectedBoxId !== null) {
        controlsEmpty.classList.add('hidden');
        controlsActive.classList.remove('hidden');
        const b = boxes.find(bx => bx.id === selectedBoxId);
        classSelect.value = b.class_label;
    } else {
        controlsEmpty.classList.remove('hidden');
        controlsActive.classList.add('hidden');
    }
}

function redrawOverlay() {
    ctxOverlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    const cw = overlayCanvas.width;
    const ch = overlayCanvas.height;
    
    // Build class color map for sandbox mode
    const allClasses = appMode === 'sandbox'
        ? [...new Set(boxes.map(b => b.class_label))]
        : [];
    
    boxes.forEach(b => {
        const bw = b.w * cw;
        const bh = b.h * ch;
        const bx = (b.x_c * cw) - (bw / 2);
        const by = (b.y_c * ch) - (bh / 2);
        
        // Pick color
        let strokeColor;
        if (appMode === 'sandbox') {
            const classIdx = allClasses.indexOf(b.class_label);
            strokeColor = CLASS_COLORS[classIdx % CLASS_COLORS.length];
        } else {
            strokeColor = b.selected ? '#fbbf24' : '#34d399';
        }

        // Styling
        ctxOverlay.lineWidth = b.selected ? 3 : 2;
        ctxOverlay.strokeStyle = strokeColor;
        
        // Shadow / Glow for selected
        if (b.selected) {
            ctxOverlay.shadowColor = 'rgba(251, 191, 36, 0.8)';
            ctxOverlay.shadowBlur = 12;
        } else {
            ctxOverlay.shadowBlur = 0;
        }

        ctxOverlay.strokeRect(bx, by, bw, bh);
        
        // Draw Label Tag
        ctxOverlay.shadowBlur = 0;
        
        let tagText = b.class_label;
        if (appMode === 'sandbox' && b.confidence !== undefined) {
            tagText += ` ${(b.confidence * 100).toFixed(0)}%`;
        }
        if (b.confidence === undefined) {
            tagText += ' ✎'; // mark user-drawn boxes
        }
        
        ctxOverlay.font = "bold 13px 'Outfit'";
        const textM = ctxOverlay.measureText(tagText);
        const tagW = textM.width + 10;
        const tagH = 22;
        const tagX = bx;
        const tagY = by - tagH;
        
        // Tag background
        ctxOverlay.fillStyle = strokeColor;
        ctxOverlay.beginPath();
        if (ctxOverlay.roundRect) {
            ctxOverlay.roundRect(tagX, tagY, tagW, tagH, [4, 4, 0, 0]);
        } else {
            ctxOverlay.rect(tagX, tagY, tagW, tagH);
        }
        ctxOverlay.fill();
        
        // Tag text
        ctxOverlay.fillStyle = "#000";
        ctxOverlay.fillText(tagText, tagX + 5, tagY + 15);
    });
}

function drawCrosshairs(x, y) {
    ctxOverlay.shadowBlur = 0;
    ctxOverlay.strokeStyle = 'rgba(255,255,255,0.3)';
    ctxOverlay.lineWidth = 1;

    ctxOverlay.beginPath();
    ctxOverlay.moveTo(x, 0);
    ctxOverlay.lineTo(x, overlayCanvas.height);
    ctxOverlay.moveTo(0, y);
    ctxOverlay.lineTo(overlayCanvas.width, y);
    ctxOverlay.stroke();
}

// --- Sync ---
async function saveAndNext() {
    if (!currentTile) return;
    
    try {
        btnSave.disabled = true;
        btnSave.querySelector('span').textContent = "Saving...";

        const payload = {
            annotations: boxes.map(b => ({
                class_label: b.class_label,
                x_center: b.x_c,
                y_center: b.y_c,
                width: b.w,
                height: b.h
            }))
        };

        const res = await fetch(`/api/tile/${currentTile.id}/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Save failed");

        btnSave.querySelector('span').textContent = "Save & Next";
        
        await fetchNextTile();

    } catch (e) {
        console.error(e);
        alert("Failed to save annotations!");
        btnSave.disabled = false;
        btnSave.querySelector('span').textContent = "Save & Next";
    }
}

// Start
window.addEventListener('DOMContentLoaded', init);
