"use strict";

const CLASS_COLORS = [
  "#16a34a", "#2563eb", "#db2777", "#d97706", "#7c3aed",
  "#ea580c", "#0f766e", "#dc2626", "#4f46e5", "#65a30d"
];

const $ = (id) => document.getElementById(id);
const els = {
  imgCanvas: $("image-canvas"), overlayCanvas: $("overlay-canvas"),
  tileStatus: $("tile-status"), statId: $("stat-id"), statBoxes: $("stat-boxes"),
  modeAl: $("mode-al"), modeSandbox: $("mode-sandbox"), themeToggle: $("theme-toggle"),
  alToolbar: $("al-toolbar"), sandboxToolbar: $("sandbox-toolbar"),
  toolDraw: $("tool-draw"), toolMove: $("tool-move"), btnClear: $("btn-clear"),
  sandboxToolDraw: $("sandbox-tool-draw"), sandboxToolMove: $("sandbox-tool-move"),
  btnClearSandbox: $("btn-clear-sandbox"), sandboxFilename: $("sandbox-filename"),
  alPanels: $("al-sidebar-panels"), sandboxPanels: $("sandbox-sidebar-panels"),
  controlsEmpty: $("controls-empty"), controlsActive: $("controls-active"), classSelect: $("class-select"),
  btnDeleteBox: $("btn-delete-box"), sandboxControlsEmpty: $("sandbox-controls-empty"),
  sandboxControlsActive: $("sandbox-controls-active"), sandboxClassSelect: $("sandbox-class-select"),
  btnDeleteSandboxBox: $("btn-delete-sandbox-box"), alActions: $("al-actions"),
  sandboxActions: $("sandbox-actions"), btnSave: $("btn-save"), btnSaveCorrections: $("btn-save-corrections"),
  btnUpload: $("btn-upload"), fileUpload: $("file-upload"), dropzone: $("dropzone"),
  inferenceSpinner: $("inference-spinner"), confSlider: $("conf-slider"), confValue: $("conf-value"),
  sandboxResultsEmpty: $("sandbox-results-empty"), sandboxResultsList: $("sandbox-results-list"),
  sandboxDetCount: $("sandbox-det-count"), sandboxVisCount: $("sandbox-vis-count"),
  sandboxBreakdown: $("sandbox-breakdown")
};

const ctxImg = els.imgCanvas.getContext("2d");
const ctxOverlay = els.overlayCanvas.getContext("2d");

let taxonomy = [];
let appMode = "al";
let alToolMode = "draw";
let sandboxToolMode = "draw";
let currentTile = null;
let alBoxes = [];
let sandboxBoxes = [];
let selectedAlId = null;
let selectedSandboxId = null;
let nextBoxId = 1;
let predictionCount = 0;
let confidenceThreshold = 0.25;
let sandboxFilenameOnServer = null;
let sandboxImageLoaded = false;
let maxUploadMb = 25;

const alImage = new Image();
const sandboxImage = new Image();
let alImageReady = false;
let sandboxImageReady = false;

let isDrawing = false;
let isDragging = false;
let pointerId = null;
let drawStart = { x: 0, y: 0 };
let drawCurrent = { x: 0, y: 0 };
let dragPrevious = { x: 0, y: 0 };

function setStatus(message, tone = "neutral") {
  els.tileStatus.textContent = message;
  const colors = {
    neutral: "var(--muted)",
    primary: "var(--primary)",
    success: "var(--success)",
    danger: "var(--danger)",
    warning: "var(--warning)"
  };
  els.tileStatus.style.color = colors[tone] || colors.neutral;
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      data.detail || data.message || `Request failed (${response.status})`
    );
  }
  return data;
}

function populateSelect(select) {
  select.replaceChildren();
  taxonomy.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });
}

function currentImage() {
  return appMode === "al"
    ? (alImageReady ? alImage : null)
    : (sandboxImageReady ? sandboxImage : null);
}

function currentBoxes() {
  return appMode === "al" ? alBoxes : sandboxBoxes;
}

function visibleBoxes() {
  if (appMode === "al") return alBoxes;
  return visibleSandboxBoxes();
}

function currentSelectedId() {
  return appMode === "al" ? selectedAlId : selectedSandboxId;
}

function setSelectedId(id) {
  if (appMode === "al") selectedAlId = id;
  else selectedSandboxId = id;
  currentBoxes().forEach((box) => {
    box.selected = box.id === id;
  });
  updatePanels();
  redrawOverlay();
}

function setCanvasForImage(image) {
  if (!image) {
    ctxImg.clearRect(
      0,
      0,
      els.imgCanvas.width,
      els.imgCanvas.height
    );
    ctxOverlay.clearRect(
      0,
      0,
      els.overlayCanvas.width,
      els.overlayCanvas.height
    );
    return;
  }
  els.imgCanvas.width = image.naturalWidth || image.width;
  els.imgCanvas.height = image.naturalHeight || image.height;
  els.overlayCanvas.width = els.imgCanvas.width;
  els.overlayCanvas.height = els.imgCanvas.height;
  ctxImg.clearRect(
    0,
    0,
    els.imgCanvas.width,
    els.imgCanvas.height
  );
  ctxImg.drawImage(image, 0, 0);
  redrawOverlay();
}

function renderCurrentImage() {
  setCanvasForImage(currentImage());
}

function loadImage(image, src) {
  return new Promise((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(
      new Error("Image could not be loaded.")
    );
    image.src = src;
  });
}

async function fetchConfig() {
  const data = await fetchJSON("/api/config");
  taxonomy = Array.isArray(data.taxonomy)
    ? data.taxonomy
    : [];
  maxUploadMb = Number(
    data.max_upload_mb || 25
  );
  populateSelect(els.classSelect);
  populateSelect(els.sandboxClassSelect);
}

async function fetchNextTile() {
  setStatus(
    "Fetching next tile…",
    "primary"
  );
  els.btnSave.disabled = true;
  alBoxes = [];
  selectedAlId = null;
  updatePanels();

  try {
    const data = await fetchJSON(
      "/api/tile/next"
    );
    if (data.status === "empty") {
      currentTile = null;
      alImageReady = false;
      els.statId.textContent = "—";
      setStatus(
        "Review queue empty",
        "success"
      );
      if (appMode === "al") {
        renderCurrentImage();
      }
      return;
    }

    currentTile = data.tile;
    els.statId.textContent = String(
      currentTile.id
    );
    alBoxes = (data.annotations || []).map(
      (ann) => ({
        id: nextBoxId++,
        class_label: ann.class_label,
        x_c: ann.x_center,
        y_c: ann.y_center,
        w: ann.width,
        h: ann.height,
        confidence: ann.score,
        selected: false,
        source: "model"
      })
    );

    await loadImage(
      alImage,
      currentTile.path
    );
    alImageReady = true;
    setStatus(
      `Reviewing tile #${currentTile.id}`
    );
    els.btnSave.disabled = false;
    updatePanels();
    if (appMode === "al") {
      renderCurrentImage();
    }
  } catch (error) {
    currentTile = null;
    alImageReady = false;
    setStatus(
      error.message,
      "danger"
    );
    showToast(error.message);
  }
}

function switchMode(mode) {
  appMode = mode;
  const isAl = mode === "al";

  els.modeAl.classList.toggle(
    "active",
    isAl
  );
  els.modeSandbox.classList.toggle(
    "active",
    !isAl
  );
  els.modeAl.setAttribute(
    "aria-pressed",
    String(isAl)
  );
  els.modeSandbox.setAttribute(
    "aria-pressed",
    String(!isAl)
  );
  els.alToolbar.classList.toggle(
    "hidden",
    !isAl
  );
  els.sandboxToolbar.classList.toggle(
    "hidden",
    isAl
  );
  els.alPanels.classList.toggle(
    "hidden",
    !isAl
  );
  els.sandboxPanels.classList.toggle(
    "hidden",
    isAl
  );
  els.alActions.classList.toggle(
    "hidden",
    !isAl
  );
  els.sandboxActions.classList.toggle(
    "hidden",
    isAl
  );
  els.dropzone.classList.toggle(
    "hidden",
    isAl || sandboxImageLoaded
  );

  setStatus(
    isAl
      ? (
        currentTile
          ? `Reviewing tile #${currentTile.id}`
          : "Review queue empty"
      )
      : "Sandbox — upload or review an image",
    isAl ? "neutral" : "warning"
  );
  updateToolButtons();
  updatePanels();
  renderCurrentImage();
}

function updateToolButtons() {
  els.toolDraw.classList.toggle(
    "active",
    alToolMode === "draw"
  );
  els.toolMove.classList.toggle(
    "active",
    alToolMode === "move"
  );
  els.sandboxToolDraw.classList.toggle(
    "active",
    sandboxToolMode === "draw"
  );
  els.sandboxToolMove.classList.toggle(
    "active",
    sandboxToolMode === "move"
  );
  els.overlayCanvas.style.cursor = (
    appMode === "al"
      ? alToolMode
      : sandboxToolMode
  ) === "draw"
    ? "crosshair"
    : "default";
}

function setTool(mode) {
  if (appMode === "al") {
    alToolMode = mode;
    if (mode === "draw") {
      selectedAlId = null;
    }
  } else {
    sandboxToolMode = mode;
    if (mode === "draw") {
      selectedSandboxId = null;
    }
  }

  currentBoxes().forEach((box) => {
    box.selected =
      box.id === currentSelectedId();
  });
  updateToolButtons();
  updatePanels();
  redrawOverlay();
}

function visibleSandboxBoxes() {
  return sandboxBoxes.filter(
    (box) => (
      box.source !== "prediction"
      || box.confidence >= confidenceThreshold
    )
  );
}

function updatePanels() {
  els.statBoxes.textContent = String(
    alBoxes.length
  );
  const alSelected = alBoxes.find(
    (box) => box.id === selectedAlId
  );
  els.controlsEmpty.classList.toggle(
    "hidden",
    Boolean(alSelected)
  );
  els.controlsActive.classList.toggle(
    "hidden",
    !alSelected
  );
  if (alSelected) {
    els.classSelect.value =
      alSelected.class_label;
  }

  const sandboxVisible =
    visibleSandboxBoxes();
  if (
    selectedSandboxId !== null
    && !sandboxVisible.some(
      (box) => box.id === selectedSandboxId
    )
  ) {
    selectedSandboxId = null;
    sandboxBoxes.forEach((box) => {
      box.selected = false;
    });
  }

  const sbSelected = sandboxBoxes.find(
    (box) => box.id === selectedSandboxId
  );
  els.sandboxControlsEmpty.classList.toggle(
    "hidden",
    Boolean(sbSelected)
  );
  els.sandboxControlsActive.classList.toggle(
    "hidden",
    !sbSelected
  );
  if (sbSelected) {
    els.sandboxClassSelect.value =
      sbSelected.class_label;
  }
  updateSandboxStats();
}

function updateSandboxStats() {
  const visible = visibleSandboxBoxes();
  els.sandboxDetCount.textContent = String(
    predictionCount
  );
  els.sandboxVisCount.textContent = String(
    visible.length
  );

  const hasResults = sandboxImageLoaded;
  els.sandboxResultsEmpty.classList.toggle(
    "hidden",
    hasResults
  );
  els.sandboxResultsList.classList.toggle(
    "hidden",
    !hasResults
  );
  els.sandboxBreakdown.replaceChildren();

  const counts = new Map();
  visible.forEach((box) => {
    counts.set(
      box.class_label,
      (counts.get(box.class_label) || 0) + 1
    );
  });

  [...counts.entries()].forEach(
    ([className, count], index) => {
      const row =
        document.createElement("div");
      row.className = "det-class-row";

      const swatch =
        document.createElement("span");
      swatch.className =
        "det-class-swatch";
      swatch.style.backgroundColor =
        CLASS_COLORS[
          index % CLASS_COLORS.length
        ];

      const name =
        document.createElement("span");
      name.className = "det-class-name";
      name.textContent = className;

      const number =
        document.createElement("span");
      number.className =
        "det-class-count";
      number.textContent = String(count);

      row.append(
        swatch,
        name,
        number
      );
      els.sandboxBreakdown.appendChild(
        row
      );
    }
  );
}

function colorForBox(box, boxes) {
  if (box.selected) {
    return "#f59e0b";
  }
  if (appMode === "sandbox") {
    const classes = [
      ...new Set(
        boxes.map(
          (item) => item.class_label
        )
      )
    ];
    return CLASS_COLORS[
      Math.max(
        0,
        classes.indexOf(
          box.class_label
        )
      ) % CLASS_COLORS.length
    ];
  }
  return "#22c55e";
}

function redrawOverlay() {
  ctxOverlay.clearRect(
    0,
    0,
    els.overlayCanvas.width,
    els.overlayCanvas.height
  );
  if (!currentImage()) {
    return;
  }

  const boxes = visibleBoxes();
  const cw = els.overlayCanvas.width;
  const ch = els.overlayCanvas.height;

  boxes.forEach((box) => {
    const bw = box.w * cw;
    const bh = box.h * ch;
    const bx = box.x_c * cw - bw / 2;
    const by = box.y_c * ch - bh / 2;
    const color = colorForBox(
      box,
      boxes
    );

    ctxOverlay.strokeStyle = color;
    ctxOverlay.lineWidth =
      box.selected ? 3 : 2;
    ctxOverlay.strokeRect(
      bx,
      by,
      bw,
      bh
    );

    let label = box.class_label;
    if (
      box.source === "prediction"
    ) {
      label += ` ${Math.round(
        box.confidence * 100
      )}%`;
    }
    if (box.source === "manual") {
      label += " *";
    }

    ctxOverlay.font =
      "600 13px system-ui";
    const tagW =
      ctxOverlay.measureText(label).width
      + 10;
    const tagH = 21;
    const tagY = Math.max(
      0,
      by - tagH
    );
    ctxOverlay.fillStyle = color;
    ctxOverlay.fillRect(
      bx,
      tagY,
      tagW,
      tagH
    );
    ctxOverlay.fillStyle = "#fff";
    ctxOverlay.fillText(
      label,
      bx + 5,
      tagY + 15
    );
  });
}

function canvasCoords(event) {
  const rect =
    els.overlayCanvas.getBoundingClientRect();
  return {
    x: (
      event.clientX - rect.left
    ) * (
      els.overlayCanvas.width
      / rect.width
    ),
    y: (
      event.clientY - rect.top
    ) * (
      els.overlayCanvas.height
      / rect.height
    )
  };
}

function boxAt(x, y) {
  const boxes = visibleBoxes();
  for (
    let i = boxes.length - 1;
    i >= 0;
    i -= 1
  ) {
    const box = boxes[i];
    const bw =
      box.w * els.overlayCanvas.width;
    const bh =
      box.h * els.overlayCanvas.height;
    const bx =
      box.x_c * els.overlayCanvas.width
      - bw / 2;
    const by =
      box.y_c * els.overlayCanvas.height
      - bh / 2;

    if (
      x >= bx
      && x <= bx + bw
      && y >= by
      && y <= by + bh
    ) {
      return box;
    }
  }
  return null;
}

function activeTool() {
  return appMode === "al"
    ? alToolMode
    : sandboxToolMode;
}

function interactionAllowed() {
  return appMode === "al"
    ? Boolean(
      currentTile && alImageReady
    )
    : sandboxImageLoaded;
}

function onPointerDown(event) {
  if (!interactionAllowed()) {
    return;
  }

  const point = canvasCoords(event);
  pointerId = event.pointerId;
  els.overlayCanvas.setPointerCapture?.(
    event.pointerId
  );

  if (activeTool() === "draw") {
    isDrawing = true;
    drawStart = point;
    drawCurrent = point;
  } else {
    const hit = boxAt(
      point.x,
      point.y
    );
    setSelectedId(
      hit ? hit.id : null
    );
    if (hit) {
      isDragging = true;
      dragPrevious = point;
    }
  }
}

function onPointerMove(event) {
  if (!interactionAllowed()) {
    return;
  }

  const point = canvasCoords(event);
  if (isDrawing) {
    drawCurrent = point;
    redrawOverlay();
    ctxOverlay.setLineDash([5, 5]);
    ctxOverlay.strokeStyle = "#fff";
    ctxOverlay.lineWidth = 2;
    ctxOverlay.strokeRect(
      drawStart.x,
      drawStart.y,
      point.x - drawStart.x,
      point.y - drawStart.y
    );
    ctxOverlay.setLineDash([]);
  } else if (
    isDragging
    && currentSelectedId() !== null
  ) {
    const box = currentBoxes().find(
      (item) => (
        item.id === currentSelectedId()
      )
    );
    if (box) {
      const dx = (
        point.x - dragPrevious.x
      ) / els.overlayCanvas.width;
      const dy = (
        point.y - dragPrevious.y
      ) / els.overlayCanvas.height;

      box.x_c = Math.max(
        box.w / 2,
        Math.min(
          1 - box.w / 2,
          box.x_c + dx
        )
      );
      box.y_c = Math.max(
        box.h / 2,
        Math.min(
          1 - box.h / 2,
          box.y_c + dy
        )
      );
    }
    dragPrevious = point;
    redrawOverlay();
  }
}

function finishPointer(event) {
  if (
    pointerId !== null
    && event.pointerId !== pointerId
  ) {
    return;
  }

  if (isDrawing) {
    const width = Math.abs(
      drawCurrent.x - drawStart.x
    );
    const height = Math.abs(
      drawCurrent.y - drawStart.y
    );

    if (
      width > 5
      && height > 5
    ) {
      const minX = Math.min(
        drawStart.x,
        drawCurrent.x
      );
      const minY = Math.min(
        drawStart.y,
        drawCurrent.y
      );
      const w =
        width / els.overlayCanvas.width;
      const h =
        height / els.overlayCanvas.height;

      const box = {
        id: nextBoxId++,
        class_label:
          taxonomy[0] || "Benign",
        x_c:
          minX
          / els.overlayCanvas.width
          + w / 2,
        y_c:
          minY
          / els.overlayCanvas.height
          + h / 2,
        w,
        h,
        selected: false,
        source: "manual"
      };

      currentBoxes().push(box);
      if (
        appMode === "sandbox"
      ) {
        sandboxToolMode = "move";
        selectedSandboxId = box.id;
        sandboxBoxes.forEach(
          (item) => {
            item.selected =
              item.id === box.id;
          }
        );
      }
      updateToolButtons();
      updatePanels();
    }
  }

  isDrawing = false;
  isDragging = false;
  pointerId = null;
  redrawOverlay();
}

function resetSandbox() {
  sandboxBoxes = [];
  predictionCount = 0;
  selectedSandboxId = null;
  sandboxFilenameOnServer = null;
  sandboxImageLoaded = false;
  sandboxImageReady = false;
  sandboxImage.removeAttribute("src");
  els.sandboxFilename.textContent =
    "No image loaded";
  els.btnSaveCorrections.disabled =
    true;
  els.dropzone.classList.remove(
    "hidden"
  );
  setStatus(
    "Sandbox — upload an image",
    "warning"
  );
  updatePanels();
  renderCurrentImage();
}

async function handleSandboxFile(file) {
  if (!file) {
    return;
  }

  const validTypes = [
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp"
  ];
  if (!validTypes.includes(file.type)) {
    showToast(
      "Unsupported image type."
    );
    return;
  }
  if (
    file.size
    > maxUploadMb * 1024 * 1024
  ) {
    showToast(
      `Image exceeds ${maxUploadMb} MB.`
    );
    return;
  }

  sandboxBoxes = [];
  selectedSandboxId = null;
  predictionCount = 0;
  sandboxFilenameOnServer = null;
  els.btnSaveCorrections.disabled =
    true;
  els.dropzone.classList.add(
    "hidden"
  );
  els.inferenceSpinner.classList.remove(
    "hidden"
  );
  els.sandboxFilename.textContent =
    file.name;
  setStatus(
    "Running inference…",
    "primary"
  );

  const localUrl =
    URL.createObjectURL(file);

  try {
    await loadImage(
      sandboxImage,
      localUrl
    );
    sandboxImageReady = true;
    renderCurrentImage();

    const form = new FormData();
    form.append("file", file);
    const data = await fetchJSON(
      "/api/test/upload",
      {
        method: "POST",
        body: form
      }
    );

    sandboxFilenameOnServer =
      data.sandbox_filename;
    sandboxBoxes = (
      data.predictions || []
    ).map((prediction) => ({
      id: nextBoxId++,
      class_label:
        prediction.class_label,
      x_c: prediction.x_center,
      y_c: prediction.y_center,
      w: prediction.width,
      h: prediction.height,
      confidence:
        prediction.confidence,
      selected: false,
      source: "prediction"
    }));
    predictionCount =
      sandboxBoxes.length;
    sandboxImageLoaded = true;
    els.btnSaveCorrections.disabled =
      false;
    setStatus(
      `Sandbox — ${predictionCount} model detection(s)`,
      "success"
    );
    updatePanels();
    redrawOverlay();
  } catch (error) {
    sandboxImageLoaded = false;
    sandboxFilenameOnServer = null;
    els.dropzone.classList.remove(
      "hidden"
    );
    setStatus(
      error.message,
      "danger"
    );
    showToast(error.message);
  } finally {
    URL.revokeObjectURL(localUrl);
    els.inferenceSpinner.classList.add(
      "hidden"
    );
  }
}

async function saveSandbox() {
  if (!sandboxFilenameOnServer) {
    return;
  }

  const visible =
    visibleSandboxBoxes();
  els.btnSaveCorrections.disabled =
    true;
  els.btnSaveCorrections.textContent =
    "Saving…";

  try {
    const data = await fetchJSON(
      "/api/test/save",
      {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json"
        },
        body: JSON.stringify({
          sandbox_filename:
            sandboxFilenameOnServer,
          annotations: visible.map(
            (box) => ({
              class_label:
                box.class_label,
              x_center: box.x_c,
              y_center: box.y_c,
              width: box.w,
              height: box.h,
              confidence: 1.0
            })
          )
        })
      }
    );

    showToast(
      data.message
      || "Saved for the next training cycle."
    );
    sandboxFilenameOnServer = null;
    els.btnSaveCorrections.textContent =
      "Saved";
    setStatus(
      "Saved for next training cycle",
      "success"
    );
  } catch (error) {
    els.btnSaveCorrections.disabled =
      false;
    setStatus(
      error.message,
      "danger"
    );
    showToast(error.message);
  } finally {
    window.setTimeout(
      () => {
        els.btnSaveCorrections.textContent =
          "Save for next training";
      },
      1200
    );
  }
}

async function saveAndNext() {
  if (!currentTile) {
    return;
  }

  els.btnSave.disabled = true;
  els.btnSave.textContent =
    "Saving…";
  try {
    await fetchJSON(
      `/api/tile/${currentTile.id}/save`,
      {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json"
        },
        body: JSON.stringify({
          annotations: alBoxes.map(
            (box) => ({
              class_label:
                box.class_label,
              x_center: box.x_c,
              y_center: box.y_c,
              width: box.w,
              height: box.h,
              confidence: 1.0
            })
          )
        })
      }
    );
    await fetchNextTile();
  } catch (error) {
    els.btnSave.disabled = false;
    setStatus(
      error.message,
      "danger"
    );
    showToast(error.message);
  } finally {
    els.btnSave.textContent =
      "Save & next";
  }
}

function showToast(message) {
  document.querySelectorAll(
    ".toast"
  ).forEach(
    (node) => node.remove()
  );

  const toast =
    document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  toast.setAttribute(
    "role",
    "status"
  );
  document.body.appendChild(toast);
  requestAnimationFrame(
    () => toast.classList.add("show")
  );
  window.setTimeout(
    () => {
      toast.classList.remove("show");
      window.setTimeout(
        () => toast.remove(),
        250
      );
    },
    3000
  );
}

function setupTheme() {
  const saved =
    localStorage.getItem(
      "blast-theme"
    );
  const preferred = saved || (
    window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches
      ? "dark"
      : "light"
  );
  document.documentElement.dataset.theme =
    preferred;
  updateThemeButton();
}

function updateThemeButton() {
  const dark =
    document.documentElement.dataset.theme
    === "dark";
  els.themeToggle.textContent =
    dark ? "☀" : "◐";
  els.themeToggle.setAttribute(
    "aria-label",
    dark
      ? "Switch to light theme"
      : "Switch to dark theme"
  );
}

function toggleTheme() {
  const next =
    document.documentElement.dataset.theme
    === "dark"
      ? "light"
      : "dark";
  document.documentElement.dataset.theme =
    next;
  localStorage.setItem(
    "blast-theme",
    next
  );
  updateThemeButton();
}

function setupEvents() {
  els.modeAl.addEventListener(
    "click",
    () => switchMode("al")
  );
  els.modeSandbox.addEventListener(
    "click",
    () => switchMode("sandbox")
  );
  els.themeToggle.addEventListener(
    "click",
    toggleTheme
  );

  els.toolDraw.addEventListener(
    "click",
    () => {
      appMode = "al";
      setTool("draw");
    }
  );
  els.toolMove.addEventListener(
    "click",
    () => {
      appMode = "al";
      setTool("move");
    }
  );
  els.sandboxToolDraw.addEventListener(
    "click",
    () => {
      appMode = "sandbox";
      setTool("draw");
    }
  );
  els.sandboxToolMove.addEventListener(
    "click",
    () => {
      appMode = "sandbox";
      setTool("move");
    }
  );

  els.btnClear.addEventListener(
    "click",
    () => {
      alBoxes = [];
      selectedAlId = null;
      updatePanels();
      redrawOverlay();
    }
  );
  els.btnClearSandbox.addEventListener(
    "click",
    resetSandbox
  );

  els.btnDeleteBox.addEventListener(
    "click",
    () => {
      alBoxes = alBoxes.filter(
        (box) => (
          box.id !== selectedAlId
        )
      );
      selectedAlId = null;
      updatePanels();
      redrawOverlay();
    }
  );
  els.btnDeleteSandboxBox.addEventListener(
    "click",
    () => {
      sandboxBoxes =
        sandboxBoxes.filter(
          (box) => (
            box.id !== selectedSandboxId
          )
        );
      selectedSandboxId = null;
      updatePanels();
      redrawOverlay();
    }
  );

  els.classSelect.addEventListener(
    "change",
    () => {
      const box = alBoxes.find(
        (item) => (
          item.id === selectedAlId
        )
      );
      if (box) {
        box.class_label =
          els.classSelect.value;
      }
      redrawOverlay();
    }
  );
  els.sandboxClassSelect.addEventListener(
    "change",
    () => {
      const box = sandboxBoxes.find(
        (item) => (
          item.id === selectedSandboxId
        )
      );
      if (box) {
        box.class_label =
          els.sandboxClassSelect.value;
      }
      updatePanels();
      redrawOverlay();
    }
  );

  els.confSlider.addEventListener(
    "input",
    () => {
      confidenceThreshold =
        Number(
          els.confSlider.value
        ) / 100;
      els.confValue.textContent =
        confidenceThreshold.toFixed(2);
      updatePanels();
      redrawOverlay();
    }
  );

  els.btnSave.addEventListener(
    "click",
    saveAndNext
  );
  els.btnSaveCorrections.addEventListener(
    "click",
    saveSandbox
  );
  els.btnUpload.addEventListener(
    "click",
    () => els.fileUpload.click()
  );
  els.dropzone.addEventListener(
    "click",
    () => els.fileUpload.click()
  );
  els.fileUpload.addEventListener(
    "change",
    () => {
      handleSandboxFile(
        els.fileUpload.files?.[0]
      );
      els.fileUpload.value = "";
    }
  );

  els.dropzone.addEventListener(
    "dragover",
    (event) => {
      event.preventDefault();
      els.dropzone.classList.add(
        "drag-over"
      );
    }
  );
  els.dropzone.addEventListener(
    "dragleave",
    () => {
      els.dropzone.classList.remove(
        "drag-over"
      );
    }
  );
  els.dropzone.addEventListener(
    "drop",
    (event) => {
      event.preventDefault();
      els.dropzone.classList.remove(
        "drag-over"
      );
      handleSandboxFile(
        event.dataTransfer
          ?.files?.[0]
      );
    }
  );

  els.overlayCanvas.addEventListener(
    "pointerdown",
    onPointerDown
  );
  els.overlayCanvas.addEventListener(
    "pointermove",
    onPointerMove
  );
  els.overlayCanvas.addEventListener(
    "pointerup",
    finishPointer
  );
  els.overlayCanvas.addEventListener(
    "pointercancel",
    finishPointer
  );
}

async function init() {
  setupTheme();
  setupEvents();
  updateToolButtons();

  try {
    await fetchConfig();
    await fetchNextTile();
  } catch (error) {
    setStatus(
      error.message,
      "danger"
    );
    showToast(error.message);
  }
}

document.addEventListener(
  "DOMContentLoaded",
  init
);
