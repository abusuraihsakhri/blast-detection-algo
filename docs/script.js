/**
 * Blast Detection Research Portal Interactions
 * Handles theme switching, tabbed visualizer, batch inspection, code tabs, and lightbox zoom.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================
  // 1. Theme Management (Dark / Light)
  // ==========================================
  const root = document.documentElement;
  const themeBtn = document.getElementById("theme");
  const savedTheme = localStorage.getItem("blast_theme");

  const setTheme = (theme) => {
    root.dataset.theme = theme;
    localStorage.setItem("blast_theme", theme);
  };

  if (savedTheme) {
    setTheme(savedTheme);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    setTheme("dark");
  } else {
    setTheme("light");
  }

  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
      setTheme(nextTheme);
    });
  }

  // ==========================================
  // 2. Interactive Visualizer Tabs
  // ==========================================
  const tabBtns = document.querySelectorAll(".tabs-nav .tab-btn");
  const tabPanes = document.querySelectorAll(".tabs-wrapper .tab-pane");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      if (!targetTab) return;

      // Update button active state
      tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      // Update pane visibility
      tabPanes.forEach((pane) => {
        if (pane.id === `tab-${targetTab}`) {
          pane.classList.add("active");
        } else {
          pane.classList.remove("active");
        }
      });
    });
  });

  // ==========================================
  // 3. Validation Batch Selector (Side-by-Side)
  // ==========================================
  const batchBtns = document.querySelectorAll(".batch-selector .batch-btn");
  const batchLabelsImg = document.getElementById("batch-labels-img");
  const batchPredsImg = document.getElementById("batch-preds-img");
  const batchLabelsTitle = document.getElementById("batch-labels-title");
  const batchPredsTitle = document.getElementById("batch-preds-title");

  const batchDescriptions = {
    "0": "Batch 0 (Heterogeneous Fields)",
    "1": "Batch 1 (Dense High-Cellularity)",
    "2": "Batch 2 (Complex Clustered Fields)"
  };

  batchBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const batchId = btn.getAttribute("data-batch");
      if (!batchId) return;

      batchBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      if (batchLabelsImg) {
        batchLabelsImg.src = `assets/val_batch${batchId}_labels.jpg`;
      }
      if (batchPredsImg) {
        batchPredsImg.src = `assets/val_batch${batchId}_pred.jpg`;
      }
      if (batchLabelsTitle) {
        batchLabelsTitle.textContent = `Validation Batch ${batchId} — Annotated Ground Truth`;
      }
      if (batchPredsTitle) {
        batchPredsTitle.textContent = `Validation Batch ${batchId} — YOLOv8n Predictions`;
      }
    });
  });

  // ==========================================
  // 4. Code Snippet Tabs
  // ==========================================
  const codeTabBtns = document.querySelectorAll(".code-tabs .code-tab-btn");
  const codePanes = document.querySelectorAll(".code-body .code-pane");

  codeTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetCode = btn.getAttribute("data-code");
      if (!targetCode) return;

      codeTabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      codePanes.forEach((pane) => {
        if (pane.id === `code-${targetCode}`) {
          pane.classList.add("active");
        } else {
          pane.classList.remove("active");
        }
      });
    });
  });

  // ==========================================
  // 5. Code Copy to Clipboard
  // ==========================================
  const copyBtn = document.getElementById("copy-btn");
  const copyText = document.getElementById("copy-text");

  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const activePane = document.querySelector(".code-body .code-pane.active pre code");
      if (!activePane) return;

      const codeString = activePane.innerText || activePane.textContent;
      try {
        await navigator.clipboard.writeText(codeString);
        if (copyText) copyText.textContent = "Copied!";
        copyBtn.style.borderColor = "var(--primary-500)";
        setTimeout(() => {
          if (copyText) copyText.textContent = "Copy Code";
          copyBtn.style.borderColor = "";
        }, 2000);
      } catch (err) {
        console.error("Clipboard copy failed:", err);
      }
    });
  }

  // ==========================================
  // 6. Lightbox Modal Zoom
  // ==========================================
  const lightbox = document.getElementById("lightbox-modal");
  const lightboxImg = document.getElementById("lightbox-img");

  const openLightbox = (src) => {
    if (lightbox && lightboxImg) {
      lightboxImg.src = src;
      lightbox.classList.add("active");
      document.body.style.overflow = "hidden";
    }
  };

  const closeLightbox = () => {
    if (lightbox) {
      lightbox.classList.remove("active");
      document.body.style.overflow = "";
    }
  };

  document.querySelectorAll(".figure-card img").forEach((img) => {
    img.addEventListener("click", () => {
      openLightbox(img.src);
    });
  });

  if (lightbox) {
    lightbox.addEventListener("click", (e) => {
      // Close if clicked on backdrop or image
      closeLightbox();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox && lightbox.classList.contains("active")) {
      closeLightbox();
    }
  });
});
