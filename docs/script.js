document.addEventListener("DOMContentLoaded", () => {
  // Theme Toggle
  const root = document.documentElement;
  const themeBtn = document.getElementById("theme");
  const savedTheme = localStorage.getItem("theme");

  if (savedTheme) {
    root.dataset.theme = savedTheme;
  } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    root.dataset.theme = "dark";
  }

  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = nextTheme;
      localStorage.setItem("theme", nextTheme);
    });
  }

  // Lightbox Modal for Fullscreen Curve/Batch Inspection
  const lightbox = document.createElement("div");
  lightbox.className = "lightbox";
  const lightboxImg = document.createElement("img");
  lightbox.appendChild(lightboxImg);
  document.body.appendChild(lightbox);

  lightbox.addEventListener("click", () => {
    lightbox.classList.remove("active");
  });

  document.querySelectorAll(".gallery-card img, .metric-figure img").forEach((img) => {
    img.addEventListener("click", () => {
      lightboxImg.src = img.src;
      lightbox.classList.add("active");
    });
  });

  // Keyboard escape closes lightbox
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox.classList.contains("active")) {
      lightbox.classList.remove("active");
    }
  });
});
