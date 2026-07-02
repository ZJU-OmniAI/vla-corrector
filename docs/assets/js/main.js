const root = document.documentElement;
const themeButton = document.querySelector("[data-theme-toggle]");
const storedTheme = window.localStorage.getItem("vla-corrector-theme");
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

root.dataset.theme = storedTheme || (prefersDark ? "dark" : "light");

themeButton?.addEventListener("click", () => {
  const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = nextTheme;
  window.localStorage.setItem("vla-corrector-theme", nextTheme);
});

document.querySelectorAll("pre[data-copy]").forEach((pre) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-button";
  button.textContent = "Copy";
  pre.parentElement?.appendChild(button);

  button.addEventListener("click", async () => {
    const code = pre.innerText.trim();
    try {
      await navigator.clipboard.writeText(code);
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1400);
    } catch {
      button.textContent = "Select";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1400);
    }
  });
});

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", (event) => {
    const target = document.querySelector(anchor.getAttribute("href"));
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});
