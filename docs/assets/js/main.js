const copyButton = document.querySelector("[data-copy-bibtex]");
const bibtex = document.querySelector("#bibtex");
const pageRoot = document.documentElement;
let didMarkReady = false;

function markPageReady() {
  if (didMarkReady) return;
  didMarkReady = true;
  pageRoot.classList.remove("page-loading");
  pageRoot.classList.add("page-ready");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", markPageReady, { once: true });
} else {
  requestAnimationFrame(markPageReady);
}

window.addEventListener("load", markPageReady, { once: true });

copyButton?.addEventListener("click", async () => {
  const copyLabel = copyButton.dataset.copyLabel ?? "Copy BibTeX";
  const copiedLabel = copyButton.dataset.copiedLabel ?? "Copied";
  const selectLabel = copyButton.dataset.selectLabel ?? "Select BibTeX";

  try {
    await navigator.clipboard.writeText(bibtex?.innerText.trim() ?? "");
    copyButton.textContent = copiedLabel;
    window.setTimeout(() => {
      copyButton.textContent = copyLabel;
    }, 1400);
  } catch {
    copyButton.textContent = selectLabel;
    window.setTimeout(() => {
      copyButton.textContent = copyLabel;
    }, 1400);
  }
});

const presentationButton = document.querySelector("#presentBtn");
const presentationEmbed = document.querySelector("#presentationEmbed");
const presentationIframe = document.querySelector("#presentIframe");
const presentationLabel = document.querySelector("[data-presentation-label]");
const presentationClose = document.querySelector("#presentClose");

function scrollToPresentation() {
  const top = (presentationEmbed?.getBoundingClientRect().top ?? 0) + window.scrollY - 8;
  window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

function openPresentation() {
  if (!presentationButton || !presentationEmbed || !presentationIframe) return;
  if (!presentationIframe.getAttribute("src")) {
    presentationIframe.setAttribute("src", presentationIframe.dataset.src ?? "presentation.html");
  }
  presentationEmbed.classList.add("open");
  presentationButton.classList.add("is-open");
  presentationButton.setAttribute("aria-expanded", "true");
  if (presentationLabel) presentationLabel.textContent = "Hide Presentation";
  requestAnimationFrame(scrollToPresentation);
}

function closePresentation(scrollBack = true) {
  if (!presentationButton || !presentationEmbed) return;
  presentationEmbed.classList.remove("open");
  presentationButton.classList.remove("is-open");
  presentationButton.setAttribute("aria-expanded", "false");
  if (presentationLabel) presentationLabel.textContent = "Presentation";
  if (scrollBack) {
    const top = presentationButton.getBoundingClientRect().top + window.scrollY - 120;
    window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }
}

presentationButton?.addEventListener("click", (event) => {
  event.preventDefault();
  if (presentationEmbed?.classList.contains("open")) {
    closePresentation(true);
  } else {
    openPresentation();
  }
});

presentationClose?.addEventListener("click", () => closePresentation(true));
