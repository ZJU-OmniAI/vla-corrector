const copyButton = document.querySelector("[data-copy-bibtex]");
const bibtex = document.querySelector("#bibtex");

copyButton?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(bibtex?.innerText.trim() ?? "");
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1400);
  } catch {
    copyButton.textContent = "Select BibTeX";
    window.setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1400);
  }
});
