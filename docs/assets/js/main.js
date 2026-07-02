const copyButton = document.querySelector("[data-copy-bibtex]");
const bibtex = document.querySelector("#bibtex");

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
