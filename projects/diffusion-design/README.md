# Classifier-Guided Diffusion for Post-Stroke Drug Design

Browser platform: one guided-diffusion engine, two alphabets (small molecules, SpCas9 guide sites), target PTPσ (PTPRS).
Serve the folder over HTTP and open `index.html` (overview), `studio.html` (the app) or `evidence.html` (results and method). Everything runs client-side.

- `engine.js` / `worker.js` / `app.js`: inference engine (mirror of `scripts/gd.py`), worker, UI.
- `models/`: exported weights. `scripts/`: data prep, training, experiments, export, parity reference.
- `parity.js`: `node parity.js` checks the JS engine against the PyTorch reference (denoiser, critics, gradients, sampling, decoder).
- `scripts/experiments_locus.js`: locus-guidance experiment (node). `results/lib_parity_reference.json`: numpy reference for the locus critic.
- `results/`: saved metrics and experiment outputs shown on `evidence.html` (see `results/READING.html`, `results/METHOD.html`).

Honest limits: small public datasets; guides are not locus-targeted; the PTP-family activity model is a proxy for PTPσ; in silico only.

- `studio.html`, `app.js`, `chats.js`: the studio. Chats are auto-saved in the browser (IndexedDB) and can be downloaded as Markdown or JSON; there is no account or server.
- `site.css`, `site.js`, `evidence.js`, `logo.svg`, `favicon.svg`: shared styles, theme toggle, evidence page, and logo.
