# ST Final — LSTM next-step prediction (TF.js)

Local static demo that generates a synthetic dataset of noisy sine-wave sequences and trains a single-layer LSTM to predict the next step.

How to run

- Open `index.html` in a browser (or serve the folder with a static server, e.g. `python -m http.server`).

Node (recommended): install dependencies and start the small Express server included in this repo:

```bash
npm install
npm start
```

Then open `http://localhost:8000` in your browser.

Python (quick alternative):

```bash
python -m http.server 8000
# then open http://localhost:8000
```

Build (bundle for distribution):

```bash
# install dev deps
npm install
# produce a bundled/minified build into `dist/`
npm run build
# serve the `dist` folder (example using Python)
python -m http.server 8000 --directory dist
# then open http://localhost:8000
```

Features

- Synthetic dataset: samples are sequences of length `n` from noisy sine waves (random amplitude, frequency, phase). Target is next point.
- Download dataset as JSON.
- Interactive plotting with Plotly: hover values, compare multiple samples (overlap option).
- Model: single LSTM layer -> dense(1). Loss: MSE. Optimizer: Adam.
- Train in-browser with TF.js; shows training loss and predictions vs targets for validation samples.

Notes

- The app uses `seedrandom` for repeatable dataset generation when a seed is provided.
- LSTM internals are summarized in the UI.
# Final-Project