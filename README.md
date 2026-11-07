# RAP Superpave Performance Predictor (PI-NGB-TSTR)

A Streamlit web app for the framework in *Next-Generation Physics-Guided AI Framework for Performance
Prediction of RAP Superpave Asphalt Mixtures*. Enter a RAP Superpave mix design and get, for each of the three
performance indicators – dry indirect tensile strength (ITSD), tensile strength ratio (TSR) and rut depth (RD):

* the PI-NGB-TSTR prediction with a split-conformal prediction interval (80, 90 or 95%);
* for TSR, the probability of meeting TSR ≥ 0.80 and a Pass / Fail / *Uncertain – test in laboratory* decision;
* engineering input checks (gradation monotonicity, AV < VMA, VFA consistency) and an applicability-domain warning when an
  input lies outside the experimental database;
* the nine physics-informed descriptors computed from the inputs;
* batch prediction from an Excel/CSV file, with downloadable results.

## Folder contents

| Path | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `rapcore/` | Physics descriptors, input checks, prediction engine (NumPy only), exporter |
| `models/ITSD.npz`, `TSR.npz`, `RD.npz` | The three PI-NGB-TSTR models in portable form |
| `models/meta.json` | Database ranges and categories, example mixtures, measured-value distributions, app texts |
| `examples/batch_template.xlsx` | Input template for batch prediction |
| `Train_and_export_PI-NGB_models.ipynb` | Colab notebook that trains the official models and exports them for the app |
| `index.html`, `.nojekyll` | Browser version of the app for free static hosting (GitHub Pages) |
| `requirements.txt`, `.streamlit/config.toml` | Dependencies and theme for running with Streamlit |

## How this repository builds the app

Every push to `main` runs the GitHub Actions workflow `.github/workflows/build.yml`, which publishes the app on GitHub Pages
(about 1 minute). To republish, push a change or open *Actions → Publish the app → Run workflow*.

The experimental and synthetic datasets are **not** stored in this repository. The app ships only the trained models
(`models/*.npz`: tree structures and calibration constants) and summary information on the database (`models/meta.json`:
input ranges, categories, four example mixtures and histograms of the measured properties).

## Retraining the models (optional, with your local data)

The app needs neither scikit-learn nor ngboost: the tree ensembles are stored as arrays and evaluated with NumPy, so the
models cannot break when library versions change. To retrain and re-export them from the three TSTR data files:

* **On your computer:**
  ```bash
  pip install "scikit-learn==1.6.1" "ngboost==0.5.11" numpy pandas openpyxl
  python tools/train_export.py /path/to/folder/with/the/TSTR/xlsx/files
  ```
* **In Google Colab:** open `Train_and_export_PI-NGB_models.ipynb`, choose *Runtime → Run all* and upload the three data files
  when asked; copy the downloaded `ITSD.npz`, `TSR.npz` and `RD.npz` into `models/`.

Both print the testing/validation R² of each model and check that the exported model reproduces `model.predict` exactly.
Then commit the new `models/*.npz` files; the app is republished automatically.

## Step 2 – publish free and permanently (GitHub Pages, recommended)

The folder includes `index.html`, which runs the whole app **inside the visitor's browser** (Streamlit compiled to
WebAssembly with [stlite](https://github.com/whitphx/stlite)). Because there is no server, nothing can go to sleep, be
paused for inactivity or run out of free hours, and visitors' inputs never leave their computer.

1. Create a free GitHub account and a new **public** repository, e.g. `rap-superpave-predictor`
   (GitHub Pages is free for public repositories).
2. Upload the **contents** of this folder to the repository root (*Add file → Upload files*; drag in everything,
   including the `models`, `rapcore` and `examples` folders and the hidden `.nojekyll` file).
3. Open *Settings → Pages* and set *Source* to **GitHub Actions**; the workflow then publishes the app.
4. After about a minute the app is live at `https://<your-user-name>.github.io/rap-superpave-predictor/`. Quote this link
   in the paper's *Data and code availability* statement. Uploading new files updates the app.

Notes: the first visit takes about 30–60 s while the browser downloads Python (it is cached afterwards); predictions are
instant.
The same static files can also be hosted on a free *Static* Hugging Face Space, Netlify or Cloudflare Pages.

Why not a server host? Free server tiers put apps to sleep or restrict them: Streamlit Community Cloud hibernates apps
without traffic, and since mid-2026 new free Hugging Face accounts can no longer create Streamlit/Gradio/Docker Spaces on
free CPU (static Spaces remain free).

## Step 3 – run on your computer (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Customising

Edit the `app` block of `models/meta.json`: `title`, `subtitle`, `authors`, `affiliation`, `citation` (shown on the
*Model information* tab) and `rct_definitions`, the rutting cycle types shown in the input form
(Type A = 8,000, Type B = 10,000, Type C = 20,000 load cycles).

## Method summary

* **Model**: NGBoost with a Normal (ITSD, TSR) or LogNormal (RD) predictive distribution, trained on 132 experimental and 924
  PI-TVAE synthetic mixtures, with nine physics-informed descriptors. The TSR model does not use aggregate type, RAP content,
  rutting cycle type or design gyrations (feature selection in the published model).
* **Intervals**: split-conformal, calibrated on the 57 held-out experimental mixtures (absolute for ITSD and TSR,
  proportional to the prediction for RD), as in Section 4.5 of the paper.
* **TSR decision**: P(TSR ≥ 0.80) from a Gaussian with the held-out error spread; Pass if P ≥ 0.90, Fail if P ≤ 0.10.

## Limitations

ITSD, TSR and rut depth are laboratory indicators for design screening, not field service-life predictions. Mixtures outside
the database ranges are extrapolations, and mixtures classified as uncertain should be tested in the laboratory.
