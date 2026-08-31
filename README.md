# Dyslexia Detection from Handwriting

Screening for dyslexia-related handwriting patterns from a photo of a handwriting
sample. Originally an AI course project (2024); rebuilt in 2025 with a cleaner
codebase, offline inference, and current models.

> ⚠️ **Research demo, not a medical device.** A dyslexia diagnosis requires a
> full assessment by a qualified professional. Every output here is a rough,
> exploratory indicator only.

---

## What it does

`DyslexiaScreener` blends up to three independent signals into one risk score:


| Signal                      | Model                                                    | Input                                                                                                        |
| ----------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Linguistic**              | Calibrated gradient-boosting / random-forest classifier  | 4 features from OCR'd text: spelling accuracy, grammatical accuracy, % corrections needed, phonetic accuracy |
| **Whole-sample**            | YOLO11n-cls (transfer-learned)                           | the full handwriting image                                                                                   |
| **Per-letter** *(optional)* | ConvNeXt-Tiny on the Gambo*Dyslexia Handwriting Dataset* | individual letter crops → Normal / Reversal / Corrected                                                     |

Blend weights live in `config.yaml`. If a model or its dependencies are missing,
that signal is dropped and the remaining weights are renormalised — the app
still runs.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                    # exposes the `dyslexia` package

python scripts/train_tabular.py                     # builds models/tabular/model.joblib
streamlit run streamlit_app.py
```

The first screening run downloads the EasyOCR weights (~100 MB), once.

## Project layout

```
config.yaml                 central config (paths, OCR engine, blend weights)
streamlit_app.py            app entry point (st.navigation)
app_pages/                  Screening · Feature explorer · About
notebooks/
  dyslexia_modeling.ipynb   trains & evaluates all three models + the ensemble
scripts/
  train_tabular.py          model selection + persistence for the tabular model
src/dyslexia/
  config.py                 config + path resolution
  ocr.py                    pluggable offline OCR (easyocr / tesseract / null)
  text_metrics.py           levenshtein & similarity helpers
  features.py               image → 4 linguistic features
  tabular.py                candidate pipelines, selection, TabularModel
  image_models.py           YOLO + Gambo wrappers (degrade gracefully)
  screener.py               DyslexiaScreener — blends the signals
  datasets.py               CSV / sample / Gambo loaders
  app_support.py            Streamlit-only helpers
data/
  linguistic_features.csv   ~100 labelled feature rows
  samples/{all,train,test}/ handwriting images (dyslexic / non_dyslexic)
  vocab/                     word lists (legacy pronunciation/dictation tests)
models/
  tabular/model.joblib      committed, ready to use
  image/yolo_samples_*.pt   2024 YOLOv8-cls weights (fallback)
tests/                      pytest suite (no heavy deps required)
legacy/                     the untouched 2024 app.py, app2.py, notebooks, runs
```

## Notebook

`notebooks/dyslexia_modeling.ipynb` is the single source of truth for training:

- **Part A** – EDA, cross-validated model selection, calibration, ROC/PR,
  permutation importance, SHAP, saves `models/tabular/model.joblib`.
- **Part B** – YOLO11n-cls on `data/samples` (run on a GPU / Colab).
- **Part C** – ConvNeXt-Tiny on the Gambo dataset (put `Gambo.zip` at the repo root).
- **Part D** – end-to-end ensemble evaluation.

## Changes from the 2024 version

- **No secrets / no network.** The Azure Computer Vision keys committed in the
  old `app.py` are gone; OCR is now `easyocr` (offline). Spell-check moved from
  the Bing API to `pyspellchecker`, phonetics from `abydos` to `jellyfish`.
- **Real model artefact.** The hand-transcribed decision tree in `app.py` is
  replaced by a persisted, calibrated, cross-validated pipeline.
- **One package, tested.** Logic moved out of the Streamlit script into
  `src/dyslexia/` with a pytest suite.
- **Ensemble.** Linguistic + whole-sample + per-letter signals are combined
  instead of used in isolation.
- The mic-based pronunciation / dictation tabs from the old app were dropped
  (they need local audio hardware and don't run on Streamlit Cloud). The old
  code is preserved under `legacy/`.

## Feature drift note

`data/linguistic_features.csv` was generated in 2024 with the Azure/Bing/abydos
stack. The current extractors produce values on a comparable 0–100 scale but not
identical. The notebook's Part A has an opt-in cell to regenerate the table from
the raw images with the offline pipeline and retrain.
