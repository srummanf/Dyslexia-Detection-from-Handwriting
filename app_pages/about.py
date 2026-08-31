"""Methodology, limitations and setup notes."""

from __future__ import annotations

import streamlit as st

from dyslexia.app_support import DISCLAIMER, get_screener

st.title("📖 About & methodology")

st.warning(DISCLAIMER)

st.markdown(
    """
### What this app does

It blends up to three independent signals into a single risk score:

| Signal | Model | Input |
|---|---|---|
| **Linguistic** | Calibrated gradient-boosting / random-forest classifier | 4 features derived from OCR'd text: spelling accuracy, grammatical accuracy, % of corrections needed, phonetic accuracy |
| **Whole-sample** | YOLOv8n-cls (transfer-learned) | The full handwriting image |
| **Per-letter** *(optional)* | ConvNeXt-Tiny on the Gambo *Dyslexia Handwriting Dataset* | Individual letter crops → Normal / Reversal / Corrected |

Weights are set in `config.yaml`. If a model or its dependencies are missing,
that signal is dropped and the remaining weights are renormalised.

### How the linguistic features are computed

Everything runs **offline**:

- **OCR** — `easyocr` (was: Azure Computer Vision Read API with committed keys)
- **Spelling / corrections** — `pyspellchecker` (was: Bing Spell-Check API)
- **Phonetics** — `jellyfish` Soundex / Metaphone / NYSIIS (was: `abydos`)
- **Grammar** — `language_tool_python` if a JRE is present, else a lightweight heuristic

Because the extractors changed, feature values differ slightly from the 2024
CSV. `notebooks/dyslexia_modeling.ipynb` can regenerate the dataset with this
exact pipeline and retrain.

### Limitations

- Training data is small (~100 samples for the linguistic model, ~200 images
  for the whole-sample CNN) and not demographically representative.
- The linguistic features are proxies; OCR errors on messy handwriting feed
  directly into them.
- A high score means "worth a proper assessment", nothing more.
    """
)

st.subheader("Current pipeline status")
st.json(get_screener().component_status)

st.subheader("Setup")
st.code(
    "pip install -r requirements.txt\n"
    "python scripts/train_tabular.py         # (re)build the tabular model\n"
    "streamlit run streamlit_app.py",
    language="bash",
)
