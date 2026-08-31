"""Upload a handwriting sample and run the blended screening pipeline."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from PIL import Image

from dyslexia.app_support import DISCLAIMER, band_for, feature_labels, get_screener

st.title("🖋️ Handwriting screening")
st.caption(DISCLAIMER)

screener = get_screener()
status = screener.component_status

with st.sidebar:
    st.subheader("Pipeline status")
    labels = {
        "features": "Linguistic features (OCR + tabular model)",
        "tabular": "Tabular classifier",
        "yolo": "Whole-sample CNN",
        "gambo": "Per-letter CNN (optional)",
    }
    for key, text in labels.items():
        st.write(("✅ " if status.get(key) else "⚠️ ") + text)
    if not any(status.values()):
        st.error("No models are available. See the About page for setup steps.")

uploaded = st.file_uploader(
    "Upload a handwriting sample (a sentence or short paragraph works best)",
    type=["jpg", "jpeg", "png"],
)
letter_files = st.file_uploader(
    "Optional: individual letter crops for the per-letter model",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

col_img, col_result = st.columns([1, 1.3])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    col_img.image(image, caption=uploaded.name, width="stretch")

    if col_img.button("Run screening", type="primary"):
        crops = [Image.open(io.BytesIO(f.getvalue())).convert("RGB") for f in letter_files]
        with st.spinner("Analysing… first run downloads the OCR model."):
            result = screener.screen(image, letter_crops=crops or None)

        with col_result:
            score = result.risk_score
            name, colour = band_for(score)
            st.markdown(
                f"<h2 style='color:{colour};margin-bottom:0'>{score:.0%}</h2>"
                f"<p style='color:{colour};margin-top:0'><b>{name}</b> · {result.label}</p>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0.0), 1.0))

            if result.signals:
                st.markdown("**Contributing signals**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "signal": list(result.signals),
                            "P(dyslexia)": [round(v, 3) for v in result.signals.values()],
                            "blend weight": [
                                round(result.weights.get(k, 0.0), 3) for k in result.signals
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )

            if result.features is not None:
                st.markdown("**Linguistic features**")
                lbl = feature_labels()
                st.dataframe(
                    pd.DataFrame(
                        [(lbl[k], round(v, 2)) for k, v in result.features.as_dict().items()],
                        columns=["feature", "value"],
                    ),
                    hide_index=True,
                    width="stretch",
                )
                if result.extracted_text:
                    st.markdown("**Text read from the image**")
                    st.info(result.extracted_text)

            for note in result.notes:
                st.caption("ℹ️ " + note)

        st.divider()
        st.caption(DISCLAIMER)
else:
    st.info("Upload an image to begin.")
