"""Entry point for the dyslexia-screening Streamlit app.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make ``src/`` importable when run via ``streamlit run`` (no install needed).
_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="Dyslexia Handwriting Screening",
    page_icon="🖋️",
    layout="wide",
)

pages = [
    st.Page("app_pages/screening.py", title="Screening", icon="🖋️", default=True),
    st.Page("app_pages/dataset.py", title="Feature explorer", icon="📊"),
    st.Page("app_pages/about.py", title="About & methodology", icon="📖"),
]
st.navigation(pages).run()
