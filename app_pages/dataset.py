"""Explore the linguistic-feature dataset the tabular model learns from."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dyslexia.app_support import feature_labels, get_dataset
from dyslexia.features import FEATURE_NAMES

st.title("📊 Linguistic feature explorer")

df = get_dataset().copy()
df["group"] = df["presence_of_dyslexia"].map({1: "dyslexic", 0: "non-dyslexic"})
labels = feature_labels()

st.write(
    f"{len(df)} handwriting samples · "
    f"{int(df.presence_of_dyslexia.sum())} dyslexic / "
    f"{int((1 - df.presence_of_dyslexia).sum())} non-dyslexic"
)

feature = st.selectbox("Feature", list(FEATURE_NAMES), format_func=labels.get)
fig = px.histogram(
    df, x=feature, color="group", barmode="overlay", nbins=30,
    color_discrete_map={"dyslexic": "#c62828", "non-dyslexic": "#2e7d32"},
    labels={feature: labels[feature]},
)
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
x_feat = c1.selectbox("x axis", list(FEATURE_NAMES), index=0, format_func=labels.get)
y_feat = c2.selectbox("y axis", list(FEATURE_NAMES), index=3, format_func=labels.get)
scatter = px.scatter(
    df, x=x_feat, y=y_feat, color="group",
    color_discrete_map={"dyslexic": "#c62828", "non-dyslexic": "#2e7d32"},
    labels={x_feat: labels[x_feat], y_feat: labels[y_feat]},
)
st.plotly_chart(scatter, width="stretch")

with st.expander("Group means"):
    st.dataframe(
        df.groupby("group")[list(FEATURE_NAMES)].mean().round(2).rename(columns=labels),
        width="stretch",
    )

with st.expander("Raw data"):
    st.dataframe(df, width="stretch", hide_index=True)
