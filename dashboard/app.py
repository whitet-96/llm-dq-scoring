"""
Stage 6: Quality Dashboard
Streamlit app to visualise scoring results.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import polars as pl
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="LLM Data Quality Pipeline",
    page_icon="🔬",
    layout="wide"
)

SCORED_DIR = Path("data/scored")

st.title("🔬 LLM Data Quality Scoring Pipeline")
st.caption("Training data quality analysis — Project 1")

# Load scored data
parquet_files = list(SCORED_DIR.glob("*.parquet"))

if not parquet_files:
    st.warning("No scored data found. Run the pipeline first: `python main.py --stage all`")
    st.stop()

selected_file = st.selectbox("Select dataset", parquet_files, format_func=lambda x: x.name)
df = pl.read_parquet(selected_file).to_pandas()

# --- Top metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total samples", f"{len(df):,}")
with col2:
    if "composite_score" in df.columns:
        passed = (df["composite_score"] >= 0.7).sum()
        st.metric("Pass threshold (≥0.7)", f"{passed:,}", f"{passed/len(df):.0%}")
with col3:
    if "security_severity" in df.columns:
        flagged = (df["security_severity"] != "NONE").sum()
        st.metric("Security flagged", f"{flagged:,}", f"{flagged/len(df):.0%}", delta_color="inverse")
with col4:
    if "content_hash" in df.columns:
        st.metric("Unique samples", f"{df['content_hash'].nunique():,}")

st.divider()

# --- Score distribution ---
if "composite_score" in df.columns:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Composite Score Distribution")
        fig = px.histogram(df, x="composite_score", nbins=50,
                          color_discrete_sequence=["#00d4ff"])
        fig.add_vline(x=0.7, line_dash="dash", line_color="#ff6b6b",
                     annotation=dict(text="Quality threshold"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Score Breakdown")
        score_cols = [c for c in ["syntactic_score", "doc_score", "security_score", "llm_score"]
                     if c in df.columns]
        if score_cols:
            melted = df[score_cols].melt(var_name="Score Type", value_name="Score")
            fig2 = px.box(melted, x="Score Type", y="Score", color="Score Type",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig2, use_container_width=True)

# --- Sample browser ---
st.subheader("Sample Browser")
threshold = st.slider("Minimum composite score", 0.0, 1.0, 0.7, 0.05)
if "composite_score" in df.columns:
    filtered = df[df["composite_score"] >= threshold]
    st.caption(f"Showing {len(filtered):,} samples above threshold")
    display_cols = [c for c in ["composite_score", "language", "num_lines",
                                "is_parseable", "content"] if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(50),
                use_container_width=True)
else:
    st.dataframe(df.head(50), use_container_width=True)
