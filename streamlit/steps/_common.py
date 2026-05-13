"""
steps/_common.py
================
Shared UI helpers used by all step modules.
"""
import streamlit as st


def section(icon: str, title: str) -> None:
    st.markdown(
        f"<h3 style='margin-bottom:2px'>{icon}&nbsp;{title}</h3>"
        f"<hr style='margin-top:0;margin-bottom:10px;border-color:#555'>",
        unsafe_allow_html=True,
    )


def steps_summary(ui: dict) -> str:
    labels = {
        "step_georef":     "Georef",
        "step_merge":      "Merge",
        "step_chunk":      "Chunk",
        "step_limatch":    "LiMatch",
        "step_gps_outage": "GPS outage",
    }
    return " → ".join(v for k, v in labels.items() if ui.get(k)) or "—"
