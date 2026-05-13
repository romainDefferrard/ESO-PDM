"""
steps/chunk.py
==============
UI section for the Chunking step.

Simplified vs previous version:
  - No min_points field (removed per design decision)
  - No gps_time field (hardcoded in YAML builder)
  - No chunk variant UI (always active, uses global outage window)
  - min_last_chunk_m is enforced as 2/3 * length_m in project_io (not shown)
"""

from __future__ import annotations
import streamlit as st


def render(ui: dict, pfx: str, expanded: bool = False) -> None:
    with st.expander("✂️ Chunking", expanded=expanded):

        c1, c2 = st.columns(2)
        ui["chunk_source"] = c1.selectbox(
            "Source chunks",
            ["generate", "existing"],
            index=["generate", "existing"].index(ui["chunk_source"]),
            key=f"{pfx}_cs",
            help=(
                "**generate** — découpe les nuages merged/ALL en chunks de longueur fixe.  \n"
                "**existing** — réutilise des chunks déjà calculés sur disque."
            ),
        )
        ui["chunk_fmt"] = c2.selectbox(
            "Format nuage",
            ["las", "laz", "txt"],
            index=["las", "laz", "txt"].index(ui["chunk_fmt"]),
            key=f"{pfx}_cf",
        )

        if ui["chunk_source"] == "generate":
            ui["chunk_merged_input_root"] = st.text_input(
                "Dossier merged/ALL source",
                value=ui["chunk_merged_input_root"],
                key=f"{pfx}_cmir",
                placeholder="Laisser vide → <root_out_dir>/<scenario>/merged/ALL (auto)",
            )
        else:
            ui["chunk_existing_root"] = st.text_input(
                "Dossier chunks existants",
                value=ui["chunk_existing_root"],
                key=f"{pfx}_cer",
            )

        c1, c2 = st.columns(2)
        ui["chunk_length_m"] = c1.number_input(
            "Longueur chunk (m)",
            value=ui["chunk_length_m"],
            key=f"{pfx}_cl",
            help="Longueur de chaque chunk en mètres le long de la trajectoire.",
        )
        min_last = round(ui["chunk_length_m"] * 2.0 / 3.0, 1)
        c2.metric(
            "Min dernier chunk (auto)",
            f"{min_last} m",
            help="Automatiquement = 2/3 × longueur chunk. Le dernier chunk est conservé uniquement s'il atteint cette longueur.",
        )

        ui["chunk_epsg"] = st.text_input(
            "EPSG sortie", value=ui["chunk_epsg"], key=f"{pfx}_ce",
        )

        st.caption(
            "ℹ️ Le chunker découpe **toujours autour de la fenêtre d'outage** définie "
            "dans la section Georeferencing (t_start / durée / buffers)."
        )
