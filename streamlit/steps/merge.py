"""
steps/merge.py
==============
UI section for the Merge step:
  - Preset (all / vux_only / puck_on_existing)
  - Output naming, source IDs, cleanup
  - Merge correspondences toggle
"""

from __future__ import annotations
import streamlit as st


def render(ui: dict, pfx: str, expanded: bool = True) -> None:
    """Render merge fields into *ui* in place."""
    with st.expander("🔀 Merge", expanded=expanded):

        c1, c2 = st.columns(2)
        ui["merge_preset"] = c1.selectbox(
            "Preset",
            ["vux_only", "all", "puck_on_existing"],
            index=["vux_only", "all", "puck_on_existing"].index(ui["merge_preset"]),
            key=f"{pfx}_mp",
            help="all = VUX (HA+LR) puis PUCK | vux_only = HA+LR seulement | puck_on_existing = ajouter PUCK sur merged/HA_LR existant",
        )
        ui["merge_chunk_size"] = c2.number_input(
            "Chunk size (pts)", value=ui["merge_chunk_size"],
            step=1_000_000, key=f"{pfx}_mcs",
        )

        c1, c2, c3 = st.columns(3)
        ui["merge_out_prefix"]    = c1.text_input("Préfixe output", value=ui["merge_out_prefix"],    key=f"{pfx}_mop")
        ui["merge_out_suffix"]    = c2.text_input("Suffixe VUX",    value=ui["merge_out_suffix"],    key=f"{pfx}_mos")
        ui["merge_output_suffix"] = c3.text_input("Suffixe ALL",    value=ui["merge_output_suffix"], key=f"{pfx}_moas")

        c1, c2, c3, c4 = st.columns(4)
        ui["merge_src_vux"]  = c1.number_input("scanner_src VUX",  value=ui["merge_src_vux"],  key=f"{pfx}_msv")
        ui["merge_src_puck"] = c2.number_input("scanner_src PUCK", value=ui["merge_src_puck"], key=f"{pfx}_msp")
        ui["merge_cleanup"]  = c3.checkbox("Cleanup après merge",  value=ui["merge_cleanup"],  key=f"{pfx}_mc")
        ui["merge_cleanup_scanner_dirs"] = c4.checkbox(
            "+ dossiers scanners",
            value=ui["merge_cleanup_scanner_dirs"],
            disabled=not ui["merge_cleanup"],
            key=f"{pfx}_mcsd",
        )

        if ui["merge_preset"] == "puck_on_existing":
            ui["merge_vux_input_dir"] = st.text_input(
                "VUX input dir (merged/HA_LR existant)",
                value=ui["merge_vux_input_dir"],
                key=f"{pfx}_mvid",
                placeholder="<root_out_dir>/<scenario>/merged/HA_LR",
            )

        st.divider()
        ui["mc_enabled"] = st.checkbox(
            "Merge correspondences LiMatch (LiDAR_p2p.txt)",
            value=ui["mc_enabled"],
            key=f"{pfx}_mce",
        )
