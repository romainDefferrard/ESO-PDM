"""
steps/limatch.py
================
UI section for the LiMatch step.
- lm_run removed (derived from step_limatch in project_io)
- mc_enabled (merge correspondences / LiDAR_p2p.txt) is here, at the bottom
- All widget keys use pfx prefix — no hardcoded suffixes that could collide
"""

from __future__ import annotations
import streamlit as st


def render(ui: dict, pfx: str, expanded: bool = False) -> None:
    with st.expander("🔍 LiMatch", expanded=expanded):

        ui["limatch_cfg"] = st.text_input(
            "Fichier config LiMatch (.yml)",
            value=ui.get("limatch_cfg", ""),
            key=f"{pfx}_lmc",
            placeholder="Patcher/submodules/limatch/configs/MLS_F2B_1.yml",
            help="Chemin relatif à la racine du repo, ou absolu.",
        )

        # ── Consecutive pairs ──────────────────────────────────────────────
        st.markdown("**Paires consécutives (F2B)**")
        c1, c2 = st.columns(2)
        ui["lm_neighbor_k"] = c1.number_input(
            "k voisins consécutifs",
            value=ui["lm_neighbor_k"],
            min_value=0,
            key=f"{pfx}_lmnk",
            help="0 = chaque chunk matchée avec la suivante uniquement (k+1 = 1 paire).",
        )
        ui["lm_do_cross_scan"] = c2.checkbox(
            "Cross-scan F2B (aller/retour)",
            value=ui["lm_do_cross_scan"],
            key=f"{pfx}_lmcs",
        )
        ui["lm_output_root"] = st.text_input(
            "Output root (vide = auto)",
            value=ui["lm_output_root"],
            key=f"{pfx}_lmor",
            placeholder="<root_out_dir>/<scenario>/limatch",
        )

        st.divider()

        # ── Spatial crossings ──────────────────────────────────────────────
        st.markdown("**Spatial crossings**")
        ui["lm_do_spatial_crossings"] = st.checkbox(
            "Activer les spatial crossings",
            value=ui["lm_do_spatial_crossings"],
            key=f"{pfx}_lmsc",
            help=(
                "Apparie les chunks de passes opposées (aller vs retour) qui se croisent "
                "spatialement. Très efficace pour contraindre la dérive latérale."
            ),
        )
        if ui["lm_do_spatial_crossings"]:
            c1, c2, c3 = st.columns(3)
            ui["lm_crossing_min_sep"] = c1.number_input(
                "Séparation min (m)",
                value=ui["lm_crossing_min_sep"],
                key=f"{pfx}_lcms",
            )
            ui["lm_crossing_overlap_m"] = c2.number_input(
                "Overlap margin (m)",
                value=ui["lm_crossing_overlap_m"],
                key=f"{pfx}_lcom",
            )
            ui["lm_crossings_output_root"] = c3.text_input(
                "Output crossings (vide = auto)",
                value=ui["lm_crossings_output_root"],
                key=f"{pfx}_lcor",
                placeholder="<limatch_output_root>_crossings",
            )

        st.divider()

        # ── Per-pair uncertainty_r overrides ───────────────────────────────
        st.markdown("**Overrides uncertainty_r**")
        ui["lm_use_overrides"] = st.checkbox(
            "Valeurs différentes pour F2B et crossings",
            value=ui["lm_use_overrides"],
            key=f"{pfx}_lmuo",
            help=(
                "Utile quand le drift APX est fort : augmenter r pour les paires F2B, "
                "garder faible pour les crossings."
            ),
        )
        if ui["lm_use_overrides"]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**F2B consécutifs**")
                ui["lm_f2b_uncertainty_r_min"] = st.number_input(
                    "r_min (m)", value=ui["lm_f2b_uncertainty_r_min"],
                    key=f"{pfx}_f2brmin",
                )
                ui["lm_f2b_uncertainty_r_max"] = st.number_input(
                    "r_max (m)", value=ui["lm_f2b_uncertainty_r_max"],
                    key=f"{pfx}_f2brmax",
                )
            with c2:
                st.markdown("**Spatial crossings**")
                ui["lm_cross_uncertainty_r_min"] = st.number_input(
                    "r_min (m)", value=ui["lm_cross_uncertainty_r_min"],
                    key=f"{pfx}_cxrmin",
                )
                ui["lm_cross_uncertainty_r_max"] = st.number_input(
                    "r_max (m)", value=ui["lm_cross_uncertainty_r_max"],
                    key=f"{pfx}_cxrmax",
                )

        st.divider()

        # ── Merge correspondences ──────────────────────────────────────────
        ui["mc_enabled"] = st.checkbox(
            "Fusionner les correspondances (LiDAR_p2p.txt)",
            value=ui["mc_enabled"],
            key=f"{pfx}_lm_mce",          # NOTE: _lm_mce to avoid collision with any old _mce key
            help=(
                "Concatène tous les fichiers de correspondances LiMatch en un seul "
                "LiDAR_p2p.txt à la racine du dossier limatch — prêt pour ODyN."
            ),
        )
