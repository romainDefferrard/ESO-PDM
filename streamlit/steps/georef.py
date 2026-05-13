"""
steps/georef.py
===============
"Georeferencing" expander:
  - Trajectory
  - Scanner summary (read-only)
  - Outage window definition (t_start, duration)
  - Temporal filtering toggle → buffers shown only when enabled
  - Distance filtering toggle → params shown only when enabled
"""

from __future__ import annotations
import streamlit as st
from project_io import list_scanner_files, load_scanner


def render(project: str, ui: dict, pfx: str, expanded: bool = True) -> None:
    with st.expander("📡 Georeferencing", expanded=expanded):

        # ── Trajectory ─────────────────────────────────────────────────────
        st.markdown("**Trajectoire ODyN (.out SBET)**")
        traj_col, type_col = st.columns([5, 1])
        ui["traj_path"] = traj_col.text_input(
            "Chemin",
            value=ui["traj_path"],
            key=f"{pfx}_tp",
            placeholder="/media/.../ODyN/.../out/traj.out",
            label_visibility="collapsed",
        )
        traj_up = traj_col.file_uploader(
            "Ou glisser-déposer le fichier .out",
            type=["out"],
            key=f"{pfx}_traj_up",
            label_visibility="visible",
        )
        if traj_up is not None:
            st.caption(
                f"ℹ️ Fichier sélectionné : **{traj_up.name}** — "
                "entre le chemin absolu sur le serveur dans le champ ci-dessus."
            )
        ui["traj_type"] = type_col.selectbox("Type", ["SBET"], key=f"{pfx}_tt")

        st.divider()

        # ── Scanner summary ────────────────────────────────────────────────
        st.markdown("**Scanners** _(éditer via sidebar → 📷 Scanners)_")
        scanners = list_scanner_files(project)
        if scanners:
            cols = st.columns(len(scanners))
            for i, sc_name in enumerate(scanners):
                sc    = load_scanner(project, sc_name)
                mount = sc.get("mount", {})
                la    = mount.get("lever_arm", [0, 0, 0])
                bs    = mount.get("boresight", {})
                sdc   = sc.get("lasvec", {}).get("path", "—")
                with cols[i]:
                    st.markdown(
                        f"**{sc_name}** "
                        f"<code style='font-size:0.7em;color:#888'>{sc_name}.yml</code>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"lever arm: `[{la[0]:.4f}, {la[1]:.4f}, {la[2]:.4f}]`")
                    st.caption(
                        f"boresight: r={bs.get('roll',0):.6f} "
                        f"p={bs.get('pitch',0):.6f} y={bs.get('yaw',0):.6f}"
                    )
                    st.caption(f"SDC: `…{str(sdc)[-38:]}`")
        else:
            st.warning("Aucun scanner — ajoute-en via la sidebar → vue **📷 Scanners**.")

        st.divider()

        # ── Outage window ──────────────────────────────────────────────────
        st.markdown("**Fenêtre d'outage GNSS**")
        st.caption("Définit la fenêtre globale partagée par le filtrage temporel et le chunker.")
        c1, c2 = st.columns(2)
        ui["outage_start"] = c1.number_input(
            "t_start GPS (s)", value=float(ui["outage_start"]),
            format="%.1f", key=f"{pfx}_os",
        )
        ui["outage_dur"] = c2.number_input(
            "Durée (s)", value=float(ui["outage_dur"]),
            format="%.1f", key=f"{pfx}_od",
        )
        t_end = ui["outage_start"] + ui["outage_dur"]
        st.caption(f"Fin d'outage : `{t_end:.1f}` s GPS")

        st.divider()

        # ── Temporal filtering ─────────────────────────────────────────────
        ui["tw_enable"] = st.checkbox(
            "Activer le filtrage temporel du géoréférencement",
            value=ui["tw_enable"],
            key=f"{pfx}_twe",
            help=(
                "Restreint le géoréférencement aux points LiDAR dont le timestamp GPS "
                "tombe dans [t_start − buf_pre, t_end + buf_post]. "
                "Réduit le volume de données traité autour de l'outage."
            ),
        )
        if ui["tw_enable"]:
            c1, c2 = st.columns(2)
            ui["buf_pre_s"] = c1.number_input(
                "Buffer pré-outage (s)", value=float(ui["buf_pre_s"]),
                format="%.1f", key=f"{pfx}_bpre",
                help="Données conservées avant le début de l'outage.",
            )
            ui["buf_post_s"] = c2.number_input(
                "Buffer post-outage (s)", value=float(ui["buf_post_s"]),
                format="%.1f", key=f"{pfx}_bpost",
                help="Données conservées après la fin de l'outage.",
            )
            st.info(
                f"✂️ Fenêtre géoréf : "
                f"`{ui['outage_start'] - ui['buf_pre_s']:.1f}` → "
                f"`{t_end + ui['buf_post_s']:.1f}` s  "
                f"({ui['outage_dur']:.0f}s outage + {ui['buf_pre_s']:.0f}s + {ui['buf_post_s']:.0f}s buffers)"
            )

        st.divider()

        # ── Distance filtering ─────────────────────────────────────────────
        st.markdown("**Filtrage distance laser**")
        st.caption(
            "Élimine les points dont la distance scanner→surface dépasse le seuil. "
            "Chaque point LAS porte un vecteur laser (lasvec) reliant le centre du scanner "
            "au point mesuré — ce filtre tronque les vecteurs trop longs, souvent bruités ou "
            "issus de réflexions parasites."
        )
        ui["dist_enable"] = st.checkbox(
            "Activer le filtrage distance", value=ui["dist_enable"], key=f"{pfx}_de",
        )
        if ui["dist_enable"]:
            c1, c2 = st.columns(2)
            ui["dist_max_m"] = c1.number_input(
                "Distance max (m)", value=ui["dist_max_m"], key=f"{pfx}_dm",
            )
            ui["dist_epsg"] = c2.text_input(
                "EPSG carte", value=ui["dist_epsg"], key=f"{pfx}_dep",
            )
