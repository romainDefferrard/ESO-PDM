"""
steps/scanners.py
=================
Scanner editor — rendered in the sidebar under the "📷 Scanners" view.

Each scanner in a project is stored as an independent YAML file:
    streamlit/projects/<PROJECT>/scanners/<NAME>.yml

The file name shown next to each scanner radio button corresponds to
the YAML file currently loaded on disk (e.g. "HA.yml").
"""

from __future__ import annotations
from pathlib import Path

import streamlit as st
import yaml

from project_io import (
    DEFAULT_CFGS,
    default_scanner,
    list_scanner_files,
    load_project_meta,
    load_scanner,
    save_project_meta,
    save_scanner,
    scanner_path,
)


# ── Public entry point ─────────────────────────────────────────────────────

def render(project: str) -> None:
    """Render the full scanner sidebar for *project*."""
    st.sidebar.subheader(f"📷 Scanners — {project}")

    _render_add_controls(project)
    st.sidebar.divider()

    scanners = list_scanner_files(project)
    if not scanners:
        st.sidebar.info("Aucun scanner. Ajoute-en via le panneau ci-dessus.")
        return

    # Radio with filename shown next to scanner name
    labels = [f"{sc}  `{sc}.yml`" for sc in scanners]
    sel_label = st.sidebar.radio(
        "Scanner actif",
        labels,
        label_visibility="collapsed",
        key="sc_radio_sel",
    )
    sel = scanners[labels.index(sel_label)]

    st.sidebar.divider()
    _render_scanner_editor(project, sel)


# ── Add / import controls ──────────────────────────────────────────────────

def _render_add_controls(project: str) -> None:
    with st.sidebar.expander("➕ Ajouter / importer un scanner"):
        # Create blank
        c1, c2 = st.columns([3, 1])
        new_name = c1.text_input("Nom", key="sc_new_name", placeholder="HA / LR / PUCK")
        if c2.button("Créer", key="sc_btn_create"):
            nm = new_name.strip().upper()
            if nm:
                if not scanner_path(project, nm).exists():
                    # Seed from default_configs if a matching file exists
                    candidates = list(DEFAULT_CFGS.glob(f"scanner_{nm}*.yml"))
                    sc_data = (yaml.safe_load(candidates[0].read_text(encoding="utf-8"))
                               if candidates else default_scanner(nm))
                    save_scanner(project, nm, sc_data)
                _add_to_meta(project, nm)
                st.rerun()

        # Upload existing YAML
        uploaded = st.file_uploader(
            "Importer un scanner .yml",
            type=["yml", "yaml"],
            key="sc_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            sc_data = yaml.safe_load(uploaded.read().decode("utf-8"))
            nm = sc_data.get("scanner_name", uploaded.name.rsplit(".", 1)[0]).upper()
            save_scanner(project, nm, sc_data)
            _add_to_meta(project, nm)
            st.sidebar.success(f"Scanner {nm} importé depuis `{uploaded.name}`.")
            st.rerun()


def _add_to_meta(project: str, sc_name: str) -> None:
    meta = load_project_meta(project)
    if sc_name not in meta.get("scanners", []):
        meta.setdefault("scanners", []).append(sc_name)
        save_project_meta(project, meta)


# ── Scanner editor ─────────────────────────────────────────────────────────

def _render_scanner_editor(project: str, sel: str) -> None:
    sc       = load_scanner(project, sel)
    yml_file = f"{sel}.yml"

    # Header with filename badge
    st.sidebar.markdown(
        f"**✏️ {sel}** &nbsp; <code style='font-size:0.75em;color:#aaa'>"
        f"projects/{project}/scanners/{yml_file}</code>",
        unsafe_allow_html=True,
    )

    # ── scanner_name ─────────────────────────────────────────────────────────
    sc["scanner_name"] = st.sidebar.text_input(
        "scanner_name",
        value=sc.get("scanner_name", sel),
        key=f"sc_{sel}_name",
    )

    # ── leapsec ──────────────────────────────────────────────────────────────
    sc["leapsec"] = st.sidebar.number_input(
        "leapsec", value=int(sc.get("leapsec", 18)), step=1, key=f"sc_{sel}_ls",
    )

    # ── lasvec ───────────────────────────────────────────────────────────────
    with st.sidebar.expander("📂 lasvec (SDC)", expanded=False):
        lv = sc.get("lasvec", {})
        lv["type"] = st.selectbox(
            "type", ["SDC", "LAS"],
            index=["SDC", "LAS"].index(lv.get("type", "SDC")),
            key=f"sc_{sel}_lvt",
        )
        lv["path"] = st.text_input(
            "SDC path", value=lv.get("path", ""), key=f"sc_{sel}_lvp",
        )
        raw_cols = ",".join(str(c) for c in lv.get("cols", [0, 3, 4, 5]))
        parsed   = st.text_input(
            "cols (comma-separated)", value=raw_cols, key=f"sc_{sel}_lvc",
        )
        lv["cols"] = [
            int(x.strip()) for x in parsed.split(",")
            if x.strip().lstrip("-").isdigit()
        ]
        sc["lasvec"] = lv

    # ── mount ─────────────────────────────────────────────────────────────────
    with st.sidebar.expander("🔩 Mount", expanded=True):
        mount = sc.get("mount", {})

        # R_mount — 3×3 grid
        st.markdown("**R_mount** (3×3, row-major)")
        R = mount.get("R_mount", [[1,0,0],[0,1,0],[0,0,1]])
        while len(R) < 3:
            R.append([0.0, 0.0, 0.0])
        new_R = []
        for ri in range(3):
            cols3 = st.columns(3)
            row   = []
            for ci in range(3):
                val = float(R[ri][ci]) if ci < len(R[ri]) else 0.0
                row.append(cols3[ci].number_input(
                    f"[{ri},{ci}]", value=val, format="%.9f",
                    key=f"sc_{sel}_R{ri}{ci}",
                    label_visibility="collapsed",
                ))
            new_R.append(row)
        mount["R_mount"] = new_R

        # Boresight
        st.markdown("**Boresight** (rad)")
        bs  = mount.get("boresight", {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
        bc1, bc2, bc3 = st.columns(3)
        bs["roll"]  = bc1.number_input("roll",  value=float(bs.get("roll",  0.0)), format="%.10f", key=f"sc_{sel}_bsr")
        bs["pitch"] = bc2.number_input("pitch", value=float(bs.get("pitch", 0.0)), format="%.10f", key=f"sc_{sel}_bsp")
        bs["yaw"]   = bc3.number_input("yaw",   value=float(bs.get("yaw",   0.0)), format="%.10f", key=f"sc_{sel}_bsy")
        mount["boresight"] = bs

        # Lever arm
        st.markdown("**Lever arm** (m, body frame)")
        la = mount.get("lever_arm", [0.0, 0.0, 0.0])
        while len(la) < 3:
            la.append(0.0)
        lc1, lc2, lc3 = st.columns(3)
        la[0] = lc1.number_input("X", value=float(la[0]), format="%.6f", key=f"sc_{sel}_lax")
        la[1] = lc2.number_input("Y", value=float(la[1]), format="%.6f", key=f"sc_{sel}_lay")
        la[2] = lc3.number_input("Z", value=float(la[2]), format="%.6f", key=f"sc_{sel}_laz")
        mount["lever_arm"] = la
        sc["mount"] = mount

    # ── manifest_path ─────────────────────────────────────────────────────────
    sc["manifest_path"] = st.sidebar.text_input(
        "manifest_path",
        value=sc.get("manifest_path", ""),
        key=f"sc_{sel}_mp",
    )

    # ── output_defaults ───────────────────────────────────────────────────────
    with st.sidebar.expander("⬇️ output_defaults", expanded=False):
        od = sc.get("output_defaults", {"type": "LAS", "lasvec": True, "lasvec_to_body": True})
        od["type"] = st.selectbox(
            "type", ["LAS", "TXT"],
            index=["LAS", "TXT"].index(od.get("type", "LAS")),
            key=f"sc_{sel}_odt",
        )
        od["lasvec"]         = st.checkbox("lasvec",         value=od.get("lasvec", True),         key=f"sc_{sel}_odlv")
        od["lasvec_to_body"] = st.checkbox("lasvec_to_body", value=od.get("lasvec_to_body", True),  key=f"sc_{sel}_odltb")
        sc["output_defaults"] = od

    # ── Actions ───────────────────────────────────────────────────────────────
    st.sidebar.divider()
    c_save, c_dl = st.sidebar.columns(2)

    if c_save.button("💾 Sauvegarder", key=f"sc_{sel}_save", use_container_width=True):
        save_scanner(project, sel, sc)
        st.sidebar.success(f"✅ {sel}.yml sauvegardé")

    c_dl.download_button(
        f"⬇️ {yml_file}",
        data=yaml.dump(sc, sort_keys=False, allow_unicode=True),
        file_name=yml_file,
        mime="text/yaml",
        key=f"sc_{sel}_dl",
        use_container_width=True,
    )

    with st.sidebar.expander("📄 YAML brut", expanded=False):
        st.code(yaml.dump(sc, sort_keys=False, allow_unicode=True), language="yaml")

    if st.sidebar.button(f"🗑 Supprimer {sel}", key=f"sc_{sel}_del"):
        scanner_path(project, sel).unlink(missing_ok=True)
        meta = load_project_meta(project)
        meta["scanners"] = [s for s in meta.get("scanners", []) if s != sel]
        save_project_meta(project, meta)
        st.rerun()
