"""
MLS Pipeline Manager
====================
Run from repo root:
    streamlit run streamlit/app.py

Key fix: tab_cfg calls scenario_form() which renders all widgets.
tab_run and tab_yaml read from disk only — no widgets, no key collisions.
"""

from __future__ import annotations
import sys
import queue
import shutil
import tempfile
import threading
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from project_io import (
    DEFAULT_CFGS,
    PIPELINE_SCRIPT,
    REPO_ROOT,
    build_pipeline_yaml,
    create_project,
    create_project_from_template,
    default_scenario_ui,
    list_projects,
    list_scanner_files,
    list_scenarios,
    load_project_meta,
    load_scenario,
    save_project_meta,
    save_scenario,
    scenario_path,
    project_dir,
    ui_from_pipeline_yaml,
)
from steps._common import section, steps_summary
from steps import scanners as sc_step
import steps.georef  as georef_step
import steps.merge   as merge_step
import steps.chunk   as chunk_step
import steps.limatch as limatch_step
import steps.batch   as batch_step

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLS Pipeline Manager",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _init():
    for k, v in {
        "active_project":  None,
        "active_scenario": None,
        "batch_queue":     [],
        "batch_running":   False,
        "batch_results":   [],
        "single_log":      [],
        "single_running":  False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("🛰️ MLS Pipeline")
    st.sidebar.caption(f"`{REPO_ROOT.name}`")
    view = st.sidebar.radio(
        "", ["🗂 Projets", "📷 Scanners"],
        horizontal=True, label_visibility="collapsed", key="sb_view",
    )
    st.sidebar.divider()
    if view == "🗂 Projets":
        _sidebar_projects()
    else:
        _sidebar_scanners()


def _sidebar_projects():
    defaults = sorted(DEFAULT_CFGS.glob("*.yml")) if DEFAULT_CFGS.exists() else []
    pipeline_tpls = [p for p in defaults if not p.stem.startswith("scanner_")]
    if pipeline_tpls:
        labels = ["— template pipeline —"] + [p.stem for p in pipeline_tpls]
        sel = st.sidebar.selectbox("Charger un template", labels, key="tmpl_sel")
        if sel != labels[0]:
            matched = next(p for p in pipeline_tpls if p.stem == sel)
            name = create_project_from_template(matched)
            if name is None:
                st.sidebar.warning(f"Projet « {matched.stem} » existe déjà.")
            else:
                st.session_state.active_project  = name
                st.session_state.active_scenario = matched.stem
            st.rerun()
        st.sidebar.divider()

    with st.sidebar.expander("➕ Nouveau projet", expanded=not bool(list_projects())):
        nm   = st.text_input("Nom *",            key="np_name", placeholder="CALIB_26_02_25")
        desc = st.text_input("Description",      key="np_desc", placeholder="Dataset calibration")
        root = st.text_input("root_out_dir",     key="np_root",
                              placeholder="/media/b085164/Elements/CALIB_26_02_25")
        gps  = st.text_input("GPS input (.txt)", key="np_gps",
                              placeholder="/media/.../GPS.txt")
        sc_n = st.text_input("Scanners (noms, virgule)", key="np_sc", value="HA,LR,PUCK")
        if st.button("Créer le projet", key="btn_np", use_container_width=True):
            name = nm.strip()
            if name and name not in list_projects():
                create_project(name, desc.strip(), root.strip(), gps.strip(),
                               [s.strip() for s in sc_n.split(",") if s.strip()])
                st.session_state.active_project  = name
                st.session_state.active_scenario = None
                st.rerun()
            elif not name: st.warning("Entrer un nom.")
            else:          st.warning("Ce projet existe déjà.")

    st.sidebar.divider()

    for proj in list_projects():
        meta    = load_project_meta(proj)
        is_proj = proj == st.session_state.active_project

        with st.sidebar.expander(f"**{proj}**" if is_proj else proj, expanded=is_proj):
            if meta.get("description"):
                st.caption(meta["description"])

            for scen in list_scenarios(proj):
                is_scen = is_proj and scen == st.session_state.active_scenario
                c1, c2  = st.columns([5, 1])
                label   = f"▶ **{scen}**" if is_scen else f"　{scen}"
                if c1.button(label, key=f"sel_{proj}_{scen}", use_container_width=True):
                    st.session_state.active_project  = proj
                    st.session_state.active_scenario = scen
                    st.rerun()
                if c2.button("✕", key=f"del_sc_{proj}_{scen}"):
                    scenario_path(proj, scen).unlink(missing_ok=True)
                    if st.session_state.active_scenario == scen:
                        st.session_state.active_scenario = None
                    st.session_state.batch_queue = [
                        x for x in st.session_state.batch_queue
                        if x != f"{proj}/{scen}"
                    ]
                    st.rerun()

            st.divider()
            c1, c2 = st.columns([3, 2])
            new_sc = c1.text_input("", key=f"ns_{proj}", placeholder="outage_1_F2B",
                                    label_visibility="collapsed")
            if c2.button("➕ Scénario", key=f"btn_ns_{proj}", use_container_width=True):
                nm2 = new_sc.strip()
                if nm2:
                    save_scenario(proj, nm2, {"_ui": default_scenario_ui(nm2)})
                    st.session_state.active_project  = proj
                    st.session_state.active_scenario = nm2
                    st.rerun()

            imp = st.file_uploader("Importer YAML scénario", type=["yml","yaml"],
                                    key=f"imp_{proj}", label_visibility="collapsed")
            if imp:
                yd   = yaml.safe_load(imp.read().decode("utf-8"))
                nm2  = imp.name.rsplit(".", 1)[0]
                ui_d = ui_from_pipeline_yaml(yd, name=nm2)
                save_scenario(proj, nm2, {"_ui": ui_d})
                paths = yd.get("paths", {})
                meta_ = load_project_meta(proj)
                if paths.get("root_out_dir"): meta_["root_out_dir"] = paths["root_out_dir"]
                if paths.get("gps_input"):    meta_["gps_input"]    = paths["gps_input"]
                save_project_meta(proj, meta_)
                st.session_state.active_project  = proj
                st.session_state.active_scenario = nm2
                st.rerun()

            if st.button(f"🗑 Supprimer {proj}", key=f"del_proj_{proj}",
                          use_container_width=True):
                shutil.rmtree(project_dir(proj), ignore_errors=True)
                if st.session_state.active_project == proj:
                    st.session_state.active_project  = None
                    st.session_state.active_scenario = None
                st.rerun()


def _sidebar_scanners():
    proj = st.session_state.active_project
    if not proj or proj not in list_projects():
        st.sidebar.info("Sélectionne un projet d'abord (vue 🗂 Projets).")
        return
    sc_step.render(proj)

# ─────────────────────────────────────────────────────────────────────────────
# Scenario form  (tab_cfg only — all widgets live here)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_form(project: str, scenario: str) -> dict:
    """
    Render all editable widgets for this scenario.
    Called ONLY from tab_cfg to avoid duplicate widget keys across tabs.
    Returns the mutated ui dict (not yet saved — user clicks Save explicitly).
    """
    data = load_scenario(project, scenario)
    ui   = data.get("_ui", {})
    for k, v in default_scenario_ui(scenario).items():
        ui.setdefault(k, v)

    # Unique prefix — based on project+scenario so keys never collide
    pfx = f"{project}__{scenario}"

    # ── Identity + root_out_dir + steps ────────────────────────────────────
    section("📋", "Scénario")
    c1, c2 = st.columns(2)
    ui["name"]          = c1.text_input("Nom", value=ui["name"], key=f"{pfx}_name")
    ui["scenario_name"] = c2.text_input(
        "scenario_name (sous-dossier de sortie)",
        value=ui["scenario_name"], key=f"{pfx}_scen",
        help="Ex: georef_ALL_traj_outage_1/georef_F2B",
    )
    ui["root_out_dir"] = st.text_input(
        "root_out_dir",
        value=ui.get("root_out_dir", ""),
        key=f"{pfx}_root",
        placeholder="Laisser vide pour hériter du projet",
        help="Répertoire racine des sorties. Vide = utilise celui défini dans le projet.",
    )

    st.markdown("**Étapes actives**")
    cols = st.columns(5)
    ui["step_georef"]     = cols[0].checkbox("Georef",     value=ui["step_georef"],     key=f"{pfx}_sg")
    ui["step_merge"]      = cols[1].checkbox("Merge",      value=ui["step_merge"],      key=f"{pfx}_sm")
    ui["step_chunk"]      = cols[2].checkbox("Chunk",      value=ui["step_chunk"],      key=f"{pfx}_sc")
    ui["step_limatch"]    = cols[3].checkbox("LiMatch",    value=ui["step_limatch"],    key=f"{pfx}_sl")
    ui["step_gps_outage"] = cols[4].checkbox("GPS outage", value=ui["step_gps_outage"], key=f"{pfx}_sg2")

    mode = "chunk" if (ui["step_chunk"] or ui["step_limatch"]) else "georef_only"
    st.caption(f"Mode dérivé : **`{mode}`**")
    st.divider()

    # ── Step expanders ─────────────────────────────────────────────────────
    georef_step.render(project, ui, pfx, expanded=ui["step_georef"])
    merge_step.render(ui, pfx,   expanded=ui["step_merge"])
    chunk_step.render(ui, pfx,   expanded=ui["step_chunk"])
    limatch_step.render(ui, pfx, expanded=ui["step_limatch"])

    st.divider()

    # ── Save / rename ──────────────────────────────────────────────────────
    c_save, _, c_rl, c_ri, c_rb = st.columns([2, 1, 1, 3, 2])
    if c_save.button("💾 Sauvegarder", type="primary", key=f"{pfx}_save",
                     use_container_width=True):
        save_scenario(project, scenario, {"_ui": ui})
        st.success("✅ Scénario sauvegardé.")

    c_rl.markdown("<br>", unsafe_allow_html=True)
    c_rl.markdown("Renommer :")
    new_nm = c_ri.text_input("", value=scenario, label_visibility="collapsed",
                               key=f"{pfx}_rennm")
    if c_rb.button("✏️ Renommer", key=f"{pfx}_do_rename", use_container_width=True):
        nm2 = new_nm.strip()
        if nm2 and nm2 != scenario:
            save_scenario(project, nm2, {"_ui": ui})
            scenario_path(project, scenario).unlink(missing_ok=True)
            st.session_state.active_scenario = nm2
            st.rerun()

    return ui

# ─────────────────────────────────────────────────────────────────────────────
# Run panel  (NO widgets — reads last saved state from disk)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_log_line(line: str) -> str | None:
    """
    Filter pipeline subprocess output before displaying in the UI.
    Returns the line to display, or None to suppress it.

    Keep:  INFO / WARNING / ERROR log lines, UI lines, tracebacks
    Drop:  tqdm progress bars, === / --- separators, blank lines,
           verbose print blocks (scanner_entries dicts, SDC reading progress…)
    """
    s = line.rstrip()

    # Always keep empty-ish lines that separate sections — but collapse runs
    if not s:
        return None

    # tqdm progress bars: contain \r or % with | characters
    if "\r" in line or ("|" in line and "%" in line and "pts/s" in line):
        return None

    # Separator lines
    if s.strip().startswith("=") and len(s.strip()) > 10 and set(s.strip()) <= {"=", "-", " "}:
        return None
    if s.strip().startswith("-") and len(s.strip()) > 10 and set(s.strip()) <= {"=", "-", " "}:
        return None

    # Verbose scanner_entries dict dumps
    if s.strip().startswith("[") and "'cfg_path'" in s:
        return None
    if s.strip().startswith("[") and "'scanner_cfg'" in s:
        return None

    # SDC reading progress (tqdm-like without %)
    if "Reading SDC" in s and ("pts/s" in s or "<" in s):
        return None

    # Raw dict/list dumps that are too long
    if len(s) > 300:
        return s[:300] + " …"

    return line
    try:
        for line in iter(proc.stdout.readline, ""):
            q.put(line)
        proc.stdout.close()
        proc.wait()
    finally:
        q.put(None)


def run_panel(project: str, scenario: str):
    """
    Read-only panel: loads last saved scenario from disk, shows YAML summary,
    and runs the pipeline. No Streamlit widgets with shared keys.
    """
    section("🚀", "Lancer le pipeline")

    # Load from disk — never re-render the form widgets here
    ui = load_scenario(project, scenario).get("_ui", {})
    for k, v in default_scenario_ui(scenario).items():
        ui.setdefault(k, v)

    yaml_dict = build_pipeline_yaml(project, ui)
    mode      = yaml_dict["mode"]

    st.info(
        f"**Étapes :** {steps_summary(ui)}   |   "
        f"**Mode :** `{mode}`   |   "
        f"**Scénario :** `{ui['scenario_name'] or '—'}`"
    )
    st.caption("_Les paramètres affichés sont ceux du dernier enregistrement (onglet Configuration → 💾 Sauvegarder)._")

    c1, c2 = st.columns([1, 1])
    run_clicked = c1.button(
        "▶ Run", type="primary",
        disabled=st.session_state.single_running or st.session_state.batch_running,
        key="btn_run",
    )
    c2.download_button(
        "⬇️ YAML",
        data=yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True),
        file_name=f"pipeline_{scenario}.yml",
        mime="text/yaml",
        key="btn_dl_run",
    )

    if run_clicked:
        # ── Validation ─────────────────────────────────────────────────────
        errors = []
        if not ui.get("traj_path", "").strip():
            errors.append("**Trajectoire** : chemin vers le fichier .out SBET manquant.")
        root = (ui.get("root_out_dir","").strip()
                or load_project_meta(project).get("root_out_dir","").strip())
        if not root:
            errors.append("**root_out_dir** : répertoire de sortie manquant (scénario ou projet).")
        if not ui.get("scenario_name","").strip():
            errors.append("**scenario_name** : nom du sous-dossier de sortie manquant.")
        if errors:
            st.error("Impossible de lancer — champs obligatoires manquants :\n\n"
                     + "\n".join(f"- {e}" for e in errors))
            st.session_state.single_running = False
            st.stop()

        st.session_state.single_running = True
        st.session_state.single_log     = []
        log_box = st.empty()
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "mls_pipeline"
            tmp_dir.mkdir(exist_ok=True)
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cfg = tmp_dir / f"pipeline_{scenario}_{ts}.yml"
            with open(cfg, "w") as f:
                yaml.safe_dump(yaml_dict, f, sort_keys=False)
            cmd = [sys.executable, "-m", "navtools_PDM.pipeline", "--config", str(cfg)]
            st.session_state.single_log += [
                f"[UI] config : {cfg}\n",
                f"[UI] cmd    : {' '.join(cmd)}\n\n",
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(REPO_ROOT),
            )
            q: queue.Queue = queue.Queue()
            t = threading.Thread(target=_stream_to_queue, args=(proc, q), daemon=True)
            t.start()
            while True:
                try:
                    line = q.get(timeout=0.05)
                except queue.Empty:
                    log_box.code("".join(st.session_state.single_log[-300:]), language="bash")
                    continue
                if line is None:
                    break
                filtered = _filter_log_line(line)
                if filtered is not None:
                    st.session_state.single_log.append(filtered)
                log_box.code("".join(st.session_state.single_log[-300:]), language="bash")
            t.join()
            rc = proc.returncode
            st.session_state.single_running = False
            if rc == 0: st.success("✅ Terminé avec succès.")
            else:       st.error(f"❌ Code de retour : {rc}")
        except Exception as e:
            st.session_state.single_running = False
            st.error(f"Erreur : {e}")

    elif st.session_state.single_log:
        with st.expander("📋 Logs (dernier run)", expanded=False):
            st.code("".join(st.session_state.single_log), language="bash")

# ─────────────────────────────────────────────────────────────────────────────
# YAML tab (read-only)
# ─────────────────────────────────────────────────────────────────────────────

def yaml_tab(project: str, scenario: str):
    ui = load_scenario(project, scenario).get("_ui", {})
    for k, v in default_scenario_ui(scenario).items():
        ui.setdefault(k, v)
    yd = build_pipeline_yaml(project, ui)
    ys = yaml.dump(yd, sort_keys=False, allow_unicode=True)
    section("📄", "YAML généré")
    st.caption("_Basé sur le dernier enregistrement. Sauvegarde dans l'onglet Configuration._")
    st.code(ys, language="yaml")
    st.download_button(
        "⬇️ Télécharger",
        data=ys,
        file_name=f"pipeline_{scenario}.yml",
        mime="text/yaml",
        key="btn_dl_yaml_tab",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sidebar()
    proj = st.session_state.active_project
    scen = st.session_state.active_scenario

    if not proj or proj not in list_projects():
        st.title("🛰️ MLS Pipeline Manager")
        st.markdown("""
**Bienvenue.** 👈 Dans la sidebar :

- **Charger un template** — config pré-remplie (CALIB AIRINS, CALIB APX, ECCR)
- **Nouveau projet** — nom, `root_out_dir`, scanners → structure créée sur disque

**Structure sur disque :**
```
streamlit/projects/<PROJET>/
├── project.yml
├── scanners/
│   ├── HA.yml
│   ├── LR.yml
│   └── PUCK.yml
└── scenarios/
    ├── outage_1_F2B.yml
    ├── outage_2_Combined.yml
    └── outage_3_F2B.yml
```
Scanners **communs** à tous les scénarios. Éditer via sidebar → **📷 Scanners**.
        """)
        st.divider()
        batch_step.render()
        return

    if not scen or scen not in list_scenarios(proj):
        st.title(f"🛰️ {proj}")
        meta = load_project_meta(proj)
        c1, c2, c3 = st.columns(3)
        c1.metric("root_out_dir", (meta.get("root_out_dir") or "—")[:40])
        c2.metric("Scanners",     ", ".join(list_scanner_files(proj)) or "—")
        c3.metric("Scénarios",    str(len(list_scenarios(proj))))
        st.info("👈 Sélectionne ou crée un scénario dans la sidebar.")
        st.divider()
        batch_step.render()
        return

    st.title(f"🛰️ {proj}  ›  {scen}")

    tab_cfg, tab_yaml, tab_run, tab_batch = st.tabs([
        "⚙️ Configuration", "📄 YAML", "🚀 Run", "⚡ Batch",
    ])

    with tab_cfg:
        # All widgets live here — unique keys guaranteed by pfx = project__scenario
        scenario_form(proj, scen)

    with tab_yaml:
        # Read-only: loads from disk, no widgets
        yaml_tab(proj, scen)

    with tab_run:
        # Read-only: loads from disk, no widgets (except Run button + download)
        run_panel(proj, scen)

    with tab_batch:
        batch_step.render()


if __name__ == "__main__":
    main()
