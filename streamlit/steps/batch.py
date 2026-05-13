"""
steps/batch.py
==============
Batch runner panel — compose an ordered queue of project/scenario pairs
and run them sequentially with live status updates.
"""

from __future__ import annotations
import sys
import time
import queue
import threading
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

from project_io import (
    PIPELINE_SCRIPT,
    REPO_ROOT,
    build_pipeline_yaml,
    default_scenario_ui,
    list_projects,
    list_scenarios,
    load_scenario,
    save_scenario,
)
from steps._common import section, steps_summary
import subprocess


# ── Internal process runner ────────────────────────────────────────────────

def _stream_to_queue(proc, q: queue.Queue) -> None:
    try:
        for line in iter(proc.stdout.readline, ""):
            q.put(line)
        proc.stdout.close()
        proc.wait()
    finally:
        q.put(None)


def _filter_log_line(line: str) -> str | None:
    """Suppress tqdm bars, separator blocks, and verbose dict dumps."""
    s = line.rstrip()
    if not s:
        return None
    if "\r" in line or ("|" in line and "%" in line and "pts/s" in line):
        return None
    if s.strip().startswith(("=", "-")) and len(s.strip()) > 10 and set(s.strip()) <= {"=", "-", " "}:
        return None
    if s.strip().startswith("[") and ("'cfg_path'" in s or "'scanner_cfg'" in s):
        return None
    if "Reading SDC" in s and ("pts/s" in s or "<" in s):
        return None
    if len(s) > 300:
        return s[:300] + " …\n"
    return line


def run_pipeline_blocking(yaml_dict: dict, label: str = "") -> tuple[int, list[str]]:
    """Write a temp YAML, run pipeline.py, return (returncode, filtered_log_lines)."""
    tmp_dir = Path(tempfile.gettempdir()) / "mls_pipeline"
    tmp_dir.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cfg = tmp_dir / f"pipeline_{label}_{ts}.yml"
    with open(cfg, "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    cmd = [sys.executable, "-m", "navtools_PDM.pipeline", "--config", str(cfg)]
    log: list[str] = [
        f"[UI] config : {cfg}\n",
        f"[UI] cmd    : {' '.join(cmd)}\n\n",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
    )
    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_stream_to_queue, args=(proc, q), daemon=True)
    t.start()
    while True:
        line = q.get()
        if line is None:
            break
        filtered = _filter_log_line(line)
        if filtered is not None:
            log.append(filtered)
    t.join()
    return proc.returncode, log


# ── Public render ──────────────────────────────────────────────────────────

def render() -> None:
    section("⚡", "Batch — enchaîner des scénarios")

    all_items = [
        f"{proj}/{scen}"
        for proj in list_projects()
        for scen in list_scenarios(proj)
    ]
    if not all_items:
        st.info("Aucun scénario disponible. Crée d'abord un projet et au moins un scénario.")
        return

    # ── Queue builder ──────────────────────────────────────────────────────
    st.markdown("#### File d'exécution")
    c_add, c_clear = st.columns([4, 1])
    available = [s for s in all_items if s not in st.session_state.batch_queue]
    if available:
        to_add = c_add.selectbox("Ajouter un scénario à la file", ["—"] + available, key="bat_add")
        if to_add != "—":
            st.session_state.batch_queue.append(to_add)
            st.rerun()
    if c_clear.button("🗑 Vider", key="bat_clear"):
        st.session_state.batch_queue = []
        st.rerun()

    q_list = st.session_state.batch_queue
    if not q_list:
        st.warning("La file est vide.")
    else:
        for i, item in enumerate(q_list):
            proj, scen = item.split("/", 1)
            ui_d = load_scenario(proj, scen).get("_ui", {})
            ci, cn, cs, cu, cd, cx = st.columns([0.4, 2.5, 4, 0.4, 0.4, 0.4])
            ci.markdown(f"**{i+1}**")
            cn.markdown(f"`{proj}` › **{scen}**")
            cs.caption(steps_summary(ui_d))
            if cu.button("↑", key=f"bu_{i}", disabled=(i == 0)):
                q_list[i-1], q_list[i] = q_list[i], q_list[i-1]; st.rerun()
            if cd.button("↓", key=f"bd_{i}", disabled=(i == len(q_list) - 1)):
                q_list[i+1], q_list[i] = q_list[i], q_list[i+1]; st.rerun()
            if cx.button("✕", key=f"bx_{i}"):
                q_list.pop(i); st.rerun()

    st.divider()

    # ── Run button ─────────────────────────────────────────────────────────
    if st.button(
        f"▶▶ Lancer le batch ({len(q_list)} scénario(s))",
        type="primary",
        disabled=st.session_state.get("batch_running", False) or not q_list,
        key="btn_bat",
    ):
        st.session_state.batch_running = True
        st.session_state.batch_results = []
        total  = len(q_list)
        prog   = st.progress(0, text="Démarrage…")
        boxes  = {item: st.empty() for item in q_list}

        for i, item in enumerate(q_list):
            proj, scen = item.split("/", 1)
            ui_d = load_scenario(proj, scen).get("_ui", {})
            for k, v in default_scenario_ui(scen).items():
                ui_d.setdefault(k, v)
            yaml_dict = build_pipeline_yaml(proj, ui_d)

            prog.progress(i / total, text=f"[{i+1}/{total}] {proj}/{scen}…")

            # Validate before launching
            val_errors = []
            if not ui_d.get("traj_path","").strip():
                val_errors.append("trajectoire manquante")
            from project_io import load_project_meta as _lpm
            root = (ui_d.get("root_out_dir","").strip()
                    or _lpm(proj).get("root_out_dir","").strip())
            if not root:
                val_errors.append("root_out_dir manquant")
            if val_errors:
                dur = 0.0
                msg = f"❌ {proj}/{scen} ignoré : {', '.join(val_errors)}"
                st.session_state.batch_results.append(
                    {"item": item, "status": "❌ config invalide",
                     "log": [msg + "\n"], "dur": dur})
                boxes[item].error(msg)
                continue

            boxes[item].info(f"⏳ **{proj}/{scen}** — en cours…")
            t0 = time.time()
            try:
                rc, log = run_pipeline_blocking(yaml_dict, label=f"{proj}_{scen}")
                dur    = time.time() - t0
                status = "✅ succès" if rc == 0 else f"❌ code {rc}"
                st.session_state.batch_results.append(
                    {"item": item, "status": status, "log": log, "dur": dur}
                )
                (boxes[item].success if rc == 0 else boxes[item].error)(
                    f"{status}  `{proj}/{scen}`  ({dur:.0f}s)"
                )
            except Exception as e:
                dur = time.time() - t0
                st.session_state.batch_results.append(
                    {"item": item, "status": "💥 exception", "log": [str(e)], "dur": dur}
                )
                boxes[item].error(f"💥 **{proj}/{scen}** — {e}")

        prog.progress(1.0, text="Batch terminé.")
        st.session_state.batch_running = False
        n_ok = sum(1 for r in st.session_state.batch_results if r["status"].startswith("✅"))
        st.success(f"Batch terminé : **{n_ok}/{total}** réussi(s).")

    # ── Previous results ───────────────────────────────────────────────────
    if st.session_state.get("batch_results"):
        st.divider()
        st.markdown("#### Résultats du dernier batch")
        for r in st.session_state.batch_results:
            with st.expander(
                f"{r['status']}  `{r['item']}`  —  {r['dur']:.0f}s",
                expanded=False,
            ):
                st.code("".join(r["log"]), language="bash")
