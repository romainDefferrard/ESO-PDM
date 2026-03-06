import streamlit as st
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib.scenario_tools import scenario_paths, scenario_status

st.title("Files")

cfg = st.session_state.app_config
paths = scenario_paths(cfg)
status = scenario_status(cfg)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Merged txt", status["merged_n_txt"])
c2.metric("Chunks", "yes" if status["chunks_exists"] else "no")
c3.metric("LiMatch", "yes" if status["limatch_exists"] else "no")
c4.metric("Tmp", "yes" if status["tmp_exists"] else "no")

st.subheader("Main folders")
main_paths = {
    "scenario_root": str(paths["scenario_root"]),
    "merged_dir": str(paths["merged_dir"]),
    "chunks_dir": str(paths["chunks_dir"]),
    "limatch_dir": str(paths["limatch_dir"]),
    "patcher_dir": str(paths["patcher_dir"]),
    "results_dir": str(paths["results_dir"]),
}
st.json(main_paths)

show_intermediate = st.checkbox(
    "Show intermediate folders",
    value=cfg["execution"].get("show_intermediate_outputs", False),
)

folder_keys = [
    "scenario_root",
    "merged_dir",
    "chunks_dir",
    "limatch_dir",
    "patcher_dir",
    "results_dir",
]

if show_intermediate:
    folder_keys.extend([
        "tmp_root",
        "georef_tmp_root",
        "ha_tmp_dir",
        "lr_tmp_dir",
        "generated_cfg_dir",
        "scenario_combined_dir",
        "scenario_gps_outage_dir",
    ])

folder_key = st.selectbox("Folder to inspect", folder_keys)
folder = paths[folder_key]

if folder.exists():
    entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

    rows = []
    for p in entries:
        rows.append(
            {
                "name": p.name,
                "type": "dir" if p.is_dir() else "file",
                "suffix": p.suffix,
                "size_mb": round(p.stat().st_size / 1e6, 3) if p.is_file() else None,
            }
        )

    st.dataframe(rows, use_container_width=True)

    previewable = [
        p for p in entries
        if p.is_file() and p.suffix in [".txt", ".yml", ".yaml", ".log", ".out"]
    ]

    if previewable:
        selected = st.selectbox("Preview file", previewable, format_func=lambda p: p.name)
        max_chars = st.slider("Preview max chars", 500, 20000, 5000, step=500)
        content = selected.read_text(errors="ignore")
        st.code(content[:max_chars], language="text")
else:
    st.warning("Folder does not exist.")