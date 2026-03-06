import streamlit as st
from pathlib import Path
import sys
import traceback

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

cfg = st.session_state.app_config

repo_root = Path(cfg["repo_root"])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from navtools_PDM.pipeline_app import run_pipeline_app

st.title("Run Pipeline")

st.write("Mode:", cfg["scenario"]["mode"])
st.write("Method:", cfg["scenario"]["method"])
st.write("Run:", cfg["scenario"]["run_name"])
st.write("Scenario root:", cfg["outputs"]["scenario_root"])

auto_sync = st.checkbox(
    "Synchronize configs before run",
    value=True,
)

if st.button("Run pipeline", type="primary", use_container_width=True):
    try:
        result = run_pipeline_app(
            app_cfg=cfg,
            auto_sync=auto_sync,
        )
        st.session_state.last_result = result
        st.session_state.run_log = "Pipeline finished successfully."
        st.success("Pipeline finished successfully.")
        st.json(result)

    except Exception as e:
        st.session_state.run_log = traceback.format_exc()
        st.error("Pipeline failed.")
        st.exception(e)

st.subheader("Run log")
st.code(st.session_state.get("run_log", ""), language="text")