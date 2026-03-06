import streamlit as st
from pathlib import Path
import sys
import yaml

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib.state import ensure_state
from lib.config_io import load_app_config, save_app_config
from lib.scenario_tools import get_effective_scenario_root

st.set_page_config(
    page_title="MLS Pipeline Control",
    page_icon="🛰️",
    layout="wide",
)

ensure_state()

cfg_path = APP_DIR / "configs" / "app_config.yml"
if "app_config" not in st.session_state:
    st.session_state.app_config = load_app_config(cfg_path)

cfg = st.session_state.app_config
scenario_root = get_effective_scenario_root(cfg)

st.title("MLS Pipeline Control")
st.caption("Centralized control panel for MLS scenarios, synchronized configs, and pipeline execution.")

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.0])

with col1:
    st.metric("Project root", cfg.get("project_root", ""))

with col2:
    st.metric("Method", cfg.get("scenario", {}).get("method", ""))

with col3:
    st.metric("Run", cfg.get("scenario", {}).get("run_name", ""))

with col4:
    if st.button("Save master config", use_container_width=True):
        save_app_config(cfg_path, cfg)
        st.success("Master config saved.")

st.subheader("Effective scenario root")
st.code(str(scenario_root), language="text")

st.subheader("Current config preview")
st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")