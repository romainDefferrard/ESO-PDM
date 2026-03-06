import streamlit as st
from pathlib import Path
import sys
import yaml

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

cfg = st.session_state.app_config

repo_root = Path(cfg["repo_root"])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from lib.config_io import save_app_config
from lib.scenario_tools import get_effective_scenario_root
from navtools_PDM.sync_configs import sync_all_configs

st.title("Configure")

MODES = [
    "GeorefOnly",
    "Chunk",
    "OutageChunk",
    "Patcher",
    "PatcherLiMatch",
    "CycleSlip",
    "None",
]

METHODS = [
    "ref",
    "outage_only",
    "chunk",
    "patcher",
    "patcher_limatch",
    "cycleslip",
]

with st.form("configure_form"):
    st.subheader("Scenario")

    cfg["scenario"]["mode"] = st.selectbox(
        "Pipeline mode",
        MODES,
        index=MODES.index(cfg["scenario"]["mode"]),
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        cfg["scenario"]["method"] = st.selectbox(
            "Method",
            METHODS,
            index=METHODS.index(cfg["scenario"]["method"]) if cfg["scenario"]["method"] in METHODS else 0,
        )

    with col2:
        cfg["scenario"]["run_name"] = st.text_input(
            "Run name",
            value=cfg["scenario"]["run_name"],
        )

    with col3:
        cfg["project_root"] = st.text_input(
            "Project root",
            value=cfg["project_root"],
        )

    cfg["scenario"]["manual_scenario_root"] = st.checkbox(
        "Override scenario root manually",
        value=cfg["scenario"].get("manual_scenario_root", False),
    )

    auto_root = Path(cfg["project_root"]) / cfg["scenario"]["method"] / cfg["scenario"]["run_name"]
    if cfg["scenario"]["manual_scenario_root"]:
        cfg["outputs"]["scenario_root"] = st.text_input(
            "Scenario root",
            value=cfg["outputs"]["scenario_root"],
        )
    else:
        cfg["outputs"]["scenario_root"] = str(auto_root)
        st.text_input(
            "Scenario root",
            value=str(auto_root),
            disabled=True,
        )

    st.subheader("Inputs")
    mode = cfg["scenario"]["mode"]

    if mode in ["GeorefOnly", "Chunk", "OutageChunk"]:
        cfg["paths"]["traj_path"] = st.text_input("Trajectory path", value=cfg["paths"]["traj_path"])
        cfg["paths"]["sdc_ha"] = st.text_input("SDC HA folder", value=cfg["paths"]["sdc_ha"])
        cfg["paths"]["sdc_lr"] = st.text_input("SDC LR folder", value=cfg["paths"]["sdc_lr"])

    if mode in ["OutageChunk", "PatcherLiMatch", "CycleSlip"]:
        cfg["paths"]["gps_in"] = st.text_input("GPS input", value=cfg["paths"]["gps_in"])

    if mode in ["Chunk", "OutageChunk", "PatcherLiMatch"]:
        cfg["paths"]["limatch_cfg_template"] = st.text_input(
            "LiMatch template config",
            value=cfg["paths"]["limatch_cfg_template"],
        )

    if mode in ["Patcher", "PatcherLiMatch"]:
        cfg["paths"]["patcher_cfg_template"] = st.text_input(
            "Patcher template config",
            value=cfg["paths"]["patcher_cfg_template"],
        )

    if mode == "PatcherLiMatch":
        cfg["paths"]["patcher_out_root"] = st.text_input(
            "Patcher output root",
            value=cfg["paths"]["patcher_out_root"],
        )

    st.subheader("Outputs")
    cfg["outputs"]["merged_dirname"] = st.text_input("Merged dirname", value=cfg["outputs"]["merged_dirname"])
    cfg["outputs"]["tmp_dirname"] = st.text_input("Temporary dirname", value=cfg["outputs"]["tmp_dirname"])

    if mode in ["Chunk", "OutageChunk"]:
        cfg["outputs"]["chunks_dirname"] = st.text_input("Chunks dirname", value=cfg["outputs"]["chunks_dirname"])

    if mode in ["Chunk", "OutageChunk", "PatcherLiMatch"]:
        cfg["outputs"]["limatch_dirname"] = st.text_input("LiMatch dirname", value=cfg["outputs"]["limatch_dirname"])

    if mode in ["Patcher", "PatcherLiMatch"]:
        cfg["outputs"]["patcher_dirname"] = st.text_input("Patcher dirname", value=cfg["outputs"]["patcher_dirname"])

    if mode in ["GeorefOnly", "Chunk", "OutageChunk"]:
        st.subheader("Execution")
        cfg["execution"]["do_georef_merge"] = st.checkbox(
            "Run georef + merge",
            value=cfg["execution"]["do_georef_merge"],
        )
        cfg["execution"]["delete_tmp_after_success"] = st.checkbox(
            "Delete tmp after successful merge",
            value=cfg["execution"]["delete_tmp_after_success"],
        )

    if mode == "Chunk":
        st.subheader("Chunk options")
        cfg["chunk"]["length_m"] = st.number_input("Chunk length (m)", value=float(cfg["chunk"]["length_m"]))
        cfg["chunk"]["min_points"] = st.number_input("Chunk min points", value=int(cfg["chunk"]["min_points"]), step=100)
        cfg["chunk"]["neighbor_k"] = st.number_input("neighbor_k", value=int(cfg["chunk"]["neighbor_k"]), step=1)
        cfg["chunk"]["do_cross_scan"] = st.checkbox("Do cross scan matching", value=cfg["chunk"]["do_cross_scan"])

    if mode == "OutageChunk":
        st.subheader("Outage window")
        cfg["outage"]["start"] = st.number_input("Outage start", value=float(cfg["outage"]["start"]))
        cfg["outage"]["duration"] = st.number_input("Outage duration", value=float(cfg["outage"]["duration"]))
        cfg["outage"]["pre"] = st.number_input("Pre-outage buffer (s)", value=float(cfg["outage"]["pre"]))
        cfg["outage"]["post"] = st.number_input("Post-outage buffer (s)", value=float(cfg["outage"]["post"]))

        st.subheader("Chunk options")
        cfg["chunk"]["length_m"] = st.number_input("Chunk length (m)", value=float(cfg["chunk"]["length_m"]))
        cfg["chunk"]["min_points"] = st.number_input("Chunk min points", value=int(cfg["chunk"]["min_points"]), step=100)
        cfg["chunk"]["neighbor_k"] = st.number_input("neighbor_k", value=int(cfg["chunk"]["neighbor_k"]), step=1)
        cfg["chunk"]["do_cross_scan"] = st.checkbox("Do cross scan matching", value=cfg["chunk"]["do_cross_scan"])

        st.subheader("Outage execution")
        cfg["execution"]["do_chunks"] = st.checkbox("Create chunks", value=cfg["execution"]["do_chunks"])
        cfg["execution"]["reuse_chunks"] = st.checkbox("Reuse existing chunks", value=cfg["execution"]["reuse_chunks"])
        cfg["execution"]["force_rebuild_chunks"] = st.checkbox(
            "Force rebuild chunks",
            value=cfg["execution"]["force_rebuild_chunks"],
        )

    if mode == "PatcherLiMatch":
        st.subheader("Optional GPS outage generation")
        cfg["execution"]["do_gps_outage"] = st.checkbox(
            "Generate GPS outage file before LiMatch",
            value=cfg["execution"]["do_gps_outage"],
        )
        if cfg["execution"]["do_gps_outage"]:
            cfg["outage"]["start"] = st.number_input("Outage start", value=float(cfg["outage"]["start"]))
            cfg["outage"]["duration"] = st.number_input("Outage duration", value=float(cfg["outage"]["duration"]))

    with st.expander("Advanced options"):
        cfg["merge"]["delimiter"] = st.text_input("Merge delimiter", value=cfg["merge"]["delimiter"])
        cfg["merge"]["skiprows"] = st.number_input("Merge skiprows", value=int(cfg["merge"]["skiprows"]), step=1)
        cfg["merge"]["sort_by_time"] = st.checkbox("Sort merged clouds by time", value=cfg["merge"]["sort_by_time"])
        cfg["merge"]["out_prefix"] = st.text_input("Merge output prefix", value=cfg["merge"]["out_prefix"])
        cfg["merge"]["out_suffix"] = st.text_input("Merge output suffix", value=cfg["merge"]["out_suffix"])

    submitted = st.form_submit_button("Save config", use_container_width=True)

if submitted:
    st.session_state.app_config = cfg
    save_app_config(APP_DIR / "configs" / "app_config.yml", cfg)
    st.success("Config saved.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Synchronize all configs", use_container_width=True):
        try:
            result = sync_all_configs(cfg)
            st.session_state.last_sync_result = {k: str(v) for k, v in result.items()}
            save_app_config(APP_DIR / "configs" / "app_config.yml", cfg)
            st.success("All configs synchronized.")
            st.json(st.session_state.last_sync_result)
        except Exception as e:
            st.error("Synchronization failed.")
            st.exception(e)

with col2:
    st.write("Effective scenario root")
    st.code(str(get_effective_scenario_root(cfg)), language="text")

st.subheader("Current YAML")
st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")