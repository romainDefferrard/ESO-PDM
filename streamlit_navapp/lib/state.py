import streamlit as st


def ensure_state():
    defaults = {
        "run_log": "",
        "last_result": {},
        "last_sync_result": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value