"""Streamlit Community Cloud entrypoint: no training or downloads at startup."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.deployment import verify_bundle

bundle = ROOT / "demo_bundle"


@st.cache_data
def check_bundle():
    return verify_bundle(bundle)


try:
    check_bundle()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Public demo bundle unavailable: {exc}")
    st.stop()

os.environ["WATER_WATCH_PUBLIC_DEMO"] = "1"
os.environ["WATER_WATCH_RESULTS"] = str(bundle / "development")
os.environ["WATER_WATCH_EXPLANATIONS"] = str(bundle / "explanations")
os.environ["WATER_WATCH_FINAL_RESULTS"] = str(bundle / "final_2019")
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
