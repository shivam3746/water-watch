import hashlib
import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.deployment import public_review_workspace, verify_bundle


def test_public_reviews_are_session_isolated_and_reused():
    first, second = {}, {}
    a = public_review_workspace(first)
    assert public_review_workspace(first) == a
    assert public_review_workspace(second) != a
    assert a.name == "reviews.sqlite"
    first["water_watch_public_workspace"].cleanup()
    second["water_watch_public_workspace"].cleanup()


def test_bundle_verification_detects_corruption_and_invalid_paths(tmp_path):
    payload = b"public results"
    (tmp_path / "result.csv").write_bytes(payload)
    manifest = {"files": {"result.csv": {"sha256": hashlib.sha256(payload).hexdigest()}}}
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path) == manifest
    (tmp_path / "result.csv").write_text("modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(tmp_path)
    manifest["files"] = {"../outside.csv": {"sha256": "invalid"}}
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid"):
        verify_bundle(tmp_path)


def test_public_entrypoint_uses_bundle_and_private_temporary_reviews(monkeypatch):
    for name in ["WATER_WATCH_PUBLIC_DEMO", "WATER_WATCH_RESULTS", "WATER_WATCH_EXPLANATIONS", "WATER_WATCH_FINAL_RESULTS"]:
        monkeypatch.setenv(name, os.environ.get(name, ""))
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "deployment/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert "water_watch_public_workspace" in app.session_state
    workspace = app.session_state["water_watch_public_workspace"]
    assert not Path(workspace.name).is_relative_to(root)
    assert any("isolated to this browser session" in item.value for item in app.info)
    assert any("2/19" in item.value for item in app.warning)
    assert len(app.tabs) == 7
    workspace.cleanup()
