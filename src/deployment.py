from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile


def verify_bundle(directory: Path) -> dict:
    manifest = json.loads((directory / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid demo bundle file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Demo bundle checksum mismatch: {name}")
    return manifest


def public_review_workspace(state) -> Path:
    """TemporaryDirectory is session-owned and cleans up when its owner is released."""
    if "water_watch_public_workspace" not in state:
        state["water_watch_public_workspace"] = tempfile.TemporaryDirectory(prefix="water-watch-demo-")
    return Path(state["water_watch_public_workspace"].name) / "reviews.sqlite"
