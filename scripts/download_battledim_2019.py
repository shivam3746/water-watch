"""Download official files without parsing held-out sensor values or labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://zenodo.org/records/4017659"
# Checksums published by the dataset authors on the official Zenodo record.
FILES = {
    "2019_SCADA_Pressures.csv": "5ea1e46d3f2f0a89a3f98d6fd39a851d",
    "2019_SCADA_Flows.csv": "28fc99fdcbf80fcd26079e7fe602d6dc",
    "2019_Leakages.csv": "e1f0a43683813a90a9fec8562dde599b",
}


def checksums(path: Path) -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def download(raw: Path = ROOT / "data/raw") -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE, "doi": "10.5281/zenodo.4017659", "contents_parsed": False, "files": {}}
    for name, expected in FILES.items():
        target = raw / name
        url = f"{SOURCE}/files/{name}?download=1"
        if not target.exists():
            temporary = target.with_suffix(".csv.part")
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "WaterWatchResearch/1.0"})
                    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as destination:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            destination.write(chunk)
                    if checksums(temporary)["md5"] != expected:
                        raise ValueError(f"Official checksum mismatch for {name}; file not installed.")
                    temporary.replace(target)
                    break
                except (OSError, ValueError):
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))
        verified = checksums(target)
        if verified["md5"] != expected:
            raise ValueError(f"Existing {name} differs from the official file; not overwritten.")
        manifest["files"][name] = {"url": url, **verified}
        print(f"Verified {name}: {verified['bytes']} bytes", flush=True)
    (raw / "2019_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    download()
