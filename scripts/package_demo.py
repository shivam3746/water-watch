from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def package(root: Path = ROOT) -> dict:
    target = root / "demo_bundle"
    if target.exists():
        raise ValueError("demo_bundle already exists. Use a reviewed new bundle rather than silently overwriting it.")
    source = root / "artifacts"
    files = []
    development = source / "results/validation_v1"
    for name in ["summary.csv", "fold_metrics.csv", "event_audit.csv", "alarm_audit.csv", "calibration_audit.csv", "run_manifest.json", "protocol_snapshot.yaml"]:
        files.append((development / name, Path("development") / name))
    development_manifest = json.loads((development / "run_manifest.json").read_text())
    for month in development_manifest["selected_folds"]:
        files.append((development / month / "timeline.csv.gz", Path("development") / month / "timeline.csv.gz"))
    explanation = source / "explanations"
    records = json.loads((explanation / "incidents.json").read_text()) + json.loads((explanation / "cases.json").read_text())
    names = {"incidents.json", "cases.json", "manifest.json"}
    for record in records:
        names.update(record[key] for key in ("evidence_file", "trace_file", "plot_file"))
    files += [(explanation / name, Path("explanations") / name) for name in sorted(names)]
    final = source / "results/final_2019"
    for name in ["evaluation.json", "frozen_manifest.json", "prediction_manifest.json", "report.md", "event_audit.csv", "alarm_audit.csv"]:
        files.append((final / name, Path("final_2019") / name))
    for path, relative in files:
        if not path.is_file() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Unavailable demo input: {path}")
        if any(part == ".." for part in relative.parts) or path.suffix not in {".csv", ".gz", ".json", ".yaml", ".md", ".png"}:
            raise ValueError("Only reviewed results, traces, plots and provenance are packaged.")
    manifest = {"schema_version": 1, "dataset_doi": "10.5281/zenodo.4017659", "dataset_license": "CC-BY-4.0",
                "source": "https://zenodo.org/records/4017659", "files": {},
                "excluded": ["raw datasets", "trained model pickles", "review databases", "reviewer decisions", "secrets"],
                "purpose": "Historical research demo; not live monitoring or an operationally validated detector."}
    for path, relative in files:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest["files"][relative.as_posix()] = {"sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "bytes": destination.stat().st_size}
    (target / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f'Packaged {len(files)} files, {sum(item["bytes"] for item in manifest["files"].values()) / 1024**2:.2f} MiB in {target}')
    return manifest


if __name__ == "__main__":
    package()
