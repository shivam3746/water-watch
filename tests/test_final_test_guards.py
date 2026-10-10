import hashlib
import json

import pytest
import yaml

import scripts.run_final_2019 as runner
from scripts.download_battledim_2019 import checksums


def test_final_test_cannot_be_silently_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "final"
    output.mkdir()
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"output_dir": "final"}))
    (output / "frozen_manifest.json").write_text("{}")
    (output / "evaluation.json").write_text("{}")
    with pytest.raises(ValueError, match="already completed"):
        runner.evaluate(protocol)


def test_changed_freeze_is_rejected_before_loading_test_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "final"
    output.mkdir()
    train = tmp_path / "train.csv"
    train.write_text("unread training placeholder")
    model = output / "baseline_model.joblib"
    model.write_bytes(b"unread model placeholder")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"output_dir": "final", "train_path": "train.csv"}))
    manifest = {"protocol_sha256": "incorrect", "train_sha256": runner.checksum(train),
                "model_sha256": runner.checksum(model), "code_sha256": {}}
    (output / "frozen_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "read_battledim_csv", lambda *args: pytest.fail("Test measurements must not be read"))
    with pytest.raises(ValueError, match="changed"):
        runner.evaluate(protocol)


def test_download_checksum_verification_does_not_parse_contents(tmp_path):
    contents = b"test\nnot a parsed dataset"
    path = tmp_path / "file.csv"
    path.write_bytes(contents)
    result = checksums(path)
    assert result["md5"] == hashlib.md5(contents).hexdigest()
    assert result["sha256"] == hashlib.sha256(contents).hexdigest()
    assert result["bytes"] == len(contents)
