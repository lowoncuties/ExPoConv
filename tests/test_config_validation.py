from pathlib import Path

import pytest

from poincare_exec.config.loader import load_yaml, validate_hashed_document

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_hashed_document_validation_accepts_matching_repository_file(tmp_path):
    document = tmp_path / "protocol.md"
    document.write_text("frozen\n", encoding="utf-8")
    config = {
        "design_document": {
            "path": "protocol.md",
            "sha256": "9c45da1b799a603c167323bd7ed4f52034ec28ea5e04373045a9c11f1ddc5446",
        }
    }

    path, digest = validate_hashed_document(
        config, field="design_document", repository_root=tmp_path
    )

    assert path == "protocol.md"
    assert digest == config["design_document"]["sha256"]


def test_hashed_document_validation_rejects_mismatch(tmp_path):
    document = tmp_path / "protocol.md"
    document.write_text("changed\n", encoding="utf-8")
    config = {
        "design_document": {
            "path": "protocol.md",
            "sha256": "0" * 64,
        }
    }

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_hashed_document(config, field="design_document", repository_root=tmp_path)


def test_hashed_document_validation_rejects_uppercase_digest(tmp_path):
    document = tmp_path / "protocol.md"
    document.write_text("frozen\n", encoding="utf-8")
    config = {
        "design_document": {
            "path": "protocol.md",
            "sha256": "A" * 64,
        }
    }

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_hashed_document(config, field="design_document", repository_root=tmp_path)


def test_e13_unet_scaling_is_the_frozen_one_factor_grid() -> None:
    config = load_yaml(REPOSITORY_ROOT / "configs/benchmarks/e13_unet_scaling.yaml")

    assert config["variants"] == ["R", "F"]
    assert config["sessions"] == 5
    assert config["warmup_calls"] == 50
    assert config["timed_calls_per_session"] == 200
    assert config["memory_budget_bytes"] == 32 * 1024**3

    workloads = config["workloads"]
    shapes = {
        (
            int(workload["shape"]["batch"]),
            int(workload["shape"]["height"]),
            int(workload["shape"]["width"]),
        )
        for workload in workloads
    }
    expected = {
        (1, 256, 256),
        (2, 256, 256),
        (4, 256, 256),
        (8, 256, 256),
        (16, 256, 256),
        (8, 128, 128),
        (8, 512, 512),
    }
    assert shapes == expected
    assert len(workloads) == len(expected)
    assert all(workload["kind"] == "unet" for workload in workloads)
    assert all(workload["mode"] == "update" for workload in workloads)
    assert all(workload["dtype"] == "float32" for workload in workloads)
