import hashlib
import json
from pathlib import Path

import pytest

from poincare_exec.config.loader import config_hash, load_yaml, validate_hashed_document

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    [
        (
            "configs/numerics/e02_structured.yaml",
            "e1b6b624be66d1a7352b4b74925ccb4876f22714ce94cd1540e658ac83ebf60b",
        ),
        (
            "docs/journal_diagnostic_protocol.md",
            "5f82bbadb1ed267a2c01506d6caa221c9313d7be3ccb7648c141beaab23833a9",
        ),
    ],
)
def test_restored_historical_protocol_inputs_are_byte_exact(
    relative_path: str, expected_sha256: str
) -> None:
    content = (REPOSITORY_ROOT / relative_path).read_bytes()
    assert hashlib.sha256(content).hexdigest() == expected_sha256


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


def test_e20_e21_protocol_pin_resolves_to_the_byte_exact_snapshot() -> None:
    manifest = load_yaml(
        REPOSITORY_ROOT / "configs" / "analysis" / "e20_e21_training_replication.yaml"
    )
    protocol_path, protocol_digest = validate_hashed_document(
        manifest, field="protocol_document"
    )
    protocol = load_yaml(REPOSITORY_ROOT / protocol_path)
    cohort = manifest["cohorts"]["E20"]
    config = load_yaml(REPOSITORY_ROOT / cohort["config_document"]["path"])

    assert protocol_path == "configs/protocols/protocol_1_14.yaml"
    assert protocol_digest == manifest["protocol_document"]["sha256"]
    assert protocol["protocol_version"] == "1.14"
    assert config_hash(protocol, config) == cohort["run_config_hash"]

    wrong_pin = {**manifest["protocol_document"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_hashed_document(
            {"protocol_document": wrong_pin}, field="protocol_document"
        )


def test_e13_analysis_uses_the_frozen_protocol_snapshot() -> None:
    manifest = load_yaml(REPOSITORY_ROOT / "configs" / "analysis" / "e13_unet_scaling.yaml")
    protocol_path, protocol_digest = validate_hashed_document(
        manifest, field="protocol_document"
    )
    benchmark_path, _ = validate_hashed_document(manifest, field="benchmark_config")
    allocator_path, allocator_digest = validate_hashed_document(
        manifest, field="allocator_limit_evidence"
    )
    protocol = load_yaml(REPOSITORY_ROOT / protocol_path)
    benchmark = load_yaml(REPOSITORY_ROOT / benchmark_path)
    environment = json.loads((REPOSITORY_ROOT / allocator_path).read_text(encoding="utf-8"))
    accepted_summary = json.loads(
        (REPOSITORY_ROOT / "results" / "e13_scaling" / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert protocol_path == "configs/protocols/protocol_1_11.yaml"
    assert protocol_digest == (
        "5dbd77ce1d6fa7e9a33114093a730208048bcf295012b66c0df00f6e3e8711fe"
    )
    assert protocol["protocol_version"] == "1.11"
    assert allocator_path == "results/environment.json"
    assert allocator_digest == (
        "3a7a509ad6f04d197ede3a75c9c9aec0d6e3778e8ab3bf0d221d9a6602cabd6a"
    )
    assert environment["hardware"]["jax_allocator_bytes_limit"] == 25_567_444_992
    assert accepted_summary["memory_reference"]["bytes"] == 25_567_444_992
    assert {
        workload["id"]: config_hash(protocol, benchmark, workload)
        for workload in benchmark["workloads"]
    } == accepted_summary["provenance_contract"]["expected_config_hashes"]
