from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import export_e35_retained_oracle as exporter

PATHS = ["blocks/0/weight", "head/bias"]
# Exactly representable in FP32, so every expected error is exact.
ORACLE = [np.array([3.0, 4.0]), np.array([2.0, 0.0])]
FP32 = {
    "R": [np.array([3.0, 4.5]), np.array([2.0, 0.0])],
    "F": [np.array([3.0, 4.0]), np.array([2.0, 1.0])],
    "T": [np.array([3.5, 4.0]), np.array([2.0, 0.5])],
}


def _save(path: Path, arrays: list[np.ndarray], dtype: Any) -> None:
    np.savez(
        path,
        paths=np.asarray(PATHS),
        loss=np.asarray(1.0, dtype=dtype),
        logits=np.zeros((2, 3), dtype=dtype),
        **{f"gradient_{i:03d}": a.astype(dtype) for i, a in enumerate(arrays)},
    )


def _write_artifacts(directory: Path, **overrides: Any) -> None:
    protocol, config, digest = exporter.load_registration()
    checkpoints = exporter.inventory_sha256(config)
    for stage, dtype in (("float32", np.float32), ("float64", np.float64)):
        sources = []
        for source in config["source_variants"]:
            results = {}
            paths = FP32 if stage == "float32" else {"R": ORACLE}
            for variant, arrays in paths.items():
                name = f"e35_{stage}_{source}_{variant}.npz"
                _save(directory / name, arrays, dtype)
                results[variant] = {
                    "file": name,
                    "sha256": exporter.sha256(directory / name),
                    "gradient_dtype": np.dtype(dtype).name,
                    "logits_bitwise_equal_to_e07": True if stage == "float32" else None,
                }
            sources.append(
                {
                    "source_variant": source,
                    "checkpoint_path": f"artifacts/training/e10/{source}/final.pkl",
                    "checkpoint_sha256": checkpoints[source],
                    "batch_equal_to_e07": {"inputs": True, "targets": True},
                    "e07_code_commit": "e07",
                    "e07_device_id": "host::gpu:0::Tesla V100-PCIE-32GB",
                    "results": results,
                }
            )
        record = {
            "experiment_id": "E35",
            "stage": stage,
            "code_commit": "abc",
            "code_dirty": False,
            "protocol_version": protocol["protocol_version"],
            "protocol_path": "configs/protocols/protocol_1_20.yaml",
            "config_path": "configs/checkpoints/e35_retained_oracle.yaml",
            "config_hash": digest,
            "evaluated_paths": ["R", "F", "T"] if stage == "float32" else [exporter.ORACLE],
            "jax_enable_x64": stage == "float64",
            "projection_margin": 0.004 if stage == "float64" else None,
            "device": "cuda(id=0)",
            "device_platform": "gpu",
            "device_kind": "Tesla V100-PCIE-32GB",
            "cuda_visible_devices": "1",
            "gpu_snapshot_before": {"uuid": "GPU-test"},
            "jax_version": "0.4.25",
            "jaxlib_version": "0.4.25",
            "batch": {"examples": 2},
            "sources": sources,
        }
        record.update(overrides.get(stage, {}))
        (directory / f"e35_{stage}_record.json").write_text(json.dumps(record))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_exports_oracle_errors_and_recomputed_f_vs_r(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    summary = exporter.export(artifacts, tmp_path / "out")

    tensors = _rows(tmp_path / "out/e35_retained_oracle_tensor_errors.csv")
    assert len(tensors) == 3 * 3 * 2
    paths = _rows(tmp_path / "out/e35_retained_oracle_path_summary.csv")
    by_path = {row["path"]: row for row in paths if row["source_variant"] == "F"}
    assert float(by_path["R"]["median_tensor_relative_l2_error"]) == pytest.approx(0.05)
    assert float(by_path["R"]["max_tensor_relative_l2_error"]) == pytest.approx(0.1)
    assert by_path["F"]["max_tensor_parameter_path"] == "head/bias"
    assert float(by_path["F"]["max_tensor_relative_l2_error"]) == pytest.approx(0.5)
    assert float(by_path["T"]["median_tensor_relative_l2_error"]) == pytest.approx(0.175)
    assert float(by_path["F"]["whole_gradient_relative_l2_error"]) == pytest.approx(
        1.0 / math.sqrt(29.0)
    )
    assert (
        by_path["F"]["tensors_closer_than_R"],
        by_path["F"]["tensors_farther_than_R"],
    ) == (
        "1",
        "1",
    )
    assert (
        by_path["T"]["tensors_closer_than_R"],
        by_path["T"]["tensors_farther_than_R"],
    ) == (
        "0",
        "1",
    )

    agreement = _rows(tmp_path / "out/e35_retained_oracle_f_vs_r.csv")
    assert [row["source_variant"] for row in agreement] == ["R", "F", "T"]
    f_source = agreement[1]
    assert float(f_source["recomputed_relative_l2"]) == pytest.approx(
        math.sqrt(1.25 / 33.25)
    )
    # E07-J2 update-0 F-versus-R value for the F source state.
    assert float(f_source["e07_j2_relative_l2"]) == pytest.approx(0.06715245961573282)
    assert summary["identity"]["code_dirty"] is False
    assert summary["devices"]["float32"]["gpu_uuid"] == "GPU-test"


@pytest.mark.parametrize(
    ("overrides", "corrupt", "match"),
    [
        ({"float32": {"code_dirty": True}}, False, "registration"),
        ({"float64": {"config_hash": "0" * 64}}, False, "registration"),
        ({}, True, "changed after the run"),
    ],
)
def test_rejects_unregistered_or_modified_runs(
    tmp_path: Path, overrides: dict[str, Any], corrupt: bool, match: str
) -> None:
    _write_artifacts(tmp_path, **overrides)
    if corrupt:
        _save(tmp_path / "e35_float32_F_F.npz", FP32["R"], np.float32)
    with pytest.raises(ValueError, match=match):
        exporter.export(tmp_path, tmp_path / "out")
