from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from poincare_exec.artifacts.provenance import precision_policy
from poincare_exec.config.loader import canonical_json
from scripts import export_e31_gather_reference as exporter

LATENCY = {"R": 4.0, "Rg": 3.0, "Mg": 2.5, "F": 2.0}
PEAK_GIB = {"R": 12, "Rg": 10, "Mg": 9, "F": 6}
TEMPORARY_GIB = {"R": 8, "Rg": 6, "Mg": 5, "F": 3}
DEVICE = exporter.EXPECTED_DEVICE
LIBRARY_PATH = "/x/nvidia/cublas/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib"


def _environment() -> dict[str, Any]:
    return {**exporter.RUNTIME_ENVIRONMENT, "LD_LIBRARY_PATH": LIBRARY_PATH}


def _git(commit: str = exporter.EXPECTED_COMMIT, dirty: bool = False) -> dict[str, Any]:
    return {"commit": commit, "dirty": dirty, "status_porcelain": []}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_runtime(root: Path) -> None:
    device = {"id": 0, "platform": "gpu", "device_kind": DEVICE["name"]}
    _write_json(
        root / "e31_runtime_gate.json",
        {
            "jax_version": "0.4.25",
            "jaxlib_version": "0.4.25",
            "default_backend": "gpu",
            "devices": [device],
            "compiled_gpu_smoke": "passed",
            "environment": _environment(),
        },
    )
    _write_json(
        root / "e31_environment.json",
        {
            "hostname": DEVICE["hostname"],
            "git": _git(),
            "jax": {
                "version": "0.4.25",
                "jaxlib_version": "0.4.25",
                "backend": "gpu",
                "devices": [{**device, "memory_stats": {}}],
            },
            "packages": {"jax": "0.4.25", "jaxlib": "0.4.25+cuda11.cudnn86"},
            "environment": _environment(),
        },
    )


def _snapshot() -> str:
    return canonical_json(
        {
            "status": "ok",
            "process_query_status": "ok",
            "index": DEVICE["physical_index"],
            "uuid": DEVICE["uuid"],
            "name": DEVICE["name"],
            "memory.used": 10,
            "memory.total": 49140,
            "utilization.gpu": 0,
            "compute_processes": [],
        }
    )


def _identity(
    registration: dict[str, Any], workload_id: str, session: int, variant: str
) -> dict[str, Any]:
    workload = registration["workloads"][workload_id]["spec"]
    return {
        "experiment_id": "E31",
        "protocol_version": exporter.EXPECTED_PROTOCOL_VERSION,
        "variant": variant,
        "config_hash": registration["workloads"][workload_id]["config_hash"],
        "code_commit": exporter.EXPECTED_COMMIT,
        "code_dirty": False,
        "reference_commit": exporter.EXPECTED_REFERENCE_COMMIT,
        "device_id": f"{DEVICE['hostname']}::gpu:0::{DEVICE['name']}",
        "precision_policy": precision_policy(workload["dtype"]),
        "seed_or_session": session,
        "workload_id": workload_id,
    }


def _raw_group(
    registration: dict[str, Any], workload_id: str, session: int, variant: str
) -> list[dict[str, Any]]:
    config = registration["config"]
    workload = registration["workloads"][workload_id]["spec"]
    shared = {
        **_identity(registration, workload_id, session, variant),
        "status": "ok",
        "shape": canonical_json(workload["shape"]),
        "coordinate_boundary": "tangent_to_tangent",
        "operation_boundary": f"{workload['kind']}:{workload['mode']}:depth=1",
        "warmup_count": config["warmup_calls"],
        "repeat_count": config["timed_calls_per_session"],
        "state_evolves_across_calls": workload["mode"] == "update",
        "state_summary_before": canonical_json({"nonfinite_count": 0}),
        "state_summary_after": canonical_json({"nonfinite_count": 0}),
        "output_nonfinite_count": 0,
        "measured_peak_bytes_in_use": PEAK_GIB[variant] * 1024**3,
        "measured_peak_pool_bytes": (PEAK_GIB[variant] + 1) * 1024**3,
        "process_reservation_mib": PEAK_GIB[variant] * 1024,
        "compiler_argument_bytes": 1024**2,
        "compiler_output_bytes": 1024**2,
        "compiler_temporary_bytes": TEMPORARY_GIB[variant] * 1024**3,
        "compiler_alias_bytes": 0,
        "gpu_snapshot_before": _snapshot(),
        "gpu_snapshot_after": _snapshot(),
    }
    rows = [
        {**shared, "observation": -2, "phase": "compilation", "latency_ms": 20.0},
        {**shared, "observation": -1, "phase": "first_compiled_call", "latency_ms": 10.0},
        {
            **shared,
            "status": "not_applicable",
            "observation": -3,
            "phase": "sustained_sequence",
            "latency_ms": "",
        },
    ]
    rows.extend(
        {
            **shared,
            "observation": observation,
            "phase": "steady_state",
            "latency_ms": LATENCY[variant] * (1.0 + 0.01 * session) + 1.0e-4 * observation,
        }
        for observation in range(int(config["timed_calls_per_session"]))
    )
    return rows


def _failure(
    registration: dict[str, Any], workload_id: str, session: int, variant: str
) -> dict[str, Any]:
    workload = registration["workloads"][workload_id]["spec"]
    snapshot = json.loads(_snapshot())
    return {
        **_identity(registration, workload_id, session, variant),
        "status": "failed",
        "shape": workload["shape"],
        "operation_boundary": f"{workload['kind']}:{workload['mode']}",
        "returncode": 1,
        "failure_kind": "oom",
        "failure_phase": "compilation",
        "gpu_snapshot_before": snapshot,
        "gpu_snapshot_after": snapshot,
        "stdout": "",
        "stderr": "RESOURCE_EXHAUSTED: Out of memory",
    }


def _comparison(equal: bool = True) -> dict[str, Any]:
    return {
        "bitwise_equal": equal,
        "differing_elements": 0 if equal else 3,
        "elements": 100,
        "max_abs_difference": 0.0 if equal else 1.0e-7,
        "max_abs_difference_over_reference_max_abs": 0.0 if equal else 1.0e-7,
    }


def _write_captures(root: Path, registration: dict[str, Any]) -> None:
    common = {
        "git": _git(),
        "jax_version": "0.4.25",
        "jaxlib_version": "0.4.25",
        "environment": _environment(),
    }
    workload_hash = registration["workloads"]["unet_b8_update"]["config_hash"]
    captures = [
        {
            "variant": variant,
            "workload_id": "unet_b8_update",
            "config_hash": workload_hash,
            "device_kind": DEVICE["name"],
            "compiled_bytes": {
                "argument": 1024**2,
                "output": 1024**2,
                "alias": 0,
                "temp": TEMPORARY_GIB[variant] * 1024**3,
                "generated_code": 1024,
            },
            "optimized_hlo": {
                "cudnn_convolution_calls": 40,
                "cudnn_scratch_bytes_sum": 1024**3,
                "cudnn_scratch_bytes_max": 1024**2,
            },
            "allocator_after_state": {"bytes_in_use": 1024**2},
            "allocator_after_compile": {"peak_bytes_in_use": 2 * 1024**3},
        }
        for variant in ("R", "Rg", "F")
    ]
    _write_json(
        root / exporter.MEMORY_CAPTURE,
        {**common, "workload_id": "unet_b8_update", "captures": captures},
    )
    patch = {
        "input_shape": [8, 256, 256, 3],
        "dtype": "float32",
        "kernel_shape": [3, 3],
        "stride": [1, 1],
        "padding": "SAME",
        "dilation": [1, 1],
    }
    _write_json(
        root / exporter.EQUIVALENCE_CAPTURE,
        {
            **common,
            "device_kind": DEVICE["name"],
            "patch_comparisons": [
                {"workload_id": workload_id, **patch, **_comparison()}
                for workload_id in registration["workloads"]
            ],
            "model_comparisons": [
                {
                    "workload_id": workload_id,
                    "mode": registration["workloads"][workload_id]["spec"]["mode"],
                    "reference_variant": reference,
                    "candidate_variant": candidate,
                    **_comparison(reference == "R"),
                }
                for workload_id in registration["workloads"]
                for reference, candidate in exporter.EQUIVALENCE_PAIRS
            ],
        },
    )


def _write_artifacts(
    root: Path,
    *,
    failed: tuple[str, int, str] | None = None,
    omitted: tuple[str, int, str] | None = None,
) -> None:
    _write_runtime(root)
    _, registration = exporter._load_registration(exporter.PROTOCOL_PATH)
    rows = []
    failures = []
    config = registration["config"]
    for workload_id in registration["workloads"]:
        for session in range(config["sessions"]):
            for variant in config["variants"]:
                key = (workload_id, session, variant)
                if key == omitted:
                    continue
                if key == failed:
                    failures.append(_failure(registration, *key))
                else:
                    rows.extend(_raw_group(registration, *key))
    with (root / exporter.RAW).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if failures:
        (root / f"{exporter.RAW}.failures.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in failures), encoding="utf-8"
        )
    _write_captures(root, registration)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_registration_binds_protocol_contrasts_and_variants() -> None:
    _, registration = exporter._load_registration(exporter.PROTOCOL_PATH)

    assert registration["config"]["variants"] == list(exporter.VARIANT_ROLES)
    assert list(registration["workloads"]) == [
        "unet_b1_forward",
        "unet_b8_forward",
        "unet_b8_update",
        "resnet20_b128_update",
    ]


def test_exports_paired_contrasts_memory_and_equivalence(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    output = tmp_path / "out"

    summary = exporter.export(artifacts, output)

    assert summary["output_rows"] == {
        "e31_session_outcomes.csv": 80,
        "e31_variant_summary.csv": 16,
        "e31_contrasts.csv": 20,
        "e31_compiled_memory.csv": 3,
        "e31_gpu_equivalence.csv": 12,
    }
    contrasts = {
        (row["workload_id"], row["reference_variant"], row["candidate_variant"]): row
        for row in _read(output / "e31_contrasts.csv")
    }
    for (reference, candidate), _ in exporter.CONTRAST_ROLES.items():
        row = contrasts[("unet_b8_update", reference, candidate)]
        expected = LATENCY[reference] / LATENCY[candidate]
        assert math.isclose(
            float(row["geometric_mean_latency_ratio"]), expected, rel_tol=5.0e-3
        )
        assert float(row["log_t_95_lower"]) <= float(row["geometric_mean_latency_ratio"])
        assert float(row["log_t_95_upper"]) >= float(row["geometric_mean_latency_ratio"])
        assert math.isclose(
            float(row["geometric_mean_peak_live_ratio"]),
            PEAK_GIB[reference] / PEAK_GIB[candidate],
        )
        assert row["accepted_paired_sessions"] == "5"
    variants = {
        (row["workload_id"], row["variant"]): row
        for row in _read(output / "e31_variant_summary.csv")
    }
    assert float(variants[("unet_b8_update", "Rg")]["median_compiler_temporary_gib"]) == 6.0
    memory = {row["variant"]: row for row in _read(output / "e31_compiled_memory.csv")}
    assert list(memory) == ["R", "Rg", "F"]
    assert float(memory["F"]["compiled_temporary_gib"]) == 3.0
    assert float(memory["F"]["benchmark_median_peak_live_gib"]) == 6.0
    audit = json.loads((output / "e31_audit.json").read_text(encoding="utf-8"))
    assert audit["gpu_equivalence"]["model_comparisons_bitwise_equal"] == 4
    assert audit["cohort"]["outcomes"] == {"ok": 80}


def test_failed_group_is_retained_and_excluded_from_pairs(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts, failed=("resnet20_b128_update", 2, "Rg"))

    exporter.export(artifacts, tmp_path / "out")

    sessions = _read(tmp_path / "out" / "e31_session_outcomes.csv")
    failed = [row for row in sessions if row["outcome"] == "failed"]
    assert [(row["workload_id"], row["session"], row["variant"]) for row in failed] == [
        ("resnet20_b128_update", "2", "Rg")
    ]
    assert failed[0]["failure_kind"] == "oom"
    contrasts = {
        (row["workload_id"], row["reference_variant"], row["candidate_variant"]): row
        for row in _read(tmp_path / "out" / "e31_contrasts.csv")
    }
    excluded = contrasts[("resnet20_b128_update", "R", "Rg")]
    assert excluded["accepted_paired_sessions"] == "4"
    assert json.loads(excluded["excluded_session_outcomes"]) == {
        "2": {"R": "ok", "Rg": "failed"}
    }
    assert contrasts[("resnet20_b128_update", "R", "F")]["accepted_paired_sessions"] == "5"


def test_rejects_missing_group(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts, omitted=("unet_b1_forward", 4, "Mg"))

    with pytest.raises(ValueError, match="outcome matrix mismatch"):
        exporter.export(artifacts, tmp_path / "out")


def test_rejects_run_from_another_commit(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)

    with pytest.raises(ValueError, match="clean"):
        exporter.export(artifacts, tmp_path / "out", expected_commit="f" * 40)


def test_rejects_memory_capture_with_wrong_variants(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    path = artifacts / exporter.MEMORY_CAPTURE
    capture = json.loads(path.read_text(encoding="utf-8"))
    capture["captures"] = capture["captures"][:2]
    path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="expected variants"):
        exporter.export(artifacts, tmp_path / "out")


def test_rejects_dirty_equivalence_capture(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    path = artifacts / exporter.EQUIVALENCE_CAPTURE
    capture = json.loads(path.read_text(encoding="utf-8"))
    capture["git"]["dirty"] = True
    path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="clean"):
        exporter.export(artifacts, tmp_path / "out")


def test_refuses_to_overwrite_existing_tables(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    output = tmp_path / "out"
    exporter.export(artifacts, output)

    with pytest.raises(FileExistsError):
        exporter.export(artifacts, output)
