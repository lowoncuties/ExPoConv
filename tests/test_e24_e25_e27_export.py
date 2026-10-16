from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from poincare_exec.artifacts.provenance import precision_policy
from poincare_exec.config.loader import canonical_json
from scripts import export_e24_e25_e27_tables as exporter
from scripts import plot_e25_cost_context as cost_figure


def _runtime_environment(ld_library_path: str | None) -> dict[str, Any]:
    return {
        "CUDA_VISIBLE_DEVICES": "1",
        "JAX_PLATFORMS": None,
        "LD_LIBRARY_PATH": ld_library_path,
        "NVIDIA_TF32_OVERRIDE": None,
        "XLA_FLAGS": None,
        "XLA_PYTHON_CLIENT_ALLOCATOR": None,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def _write_runtime_capture(
    root: Path,
    *,
    stem: str,
    jax_version: str,
    ld_library_path: str | None,
) -> None:
    device = {
        "id": 0,
        "platform": "gpu",
        "device_kind": exporter.EXPECTED_DEVICE["name"],
    }
    variables = _runtime_environment(ld_library_path)
    (root / f"{stem}_runtime_gate.json").write_text(
        json.dumps(
            {
                "jax_version": jax_version,
                "jaxlib_version": jax_version,
                "default_backend": "gpu",
                "devices": [device],
                "compiled_gpu_smoke": "passed",
                "environment": variables,
            }
        ),
        encoding="utf-8",
    )
    (root / f"{stem}_environment.json").write_text(
        json.dumps(
            {
                "hostname": exporter.EXPECTED_DEVICE["hostname"],
                "git": {
                    "commit": exporter.EXPECTED_COMMIT,
                    "dirty": False,
                    "status_porcelain": [],
                },
                "jax": {
                    "version": jax_version,
                    "jaxlib_version": jax_version,
                    "backend": "gpu",
                    "devices": [{**device, "memory_stats": {}}],
                },
                "packages": {
                    "jax": jax_version,
                    "jaxlib": (
                        "0.4.25+cuda11.cudnn86" if jax_version == "0.4.25" else jax_version
                    ),
                },
                "environment": variables,
            }
        ),
        encoding="utf-8",
    )


def _snapshot() -> str:
    return canonical_json(
        {
            "status": "ok",
            "process_query_status": "ok",
            "index": exporter.EXPECTED_DEVICE["physical_index"],
            "uuid": exporter.EXPECTED_DEVICE["uuid"],
            "name": exporter.EXPECTED_DEVICE["name"],
            "memory.used": 10,
            "memory.total": 49140,
            "utilization.gpu": 0,
            "compute_processes": [],
        }
    )


def _raw_group(
    *,
    experiment_id: str,
    registration: dict[str, Any],
    workload_id: str,
    session: int,
    variant: str,
) -> list[dict[str, Any]]:
    config = registration["config"]
    workload = registration["workloads"][workload_id]["spec"]
    variant_latency = {"R": 4.0, "M": 3.0, "F": 2.0, "E": 1.0}[variant]
    peak_gib = {"R": 12, "M": 10, "F": 6, "E": 4}[variant]
    shared = {
        "experiment_id": experiment_id,
        "protocol_version": exporter.EXPECTED_PROTOCOL_VERSION,
        "variant": variant,
        "config_hash": registration["workloads"][workload_id]["config_hash"],
        "code_commit": exporter.EXPECTED_COMMIT,
        "code_dirty": False,
        "reference_commit": exporter.EXPECTED_REFERENCE_COMMIT,
        "device_id": (
            f"{exporter.EXPECTED_DEVICE['hostname']}::gpu:0::"
            f"{exporter.EXPECTED_DEVICE['name']}"
        ),
        "precision_policy": precision_policy(workload["dtype"]),
        "seed_or_session": session,
        "status": "ok",
        "workload_id": workload_id,
        "shape": canonical_json(workload["shape"]),
        "coordinate_boundary": (
            "euclidean_to_euclidean" if variant == "E" else "tangent_to_tangent"
        ),
        "operation_boundary": f"{workload['kind']}:{workload['mode']}:depth=1",
        "warmup_count": config["warmup_calls"],
        "repeat_count": config["timed_calls_per_session"],
        "state_evolves_across_calls": workload["mode"] == "update",
        "state_summary_before": canonical_json({"nonfinite_count": 0}),
        "state_summary_after": canonical_json({"nonfinite_count": 0}),
        "output_nonfinite_count": 0,
        "measured_peak_bytes_in_use": peak_gib * 1024**3,
        "measured_peak_pool_bytes": (peak_gib + 1) * 1024**3,
        "process_reservation_mib": peak_gib * 1024,
        "gpu_snapshot_before": _snapshot(),
        "gpu_snapshot_after": _snapshot(),
    }
    rows = [
        {**shared, "observation": -2, "phase": "compilation", "latency_ms": 20.0},
        {
            **shared,
            "observation": -1,
            "phase": "first_compiled_call",
            "latency_ms": 10.0,
        },
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
            "latency_ms": (variant_latency + 0.01 * session + 0.0001 * observation),
        }
        for observation in range(int(config["timed_calls_per_session"]))
    )
    return rows


def _failure(
    *,
    experiment_id: str,
    registration: dict[str, Any],
    workload_id: str,
    session: int,
    variant: str,
) -> dict[str, Any]:
    workload = registration["workloads"][workload_id]["spec"]
    snapshot = json.loads(_snapshot())
    return {
        "experiment_id": experiment_id,
        "protocol_version": exporter.EXPECTED_PROTOCOL_VERSION,
        "variant": variant,
        "config_hash": registration["workloads"][workload_id]["config_hash"],
        "code_commit": exporter.EXPECTED_COMMIT,
        "code_dirty": False,
        "reference_commit": exporter.EXPECTED_REFERENCE_COMMIT,
        "device_id": (
            f"{exporter.EXPECTED_DEVICE['hostname']}::gpu:0::"
            f"{exporter.EXPECTED_DEVICE['name']}"
        ),
        "precision_policy": precision_policy(workload["dtype"]),
        "seed_or_session": session,
        "status": "failed",
        "workload_id": workload_id,
        "shape": workload["shape"],
        "operation_boundary": f"{workload['kind']}:{workload['mode']}",
        "returncode": 1,
        "failure_kind": "failed",
        "failure_phase": "compilation",
        "gpu_snapshot_before": snapshot,
        "gpu_snapshot_after": snapshot,
        "stdout": "retained standard output",
        "stderr": "retained failure diagnostic",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(
    root: Path,
    *,
    failed_group: tuple[str, str, int, str] | None = None,
    omitted_group: tuple[str, str, int, str] | None = None,
) -> None:
    _write_runtime_capture(
        root,
        stem="e24_e25",
        jax_version="0.4.25",
        ld_library_path="/usr/local/cuda-11.8/targets/x86_64-linux/lib",
    )
    _write_runtime_capture(
        root,
        stem="e27",
        jax_version="0.11.2",
        ld_library_path=None,
    )
    _, registrations = exporter._load_registration(Path("protocol.yaml").resolve())
    for experiment_id, cohort in exporter.COHORTS.items():
        registration = registrations[experiment_id]
        rows = []
        failures = []
        for workload_id in registration["workloads"]:
            for session in range(int(registration["config"]["sessions"])):
                for variant in registration["config"]["variants"]:
                    key = (experiment_id, workload_id, session, variant)
                    if key == omitted_group:
                        continue
                    if key == failed_group:
                        failures.append(
                            _failure(
                                experiment_id=experiment_id,
                                registration=registration,
                                workload_id=workload_id,
                                session=session,
                                variant=variant,
                            )
                        )
                    else:
                        rows.extend(
                            _raw_group(
                                experiment_id=experiment_id,
                                registration=registration,
                                workload_id=workload_id,
                                session=session,
                                variant=variant,
                            )
                        )
        _write_csv(root / cohort["raw"], rows)
        if failures:
            failure_path = root / f"{cohort['raw']}.failures.jsonl"
            failure_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures),
                encoding="utf-8",
            )


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_exports_complete_matrices_roles_and_registered_contrasts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    _write_artifacts(artifacts)

    summary = exporter.export(artifacts, output)

    assert summary["output_rows"] == {
        "e24_e25_e27_session_outcomes.csv": 145,
        "e24_e25_e27_variant_summary.csv": 29,
        "e24_e25_e27_contrasts.csv": 25,
    }
    sessions = _csv(output / exporter.OUTPUT_FILES["sessions"])
    variants = _csv(output / exporter.OUTPUT_FILES["variants"])
    contrasts = _csv(output / exporter.OUTPUT_FILES["contrasts"])
    assert {row["outcome"] for row in sessions} == {"ok"}
    assert len({row["config_hash"] for row in sessions}) == 10
    assert {row["variant_role"] for row in variants if row["experiment_id"] == "E24"} == {
        "reference_materialized_ball_geometry",
        "materialized_patch_exact_score_control",
        "factorized_primary_method",
    }
    assert {row["contrast_role"] for row in contrasts if row["experiment_id"] == "E24"} == {
        "geometry_removal_with_patch_materialization_fixed",
        "patch_materialization_and_backend_with_geometry_removed",
    }
    assert {row["contrast_role"] for row in contrasts if row["experiment_id"] == "E25"} == {
        "factorized_vs_reference_poincare",
        "reference_poincare_cost_relative_to_euclidean",
        "factorized_poincare_cost_relative_to_euclidean",
    }
    e27 = next(row for row in contrasts if row["experiment_id"] == "E27")
    assert e27["runtime_role"] == "current_jax_0_11_2_cuda_12_portability"
    assert e27["jax_version"] == "0.11.2"
    assert int(e27["accepted_paired_sessions"]) == 5
    assert float(e27["geometric_mean_latency_ratio"]) > 1.0


def test_retains_failure_and_excludes_only_affected_pair(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    failed = ("E24", "unet_b1_forward", 0, "M")
    _write_artifacts(artifacts, failed_group=failed)

    exporter.export(artifacts, output)

    sessions = _csv(output / exporter.OUTPUT_FILES["sessions"])
    failure = next(
        row
        for row in sessions
        if (
            row["experiment_id"],
            row["workload_id"],
            int(row["session"]),
            row["variant"],
        )
        == failed
    )
    assert failure["outcome"] == "failed"
    assert failure["failure_phase"] == "compilation"
    assert failure["failure_stderr"] == "retained failure diagnostic"
    variants = _csv(output / exporter.OUTPUT_FILES["variants"])
    materialized = next(
        row
        for row in variants
        if row["experiment_id"] == "E24"
        and row["workload_id"] == "unet_b1_forward"
        and row["variant"] == "M"
    )
    assert int(materialized["failed_sessions"]) == 1
    assert int(materialized["accepted_finite_sessions"]) == 4
    contrasts = _csv(output / exporter.OUTPUT_FILES["contrasts"])
    geometry = next(
        row
        for row in contrasts
        if row["experiment_id"] == "E24"
        and row["workload_id"] == "unet_b1_forward"
        and row["contrast_role"] == "geometry_removal_with_patch_materialization_fixed"
    )
    assert int(geometry["accepted_paired_sessions"]) == 4
    assert json.loads(geometry["excluded_session_outcomes"]) == {
        "0": {"M": "failed", "R": "ok"}
    }


def test_rejects_missing_outcome_or_wrong_config_hash(tmp_path: Path) -> None:
    artifacts = tmp_path / "missing"
    artifacts.mkdir()
    missing = ("E27", "unet_update_b8_r256", 4, "F")
    _write_artifacts(artifacts, omitted_group=missing)
    with pytest.raises(ValueError, match="outcome matrix mismatch"):
        exporter.export(artifacts, tmp_path / "missing-output")

    artifacts = tmp_path / "wrong-config"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    raw_path = artifacts / exporter.COHORTS["E27"]["raw"]
    rows = _csv(raw_path)
    rows[0]["config_hash"] = "0" * 64
    _write_csv(raw_path, rows)
    with pytest.raises(ValueError, match="config_hash"):
        exporter.export(artifacts, tmp_path / "wrong-config-output")


def test_rejects_mixed_runtime_stack(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    gate_path = artifacts / "e27_runtime_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["jax_version"] = "0.4.25"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="jax_version"):
        exporter.export(artifacts, tmp_path / "output")


def test_role_specs_cover_exact_frozen_variant_matrices() -> None:
    _, registrations = exporter._load_registration(Path("protocol.yaml").resolve())
    observed = {
        experiment_id: tuple(registration["config"]["variants"])
        for experiment_id, registration in registrations.items()
    }
    assert observed == {
        "E24": ("R", "M", "F"),
        "E25": ("R", "F", "E"),
        "E27": ("R", "F"),
    }
    assert {
        experiment_id: len(exporter._expected_groups(experiment_id, registration))
        for experiment_id, registration in registrations.items()
    } == {"E24": 45, "E25": 90, "E27": 10}


def test_e25_cost_figure_is_exact_deterministic_and_fails_closed(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    validated = tmp_path / "validated"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    exporter.export(artifacts, validated)

    audit_path = validated / exporter.OUTPUT_FILES["audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["analysis_code"] = {
        "commit": "a" * 40,
        "dirty": False,
        "status_porcelain": [],
    }
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")

    pdf, png, figure_csv = cost_figure.render(validated, tmp_path / "figure-a")
    pdf_repeat, png_repeat, figure_csv_repeat = cost_figure.render(
        validated, tmp_path / "figure-b"
    )
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes() == pdf_repeat.read_bytes()
    assert png.read_bytes() == png_repeat.read_bytes()
    assert figure_csv.read_bytes() == figure_csv_repeat.read_bytes()

    figure_rows = _csv(figure_csv)
    assert len(figure_rows) == 9
    assert [(row["workload_id"], row["variant"]) for row in figure_rows] == [
        (workload_id, variant)
        for workload_id, _label, _mode in cost_figure.WORKLOADS
        for variant, _label, _role in cost_figure.VARIANTS
    ]
    e = next(
        row
        for row in figure_rows
        if row["workload_id"] == "unet_b1_forward" and row["variant"] == "E"
    )
    assert json.loads(e["session_ids"]) == [0, 1, 2, 3, 4]
    assert json.loads(e["session_median_latency_ms"]) == pytest.approx(
        [1.00995, 1.01995, 1.02995, 1.03995, 1.04995]
    )
    assert float(e["median_latency_ms"]) == pytest.approx(1.02995)
    assert json.loads(e["peak_live_gib_by_session"]) == [4.0] * 5
    assert float(e["median_peak_live_gib"]) == 4.0
    assert {row["physical_gpu_uuid"] for row in figure_rows} == {
        exporter.EXPECTED_DEVICE["uuid"]
    }
    assert {row["runtime_role"] for row in figure_rows} == {"pinned_jax_0_4_25_cuda_11_8"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cost_figure.render(validated, tmp_path / "figure-a")

    sessions_path = validated / exporter.OUTPUT_FILES["sessions"]
    session_rows = _csv(sessions_path)
    target = next(
        row
        for row in session_rows
        if row["experiment_id"] == "E25"
        and row["workload_id"] == "unet_b1_forward"
        and row["session"] == "0"
        and row["variant"] == "E"
    )
    target["jax_version"] = "0.11.2"
    _write_csv(sessions_path, session_rows)
    with pytest.raises(ValueError, match="audit evidence mismatch"):
        cost_figure.render(validated, tmp_path / "tampered-evidence")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    session_evidence = audit["output_evidence"][sessions_path.name]
    session_evidence["bytes"] = sessions_path.stat().st_size
    session_evidence["sha256"] = hashlib.sha256(sessions_path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected jax_version"):
        cost_figure.render(validated, tmp_path / "tampered-provenance")
