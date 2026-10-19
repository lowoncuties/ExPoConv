from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from poincare_exec.artifacts.provenance import precision_policy
from poincare_exec.config.loader import canonical_json
from scripts import export_e29_e30_tables as exporter

ALLOCATOR_LIMIT = 25567444992
LIBRARY_PATH = "/venv/nvidia/cudnn/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib"
OOM_STDERR = (
    "jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while "
    "trying to allocate 30064771072 bytes."
)
E_LATENCY = {"unet_b8_update": 20.0, "unet_update_b8_r512": 80.0}


def _variables(index: int) -> dict[str, Any]:
    return {
        "CUDA_VISIBLE_DEVICES": str(index),
        "JAX_PLATFORMS": None,
        "LD_LIBRARY_PATH": LIBRARY_PATH,
        "NVIDIA_TF32_OVERRIDE": None,
        "XLA_FLAGS": None,
        "XLA_PYTHON_CLIENT_ALLOCATOR": None,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": None,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def _write_runtime(root: Path, stem: str, index: int) -> None:
    device = {"id": 0, "platform": "gpu", "device_kind": exporter.EXPECTED_GPU_NAME}
    (root / f"{stem}_runtime_gate.json").write_text(
        json.dumps(
            {
                "jax_version": "0.4.25",
                "jaxlib_version": "0.4.25",
                "default_backend": "gpu",
                "devices": [device],
                "compiled_gpu_smoke": "passed",
                "environment": _variables(index),
            }
        ),
        encoding="utf-8",
    )
    (root / f"{stem}_environment.json").write_text(
        json.dumps(
            {
                "hostname": exporter.EXPECTED_HOSTNAME,
                "git": {
                    "commit": exporter.EXPECTED_COMMIT,
                    "dirty": False,
                    "status_porcelain": [],
                },
                "jax": {
                    "version": "0.4.25",
                    "jaxlib_version": "0.4.25",
                    "backend": "gpu",
                    "devices": [
                        {**device, "memory_stats": {"bytes_limit": ALLOCATOR_LIMIT}}
                    ],
                },
                "packages": {"jax": "0.4.25", "jaxlib": "0.4.25+cuda11.cudnn86"},
                "environment": _variables(index),
            }
        ),
        encoding="utf-8",
    )


def _snapshot(index: int) -> dict[str, Any]:
    return {
        "status": "ok",
        "process_query_status": "ok",
        "index": index,
        "uuid": exporter.PHYSICAL_GPU_UUIDS[index],
        "name": exporter.EXPECTED_GPU_NAME,
        "memory.used": 0,
        "memory.total": 32768,
        "utilization.gpu": 0,
        "compute_processes": [],
    }


def _identity(
    experiment_id: str, workload: dict[str, Any], session: int, variant: str, timestamp: str
):
    return {
        "timestamp_utc": timestamp,
        "experiment_id": experiment_id,
        "protocol_version": exporter.EXPECTED_PROTOCOL_VERSION,
        "variant": variant,
        "config_hash": workload["config_hash"],
        "code_commit": exporter.EXPECTED_COMMIT,
        "code_dirty": False,
        "reference_commit": exporter.EXPECTED_REFERENCE_COMMIT,
        "device_id": exporter.EXPECTED_DEVICE_ID,
        "precision_policy": precision_policy(workload["spec"]["dtype"]),
        "seed_or_session": session,
        "workload_id": workload["spec"]["id"],
    }


def _success(
    experiment_id: str,
    workload: dict[str, Any],
    session: int,
    variant: str,
    index: int,
    timestamp: str,
) -> list[dict[str, Any]]:
    spec = workload["spec"]
    latency = (
        E_LATENCY.get(spec["id"], 10.0)
        if variant == "E"
        else {"R": 4.0, "F": 2.0}[variant] * workload["initial_width"]
    )
    peak_gib = {"E": 1, "R": 12, "F": 6}[variant]
    shared = {
        **_identity(experiment_id, workload, session, variant, timestamp),
        "status": "ok",
        "shape": canonical_json(spec["shape"]),
        "coordinate_boundary": (
            "euclidean_to_euclidean" if variant == "E" else "tangent_to_tangent"
        ),
        "operation_boundary": f"unet:{spec['mode']}:depth=1",
        "warmup_count": 50,
        "repeat_count": 200,
        "state_evolves_across_calls": spec["mode"] == "update",
        "state_summary_before": canonical_json({"nonfinite_count": 0}),
        "state_summary_after": canonical_json({"nonfinite_count": 0}),
        "output_nonfinite_count": 0,
        "measured_peak_bytes_in_use": peak_gib * 1024**3,
        "measured_peak_pool_bytes": (peak_gib + 1) * 1024**3,
        "process_reservation_mib": peak_gib * 1024,
        "gpu_snapshot_before": canonical_json(_snapshot(index)),
        "gpu_snapshot_after": canonical_json(_snapshot(index)),
    }
    rows = [
        {**shared, "observation": -2, "phase": "compilation", "latency_ms": 900.0},
        {**shared, "observation": -1, "phase": "first_compiled_call", "latency_ms": 50.0},
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
            "latency_ms": latency * (1.0 + 0.01 * session) + 0.0001 * observation,
        }
        for observation in range(200)
    )
    return rows


def _oom(
    experiment_id: str,
    workload: dict[str, Any],
    session: int,
    variant: str,
    index: int,
    timestamp: str,
):
    return {
        **_identity(experiment_id, workload, session, variant, timestamp),
        "status": "failed",
        "shape": workload["spec"]["shape"],
        "operation_boundary": f"unet:{workload['spec']['mode']}",
        "returncode": 1,
        "failure_kind": "oom",
        "failure_phase": "first_update",
        "gpu_snapshot_before": _snapshot(index),
        "gpu_snapshot_after": _snapshot(index),
        "stdout": "",
        "stderr": f"Traceback (most recent call last):\n{OOM_STDERR}\n",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(
    root: Path,
    *,
    oom_workload: str | None = None,
    omitted: tuple[str, str, int, str] | None = None,
    parallel: bool = True,
    lockup: bool = False,
) -> None:
    """Write raw groups; E29 completes one group per minute from 10:00 UTC.

    With ``parallel`` E30 runs from 10:10 (GPU-1 activity from 10:09:30), one
    group every two minutes, while E29 is still running; otherwise from 11:00.
    """

    _, registrations = exporter._load_registrations(exporter.PROTOCOL_PATH)
    origin = datetime(2026, 10, 18, 10, tzinfo=UTC)
    e30_start = origin + timedelta(minutes=10 if parallel else 60)
    starts = {"E29": (origin, 1), "E30": (e30_start, 2)}

    def stamp(moment: datetime) -> str:
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    logs = root / "logs"
    logs.mkdir()
    (logs / "e29.started").write_text(stamp(origin) + "\n", encoding="utf-8")
    (logs / "e30.started").write_text(stamp(e30_start) + "\n", encoding="utf-8")
    (logs / "e30.finished").write_text(
        stamp(e30_start + timedelta(minutes=41)) + "\n", encoding="utf-8"
    )
    (logs / "driver_stall_rule.txt").write_text("fixed rule\n", encoding="utf-8")
    for name, moment in (("e29", origin), ("e30", e30_start)):
        (logs / f"{name}_resume.log").write_text(
            f"=== attempt 1 {stamp(moment)}: ./scripts/with_cuda_11_8.sh\n",
            encoding="utf-8",
        )
    lockups = (
        "2026-10-18T10:30:00Z [123.000000] watchdog: BUG: soft lockup - CPU#1 "
        "stuck for 60s! [nvidia-smi:1]\n"
        if lockup
        else ""
    )
    (logs / "kernel_driver_lockups.log").write_text(lockups, encoding="utf-8")
    if parallel:
        (logs / "e30_parallel_note.log").write_text(
            stamp(e30_start - timedelta(seconds=30)) + " E30 started on GPU 1\n",
            encoding="utf-8",
        )
    for experiment_id, cohort in exporter.COHORTS.items():
        start, step = starts[experiment_id]
        group = 0
        index = int(cohort["physical_index"])
        _write_runtime(root, experiment_id.lower(), index)
        registration = registrations[experiment_id]
        rows: list[dict[str, Any]] = []
        failures = []
        for workload_id, workload in registration["workloads"].items():
            for session in range(5):
                for variant in registration["config"]["variants"]:
                    group += 1
                    timestamp = stamp(start + timedelta(minutes=step * group))
                    arguments = (
                        experiment_id,
                        workload,
                        session,
                        variant,
                        index,
                        timestamp,
                    )
                    if (experiment_id, workload_id, session, variant) == omitted:
                        continue
                    if workload_id == oom_workload and variant == "R":
                        failures.append(_oom(*arguments))
                    else:
                        rows.extend(_success(*arguments))
        _write_csv(root / cohort["raw"], rows)
        if failures:
            (root / f"{cohort['raw']}.failures.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in failures), encoding="utf-8"
            )


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_exports_e29_reference_and_e30_contrasts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)

    summaries = exporter.export(artifacts, tmp_path / "output")

    assert summaries["E29"]["output_rows"] == {
        "e29_session_outcomes.csv": 50,
        "e29_variant_summary.csv": 10,
        "e29_published_reference_ratios.csv": 10,
    }
    assert summaries["E30"]["output_rows"] == {
        "e30_session_outcomes.csv": 20,
        "e30_variant_summary.csv": 4,
        "e30_contrasts.csv": 2,
        "e30_width_series.csv": 3,
    }
    assert summaries["E29"]["runtime_capture"]["allocator_bytes_limit"] == ALLOCATOR_LIMIT
    published = {
        row["workload_id"]: row
        for row in _csv(tmp_path / "output" / "e29_published_reference_ratios.csv")
    }
    table4 = published["unet_b8_update"]
    e_median = 20.0 * 1.02 + 0.0001 * 99.5
    assert float(table4["E29_median_latency_ms"]) == pytest.approx(e_median)
    assert float(table4["R_over_E_latency"]) == pytest.approx(124.062311 / e_median)
    assert float(table4["F_over_E_latency"]) == pytest.approx(64.324783 / e_median)
    assert float(table4["E29_median_throughput_samples_per_s"]) == pytest.approx(
        8000.0 / e_median
    )
    assert float(table4["R_over_E_peak_live"]) == pytest.approx(5.9552161693573)
    oom = published["unet_update_b8_r512"]
    assert oom["published_R_outcome"].startswith("oom 5/5")
    assert oom["R_over_E_latency"] == ""
    assert float(oom["F_over_E_latency"]) == pytest.approx(
        197.300659 / (80.0 * 1.02 + 0.00995)
    )

    contrasts = _csv(tmp_path / "output" / "e30_contrasts.csv")
    assert [int(row["model_initial_width"]) for row in contrasts] == [16, 32]
    assert all(int(row["accepted_paired_sessions"]) == 5 for row in contrasts)
    assert float(contrasts[0]["geometric_mean_latency_ratio"]) == pytest.approx(
        2.0, rel=1e-3
    )
    series = _csv(tmp_path / "output" / "e30_width_series.csv")
    assert [row["model_initial_width"] for row in series] == ["8", "16", "32"]
    assert float(series[0]["geometric_mean_latency_ratio"]) == pytest.approx(
        1.9351845003656778
    )
    assert float(series[0]["log_t_95_lower"]) == pytest.approx(1.9071679299197057)
    assert float(series[0]["R_median_peak_live_gib"]) == pytest.approx(5.9552161693573)


def test_retains_oom_and_excludes_unpaired_sessions(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts, oom_workload="unet_update_b8_r256_iw32")

    exporter.export(artifacts, tmp_path / "output")

    sessions = [
        row
        for row in _csv(tmp_path / "output" / "e30_session_outcomes.csv")
        if row["workload_id"] == "unet_update_b8_r256_iw32" and row["variant"] == "R"
    ]
    assert {row["outcome"] for row in sessions} == {"oom"}
    assert {row["failure_phase"] for row in sessions} == {"first_update"}
    assert {row["failure_requested_allocation_bytes"] for row in sessions} == {
        "30064771072"
    }
    assert all("Out of memory" in row["failure_diagnostic_excerpt"] for row in sessions)
    contrast = _csv(tmp_path / "output" / "e30_contrasts.csv")[1]
    assert contrast["accepted_paired_sessions"] == "0"
    assert contrast["geometric_mean_latency_ratio"] == ""
    assert json.loads(contrast["excluded_session_outcomes"])["0"] == {"F": "ok", "R": "oom"}
    width32 = _csv(tmp_path / "output" / "e30_width_series.csv")[2]
    assert width32["R_outcome"] == "ok 0/5; oom 5/5; failed 0/5"
    assert width32["R_median_latency_ms"] == ""


def test_export_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    exporter.export(artifacts, tmp_path / "first")
    exporter.export(artifacts, tmp_path / "second")

    for files in exporter.OUTPUT_FILES.values():
        for name in files.values():
            if name.endswith(".csv"):
                assert (tmp_path / "first" / name).read_bytes() == (
                    tmp_path / "second" / name
                ).read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export(artifacts, tmp_path / "first")


def test_rejects_raised_allocator_fraction(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    path = artifacts / "e29_environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["environment"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    path.write_text(json.dumps(environment), encoding="utf-8")

    with pytest.raises(ValueError, match="XLA_PYTHON_CLIENT_MEM_FRACTION"):
        exporter.export(artifacts, tmp_path / "output")


def test_rejects_wrong_physical_gpu_or_missing_group(tmp_path: Path) -> None:
    artifacts = tmp_path / "wrong-gpu"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    raw = artifacts / exporter.COHORTS["E29"]["raw"]
    rows = _csv(raw)
    rows[0]["gpu_snapshot_before"] = canonical_json(_snapshot(1))
    _write_csv(raw, rows)
    with pytest.raises(ValueError, match="GPU snapshot failed"):
        exporter.export(artifacts, tmp_path / "wrong-gpu-output")

    artifacts = tmp_path / "missing"
    artifacts.mkdir()
    _write_artifacts(artifacts, omitted=("E30", "unet_update_b8_r256_iw16", 3, "F"))
    with pytest.raises(ValueError, match="outcome matrix mismatch"):
        exporter.export(artifacts, tmp_path / "missing-output")


def test_registration_matches_frozen_protocol_1_18() -> None:
    protocol, registrations = exporter._load_registrations(exporter.PROTOCOL_PATH)

    assert protocol["protocol_version"] == "1.18"
    assert list(registrations["E29"]["workloads"]) == [
        "unet_b1_forward",
        "unet_b8_forward",
        "unet_b8_update",
        "unet_update_b1_r256",
        "unet_update_b2_r256",
        "unet_update_b4_r256",
        "unet_update_b8_r256",
        "unet_update_b16_r256",
        "unet_update_b8_r128",
        "unet_update_b8_r512",
    ]
    assert {
        workload_id: value["initial_width"]
        for workload_id, value in registrations["E30"]["workloads"].items()
    } == {"unet_update_b8_r256_iw16": 16, "unet_update_b8_r256_iw32": 32}
    assert all(
        value["initial_width"] == 8 for value in registrations["E29"]["workloads"].values()
    )


def test_records_concurrent_e29_and_e30_execution(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)

    summaries = exporter.export(artifacts, tmp_path / "output")

    for experiment_id in ("E29", "E30"):
        schedule = summaries[experiment_id]["execution_schedule"]
        assert schedule["e29_and_e30_ran_concurrently"] is True
        assert schedule["e30_gpu1_activity_window_utc"][0] == "2026-10-18T10:09:30Z"
    assert summaries["E29"]["execution_schedule"]["groups_with_other_gpu_activity"] == 41
    assert summaries["E30"]["execution_schedule"]["groups_with_other_gpu_activity"] == 20
    variants = {
        row["workload_id"]: row
        for row in _csv(tmp_path / "output" / "e29_variant_summary.csv")
    }
    assert variants["unet_b1_forward"]["isolated_accepted_sessions"] == "5"
    assert (
        variants["unet_b1_forward"]["median_latency_ms_isolated_sessions"]
        == (variants["unet_b1_forward"]["median_latency_ms"])
    )
    assert variants["unet_b8_forward"]["sessions_with_other_gpu_activity"] == "1"
    assert variants["unet_b8_update"]["isolated_accepted_sessions"] == "0"
    assert variants["unet_b8_update"]["median_latency_ms_isolated_sessions"] == ""
    sessions = _csv(tmp_path / "output" / "e30_session_outcomes.csv")
    assert {row["other_gpu_activity"] for row in sessions} == {"E29 on physical GPU 0"}


def test_sequential_schedule_flags_no_group(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts, parallel=False)

    summaries = exporter.export(artifacts, tmp_path / "output")

    assert summaries["E30"]["execution_schedule"]["e29_and_e30_ran_concurrently"] is False
    for experiment_id in ("E29", "E30"):
        assert (
            summaries[experiment_id]["execution_schedule"]["groups_with_other_gpu_activity"]
            == 0
        )


def test_driver_stall_groups_leave_primary_estimates(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts, lockup=True)

    summaries = exporter.export(artifacts, tmp_path / "output")

    assert summaries["E29"]["execution_schedule"]["driver_stall_intervals_utc"] == [
        ["2026-10-18T10:26:00Z", "2026-10-18T10:33:00Z"]
    ]
    variants = {
        row["workload_id"]: row
        for row in _csv(tmp_path / "output" / "e29_variant_summary.csv")
    }
    flagged = variants["unet_update_b4_r256"]
    assert flagged["driver_stall_flagged_sessions"] == "4"
    assert flagged["accepted_finite_sessions"] == "1"
    assert flagged["finite_sessions"] == "5"
    assert float(flagged["median_latency_ms"]) == pytest.approx(10.0 + 0.00995)
    assert float(flagged["all_finite_sessions_median_latency_ms"]) == pytest.approx(
        10.0 * 1.02 + 0.00995
    )
    contrasts = {
        row["workload_id"]: row for row in _csv(tmp_path / "output" / "e30_contrasts.csv")
    }
    width16 = contrasts["unet_update_b8_r256_iw16"]
    assert width16["accepted_paired_sessions"] == "4"
    assert width16["all_finite_pairs"] == "5"
    assert json.loads(width16["excluded_session_outcomes"]) == {
        "4": {"F": "driver_stall_overlap", "R": "driver_stall_overlap"}
    }
    assert json.loads(contrasts["unet_update_b8_r256_iw32"]["paired_session_ids"]) == [
        1,
        2,
        3,
        4,
    ]
