from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.export_e22_allocator_fraction import (
    EXPECTED_COMMIT,
    EXPECTED_COMPILER_MODE,
    EXPECTED_DEVICE,
    EXPECTED_GPU_NAME,
    EXPECTED_GPU_UUID,
    EXPECTED_MEASURED_PEAK_PHASE,
    EXPECTED_OUTPUT_RETENTION,
    EXPECTED_PRECISION_POLICY,
    EXPECTED_REFERENCE_COMMIT,
    EXPECTED_SYNCHRONIZATION,
    WORKLOADS,
    export,
)


def _snapshot(*, contaminated: bool = False) -> dict[str, object]:
    return {
        "status": "ok",
        "process_query_status": "ok",
        "index": 0,
        "uuid": EXPECTED_GPU_UUID,
        "name": EXPECTED_GPU_NAME,
        "memory.total": 32768,
        "memory.used": 4,
        "utilization.gpu": 0,
        "compute_processes": ([{"pid": 123}] if contaminated else []),
    }


def _identity(workload: str, session: int, variant: str) -> dict[str, object]:
    specification = WORKLOADS[workload]
    return {
        "experiment_id": "E22",
        "protocol_version": "1.15",
        "code_commit": EXPECTED_COMMIT,
        "code_dirty": False,
        "config_hash": specification["config_hash"],
        "device_id": EXPECTED_DEVICE,
        "reference_commit": EXPECTED_REFERENCE_COMMIT,
        "precision_policy": EXPECTED_PRECISION_POLICY,
        "workload_id": workload,
        "variant": variant,
        "seed_or_session": session,
        "shape": json.dumps(
            {
                "batch": specification["batch"],
                "height": specification["height"],
                "width": specification["width"],
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


def _success_rows(workload: str, session: int, variant: str) -> list[dict[str, object]]:
    before = json.dumps(
        {
            "l2_norm": 10.0 + session,
            "leaf_count": 250,
            "nonfinite_count": 0,
            "scalar_count": 1460824,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    after = json.dumps(
        {
            "l2_norm": 20.0 + session,
            "leaf_count": 250,
            "nonfinite_count": 0,
            "scalar_count": 1460824,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    peak = (10_000 if variant == "R" else 6_000) + session
    common: dict[str, object] = {
        **_identity(workload, session, variant),
        "coordinate_boundary": "tangent_to_tangent",
        "operation_boundary": "unet:update:depth=1",
        "layout": "BHWC/model-pytree",
        "kernel": "None",
        "compiler_mode": EXPECTED_COMPILER_MODE,
        "measured_peak_phase": EXPECTED_MEASURED_PEAK_PHASE,
        "output_retention_policy": EXPECTED_OUTPUT_RETENTION,
        "synchronization_method": EXPECTED_SYNCHRONIZATION,
        "warmup_count": 50,
        "repeat_count": 200,
        "output_nonfinite_count": 0,
        "sustained_output_nonfinite_count": "",
        "sustained_sequence_call_count": 0,
        "state_evolves_across_calls": True,
        "state_summary_before": before,
        "state_summary_after": after,
        "measured_peak_bytes_in_use": peak,
        "measured_peak_pool_bytes": peak + 1_000,
        "process_reservation_mib": 100,
        "gpu_snapshot_before": json.dumps(_snapshot(), sort_keys=True),
        "gpu_snapshot_after": json.dumps(_snapshot(), sort_keys=True),
        "timestamp_utc": "2026-10-16T00:00:00.000Z",
    }
    rows = [
        {
            **common,
            "phase": "compilation",
            "observation": -2,
            "status": "ok",
            "latency_ms": 100.0,
        },
        {
            **common,
            "phase": "first_compiled_call",
            "observation": -1,
            "status": "ok",
            "latency_ms": 10.0,
        },
        {
            **common,
            "phase": "sustained_sequence",
            "observation": -3,
            "status": "not_applicable",
            "latency_ms": "",
        },
    ]
    rows.extend(
        {
            **common,
            "phase": "steady_state",
            "observation": observation,
            "status": "ok",
            "latency_ms": (20.0 if variant == "R" else 10.0) + observation / 1000,
        }
        for observation in range(200)
    )
    return rows


def _failure(
    workload: str,
    session: int,
    variant: str,
    *,
    kind: str = "oom",
) -> dict[str, object]:
    if kind == "oom":
        returncode = 1
        stderr = (
            "Allocator ran out of memory trying to allocate 35.00GiB "
            "(rounded to 37580963840)requested by op\n"
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
            "37580963840 bytes.\n"
            "total allocation: 35.50GiB\n"
        )
    elif kind == "terminated_by_signal":
        returncode = -9
        stderr = ""
    else:
        returncode = 1
        stderr = "shape mismatch"
    identity = _identity(workload, session, variant)
    identity["shape"] = json.loads(str(identity["shape"]))
    return {
        **identity,
        "status": "failed",
        "operation_boundary": "unet:update",
        "returncode": returncode,
        "failure_kind": kind,
        "failure_phase": "first_update",
        "gpu_snapshot_before": _snapshot(),
        "gpu_snapshot_after": _snapshot(),
        "stdout": "",
        "stderr": stderr,
        "timestamp_utc": "2026-10-16T00:00:00.000Z",
    }


def _campaign() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    successes = []
    failures = []
    for workload in WORKLOADS:
        for session in range(5):
            for variant in ("R", "F"):
                if workload == "unet_update_b16_r512" and variant == "R":
                    failures.append(_failure(workload, session, variant))
                elif workload == "unet_update_b8_r256" and session == 4 and variant == "F":
                    failures.append(_failure(workload, session, variant, kind="failed"))
                else:
                    successes.extend(_success_rows(workload, session, variant))
    return successes, failures


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    success_rows, failure_rows = _campaign()
    success = tmp_path / "e22.csv"
    failures = tmp_path / "e22.csv.failures.jsonl"
    _write_csv(success, success_rows)
    _write_jsonl(failures, failure_rows)
    return success, failures


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_export_preserves_failures_and_pairs_only_complete_paths(tmp_path: Path) -> None:
    success, failures = _sources(tmp_path)
    output = tmp_path / "output"
    summary = export(success, failures, output)
    assert summary["output_rows"] == {
        "e22_session_outcomes.csv": 30,
        "e22_workload_variant_summary.csv": 6,
        "e22_paired_contrasts.csv": 9,
        "e22_failure_diagnostics.csv": 6,
    }
    assert summary["matrix"]["outcome_counts"] == {
        "failed": 1,
        "ok": 24,
        "oom": 5,
    }
    assert summary["matrix"]["paired_complete_sessions_by_workload"] == {
        "unet_update_b8_r512": 5,
        "unet_update_b16_r512": 0,
        "unet_update_b8_r256": 4,
    }
    assert summary["allocator_provenance"]["allocator_fraction"] == 0.95
    assert summary["allocator_provenance"]["preallocate"] is False
    assert summary["allocator_provenance"]["provenance_status"] == (
        "frozen_wrapper_contract_not_raw_row_embedded"
    )
    outcomes = _read(output / "e22_session_outcomes.csv")
    oom = next(row for row in outcomes if row["outcome"] == "oom")
    assert oom["failure_phase"] == "first_update"
    assert oom["requested_allocation_bytes"] == "37580963840"
    assert oom["reported_total_allocation_gib"] == "35.5"
    assert oom["requested_exceeds_physical_memory"] == "True"
    assert oom["requested_exceeds_nominal_fraction_capacity"] == "True"
    assert oom["reported_total_exceeds_physical_memory"] == "True"
    contrasts = _read(output / "e22_paired_contrasts.csv")
    assert len(contrasts) == 9
    assert all(
        row["pairing_rule"] == "included_only_when_both_R_and_F_completed"
        for row in contrasts
    )
    assert all(float(row["R_over_F_latency_ratio"]) > 1 for row in contrasts)


def test_export_excludes_protocol_111_pilot_by_identity(tmp_path: Path) -> None:
    success, failures = _sources(tmp_path)
    success_rows, _ = _campaign()
    success_rows[0]["protocol_version"] = "1.11"
    _write_csv(success, success_rows)
    with pytest.raises(ValueError, match="protocol_version"):
        export(success, failures, tmp_path / "pilot-success")

    success, failures = _sources(tmp_path)
    _, failure_rows = _campaign()
    failure_rows[0]["protocol_version"] = "1.11"
    _write_jsonl(failures, failure_rows)
    with pytest.raises(ValueError, match="protocol_version"):
        export(success, failures, tmp_path / "pilot-failure")


def test_export_rejects_incomplete_and_overlapping_group_matrices(tmp_path: Path) -> None:
    success, failures = _sources(tmp_path)
    success_rows, _ = _campaign()
    _write_csv(success, success_rows[:-1])
    with pytest.raises(ValueError, match="success group"):
        export(success, failures, tmp_path / "partial")

    success, failures = _sources(tmp_path)
    success_rows, _ = _campaign()
    success_rows.extend(_success_rows("unet_update_b16_r512", 0, "R"))
    _write_csv(success, success_rows)
    with pytest.raises(ValueError, match="both success and failure"):
        export(success, failures, tmp_path / "overlap")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("code_dirty", True, "code_dirty"),
        ("code_commit", "wrong", "code_commit"),
        ("config_hash", "wrong", "config_hash"),
        ("device_id", "wrong", "device_id"),
        ("status", "wrong", "status"),
        ("latency_ms", "nan", "finite positive"),
    ],
)
def test_export_rejects_invalid_success_identity_and_phases(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    success, failures = _sources(tmp_path)
    success_rows, _ = _campaign()
    success_rows[0][field] = value
    _write_csv(success, success_rows)
    with pytest.raises(ValueError, match=message):
        export(success, failures, tmp_path / field)


def test_export_rejects_fractional_shape_dimensions(tmp_path: Path) -> None:
    success, failures = _sources(tmp_path)
    success_rows, _ = _campaign()
    shape = json.loads(str(success_rows[0]["shape"]))
    shape["width"] = 512.5
    success_rows[0]["shape"] = json.dumps(shape, separators=(",", ":"), sort_keys=True)
    _write_csv(success, success_rows)
    with pytest.raises(ValueError, match="shape width must be an integer"):
        export(success, failures, tmp_path / "fractional-shape")


def test_export_recomputes_failure_class_and_requires_oom_diagnostics(
    tmp_path: Path,
) -> None:
    success, failures = _sources(tmp_path)
    _, failure_rows = _campaign()
    failure_rows[0]["failure_kind"] = "failed"
    _write_jsonl(failures, failure_rows)
    with pytest.raises(ValueError, match="disagrees with explicit worker diagnostic"):
        export(success, failures, tmp_path / "class")

    success, failures = _sources(tmp_path)
    _, failure_rows = _campaign()
    failure_rows[0]["stderr"] = "out of memory"
    _write_jsonl(failures, failure_rows)
    with pytest.raises(ValueError, match="lacks requested/total allocation diagnostics"):
        export(success, failures, tmp_path / "diagnostics")


def test_export_rejects_contaminated_snapshot_and_wrong_runner(tmp_path: Path) -> None:
    success, failures = _sources(tmp_path)
    _, failure_rows = _campaign()
    failure_rows[0]["gpu_snapshot_before"] = _snapshot(contaminated=True)
    _write_jsonl(failures, failure_rows)
    with pytest.raises(ValueError, match="compute_processes"):
        export(success, failures, tmp_path / "contaminated")

    success, failures = _sources(tmp_path)
    runner = tmp_path / "run_e22.sh"
    runner.write_text(
        "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75\n"
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runner SHA-256"):
        export(success, failures, tmp_path / "runner", runner_path=runner)


def test_export_accepts_explicit_signal_failure_without_labeling_it_oom(
    tmp_path: Path,
) -> None:
    success_rows, failure_rows = _campaign()
    target = next(row for row in failure_rows if row["failure_kind"] == "failed")
    replacement = _failure(
        str(target["workload_id"]),
        int(target["seed_or_session"]),
        str(target["variant"]),
        kind="terminated_by_signal",
    )
    failure_rows[failure_rows.index(target)] = replacement
    success = tmp_path / "e22.csv"
    failures = tmp_path / "e22.csv.failures.jsonl"
    _write_csv(success, success_rows)
    _write_jsonl(failures, failure_rows)
    summary = export(success, failures, tmp_path / "output")
    assert summary["matrix"]["outcome_counts"] == {
        "ok": 24,
        "oom": 5,
        "terminated_by_signal": 1,
    }


def test_export_accepts_complete_campaign_with_no_failures(tmp_path: Path) -> None:
    success_rows = [
        row
        for workload in WORKLOADS
        for session in range(5)
        for variant in ("R", "F")
        for row in _success_rows(workload, session, variant)
    ]
    success = tmp_path / "e22.csv"
    failures = tmp_path / "e22.csv.failures.jsonl"
    _write_csv(success, success_rows)
    _write_jsonl(failures, [])
    output = tmp_path / "output"
    summary = export(success, failures, output)
    assert summary["output_rows"]["e22_session_outcomes.csv"] == 30
    assert summary["output_rows"]["e22_paired_contrasts.csv"] == 15
    assert summary["output_rows"]["e22_failure_diagnostics.csv"] == 0
    assert _read(output / "e22_failure_diagnostics.csv") == []


def test_export_writes_header_only_contrasts_when_no_pair_completes(
    tmp_path: Path,
) -> None:
    template = _success_rows("unet_update_b8_r512", 0, "R")[0]
    success = tmp_path / "e22.csv"
    with success.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=list(template), lineterminator="\n").writeheader()
    failure_rows = [
        _failure(workload, session, variant)
        for workload in WORKLOADS
        for session in range(5)
        for variant in ("R", "F")
    ]
    failures = tmp_path / "e22.csv.failures.jsonl"
    _write_jsonl(failures, failure_rows)
    output = tmp_path / "output"
    summary = export(success, failures, output)
    assert summary["output_rows"]["e22_paired_contrasts.csv"] == 0
    assert summary["output_rows"]["e22_failure_diagnostics.csv"] == 30
    assert _read(output / "e22_paired_contrasts.csv") == []
