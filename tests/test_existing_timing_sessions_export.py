from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import export_existing_timing_sessions as exporter

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40


def _snapshot(*, process_query_status: str = "ok") -> str:
    return json.dumps(
        {
            "status": "ok",
            "process_query_status": process_query_status,
            "index": 1,
            "uuid": "GPU-test",
            "compute_processes": [],
        }
    )


def _base_row(*, phase: str, status: str, observation: int, latency: str) -> dict[str, str]:
    return {
        "experiment_id": "E08",
        "workload_id": "unet_b8_update",
        "config_hash": "config-a",
        "code_commit": COMMIT,
        "code_dirty": "False",
        "reference_commit": "{}",
        "device_id": "host::gpu:0::V100",
        "precision_policy": "{}",
        "variant": "F",
        "seed_or_session": "0",
        "phase": phase,
        "coordinate_boundary": "tangent_to_tangent",
        "operation_boundary": "unet:update:depth=1",
        "gpu_snapshot_before": _snapshot(),
        "gpu_snapshot_after": _snapshot(),
        "latency_ms": latency,
        "observation": str(observation),
        "repeat_count": "2" if phase == "steady_state" else "1",
        "status": status,
        "timestamp_utc": "2026-10-07T00:00:00Z",
    }


def _valid_rows() -> list[dict[str, str]]:
    return [
        _base_row(phase="compilation", status="ok", observation=0, latency="10"),
        _base_row(phase="first_compiled_call", status="ok", observation=0, latency="8"),
        _base_row(
            phase="sustained_sequence",
            status="not_applicable",
            observation=0,
            latency="",
        ),
        _base_row(phase="steady_state", status="ok", observation=0, latency="2"),
        _base_row(phase="steady_state", status="ok", observation=1, latency="4"),
    ]


def _write_environment(path: Path, *, commit: str = COMMIT) -> None:
    path.write_text(
        json.dumps(
            {
                "git": {
                    "commit": commit,
                    "dirty": False,
                    "status_porcelain": [],
                },
                "environment": {"CUDA_VISIBLE_DEVICES": "1"},
            }
        ),
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[*exporter.GROUP_FIELDS, *exporter.AUDIT_FIELDS],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _run_export(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    environment_commit: str = COMMIT,
) -> tuple[dict[str, object], Path]:
    raw = tmp_path / "raw.csv"
    environment = tmp_path / "environment.json"
    output = tmp_path / "sessions.csv"
    _write_rows(raw, rows)
    _write_environment(environment, commit=environment_commit)
    metadata = exporter.export(
        raw,
        output,
        environment_path=environment,
        expected_code_commit=COMMIT,
    )
    return metadata, output


def test_exports_complete_phase_aware_groups(tmp_path: Path) -> None:
    metadata, output = _run_export(tmp_path, _valid_rows())

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    steady = next(row for row in rows if row["phase"] == "steady_state")

    assert len(rows) == 4
    assert steady["complete"] == "True"
    assert float(steady["median_latency_ms"]) == 3.0
    assert metadata["source_rows"] == 5
    assert metadata["audited_snapshots"] == 10
    assert metadata["incomplete_groups"] == 0
    assert metadata["invalid_latency_count_groups"] == 0
    assert metadata["mixed_identity_groups"] == 0


def test_rejects_non_ok_status_for_timed_phase(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[-1]["status"] = "failed"

    with pytest.raises(ValueError, match="invalid status"):
        _run_export(tmp_path, rows)


def test_rejects_incomplete_group(tmp_path: Path) -> None:
    rows = _valid_rows()[:-1]

    with pytest.raises(ValueError, match="incomplete groups"):
        _run_export(tmp_path, rows)


def test_rejects_missing_expected_latency(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[-1]["latency_ms"] = ""

    with pytest.raises(ValueError, match="invalid latency counts"):
        _run_export(tmp_path, rows)


def test_rejects_mixed_structural_identity(tmp_path: Path) -> None:
    rows = _valid_rows()
    second_identity = [dict(row) for row in rows if row["phase"] == "steady_state"]
    for row in second_identity:
        row["config_hash"] = "config-b"
    rows.extend(second_identity)

    with pytest.raises(ValueError, match="mixed structural identity"):
        _run_export(tmp_path, rows)


def test_rejects_failed_process_query(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0]["gpu_snapshot_after"] = _snapshot(process_query_status="failed")

    with pytest.raises(ValueError, match="compute-process query status"):
        _run_export(tmp_path, rows)


@pytest.mark.parametrize(
    ("environment_commit", "raw_commit", "message"),
    [
        (OTHER_COMMIT, COMMIT, "environment commit"),
        (COMMIT, OTHER_COMMIT, "raw-row code commits"),
    ],
)
def test_rejects_mismatched_environment_or_raw_commit(
    tmp_path: Path,
    environment_commit: str,
    raw_commit: str,
    message: str,
) -> None:
    rows = _valid_rows()
    for row in rows:
        row["code_commit"] = raw_commit

    with pytest.raises(ValueError, match=message):
        _run_export(tmp_path, rows, environment_commit=environment_commit)
