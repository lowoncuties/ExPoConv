from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.export_e23_autotune_memory import (
    EXPECTED_COMMIT,
    EXPECTED_COMPILER_MODE,
    EXPECTED_CONFIG_HASH,
    EXPECTED_DEVICE,
    EXPECTED_GPU_NAME,
    EXPECTED_GPU_UUID,
    EXPECTED_MEASURED_PEAK_PHASE,
    EXPECTED_OUTPUT_RETENTION,
    EXPECTED_PRECISION_POLICY,
    EXPECTED_REFERENCE_COMMIT,
    EXPECTED_SHAPE,
    EXPECTED_SYNCHRONIZATION,
    export,
)


def _snapshot(*, contaminated: bool = False) -> str:
    return json.dumps(
        {
            "status": "ok",
            "process_query_status": "ok",
            "index": 1,
            "uuid": EXPECTED_GPU_UUID,
            "name": EXPECTED_GPU_NAME,
            "memory.total": 32768,
            "memory.used": 4,
            "utilization.gpu": 0,
            "compute_processes": ([{"pid": 123}] if contaminated else []),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _rows(mode: str) -> list[dict[str, object]]:
    rows = []
    for session in range(5):
        state = json.dumps(
            {
                "l2_norm": 10.0 + session,
                "leaf_count": 83,
                "nonfinite_count": 0,
                "scalar_count": 486941,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        for variant in ("R", "F"):
            if mode == "default":
                peak = (2000 if variant == "R" else 3500) + session
            else:
                peak = (1800 if variant == "R" else 1900) + session
            common: dict[str, object] = {
                "experiment_id": "E23",
                "protocol_version": "1.15",
                "code_commit": EXPECTED_COMMIT,
                "code_dirty": False,
                "config_hash": EXPECTED_CONFIG_HASH,
                "device_id": EXPECTED_DEVICE,
                "reference_commit": EXPECTED_REFERENCE_COMMIT,
                "precision_policy": EXPECTED_PRECISION_POLICY,
                "workload_id": "unet_b1_forward",
                "shape": EXPECTED_SHAPE,
                "coordinate_boundary": "tangent_to_tangent",
                "operation_boundary": "unet:forward:depth=1",
                "layout": "BHWC/model-pytree",
                "kernel": "None",
                "compiler_mode": EXPECTED_COMPILER_MODE,
                "measured_peak_phase": EXPECTED_MEASURED_PEAK_PHASE,
                "output_retention_policy": EXPECTED_OUTPUT_RETENTION,
                "synchronization_method": EXPECTED_SYNCHRONIZATION,
                "warmup_count": 50,
                "repeat_count": 200,
                "state_evolves_across_calls": False,
                "output_nonfinite_count": 0,
                "sustained_output_nonfinite_count": "",
                "sustained_sequence_call_count": 0,
                "seed_or_session": session,
                "variant": variant,
                "state_summary_before": state,
                "state_summary_after": state,
                "measured_peak_bytes_in_use": peak,
                "measured_peak_pool_bytes": peak + 500,
                "process_reservation_mib": 100,
                "gpu_snapshot_before": _snapshot(),
                "gpu_snapshot_after": _snapshot(),
                "timestamp_utc": "2026-10-16T00:00:00.000Z",
            }
            phase_rows = [
                ("compilation", -2, "ok", 100.0),
                ("first_compiled_call", -1, "ok", 10.0),
                ("sustained_sequence", -3, "not_applicable", ""),
            ]
            phase_rows.extend(
                ("steady_state", observation, "ok", 1.0 + observation / 1000)
                for observation in range(200)
            )
            for phase, observation, status, latency in phase_rows:
                rows.append(
                    {
                        **common,
                        "phase": phase,
                        "observation": observation,
                        "status": status,
                        "latency_ms": latency,
                    }
                )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    default = tmp_path / "default.csv"
    autotune0 = tmp_path / "autotune0.csv"
    _write(default, _rows("default"))
    _write(autotune0, _rows("autotune0"))
    return default, autotune0


def test_export_writes_complete_paired_memory_tables(tmp_path: Path) -> None:
    default, autotune0 = _sources(tmp_path)
    output = tmp_path / "output"
    summary = export(default, autotune0, output)
    assert summary["output_rows"] == {
        "e23_memory_sessions.csv": 20,
        "e23_memory_summary.csv": 4,
        "e23_memory_contrasts.csv": 5,
    }
    assert summary["compiler_setting_provenance"]["status"] == (
        "wrapper_and_source_file_role_not_raw_embedded"
    )
    with (output / "e23_memory_contrasts.csv").open(encoding="utf-8", newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    assert len(contrasts) == 5
    assert {int(row["default_F_minus_R_bytes"]) for row in contrasts} == {1500}
    assert {int(row["autotune0_F_minus_R_bytes"]) for row in contrasts} == {100}
    assert {int(row["difference_in_differences_bytes"]) for row in contrasts} == {1400}
    with (output / "e23_memory_summary.csv").open(encoding="utf-8", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    assert {(row["compiler_setting"], row["variant"]) for row in summaries} == {
        ("default", "R"),
        ("default", "F"),
        ("autotune0", "R"),
        ("autotune0", "F"),
    }


def test_export_rejects_incomplete_duplicate_and_same_role_inputs(tmp_path: Path) -> None:
    default_rows = _rows("default")
    default = tmp_path / "default.csv"
    autotune0 = tmp_path / "autotune0.csv"
    _write(default, default_rows[:-1])
    _write(autotune0, _rows("autotune0"))
    with pytest.raises(ValueError, match="row count"):
        export(default, autotune0, tmp_path / "incomplete")

    rows = _rows("default")
    rows[-1] = deepcopy(rows[-2])
    _write(default, rows)
    with pytest.raises(ValueError, match="duplicate E23 phase observation"):
        export(default, autotune0, tmp_path / "duplicate")

    with pytest.raises(ValueError, match="distinct files"):
        export(autotune0, autotune0, tmp_path / "same")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("code_dirty", True, "code_dirty"),
        ("config_hash", "wrong", "config_hash"),
        ("status", "wrong", "phase/status"),
        ("latency_ms", "nan", "finite positive"),
    ],
)
def test_export_rejects_identity_phase_and_latency_errors(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    default, autotune0 = _sources(tmp_path)
    rows = _rows("default")
    rows[0][field] = value
    _write(default, rows)
    with pytest.raises(ValueError, match=message):
        export(default, autotune0, tmp_path / field)


def test_export_rejects_nonfinite_state_inconsistent_peak_and_gpu_contamination(
    tmp_path: Path,
) -> None:
    default, autotune0 = _sources(tmp_path)
    rows = _rows("default")
    state = json.loads(str(rows[0]["state_summary_before"]))
    state["nonfinite_count"] = 1
    rows[0]["state_summary_before"] = json.dumps(state)
    rows[0]["state_summary_after"] = json.dumps(state)
    _write(default, rows)
    with pytest.raises(ValueError, match="nonfinite E23 state"):
        export(default, autotune0, tmp_path / "state")

    rows = _rows("default")
    rows[1]["measured_peak_bytes_in_use"] = 9999
    _write(default, rows)
    with pytest.raises(ValueError, match="varies within"):
        export(default, autotune0, tmp_path / "peak")

    rows = _rows("default")
    rows[0]["gpu_snapshot_before"] = _snapshot(contaminated=True)
    _write(default, rows)
    with pytest.raises(ValueError, match="compute_processes"):
        export(default, autotune0, tmp_path / "gpu")


def test_export_rejects_cross_mode_state_mismatch(tmp_path: Path) -> None:
    default, autotune0 = _sources(tmp_path)
    rows = _rows("autotune0")
    for row in rows:
        if row["seed_or_session"] == 0:
            state = json.loads(str(row["state_summary_before"]))
            state["l2_norm"] = 999.0
            row["state_summary_before"] = json.dumps(state, sort_keys=True)
            row["state_summary_after"] = json.dumps(state, sort_keys=True)
    _write(autotune0, rows)
    with pytest.raises(ValueError, match="differ across compiler settings"):
        export(default, autotune0, tmp_path / "state-mismatch")
