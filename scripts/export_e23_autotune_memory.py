#!/usr/bin/env python3
"""Validate E23 compiler-mode artifacts and export paired memory summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from scripts.export_e26_elementwise_tables import _atomic_csv, _atomic_json, _sha256

EXPECTED_COMMIT = "30142ce2c9f60a115fa77cf844cac14166446b27"
EXPECTED_PROTOCOL = "1.15"
EXPECTED_CONFIG_HASH = "0d9fa81f314e3f825fe06400866f53c119aa34f82aded04eff5926e3cd3b3f33"
EXPECTED_DEVICE = "argexpr3.vsb.cz::gpu:0::Tesla V100-PCIE-32GB"
EXPECTED_GPU_INDEX = 1
EXPECTED_GPU_UUID = "GPU-4e0547c3-d65d-7d8b-9f6a-d40ad61a0f0f"
EXPECTED_GPU_NAME = "Tesla V100-PCIE-32GB"
EXPECTED_REFERENCE_COMMIT = (
    '{"hnnpp":"28737e22822562ac18d5ea03f8c3e3929e945a83",'
    '"hyperbolic_unet":"bec504e2f4d198ee4c438579b86d72f9d8186d78",'
    '"poincare_resnet":"fbbbbf890571115444b89b114f9ed52d128d5aa3"}'
)
EXPECTED_PRECISION_POLICY = (
    '{"matmul_precision":"highest","reduced_precision_reduction":false,'
    '"storage_dtype":"float32","tf32":false}'
)
EXPECTED_SHAPE = '{"batch":1,"height":256,"width":256}'
EXPECTED_COMPILER_MODE = "jax.jit lowered+compiled before steady-state timing"
EXPECTED_MEASURED_PEAK_PHASE = (
    "process lifetime through final synchronized steady-state output; includes "
    "initialization, compilation, and warmup; excludes post-run state summary"
)
EXPECTED_OUTPUT_RETENTION = (
    "dereference each completed output before the next invocation; retain only "
    "the final steady-state output for finite-value validation"
)
EXPECTED_SYNCHRONIZATION = "block_until_ready on every output/state leaf"
REGISTERED_WRAPPER_SHA256 = (
    "b79eba9fd5970f5d1c80ed3d4510cb5e3e231d63ad2dda71031802fd4019787f"
)
MODES = ("default", "autotune0")
VARIANTS = ("R", "F")
SESSIONS = tuple(range(5))
TIMED_CALLS = 200
EXPECTED_ROWS_PER_FILE = len(VARIANTS) * len(SESSIONS) * (TIMED_CALLS + 3)
MODE_PROVENANCE = (
    "Compiler setting is assigned from the explicit CLI file role and the "
    "registered run_e23.sh contract; XLA_FLAGS is not embedded in E23 rows."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", required=True, type=Path)
    parser.add_argument("--autotune0", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/reviewer_requested")
    )
    return parser


def _truth(value: Any, *, context: str) -> bool:
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    raise ValueError(f"{context}: expected a boolean")


def _finite(value: Any, *, context: str, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected numeric value") from error
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{context}: expected finite {qualifier}value")
    return number


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected an integer") from error
    if str(number) != str(value).strip() or number < minimum:
        raise ValueError(f"{context}: invalid integer")
    return number


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty E23 input: {path}")
    return rows


def _check_snapshot(value: str, *, context: str, before: bool) -> None:
    try:
        snapshot = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context}: invalid GPU snapshot JSON") from error
    expected = {
        "status": "ok",
        "process_query_status": "ok",
        "index": EXPECTED_GPU_INDEX,
        "uuid": EXPECTED_GPU_UUID,
        "name": EXPECTED_GPU_NAME,
        "compute_processes": [],
    }
    for field, wanted in expected.items():
        if snapshot.get(field) != wanted:
            raise ValueError(f"{context}: unexpected GPU snapshot {field}")
    if before:
        memory = _finite(snapshot.get("memory.used"), context=f"{context} memory")
        utilization = _finite(
            snapshot.get("utilization.gpu"), context=f"{context} utilization"
        )
        if memory > 512 or utilization > 5:
            raise ValueError(f"{context}: pre-run GPU is not idle")


def _constant_fields() -> dict[str, Any]:
    return {
        "experiment_id": "E23",
        "protocol_version": EXPECTED_PROTOCOL,
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
        "repeat_count": TIMED_CALLS,
        "state_evolves_across_calls": False,
        "output_nonfinite_count": 0,
        "sustained_sequence_call_count": 0,
    }


def _validate_file(path: Path, *, mode: str, source_hash: str) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    if len(rows) != EXPECTED_ROWS_PER_FILE:
        raise ValueError(
            f"E23 {mode} row count is {len(rows)}, expected {EXPECTED_ROWS_PER_FILE}"
        )
    fixed = _constant_fields()
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    keys: set[tuple[int, str, str, int]] = set()
    for index, row in enumerate(rows, start=1):
        for field, expected in fixed.items():
            observed: Any = row.get(field)
            if isinstance(expected, bool):
                observed = _truth(observed, context=f"row {index} {field}")
            elif isinstance(expected, int):
                observed = _integer(observed, context=f"row {index} {field}")
            if observed != expected:
                raise ValueError(
                    f"row {index}: E23 {field} is {observed!r}, expected {expected!r}"
                )
        session = _integer(row.get("seed_or_session"), context=f"row {index} session")
        variant = str(row.get("variant"))
        if session not in SESSIONS or variant not in VARIANTS:
            raise ValueError(f"row {index}: unexpected E23 session/variant")
        phase = str(row.get("phase"))
        observation = _integer(
            row.get("observation"), context=f"row {index} observation", minimum=-3
        )
        key = (session, variant, phase, observation)
        if key in keys:
            raise ValueError(f"row {index}: duplicate E23 phase observation {key}")
        keys.add(key)
        expected_status = "not_applicable" if phase == "sustained_sequence" else "ok"
        if row.get("status") != expected_status:
            raise ValueError(f"row {index}: invalid E23 phase/status")
        if phase == "compilation":
            expected_observation = -2
        elif phase == "first_compiled_call":
            expected_observation = -1
        elif phase == "sustained_sequence":
            expected_observation = -3
        elif phase == "steady_state" and 0 <= observation < TIMED_CALLS:
            expected_observation = observation
        else:
            raise ValueError(f"row {index}: invalid E23 phase/observation")
        if observation != expected_observation:
            raise ValueError(f"row {index}: invalid E23 phase/observation")
        if phase == "sustained_sequence":
            if str(row.get("latency_ms", "")).strip():
                raise ValueError(f"row {index}: sustained latency must be empty")
        else:
            _finite(row.get("latency_ms"), context=f"row {index} latency", positive=True)
        state_before = json.loads(row["state_summary_before"])
        state_after = json.loads(row["state_summary_after"])
        if state_before != state_after:
            raise ValueError(f"row {index}: stateless E23 state summaries differ")
        if int(state_before.get("nonfinite_count", -1)) != 0:
            raise ValueError(f"row {index}: nonfinite E23 state")
        _finite(state_before.get("l2_norm"), context=f"row {index} state l2")
        if (
            int(state_before.get("leaf_count", 0)) <= 0
            or int(state_before.get("scalar_count", 0)) <= 0
        ):
            raise ValueError(f"row {index}: invalid E23 state counts")
        sustained_nonfinite = str(row.get("sustained_output_nonfinite_count", "")).strip()
        if sustained_nonfinite not in {"", "0"}:
            raise ValueError(f"row {index}: nonfinite sustained output")
        _check_snapshot(
            row["gpu_snapshot_before"], context=f"row {index} before", before=True
        )
        _check_snapshot(
            row["gpu_snapshot_after"], context=f"row {index} after", before=False
        )
        groups[(session, variant)].append(row)

    expected_groups = {(session, variant) for session in SESSIONS for variant in VARIANTS}
    if set(groups) != expected_groups:
        raise ValueError("E23 session/variant matrix is incomplete")
    prepared = []
    state_by_cell: dict[tuple[int, str], str] = {}
    for (session, variant), selected in sorted(groups.items()):
        phases = Counter(row["phase"] for row in selected)
        if phases != Counter(
            {
                "compilation": 1,
                "first_compiled_call": 1,
                "sustained_sequence": 1,
                "steady_state": TIMED_CALLS,
            }
        ):
            raise ValueError(f"E23 phase matrix is incomplete for {(session, variant)}")
        steady_observations = {
            int(row["observation"]) for row in selected if row["phase"] == "steady_state"
        }
        if steady_observations != set(range(TIMED_CALLS)):
            raise ValueError(
                f"E23 steady observations are incomplete for {(session, variant)}"
            )
        for field in (
            "measured_peak_bytes_in_use",
            "measured_peak_pool_bytes",
            "process_reservation_mib",
            "state_summary_before",
            "state_summary_after",
        ):
            if len({row[field] for row in selected}) != 1:
                raise ValueError(f"E23 {field} varies within {(session, variant)}")
        exemplar = selected[0]
        state_by_cell[(session, variant)] = exemplar["state_summary_before"]
        peak_live = _integer(
            exemplar["measured_peak_bytes_in_use"], context="peak live", minimum=1
        )
        peak_pool = _integer(
            exemplar["measured_peak_pool_bytes"], context="peak pool", minimum=1
        )
        if peak_pool < peak_live:
            raise ValueError(f"E23 peak pool is below live bytes for {(session, variant)}")
        reservation = _integer(
            exemplar["process_reservation_mib"], context="reservation", minimum=1
        )
        phase_row = {
            row["phase"]: row for row in selected if row["phase"] != "steady_state"
        }
        steady_latencies = sorted(
            _finite(row["latency_ms"], context="steady latency", positive=True)
            for row in selected
            if row["phase"] == "steady_state"
        )
        prepared.append(
            {
                "experiment_id": "E23",
                "protocol_version": EXPECTED_PROTOCOL,
                "compiler_setting": mode,
                "session": session,
                "variant": variant,
                "peak_live_bytes": peak_live,
                "peak_live_gib": peak_live / 1024**3,
                "peak_pool_bytes": peak_pool,
                "peak_pool_gib": peak_pool / 1024**3,
                "process_reservation_mib": reservation,
                "compilation_ms": float(phase_row["compilation"]["latency_ms"]),
                "first_compiled_call_ms": float(
                    phase_row["first_compiled_call"]["latency_ms"]
                ),
                "steady_state_median_latency_ms": statistics.median(steady_latencies),
                "steady_state_observations": len(steady_latencies),
                "latency_interpretation": (
                    "diagnostic_nonperformance" if mode == "autotune0" else "descriptive"
                ),
                "code_commit": EXPECTED_COMMIT,
                "config_hash": EXPECTED_CONFIG_HASH,
                "device_id": EXPECTED_DEVICE,
                "physical_gpu_uuid": EXPECTED_GPU_UUID,
                "source_file": path.name,
                "source_sha256": source_hash,
                "compiler_setting_provenance": "explicit_cli_file_role",
            }
        )
    for session in SESSIONS:
        if state_by_cell[(session, "R")] != state_by_cell[(session, "F")]:
            raise ValueError(f"E23 R/F initial states differ for session {session}")
    return prepared


def _summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["compiler_setting"]), str(row["variant"]))].append(row)
    output = []
    for key in sorted(grouped):
        selected = sorted(grouped[key], key=lambda row: int(row["session"]))
        if [int(row["session"]) for row in selected] != list(SESSIONS):
            raise ValueError(f"E23 summary sessions are incomplete for {key}")
        live = [float(row["peak_live_gib"]) for row in selected]
        pool = [float(row["peak_pool_gib"]) for row in selected]
        reservation = [int(row["process_reservation_mib"]) for row in selected]
        compilation = [float(row["compilation_ms"]) for row in selected]
        first = [float(row["first_compiled_call_ms"]) for row in selected]
        output.append(
            {
                "experiment_id": "E23",
                "protocol_version": EXPECTED_PROTOCOL,
                "compiler_setting": key[0],
                "variant": key[1],
                "sessions": len(selected),
                "median_peak_live_gib": statistics.median(live),
                "min_peak_live_gib": min(live),
                "max_peak_live_gib": max(live),
                "session_peak_live_gib": json.dumps(live, separators=(",", ":")),
                "median_peak_pool_gib": statistics.median(pool),
                "min_peak_pool_gib": min(pool),
                "max_peak_pool_gib": max(pool),
                "session_peak_pool_gib": json.dumps(pool, separators=(",", ":")),
                "median_process_reservation_mib": statistics.median(reservation),
                "session_process_reservation_mib": json.dumps(
                    reservation, separators=(",", ":")
                ),
                "median_compilation_ms": statistics.median(compilation),
                "min_compilation_ms": min(compilation),
                "max_compilation_ms": max(compilation),
                "median_first_compiled_call_ms": statistics.median(first),
                "min_first_compiled_call_ms": min(first),
                "max_first_compiled_call_ms": max(first),
                "source_file": selected[0]["source_file"],
                "source_sha256": selected[0]["source_sha256"],
                "compiler_setting_provenance": "explicit_cli_file_role",
            }
        )
    return output


def _contrasts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(row["compiler_setting"]), int(row["session"]), str(row["variant"])): row
        for row in rows
    }
    output = []
    for session in SESSIONS:
        values = {
            (mode, variant): int(indexed[(mode, session, variant)]["peak_live_bytes"])
            for mode in MODES
            for variant in VARIANTS
        }
        default_difference = values[("default", "F")] - values[("default", "R")]
        autotune0_difference = values[("autotune0", "F")] - values[("autotune0", "R")]
        output.append(
            {
                "experiment_id": "E23",
                "protocol_version": EXPECTED_PROTOCOL,
                "session": session,
                "default_R_peak_live_bytes": values[("default", "R")],
                "default_F_peak_live_bytes": values[("default", "F")],
                "autotune0_R_peak_live_bytes": values[("autotune0", "R")],
                "autotune0_F_peak_live_bytes": values[("autotune0", "F")],
                "default_F_minus_R_bytes": default_difference,
                "autotune0_F_minus_R_bytes": autotune0_difference,
                "R_default_minus_autotune0_bytes": (
                    values[("default", "R")] - values[("autotune0", "R")]
                ),
                "F_default_minus_autotune0_bytes": (
                    values[("default", "F")] - values[("autotune0", "F")]
                ),
                "difference_in_differences_bytes": (
                    default_difference - autotune0_difference
                ),
                "default_source_sha256": indexed[("default", session, "R")][
                    "source_sha256"
                ],
                "autotune0_source_sha256": indexed[("autotune0", session, "R")][
                    "source_sha256"
                ],
            }
        )
    return output


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    materialized = list(values)
    return {
        "median": statistics.median(materialized),
        "min": min(materialized),
        "max": max(materialized),
    }


def export(default_path: Path, autotune0_path: Path, output_dir: Path) -> dict[str, Any]:
    paths = {"default": default_path.resolve(), "autotune0": autotune0_path.resolve()}
    if len(set(paths.values())) != 2:
        raise ValueError("E23 default and autotune0 inputs must be distinct files")
    hashes = {mode: _sha256(path) for mode, path in paths.items()}
    sessions = []
    for mode in MODES:
        sessions.extend(_validate_file(paths[mode], mode=mode, source_hash=hashes[mode]))
    # The same deterministic parameter realization is required across modes.
    states_by_mode = {}
    for mode, path in paths.items():
        raw = _read_csv(path)
        states_by_mode[mode] = {
            (int(row["seed_or_session"]), row["variant"]): row["state_summary_before"]
            for row in raw
        }
    if states_by_mode["default"] != states_by_mode["autotune0"]:
        raise ValueError(
            "E23 deterministic state summaries differ across compiler settings"
        )
    summaries = _summaries(sessions)
    contrasts = _contrasts(sessions)
    if len(sessions) != 20 or len(summaries) != 4 or len(contrasts) != 5:
        raise ValueError("E23 exported table matrix is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "e23_memory_sessions.csv", sessions)
    _atomic_csv(output_dir / "e23_memory_summary.csv", summaries)
    _atomic_csv(output_dir / "e23_memory_contrasts.csv", contrasts)
    summary = {
        "schema_version": "1.0",
        "analysis_status": "post_hoc_compiler_workspace_diagnostic",
        "scientific_behavior_change": False,
        "identity": {
            "experiment_id": "E23",
            "protocol_version": EXPECTED_PROTOCOL,
            "code_commit": EXPECTED_COMMIT,
            "code_dirty": False,
            "config_hash": EXPECTED_CONFIG_HASH,
            "device_id": EXPECTED_DEVICE,
            "physical_gpu_uuid": EXPECTED_GPU_UUID,
        },
        "sources": {
            mode: {
                "file": paths[mode].name,
                "bytes": paths[mode].stat().st_size,
                "sha256": hashes[mode],
                "rows": EXPECTED_ROWS_PER_FILE,
            }
            for mode in MODES
        },
        "compiler_setting_provenance": {
            "status": "wrapper_and_source_file_role_not_raw_embedded",
            "note": MODE_PROVENANCE,
            "registered_wrapper": "scripts/run_e23.sh",
            "registered_wrapper_sha256": REGISTERED_WRAPPER_SHA256,
            "default_contract": "XLA_FLAGS unset",
            "autotune0_contract": "XLA_FLAGS=--xla_gpu_autotune_level=0",
        },
        "memory_estimand": EXPECTED_MEASURED_PEAK_PHASE,
        "latency_interpretation": (
            "Autotune-disabled latency is a diagnostic and is not a performance result."
        ),
        "contrast_summary_bytes": {
            field: _distribution(int(row[field]) for row in contrasts)
            for field in (
                "default_F_minus_R_bytes",
                "autotune0_F_minus_R_bytes",
                "R_default_minus_autotune0_bytes",
                "F_default_minus_autotune0_bytes",
                "difference_in_differences_bytes",
            )
        },
        "interpretation": (
            "Changes between settings are consistent with compiler autotuning-workspace "
            "involvement but do not uniquely attribute every process-lifetime allocation."
        ),
        "output_rows": {
            "e23_memory_sessions.csv": len(sessions),
            "e23_memory_summary.csv": len(summaries),
            "e23_memory_contrasts.csv": len(contrasts),
        },
    }
    _atomic_json(output_dir / "e23_memory_summary.json", summary)
    return summary


def main() -> int:
    arguments = _parser().parse_args()
    summary = export(arguments.default, arguments.autotune0, arguments.output_dir.resolve())
    print(json.dumps(summary["output_rows"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
