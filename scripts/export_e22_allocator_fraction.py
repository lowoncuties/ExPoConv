#!/usr/bin/env python3
"""Validate clean E22 evidence and export allocator-sensitivity summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.export_e26_elementwise_tables import _atomic_csv, _atomic_json, _sha256

EXPECTED_COMMIT = "30142ce2c9f60a115fa77cf844cac14166446b27"
EXPECTED_PROTOCOL = "1.15"
EXPECTED_DEVICE = "argexpr3.vsb.cz::gpu:0::Tesla V100-PCIE-32GB"
EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-a37e9fd0-2d52-1766-149e-64287a10103d"
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
EXPECTED_RUNNER_SHA256 = "eabac2a0599a6c830531693dfcd4b5ed46b82b45a32b31eb5b50355642351426"
ALLOCATOR_FRACTION = 0.95
PREALLOCATE = False
PHYSICAL_MEMORY_BYTES = 32 * 1024**3
NOMINAL_FRACTION_CAPACITY_GIB = 32.0 * ALLOCATOR_FRACTION
VARIANTS = ("R", "F")
SESSIONS = tuple(range(5))
TIMED_CALLS = 200
SUCCESS_ROWS_PER_GROUP = TIMED_CALLS + 3
WORKLOADS = {
    "unet_update_b8_r512": {
        "batch": 8,
        "height": 512,
        "width": 512,
        "config_hash": "7174ae527bc9e4f729c589b67e7b0ed181a063128810270903cf9f26b3ade969",
    },
    "unet_update_b16_r512": {
        "batch": 16,
        "height": 512,
        "width": 512,
        "config_hash": "5db29c5d6155744983c81acba7a32fb764ae4c96070b641954bd675d55a8e715",
    },
    "unet_update_b8_r256": {
        "batch": 8,
        "height": 256,
        "width": 256,
        "config_hash": "c5ba10a857be2f2552695edb281676721a9b387c69503ed189ce8734f7ceb204",
    },
}
FAILURE_PHASES = {
    "initialization",
    "compilation",
    "first_update",
    "warmup_updates",
    "measured_updates",
    "post_measurement",
    "unknown",
}
OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "failed to allocate",
    "ran out of memory",
)
REQUESTED_BYTES_PATTERN = re.compile(
    r"out of memory while trying to allocate\s+(\d+)\s+bytes", re.IGNORECASE
)
ROUNDED_BYTES_PATTERN = re.compile(r"rounded to\s+(\d+)\)", re.IGNORECASE)
TOTAL_ALLOCATION_PATTERN = re.compile(
    r"total allocation:\s*([0-9]+(?:\.[0-9]+)?)(KiB|MiB|GiB)", re.IGNORECASE
)
ALLOCATOR_PROVENANCE_NOTE = (
    "XLA_PYTHON_CLIENT_MEM_FRACTION and XLA_PYTHON_CLIENT_PREALLOCATE are not "
    "embedded in E22 result rows. Their values are verified against the frozen "
    "run_e22.sh contract and are therefore wrapper-derived provenance."
)
PAIRED_FIELDS = (
    "experiment_id",
    "protocol_version",
    "workload_id",
    "batch",
    "resolution",
    "session",
    "R_median_latency_ms",
    "F_median_latency_ms",
    "R_over_F_latency_ratio",
    "R_peak_live_bytes",
    "F_peak_live_bytes",
    "F_minus_R_peak_live_bytes",
    "F_peak_live_reduction_vs_R_percent",
    "pairing_rule",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument(
        "--runner", type=Path, default=Path(__file__).resolve().with_name("run_e22.sh")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/reviewer_requested")
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _truth(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{context}: expected a boolean")


def _integer(value: Any, *, context: str, minimum: int | None = None) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected an integer") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{context}: expected an integer")
    result = int(number)
    if minimum is not None and result < minimum:
        raise ValueError(f"{context}: integer is below {minimum}")
    return result


def _finite(value: Any, *, context: str, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected numeric value") from error
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{context}: expected finite {qualifier}value")
    return number


def _canonical_shape(value: Any, *, context: str) -> dict[str, int]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: invalid shape JSON") from error
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{context}: shape must be a mapping")
    try:
        return {
            "batch": int(decoded["batch"]),
            "height": int(decoded["height"]),
            "width": int(decoded["width"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid shape") from error


def _check_snapshot(value: Any, *, context: str, before: bool) -> None:
    try:
        snapshot = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: invalid snapshot JSON") from error
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"{context}: snapshot must be a mapping")
    expected = {
        "status": "ok",
        "process_query_status": "ok",
        "index": EXPECTED_GPU_INDEX,
        "uuid": EXPECTED_GPU_UUID,
        "name": EXPECTED_GPU_NAME,
        "compute_processes": [],
        "memory.total": 32768,
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


def _validate_identity(row: Mapping[str, Any], *, context: str) -> tuple[str, int, str]:
    workload = str(row.get("workload_id"))
    variant = str(row.get("variant"))
    session = _integer(row.get("seed_or_session"), context=f"{context} session")
    if workload not in WORKLOADS or variant not in VARIANTS or session not in SESSIONS:
        raise ValueError(f"{context}: unexpected workload/session/variant")
    expected = {
        "experiment_id": "E22",
        "protocol_version": EXPECTED_PROTOCOL,
        "code_commit": EXPECTED_COMMIT,
        "config_hash": WORKLOADS[workload]["config_hash"],
        "device_id": EXPECTED_DEVICE,
        "reference_commit": EXPECTED_REFERENCE_COMMIT,
        "precision_policy": EXPECTED_PRECISION_POLICY,
    }
    for field, wanted in expected.items():
        if row.get(field) != wanted:
            raise ValueError(
                f"{context}: E22 {field} is {row.get(field)!r}, expected {wanted!r}"
            )
    if _truth(row.get("code_dirty"), context=f"{context} code_dirty"):
        raise ValueError(f"{context}: E22 code_dirty must be false")
    shape = _canonical_shape(row.get("shape"), context=context)
    planned = WORKLOADS[workload]
    expected_shape = {
        "batch": planned["batch"],
        "height": planned["height"],
        "width": planned["width"],
    }
    if shape != expected_shape:
        raise ValueError(f"{context}: E22 shape does not match {workload}")
    _check_snapshot(
        row.get("gpu_snapshot_before"), context=f"{context} before", before=True
    )
    _check_snapshot(row.get("gpu_snapshot_after"), context=f"{context} after", before=False)
    return workload, session, variant


def _validate_runner(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = _sha256(resolved)
    if digest != EXPECTED_RUNNER_SHA256:
        raise ValueError(
            f"E22 runner SHA-256 is {digest}, expected {EXPECTED_RUNNER_SHA256}"
        )
    text = resolved.read_text(encoding="utf-8")
    required = {
        "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "configs/benchmarks/e22_unet_allocator_fraction.yaml",
    }
    missing = sorted(value for value in required if value not in text)
    if missing:
        raise ValueError(f"E22 runner is missing frozen allocator contract: {missing}")
    return {
        "file": resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": digest,
        "allocator_fraction": ALLOCATOR_FRACTION,
        "preallocate": PREALLOCATE,
        "provenance_status": "frozen_wrapper_contract_not_raw_row_embedded",
        "note": ALLOCATOR_PROVENANCE_NOTE,
    }


def _atomic_csv_or_header(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    if rows:
        _atomic_csv(path, rows)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_success_rows(
    rows: Sequence[dict[str, str]],
) -> dict[tuple[str, int, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    observed: set[tuple[str, int, str, str, int]] = set()
    for index, row in enumerate(rows, start=1):
        context = f"success row {index}"
        key = _validate_identity(row, context=context)
        constants: dict[str, Any] = {
            "status": "ok"
            if row.get("phase") != "sustained_sequence"
            else "not_applicable",
            "coordinate_boundary": "tangent_to_tangent",
            "operation_boundary": "unet:update:depth=1",
            "layout": "BHWC/model-pytree",
            "kernel": "None",
            "compiler_mode": EXPECTED_COMPILER_MODE,
            "measured_peak_phase": EXPECTED_MEASURED_PEAK_PHASE,
            "output_retention_policy": EXPECTED_OUTPUT_RETENTION,
            "synchronization_method": EXPECTED_SYNCHRONIZATION,
        }
        for field, wanted in constants.items():
            if row.get(field) != wanted:
                raise ValueError(f"{context}: invalid E22 {field}")
        integer_constants = {
            "warmup_count": 50,
            "repeat_count": TIMED_CALLS,
            "output_nonfinite_count": 0,
            "sustained_sequence_call_count": 0,
        }
        for field, wanted in integer_constants.items():
            if _integer(row.get(field), context=f"{context} {field}") != wanted:
                raise ValueError(f"{context}: invalid E22 {field}")
        sustained_nonfinite = str(row.get("sustained_output_nonfinite_count", "")).strip()
        if sustained_nonfinite not in {"", "0"}:
            raise ValueError(f"{context}: nonfinite sustained output")
        if not _truth(
            row.get("state_evolves_across_calls"), context=f"{context} state evolution"
        ):
            raise ValueError(f"{context}: update state must evolve across calls")
        phase = str(row.get("phase"))
        observation = _integer(row.get("observation"), context=f"{context} observation")
        phase_observation = {
            "compilation": -2,
            "first_compiled_call": -1,
            "sustained_sequence": -3,
        }
        if phase == "steady_state":
            if observation not in range(TIMED_CALLS):
                raise ValueError(f"{context}: invalid steady-state observation")
        elif phase_observation.get(phase) != observation:
            raise ValueError(f"{context}: invalid E22 phase/observation")
        record_key = (*key, phase, observation)
        if record_key in observed:
            raise ValueError(f"{context}: duplicate phase/observation")
        observed.add(record_key)
        if phase == "sustained_sequence":
            if str(row.get("latency_ms", "")).strip():
                raise ValueError(f"{context}: sustained latency must be empty")
        else:
            _finite(row.get("latency_ms"), context=f"{context} latency", positive=True)
        before = json.loads(row["state_summary_before"])
        after = json.loads(row["state_summary_after"])
        for label, state in (("before", before), ("after", after)):
            if not isinstance(state, Mapping):
                raise ValueError(f"{context}: {label} state must be a mapping")
            if int(state.get("nonfinite_count", -1)) != 0:
                raise ValueError(f"{context}: nonfinite {label} state")
            _finite(state.get("l2_norm"), context=f"{context} {label} state L2")
            if (
                int(state.get("leaf_count", 0)) <= 0
                or int(state.get("scalar_count", 0)) <= 0
            ):
                raise ValueError(f"{context}: invalid {label} state counts")
        grouped[key].append(row)
    for key, selected in grouped.items():
        if len(selected) != SUCCESS_ROWS_PER_GROUP:
            raise ValueError(f"E22 success group {key} is incomplete")
        counts = Counter(row["phase"] for row in selected)
        if counts != Counter(
            {
                "compilation": 1,
                "first_compiled_call": 1,
                "sustained_sequence": 1,
                "steady_state": TIMED_CALLS,
            }
        ):
            raise ValueError(f"E22 success phase inventory is incomplete for {key}")
        steady = {
            int(row["observation"]) for row in selected if row["phase"] == "steady_state"
        }
        if steady != set(range(TIMED_CALLS)):
            raise ValueError(f"E22 steady observations are incomplete for {key}")
        for field in (
            "measured_peak_bytes_in_use",
            "measured_peak_pool_bytes",
            "process_reservation_mib",
            "state_summary_before",
            "state_summary_after",
        ):
            if len({row[field] for row in selected}) != 1:
                raise ValueError(f"E22 {field} varies within {key}")
        exemplar = selected[0]
        live = _integer(
            exemplar["measured_peak_bytes_in_use"], context=f"{key} peak live", minimum=1
        )
        pool = _integer(
            exemplar["measured_peak_pool_bytes"], context=f"{key} peak pool", minimum=1
        )
        if pool < live:
            raise ValueError(f"E22 peak pool is below live bytes for {key}")
        _integer(
            exemplar["process_reservation_mib"], context=f"{key} reservation", minimum=1
        )
    for workload in WORKLOADS:
        for session in SESSIONS:
            r_key = (workload, session, "R")
            f_key = (workload, session, "F")
            if r_key in grouped and f_key in grouped:
                r_state = grouped[r_key][0]["state_summary_before"]
                f_state = grouped[f_key][0]["state_summary_before"]
                if r_state != f_state:
                    raise ValueError(
                        f"E22 paired R/F initial states differ for {(workload, session)}"
                    )
    return grouped


def _classify_failure(stdout: str, stderr: str, returncode: int) -> str:
    diagnostic = f"{stdout}\n{stderr}".casefold()
    if any(marker in diagnostic for marker in OOM_MARKERS):
        return "oom"
    if returncode < 0:
        return "terminated_by_signal"
    return "failed"


def _unique_match(pattern: re.Pattern[str], text: str, *, field: str) -> str | None:
    values = set(pattern.findall(text))
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"E22 failure has inconsistent {field}: {sorted(values)}")
    value = next(iter(values))
    return value if isinstance(value, str) else value[0]


def _failure_diagnostics(row: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    stdout = str(row.get("stdout") or "")
    stderr = str(row.get("stderr") or "")
    returncode = _integer(row.get("returncode"), context=f"{context} returncode")
    if returncode == 0:
        raise ValueError(f"{context}: worker failure has zero returncode")
    classified = _classify_failure(stdout, stderr, returncode)
    if row.get("failure_kind") != classified:
        raise ValueError(
            f"{context}: failure_kind {row.get('failure_kind')!r} disagrees with "
            f"explicit worker diagnostic {classified!r}"
        )
    phase = str(row.get("failure_phase"))
    if phase not in FAILURE_PHASES:
        raise ValueError(f"{context}: invalid failure phase {phase!r}")
    requested_text = _unique_match(
        REQUESTED_BYTES_PATTERN, stderr, field="requested allocation bytes"
    )
    rounded_text = _unique_match(
        ROUNDED_BYTES_PATTERN, stderr, field="rounded requested bytes"
    )
    # _unique_match returns the first regex group for multi-group matches; parse
    # total allocation separately to preserve its unit.
    total_values = set(TOTAL_ALLOCATION_PATTERN.findall(stderr))
    if len(total_values) > 1:
        raise ValueError(f"{context}: inconsistent reported total allocations")
    if classified == "oom" and (requested_text is None or not total_values):
        raise ValueError(f"{context}: OOM lacks requested/total allocation diagnostics")
    requested = int(requested_text) if requested_text is not None else None
    rounded = int(rounded_text) if rounded_text is not None else None
    total_value, total_unit = next(iter(total_values)) if total_values else (None, None)
    if total_unit is not None and total_unit.casefold() != "gib":
        raise ValueError(f"{context}: unexpected total-allocation unit {total_unit}")
    total_gib = float(total_value) if total_value is not None else None
    return {
        "failure_kind": classified,
        "failure_phase": phase,
        "returncode": returncode,
        "requested_allocation_bytes": requested,
        "allocator_rounded_requested_bytes": rounded,
        "reported_total_allocation_gib": total_gib,
        "physical_memory_bytes": PHYSICAL_MEMORY_BYTES,
        "nominal_allocator_fraction_capacity_gib": NOMINAL_FRACTION_CAPACITY_GIB,
        "requested_exceeds_physical_memory": (
            requested > PHYSICAL_MEMORY_BYTES if requested is not None else None
        ),
        "requested_exceeds_nominal_fraction_capacity": (
            requested / 1024**3 > NOMINAL_FRACTION_CAPACITY_GIB
            if requested is not None
            else None
        ),
        "reported_total_exceeds_physical_memory": (
            total_gib > 32.0 if total_gib is not None else None
        ),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "diagnostic_excerpt": next(
            (line.strip() for line in stderr.splitlines() if line.strip()), ""
        )[:500],
    }


def _validate_failures(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    output = {}
    for index, row in enumerate(rows, start=1):
        context = f"failure row {index}"
        key = _validate_identity(row, context=context)
        if key in output:
            raise ValueError(f"{context}: duplicate E22 worker failure for {key}")
        if row.get("status") != "failed":
            raise ValueError(f"{context}: failure status must be 'failed'")
        if row.get("operation_boundary") != "unet:update":
            raise ValueError(f"{context}: invalid failure operation boundary")
        output[key] = {**row, **_failure_diagnostics(row, context=context)}
    return output


def _session_outcomes(
    successes: Mapping[tuple[str, int, str], Sequence[dict[str, str]]],
    failures: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    success_source: Path,
    failure_source: Path,
    success_hash: str,
    failure_hash: str,
) -> list[dict[str, Any]]:
    expected = {
        (workload, session, variant)
        for workload in WORKLOADS
        for session in SESSIONS
        for variant in VARIANTS
    }
    overlap = set(successes) & set(failures)
    if overlap:
        raise ValueError(
            f"E22 groups have both success and failure records: {sorted(overlap)}"
        )
    observed = set(successes) | set(failures)
    if observed != expected:
        raise ValueError(
            "E22 workload/variant/session matrix is incomplete; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    output = []
    for key in sorted(expected):
        workload, session, variant = key
        common = {
            "experiment_id": "E22",
            "protocol_version": EXPECTED_PROTOCOL,
            "workload_id": workload,
            "batch": WORKLOADS[workload]["batch"],
            "resolution": WORKLOADS[workload]["height"],
            "variant": variant,
            "session": session,
            "allocator_fraction": ALLOCATOR_FRACTION,
            "preallocate": PREALLOCATE,
            "physical_memory_bytes": PHYSICAL_MEMORY_BYTES,
            "nominal_allocator_fraction_capacity_gib": NOMINAL_FRACTION_CAPACITY_GIB,
            "allocator_setting_provenance": (
                "frozen_wrapper_contract_not_raw_row_embedded"
            ),
        }
        if key in successes:
            rows = successes[key]
            exemplar = rows[0]
            phases = {row["phase"]: row for row in rows if row["phase"] != "steady_state"}
            steady = [
                float(row["latency_ms"]) for row in rows if row["phase"] == "steady_state"
            ]
            live = int(exemplar["measured_peak_bytes_in_use"])
            pool = int(exemplar["measured_peak_pool_bytes"])
            output.append(
                {
                    **common,
                    "outcome": "ok",
                    "steady_state_observations": len(steady),
                    "median_latency_ms": statistics.median(steady),
                    "peak_live_bytes": live,
                    "peak_live_gib": live / 1024**3,
                    "peak_pool_bytes": pool,
                    "peak_pool_gib": pool / 1024**3,
                    "process_reservation_mib": int(exemplar["process_reservation_mib"]),
                    "compilation_ms": float(phases["compilation"]["latency_ms"]),
                    "first_compiled_call_ms": float(
                        phases["first_compiled_call"]["latency_ms"]
                    ),
                    "failure_kind": None,
                    "failure_phase": None,
                    "returncode": None,
                    "requested_allocation_bytes": None,
                    "allocator_rounded_requested_bytes": None,
                    "reported_total_allocation_gib": None,
                    "requested_exceeds_physical_memory": None,
                    "requested_exceeds_nominal_fraction_capacity": None,
                    "reported_total_exceeds_physical_memory": None,
                    "stderr_sha256": None,
                    "stdout_sha256": None,
                    "diagnostic_excerpt": None,
                    "config_hash": WORKLOADS[workload]["config_hash"],
                    "device_id": EXPECTED_DEVICE,
                    "physical_gpu_uuid": EXPECTED_GPU_UUID,
                    "source_file": success_source.name,
                    "source_sha256": success_hash,
                }
            )
        else:
            failure = failures[key]
            output.append(
                {
                    **common,
                    "outcome": failure["failure_kind"],
                    "steady_state_observations": 0,
                    "median_latency_ms": None,
                    "peak_live_bytes": None,
                    "peak_live_gib": None,
                    "peak_pool_bytes": None,
                    "peak_pool_gib": None,
                    "process_reservation_mib": None,
                    "compilation_ms": None,
                    "first_compiled_call_ms": None,
                    "failure_kind": failure["failure_kind"],
                    "failure_phase": failure["failure_phase"],
                    "returncode": failure["returncode"],
                    "requested_allocation_bytes": failure["requested_allocation_bytes"],
                    "allocator_rounded_requested_bytes": failure[
                        "allocator_rounded_requested_bytes"
                    ],
                    "reported_total_allocation_gib": failure[
                        "reported_total_allocation_gib"
                    ],
                    "requested_exceeds_physical_memory": failure[
                        "requested_exceeds_physical_memory"
                    ],
                    "requested_exceeds_nominal_fraction_capacity": failure[
                        "requested_exceeds_nominal_fraction_capacity"
                    ],
                    "reported_total_exceeds_physical_memory": failure[
                        "reported_total_exceeds_physical_memory"
                    ],
                    "stderr_sha256": failure["stderr_sha256"],
                    "stdout_sha256": failure["stdout_sha256"],
                    "diagnostic_excerpt": failure["diagnostic_excerpt"],
                    "config_hash": WORKLOADS[workload]["config_hash"],
                    "device_id": EXPECTED_DEVICE,
                    "physical_gpu_uuid": EXPECTED_GPU_UUID,
                    "source_file": failure_source.name,
                    "source_sha256": failure_hash,
                }
            )
    return output


def _median_range(values: Sequence[float]) -> tuple[Any, Any, Any]:
    if not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


def _workload_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["workload_id"]), str(row["variant"]))].append(row)
    output = []
    for key in sorted(grouped):
        selected = sorted(grouped[key], key=lambda row: int(row["session"]))
        successful = [row for row in selected if row["outcome"] == "ok"]
        latencies = [float(row["median_latency_ms"]) for row in successful]
        peaks = [float(row["peak_live_gib"]) for row in successful]
        pools = [float(row["peak_pool_gib"]) for row in successful]
        latency = _median_range(latencies)
        peak = _median_range(peaks)
        pool = _median_range(pools)
        counts = Counter(str(row["outcome"]) for row in selected)
        output.append(
            {
                "experiment_id": "E22",
                "protocol_version": EXPECTED_PROTOCOL,
                "workload_id": key[0],
                "batch": WORKLOADS[key[0]]["batch"],
                "resolution": WORKLOADS[key[0]]["height"],
                "variant": key[1],
                "expected_sessions": len(SESSIONS),
                "successful_sessions": len(successful),
                "oom_sessions": counts["oom"],
                "terminated_by_signal_sessions": counts["terminated_by_signal"],
                "other_failed_sessions": counts["failed"],
                "all_sessions_successful": len(successful) == len(SESSIONS),
                "capacity_outcome": (
                    "feasible_all_sessions"
                    if len(successful) == len(SESSIONS)
                    else "observed_oom"
                    if counts["oom"]
                    else "observed_worker_failure"
                ),
                "median_latency_ms_successful_sessions": latency[0],
                "min_latency_ms_successful_sessions": latency[1],
                "max_latency_ms_successful_sessions": latency[2],
                "median_peak_live_gib_successful_sessions": peak[0],
                "min_peak_live_gib_successful_sessions": peak[1],
                "max_peak_live_gib_successful_sessions": peak[2],
                "median_peak_pool_gib_successful_sessions": pool[0],
                "min_peak_pool_gib_successful_sessions": pool[1],
                "max_peak_pool_gib_successful_sessions": pool[2],
                "allocator_fraction": ALLOCATOR_FRACTION,
                "preallocate": PREALLOCATE,
                "allocator_setting_provenance": (
                    "frozen_wrapper_contract_not_raw_row_embedded"
                ),
                "config_hash": WORKLOADS[key[0]]["config_hash"],
                "device_id": EXPECTED_DEVICE,
                "physical_gpu_uuid": EXPECTED_GPU_UUID,
            }
        )
    if len(output) != len(WORKLOADS) * len(VARIANTS):
        raise ValueError("E22 workload/variant summary matrix is incomplete")
    return output


def _paired_contrasts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(row["workload_id"]), int(row["session"]), str(row["variant"])): row
        for row in rows
    }
    output = []
    for workload in WORKLOADS:
        for session in SESSIONS:
            reference = indexed[(workload, session, "R")]
            factorized = indexed[(workload, session, "F")]
            if reference["outcome"] != "ok" or factorized["outcome"] != "ok":
                continue
            r_latency = float(reference["median_latency_ms"])
            f_latency = float(factorized["median_latency_ms"])
            r_peak = int(reference["peak_live_bytes"])
            f_peak = int(factorized["peak_live_bytes"])
            output.append(
                {
                    "experiment_id": "E22",
                    "protocol_version": EXPECTED_PROTOCOL,
                    "workload_id": workload,
                    "batch": WORKLOADS[workload]["batch"],
                    "resolution": WORKLOADS[workload]["height"],
                    "session": session,
                    "R_median_latency_ms": r_latency,
                    "F_median_latency_ms": f_latency,
                    "R_over_F_latency_ratio": r_latency / f_latency,
                    "R_peak_live_bytes": r_peak,
                    "F_peak_live_bytes": f_peak,
                    "F_minus_R_peak_live_bytes": f_peak - r_peak,
                    "F_peak_live_reduction_vs_R_percent": 100.0 * (1.0 - f_peak / r_peak),
                    "pairing_rule": "included_only_when_both_R_and_F_completed",
                }
            )
    return output


def export(
    success_path: Path,
    failure_path: Path,
    output_dir: Path,
    *,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    success_path = success_path.resolve()
    failure_path = failure_path.resolve()
    if success_path == failure_path:
        raise ValueError("E22 success and failure inputs must be distinct files")
    runner = _validate_runner(
        runner_path
        if runner_path is not None
        else Path(__file__).resolve().with_name("run_e22.sh")
    )
    success_raw = _read_csv(success_path)
    failure_raw = _read_jsonl(failure_path)
    if not success_raw and not failure_raw:
        raise ValueError("E22 evidence contains no records")
    successes = _validate_success_rows(success_raw)
    failures = _validate_failures(failure_raw)
    success_hash = _sha256(success_path)
    failure_hash = _sha256(failure_path)
    sessions = _session_outcomes(
        successes,
        failures,
        success_source=success_path,
        failure_source=failure_path,
        success_hash=success_hash,
        failure_hash=failure_hash,
    )
    summaries = _workload_summaries(sessions)
    contrasts = _paired_contrasts(sessions)
    failure_diagnostics = [row for row in sessions if row["outcome"] != "ok"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "e22_session_outcomes.csv", sessions)
    _atomic_csv(output_dir / "e22_workload_variant_summary.csv", summaries)
    _atomic_csv_or_header(output_dir / "e22_paired_contrasts.csv", contrasts, PAIRED_FIELDS)
    _atomic_csv_or_header(
        output_dir / "e22_failure_diagnostics.csv",
        failure_diagnostics,
        list(sessions[0]),
    )
    outcome_counts = Counter(str(row["outcome"]) for row in sessions)
    paired_by_workload = Counter(str(row["workload_id"]) for row in contrasts)
    summary = {
        "schema_version": "1.0",
        "analysis_status": "post_hoc_allocator_fraction_and_capacity_diagnostic",
        "scientific_behavior_change": False,
        "identity": {
            "experiment_id": "E22",
            "protocol_version": EXPECTED_PROTOCOL,
            "code_commit": EXPECTED_COMMIT,
            "code_dirty": False,
            "config_hashes": {
                workload: WORKLOADS[workload]["config_hash"] for workload in WORKLOADS
            },
            "device_id": EXPECTED_DEVICE,
            "physical_gpu_uuid": EXPECTED_GPU_UUID,
        },
        "allocator_provenance": runner,
        "sources": {
            "success": {
                "file": success_path.name,
                "bytes": success_path.stat().st_size,
                "sha256": success_hash,
                "rows": len(success_raw),
            },
            "failures": {
                "file": failure_path.name,
                "bytes": failure_path.stat().st_size,
                "sha256": failure_hash,
                "rows": len(failure_raw),
            },
        },
        "pilot_exclusion": {
            "excluded_protocol_version": "1.11",
            "rule": (
                "Every accepted CSV and failure row must have protocol_version=1.15, "
                "code_commit=30142ce2c9f60a115fa77cf844cac14166446b27, "
                "code_dirty=false, and the frozen workload-specific config hash."
            ),
        },
        "matrix": {
            "expected_groups": len(WORKLOADS) * len(SESSIONS) * len(VARIANTS),
            "successful_groups": outcome_counts["ok"],
            "failure_groups": len(failure_diagnostics),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "paired_complete_sessions_by_workload": {
                workload: paired_by_workload[workload] for workload in WORKLOADS
            },
        },
        "failure_diagnostic_note": (
            "Requested bytes and total-allocation GiB are parsed from explicit worker "
            "stderr. The 30.4-GiB capacity is the nominal 0.95 fraction of the "
            "registered 32-GiB budget, not a measured free-memory value."
        ),
        "interpretation": (
            "Paired contrasts include only session/workload cells in which both R and F "
            "completed. Every explicit worker failure remains in the session and failure "
            "diagnostic tables."
        ),
        "output_rows": {
            "e22_session_outcomes.csv": len(sessions),
            "e22_workload_variant_summary.csv": len(summaries),
            "e22_paired_contrasts.csv": len(contrasts),
            "e22_failure_diagnostics.csv": len(failure_diagnostics),
        },
    }
    _atomic_json(output_dir / "e22_audit.json", summary)
    return summary


def main() -> int:
    arguments = _parser().parse_args()
    summary = export(
        arguments.success,
        arguments.failures,
        arguments.output_dir.resolve(),
        runner_path=arguments.runner,
    )
    print(json.dumps(summary["output_rows"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
