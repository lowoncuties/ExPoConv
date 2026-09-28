"""Audit and summarize a paired U-Net workload-scaling campaign.

This module is deliberately analysis-only.  It reads immutable benchmark CSV,
JSON, or JSONL records, validates their experiment identities, and writes
compact tables and one four-panel figure.  It never imports JAX or launches a
benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poincare_exec.config.constants import (
    FIGURE_PDF_METADATA,
    PLOT_COLORS,
    PLOT_MARKERS,
    REPOSITORY_ROOT,
)
from poincare_exec.config.loader import (
    canonical_json,
    config_hash,
    load_yaml,
    require_keys,
    validate_hashed_document,
)

_SUCCESS_STATUSES = {"ok", "not_applicable"}
_FAILURE_PHASES = {
    "initialization",
    "compilation",
    "first_update",
    "warmup_updates",
    "measured_updates",
    "post_measurement",
    # Legacy analysis labels remain accepted for older failure sidecars.
    "warmup",
    "steady_state",
    "later_update",
    "unknown",
}
_PANELS = {"batch", "resolution"}
_E13_BATCH_GRID = (1, 2, 4, 8, 16)
_E13_RESOLUTION_GRID = (128, 256, 512)
_E13_BATCH_FIXED_RESOLUTION = 256
_E13_RESOLUTION_FIXED_BATCH = 8
_E13_SHAPES = {
    *((batch, _E13_BATCH_FIXED_RESOLUTION) for batch in _E13_BATCH_GRID),
    *((_E13_RESOLUTION_FIXED_BATCH, resolution) for resolution in _E13_RESOLUTION_GRID),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be a boolean, received {value!r}")


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, received {value!r}")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"expected an integer-valued number, received {value!r}")
    return int(number)


def _load_one(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            value = value.get("rows")
        if not isinstance(value, list) or any(
            not isinstance(row, Mapping) for row in value
        ):
            raise ValueError(f"JSON input must be a row list or {{'rows': [...]}}: {path}")
        return [dict(row) for row in value]
    raise ValueError(f"unsupported raw evidence format: {path}")


def load_raw_records(
    paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load records and source metadata without changing their payloads."""

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _load_one(path)
        for row_number, row in enumerate(rows, start=1):
            materialized = dict(row)
            materialized["_source_path"] = str(path)
            materialized["_source_row"] = row_number
            records.append(materialized)
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "byte_size": path.stat().st_size,
                "rows": len(rows),
            }
        )
    if not records:
        raise ValueError("raw evidence contains no records")
    return records, sources


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require_keys(
        manifest,
        (
            "schema_version",
            "analysis_id",
            "experiment_id",
            "variants",
            "baseline_variant",
            "candidate_variant",
            "sessions",
            "timed_calls_per_session",
            "memory_budget_bytes",
            "precision_contract",
            "protocol_document",
            "benchmark_config",
            "sweeps",
            "workloads",
            "outputs",
        ),
        name="scaling analysis manifest",
    )
    variants = tuple(str(value) for value in manifest["variants"])
    if variants != ("R", "F"):
        raise ValueError("E13 scaling variants must be ordered exactly as [R, F]")
    baseline = str(manifest["baseline_variant"])
    candidate = str(manifest["candidate_variant"])
    if baseline != "R" or candidate != "F":
        raise ValueError("E13 scaling must pair baseline R against candidate F")
    if int(manifest["sessions"]) < 1 or int(manifest["timed_calls_per_session"]) < 1:
        raise ValueError("sessions and timed_calls_per_session must be positive")
    if int(manifest["memory_budget_bytes"]) <= 0:
        raise ValueError("memory_budget_bytes must be positive")

    sweeps = manifest["sweeps"]
    if not isinstance(sweeps, Mapping) or set(sweeps) != _PANELS:
        raise ValueError("sweeps must define exactly batch and resolution")
    workloads: dict[str, dict[str, Any]] = {}
    for item in manifest["workloads"]:
        require_keys(
            item,
            ("id", "batch", "resolution", "panels"),
            name="scaling workload",
        )
        workload_id = str(item["id"])
        if workload_id in workloads:
            raise ValueError(f"duplicate workload id: {workload_id}")
        panels = tuple(str(value) for value in item["panels"])
        if not panels or set(panels) - _PANELS or len(panels) != len(set(panels)):
            raise ValueError(f"invalid panel membership for {workload_id}: {panels}")
        batch = int(item["batch"])
        resolution = int(item["resolution"])
        if batch <= 0 or resolution <= 0:
            raise ValueError(f"nonpositive shape for {workload_id}")
        workloads[workload_id] = {
            **dict(item),
            "id": workload_id,
            "batch": batch,
            "resolution": resolution,
            "panels": panels,
        }

    for panel in sorted(_PANELS):
        specification = sweeps[panel]
        require_keys(specification, ("values", "fixed"), name=f"{panel} sweep")
        values = [int(value) for value in specification["values"]]
        if (
            not values
            or len(values) != len(set(values))
            or any(value <= 0 for value in values)
        ):
            raise ValueError(f"{panel} sweep values must be unique positive integers")
        fixed = specification["fixed"]
        if not isinstance(fixed, Mapping):
            raise ValueError(f"{panel} sweep fixed settings must be a mapping")
        selected = [item for item in workloads.values() if panel in item["panels"]]
        observed = [
            item["batch" if panel == "batch" else "resolution"] for item in selected
        ]
        if sorted(observed) != sorted(values):
            raise ValueError(
                f"{panel} workload values {sorted(observed)} do not match "
                f"declared grid {sorted(values)}"
            )
        fixed_field = "resolution" if panel == "batch" else "batch"
        fixed_value = int(fixed[fixed_field])
        if any(item[fixed_field] != fixed_value for item in selected):
            raise ValueError(f"{panel} sweep does not preserve {fixed_field}={fixed_value}")
        expected_values = _E13_BATCH_GRID if panel == "batch" else _E13_RESOLUTION_GRID
        expected_fixed = (
            _E13_BATCH_FIXED_RESOLUTION if panel == "batch" else _E13_RESOLUTION_FIXED_BATCH
        )
        if tuple(values) != expected_values or fixed_value != expected_fixed:
            raise ValueError(f"{panel} sweep does not match the frozen E13 R/F grid")
    observed_shapes = {
        (int(item["batch"]), int(item["resolution"])) for item in workloads.values()
    }
    if len(workloads) != 7 or observed_shapes != _E13_SHAPES:
        raise ValueError("workloads must be the seven unique shapes in the E13 R/F grid")
    return workloads


def _provenance_contract(
    manifest: Mapping[str, Any], workloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    protocol_relative, protocol_sha256 = validate_hashed_document(
        manifest, field="protocol_document"
    )
    config_relative, config_sha256 = validate_hashed_document(
        manifest, field="benchmark_config"
    )
    protocol = load_yaml(REPOSITORY_ROOT / protocol_relative)
    benchmark = load_yaml(REPOSITORY_ROOT / config_relative)
    require_keys(
        benchmark,
        (
            "experiment_id",
            "variants",
            "sessions",
            "timed_calls_per_session",
            "memory_budget_bytes",
            "workloads",
        ),
        name="E13 benchmark config",
    )
    scalar_checks = {
        "experiment_id": str(benchmark["experiment_id"]) == str(manifest["experiment_id"]),
        "variants": list(benchmark["variants"]) == list(manifest["variants"]),
        "sessions": int(benchmark["sessions"]) == int(manifest["sessions"]),
        "timed_calls": int(benchmark["timed_calls_per_session"])
        == int(manifest["timed_calls_per_session"]),
        "memory_budget": int(benchmark["memory_budget_bytes"])
        == int(manifest["memory_budget_bytes"]),
    }
    failed = [name for name, passed in scalar_checks.items() if not passed]
    if failed:
        raise ValueError(
            "analysis manifest disagrees with its hashed benchmark config: "
            + ", ".join(failed)
        )

    benchmark_workloads = {str(item["id"]): item for item in benchmark["workloads"]}
    if set(benchmark_workloads) != set(workloads):
        raise ValueError("analysis and benchmark configs declare different workloads")
    expected_hashes = {}
    for workload_id, planned in workloads.items():
        registered = benchmark_workloads[workload_id]
        shape = registered.get("shape", {})
        expected_shape = {
            "batch": int(planned["batch"]),
            "height": int(planned["resolution"]),
            "width": int(planned["resolution"]),
        }
        observed_shape = {
            "batch": int(shape.get("batch", -1)),
            "height": int(shape.get("height", -1)),
            "width": int(shape.get("width", -1)),
        }
        if observed_shape != expected_shape:
            raise ValueError(
                f"hashed benchmark shape mismatch for {workload_id}: "
                f"{observed_shape} != {expected_shape}"
            )
        if registered.get("kind") != "unet" or registered.get("mode") != "update":
            raise ValueError(f"{workload_id} is not a U-Net complete-update workload")
        expected_hashes[workload_id] = config_hash(protocol, benchmark, registered)
    return {
        "protocol_path": protocol_relative,
        "protocol_sha256": protocol_sha256,
        "protocol_version": str(protocol["protocol_version"]),
        "benchmark_config_path": config_relative,
        "benchmark_config_sha256": config_sha256,
        "expected_config_hashes": expected_hashes,
    }


def _shape(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("shape")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("every raw record must contain a mapping-valued shape")
    return dict(value)


def _snapshot_gpu_uuid(row: Mapping[str, Any]) -> str | None:
    """Resolve one physical UUID from the available before/after snapshots."""

    values = set()
    for field in ("gpu_snapshot_before", "gpu_snapshot_after"):
        snapshot = row.get(field)
        if isinstance(snapshot, str) and snapshot.strip():
            snapshot = json.loads(snapshot)
        if isinstance(snapshot, Mapping) and snapshot.get("uuid"):
            values.add(str(snapshot["uuid"]))
    if len(values) > 1:
        raise ValueError(f"one record contains multiple physical GPU UUIDs: {values}")
    return next(iter(values)) if values else None


def _normalized_precision_policy(value: Any) -> dict[str, Any]:
    """Normalize worker and coordinator encodings of the same FP32 policy."""

    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("precision_policy must be a JSON mapping")
    return {
        "storage_dtype": value.get("storage_dtype", value.get("primary")),
        "matmul_precision": value.get("matmul_precision"),
        "reduced_precision_reduction": value.get(
            "reduced_precision_reduction",
            value.get("reduced_precision_enabled"),
        ),
        "tf32": value.get("tf32"),
    }


def _failure_kind(row: Mapping[str, Any]) -> str:
    reported = str(row.get("failure_kind") or row.get("error_kind") or "").strip().lower()
    status = str(row.get("status", "")).strip().lower()
    if reported in {"oom", "out_of_memory", "resource_exhausted"}:
        return "oom"
    if status in {"oom", "out_of_memory", "resource_exhausted"}:
        return "oom"
    if status == "nonfinite" or reported == "nonfinite":
        return "nonfinite"
    return reported or "failed"


def _failure_message(row: Mapping[str, Any]) -> str | None:
    for field in ("failure_reason", "reason", "exception_message", "message", "stderr"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _record_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["workload_id"]),
        int(row["seed_or_session"]),
        str(row["variant"]),
    )


def _validate_records(
    records: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    workloads: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "experiment_id",
        "protocol_version",
        "variant",
        "config_hash",
        "code_commit",
        "code_dirty",
        "device_id",
        "seed_or_session",
        "status",
        "workload_id",
        "shape",
        "operation_boundary",
        "precision_policy",
    }
    variants = {str(value) for value in manifest["variants"]}
    sessions = set(range(int(manifest["sessions"])))
    identities: dict[str, set[str]] = defaultdict(set)
    device_id_encodings: set[str] = set()
    precision_policy_encodings: set[str] = set()
    physical_gpu_uuids: set[str] = set()
    records_without_gpu_uuid = 0
    config_by_workload: dict[str, set[str]] = defaultdict(set)
    observed_record_ids: set[tuple[str, int, str, str, str]] = set()

    for row in records:
        missing = sorted(field for field in required if field not in row)
        if missing:
            location = f"{row.get('_source_path')}:{row.get('_source_row')}"
            raise ValueError(
                f"raw record {location} is missing fields: {', '.join(missing)}"
            )
        workload_id, session, variant = _record_key(row)
        if str(row["experiment_id"]) != str(manifest["experiment_id"]):
            raise ValueError(f"unexpected experiment_id in {_record_key(row)}")
        if str(row["protocol_version"]) != str(provenance["protocol_version"]):
            raise ValueError(f"unexpected protocol_version in {_record_key(row)}")
        if (
            workload_id not in workloads
            or variant not in variants
            or session not in sessions
        ):
            raise ValueError(f"unexpected workload/session/variant: {_record_key(row)}")
        if bool(manifest.get("require_clean_code", True)) and _as_bool(
            row["code_dirty"], field="code_dirty"
        ):
            raise ValueError(f"dirty code identity in {_record_key(row)}")
        shape = _shape(row)
        expected = workloads[workload_id]
        observed_shape = {
            "batch": int(shape.get("batch", -1)),
            "height": int(shape.get("height", -1)),
            "width": int(shape.get("width", -1)),
        }
        expected_shape = {
            "batch": int(expected["batch"]),
            "height": int(expected["resolution"]),
            "width": int(expected["resolution"]),
        }
        if observed_shape != expected_shape:
            raise ValueError(
                f"shape mismatch for {workload_id}: {observed_shape} != {expected_shape}"
            )
        boundary = str(row["operation_boundary"])
        if not boundary.startswith(
            str(manifest.get("operation_boundary_prefix", "unet:update"))
        ):
            raise ValueError(
                f"unexpected operation boundary for {_record_key(row)}: {boundary}"
            )
        for field in ("protocol_version", "code_commit"):
            if field in row and row[field] not in (None, ""):
                identities[field].add(str(row[field]))
        device_id_encodings.add(str(row["device_id"]))
        physical_uuid = _snapshot_gpu_uuid(row)
        if physical_uuid is None:
            records_without_gpu_uuid += 1
        else:
            physical_gpu_uuids.add(physical_uuid)
        precision_policy_encodings.add(str(row["precision_policy"]))
        observed_precision = _normalized_precision_policy(row["precision_policy"])
        expected_precision = dict(manifest["precision_contract"])
        if observed_precision != expected_precision:
            raise ValueError(
                f"precision policy mismatch for {_record_key(row)}: "
                f"{observed_precision} != {expected_precision}"
            )
        config_by_workload[workload_id].add(str(row["config_hash"]))
        expected_hash = provenance["expected_config_hashes"][workload_id]
        if str(row["config_hash"]) != expected_hash:
            raise ValueError(
                f"config hash mismatch for {_record_key(row)}: "
                f"{row['config_hash']} != {expected_hash}"
            )

        phase = str(row.get("phase", ""))
        observation = str(row.get("observation", ""))
        record_id = (workload_id, session, variant, phase, observation)
        if phase and record_id in observed_record_ids:
            raise ValueError(f"duplicate phase/observation record: {record_id}")
        if phase:
            observed_record_ids.add(record_id)

    for field in ("protocol_version", "code_commit"):
        count = len(identities[field])
        if count != 1:
            raise ValueError(
                f"campaign must have one {field}; observed {sorted(identities[field])}"
            )
    if len(physical_gpu_uuids) != 1:
        raise ValueError(
            "campaign must resolve to one physical GPU UUID; observed "
            f"{sorted(physical_gpu_uuids)}"
        )
    if bool(manifest.get("require_physical_gpu_uuid", True)) and records_without_gpu_uuid:
        raise ValueError(
            f"{records_without_gpu_uuid} raw records lack a physical GPU UUID snapshot"
        )
    mixed_configs = {
        workload_id: sorted(values)
        for workload_id, values in config_by_workload.items()
        if len(values) != 1
    }
    if mixed_configs:
        raise ValueError(f"workloads have mixed config identities: {mixed_configs}")
    return {field: next(iter(values)) for field, values in identities.items()} | {
        "physical_gpu_uuid": next(iter(physical_gpu_uuids)),
        "device_id_encodings": sorted(device_id_encodings),
        "precision_policy_encodings": sorted(precision_policy_encodings),
        "normalized_precision_policy": dict(manifest["precision_contract"]),
        "config_hashes": {
            workload_id: next(iter(values))
            for workload_id, values in sorted(config_by_workload.items())
        },
    }


def _failure_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kinds = {_failure_kind(row) for row in records}
    phases = {
        str(row.get("failure_phase") or row.get("failure_stage") or "unknown")
        .strip()
        .lower()
        for row in records
    }
    invalid = phases - _FAILURE_PHASES
    if invalid:
        raise ValueError(f"unrecognized failure phases: {sorted(invalid)}")
    messages = [message for row in records if (message := _failure_message(row))]
    return {
        "failure_kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
        "failure_phase": next(iter(phases)) if len(phases) == 1 else "mixed",
        "failure_message": " | ".join(dict.fromkeys(messages)) or None,
        "returncodes": sorted(
            {
                value
                for row in records
                if (value := _optional_int(row.get("returncode"))) is not None
            }
        ),
    }


def _session_outcome(
    records: Sequence[dict[str, Any]],
    *,
    workload: Mapping[str, Any],
    variant: str,
    session: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected_calls = int(manifest["timed_calls_per_session"])
    base = {
        "experiment_id": manifest["experiment_id"],
        "workload_id": workload["id"],
        "panels": "|".join(workload["panels"]),
        "batch": int(workload["batch"]),
        "resolution": int(workload["resolution"]),
        "variant": variant,
        "session": session,
        "expected_steady_state_observations": expected_calls,
    }
    if not records:
        return {
            **base,
            "outcome": "missing",
            "steady_state_observations": 0,
            "median_latency_ms": None,
            "peak_live_gib": None,
            "peak_pool_gib": None,
            "failure_kind": "missing",
            "failure_phase": "unknown",
            "failure_message": "no raw record for expected group",
            "returncodes": "[]",
            "protocol_version": None,
            "code_commit": None,
            "config_hash": None,
            "device_id": None,
            "physical_gpu_uuid": None,
            "operation_boundary": None,
            "source_files": "[]",
        }

    statuses = [str(row.get("status", "")).strip().lower() for row in records]
    failures = [
        row
        for row, status in zip(records, statuses, strict=True)
        if status not in _SUCCESS_STATUSES
    ]
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        phases[str(row.get("phase", ""))].append(row)
    steady = phases.get("steady_state", [])
    latency_values = [_optional_float(row.get("latency_ms")) for row in steady]
    valid_latencies = [value for value in latency_values if value is not None and value > 0]

    structural_complete = (
        len(phases.get("compilation", [])) == 1
        and len(phases.get("first_compiled_call", [])) == 1
        and len(steady) == expected_calls
        and len(valid_latencies) == expected_calls
    )
    if len(steady) > expected_calls:
        raise ValueError(
            f"too many steady-state rows for {(workload['id'], session, variant)}"
        )
    if failures:
        failure = _failure_summary(failures)
        if any(status in _SUCCESS_STATUSES for status in statuses):
            outcome = "partial"
        elif failure["failure_kind"] in {"oom", "nonfinite"}:
            outcome = failure["failure_kind"]
        else:
            outcome = "failed"
    elif structural_complete:
        failure = {
            "failure_kind": None,
            "failure_phase": None,
            "failure_message": None,
            "returncodes": [],
        }
        outcome = "ok"
    else:
        failure = {
            "failure_kind": "partial",
            "failure_phase": "unknown",
            "failure_message": "incomplete required phase or observation count",
            "returncodes": [],
        }
        outcome = "partial"

    peak_live_bytes = [
        value
        for row in records
        if (value := _optional_int(row.get("measured_peak_bytes_in_use"))) is not None
    ]
    peak_pool_bytes = [
        value
        for row in records
        if (value := _optional_int(row.get("measured_peak_pool_bytes"))) is not None
    ]
    exemplar = records[0]
    return {
        **base,
        "outcome": outcome,
        "steady_state_observations": len(steady),
        "median_latency_ms": statistics.median(valid_latencies)
        if outcome == "ok"
        else None,
        "peak_live_gib": max(peak_live_bytes) / 1024**3
        if outcome == "ok" and peak_live_bytes
        else None,
        "peak_pool_gib": max(peak_pool_bytes) / 1024**3
        if outcome == "ok" and peak_pool_bytes
        else None,
        "failure_kind": failure["failure_kind"],
        "failure_phase": failure["failure_phase"],
        "failure_message": failure["failure_message"],
        "returncodes": canonical_json(failure["returncodes"]),
        "protocol_version": exemplar["protocol_version"],
        "code_commit": exemplar["code_commit"],
        "config_hash": exemplar["config_hash"],
        "device_id": exemplar["device_id"],
        "physical_gpu_uuid": _snapshot_gpu_uuid(exemplar),
        "operation_boundary": exemplar["operation_boundary"],
        "source_files": canonical_json(
            sorted({str(row["_source_path"]) for row in records})
        ),
    }


def build_session_outcomes(
    records: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    workloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one explicit outcome for every planned workload/session/path."""

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_record_key(row)].append(row)
    output = []
    for workload in workloads.values():
        for session in range(int(manifest["sessions"])):
            for variant in manifest["variants"]:
                key = (workload["id"], session, str(variant))
                output.append(
                    _session_outcome(
                        grouped.get(key, []),
                        workload=workload,
                        variant=str(variant),
                        session=session,
                        manifest=manifest,
                    )
                )
    return output


def _median_range(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


def summarize_workloads(
    sessions: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Summarize successful sessions while retaining every failure count."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sessions:
        grouped[(str(row["workload_id"]), str(row["variant"]))].append(row)
    output = []
    expected = int(manifest["sessions"])
    for key in sorted(grouped):
        selected = sorted(grouped[key], key=lambda row: int(row["session"]))
        exemplar = selected[0]
        succeeded = [row for row in selected if row["outcome"] == "ok"]
        latencies = [float(row["median_latency_ms"]) for row in succeeded]
        peaks = [
            float(row["peak_live_gib"])
            for row in succeeded
            if row["peak_live_gib"] is not None
        ]
        latency, latency_min, latency_max = _median_range(latencies)
        peak, peak_min, peak_max = _median_range(peaks)
        counts = {
            outcome: sum(row["outcome"] == outcome for row in selected)
            for outcome in {str(row["outcome"]) for row in selected}
        }
        output.append(
            {
                "experiment_id": exemplar["experiment_id"],
                "workload_id": exemplar["workload_id"],
                "panels": exemplar["panels"],
                "batch": exemplar["batch"],
                "resolution": exemplar["resolution"],
                "variant": exemplar["variant"],
                "expected_sessions": expected,
                "successful_sessions": len(succeeded),
                "oom_sessions": counts.get("oom", 0),
                "failed_sessions": counts.get("failed", 0)
                + counts.get("nonfinite", 0)
                + counts.get("mixed", 0),
                "partial_sessions": counts.get("partial", 0),
                "missing_sessions": counts.get("missing", 0),
                "all_sessions_successful": len(succeeded) == expected,
                "latency_complete": len(latencies) == expected,
                "memory_complete": len(peaks) == expected,
                "median_latency_ms_successful_sessions": latency,
                "min_latency_ms_successful_sessions": latency_min,
                "max_latency_ms_successful_sessions": latency_max,
                "median_peak_live_gib_successful_sessions": peak,
                "min_peak_live_gib_successful_sessions": peak_min,
                "max_peak_live_gib_successful_sessions": peak_max,
                "capacity_outcome": (
                    "feasible"
                    if len(succeeded) == expected
                    else "infeasible_oom"
                    if counts.get("oom", 0)
                    else "incomplete"
                ),
                "protocol_version": next(
                    (
                        row["protocol_version"]
                        for row in selected
                        if row["protocol_version"]
                    ),
                    None,
                ),
                "code_commit": next(
                    (row["code_commit"] for row in selected if row["code_commit"]), None
                ),
                "config_hash": next(
                    (row["config_hash"] for row in selected if row["config_hash"]), None
                ),
                "device_id": next(
                    (row["device_id"] for row in selected if row["device_id"]), None
                ),
                "physical_gpu_uuid": next(
                    (
                        row["physical_gpu_uuid"]
                        for row in selected
                        if row["physical_gpu_uuid"]
                    ),
                    None,
                ),
            }
        )
    return output


def _ratio_interval(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    logs = [math.log(value) for value in values]
    mean_log = statistics.mean(logs)
    estimate = math.exp(mean_log)
    if len(logs) < 2:
        return estimate, None, None
    half_width = float(
        student_t.ppf(0.975, df=len(logs) - 1)
        * statistics.stdev(logs)
        / math.sqrt(len(logs))
    )
    return estimate, math.exp(mean_log - half_width), math.exp(mean_log + half_width)


def summarize_paired_ratios(
    sessions: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Compute matched ratios only for sessions in which both paths succeeded."""

    baseline = str(manifest["baseline_variant"])
    candidate = str(manifest["candidate_variant"])
    indexed = {
        (str(row["workload_id"]), int(row["session"]), str(row["variant"])): row
        for row in sessions
    }
    workload_ids = list(dict.fromkeys(str(row["workload_id"]) for row in sessions))
    output = []
    for workload_id in workload_ids:
        latency_pairs = []
        memory_pairs = []
        paired_ids = []
        memory_ids = []
        for session in range(int(manifest["sessions"])):
            reference = indexed[(workload_id, session, baseline)]
            factorized = indexed[(workload_id, session, candidate)]
            if reference["outcome"] != "ok" or factorized["outcome"] != "ok":
                continue
            reference_latency = float(reference["median_latency_ms"])
            candidate_latency = float(factorized["median_latency_ms"])
            latency_pairs.append(reference_latency / candidate_latency)
            paired_ids.append(session)
            if (
                reference["peak_live_gib"] is not None
                and factorized["peak_live_gib"] is not None
            ):
                memory_pairs.append(
                    float(reference["peak_live_gib"]) / float(factorized["peak_live_gib"])
                )
                memory_ids.append(session)
        speedup, speedup_lower, speedup_upper = _ratio_interval(latency_pairs)
        memory_ratio, memory_lower, memory_upper = _ratio_interval(memory_pairs)
        exemplar = indexed[(workload_id, 0, baseline)]
        output.append(
            {
                "experiment_id": exemplar["experiment_id"],
                "workload_id": workload_id,
                "panels": exemplar["panels"],
                "batch": exemplar["batch"],
                "resolution": exemplar["resolution"],
                "baseline_variant": baseline,
                "candidate_variant": candidate,
                "expected_paired_sessions": int(manifest["sessions"]),
                "paired_latency_sessions": len(latency_pairs),
                "paired_session_ids": canonical_json(paired_ids),
                "session_latency_ratios": canonical_json(latency_pairs),
                "geometric_mean_latency_ratio": speedup,
                "log_t_95_interval_lower": speedup_lower,
                "log_t_95_interval_upper": speedup_upper,
                "latency_classification": (
                    "candidate_faster"
                    if speedup_lower is not None and speedup_lower > 1
                    else "candidate_slower"
                    if speedup_upper is not None and speedup_upper < 1
                    else "inconclusive"
                    if speedup is not None
                    else "not_estimable"
                ),
                "paired_memory_sessions": len(memory_pairs),
                "paired_memory_session_ids": canonical_json(memory_ids),
                "session_peak_live_ratios": canonical_json(memory_pairs),
                "geometric_mean_peak_live_ratio": memory_ratio,
                "peak_live_log_t_95_interval_lower": memory_lower,
                "peak_live_log_t_95_interval_upper": memory_upper,
                "all_expected_sessions_paired": len(latency_pairs)
                == int(manifest["sessions"]),
                "estimand": (
                    "geometric mean of paired session-level R/F ratios; "
                    "values above one favor F"
                ),
            }
        )
    return output


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(materialized[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(materialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _plot_panel(
    axis: Any,
    *,
    panel: str,
    metric: str,
    workloads: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
    variants: Sequence[str],
    memory_budget_gib: float,
) -> None:
    selected = [item for item in workloads.values() if panel in item["panels"]]
    field = "batch" if panel == "batch" else "resolution"
    selected.sort(key=lambda item: int(item[field]))
    positions = np.arange(len(selected), dtype=float)
    width = 0.08
    panel_metric_values: list[float] = []
    for variant_index, variant in enumerate(variants):
        x_values = positions + (variant_index - (len(variants) - 1) / 2) * width
        y_values: list[float] = []
        eligible_x: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        for x_value, workload in zip(x_values, selected, strict=True):
            row = summaries[(workload["id"], variant)]
            complete_field = (
                "latency_complete" if metric == "latency" else "memory_complete"
            )
            value_field = (
                "median_latency_ms_successful_sessions"
                if metric == "latency"
                else "median_peak_live_gib_successful_sessions"
            )
            minimum_field = (
                "min_latency_ms_successful_sessions"
                if metric == "latency"
                else "min_peak_live_gib_successful_sessions"
            )
            maximum_field = (
                "max_latency_ms_successful_sessions"
                if metric == "latency"
                else "max_peak_live_gib_successful_sessions"
            )
            if row[complete_field] and row[value_field] is not None:
                value = float(row[value_field])
                eligible_x.append(float(x_value))
                y_values.append(value)
                panel_metric_values.append(value)
                lower_errors.append(value - float(row[minimum_field]))
                upper_errors.append(float(row[maximum_field]) - value)
                continue
            marker = (
                "X"
                if int(row["oom_sessions"])
                else "P"
                if int(row["failed_sessions"])
                else "|"
            )
            axis.scatter(
                [x_value],
                [0.96],
                marker=marker,
                color=PLOT_COLORS.get(variant, "black"),
                s=45,
                transform=axis.get_xaxis_transform(),
                clip_on=False,
                zorder=5,
            )
        if eligible_x:
            axis.errorbar(
                eligible_x,
                y_values,
                yerr=np.asarray([lower_errors, upper_errors]),
                color=PLOT_COLORS.get(variant, "black"),
                marker=PLOT_MARKERS.get(variant, "o"),
                linewidth=1.5,
                markersize=5,
                capsize=2.5,
                label=variant,
                zorder=3,
            )
    axis.set_xticks(positions, [str(item[field]) for item in selected])
    axis.set_xlabel(
        "Batch size" if panel == "batch" else "Input resolution (pixels per side)"
    )
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.set_axisbelow(True)
    if metric == "latency":
        axis.set_ylabel("Complete-update latency (ms)")
        axis.set_yscale("log")
    else:
        axis.set_ylabel("Process-lifetime peak live allocation (GiB)")
        axis.set_yscale("log")
        axis.axhline(
            memory_budget_gib,
            color="#555555",
            linestyle=(0, (3, 3)),
            linewidth=1,
            label="VRAM budget",
        )
        positive_minimum = min(panel_metric_values) if panel_metric_values else 0.25
        positive_maximum = max(panel_metric_values) if panel_metric_values else 1.0
        axis.set_ylim(
            bottom=max(positive_minimum / 1.5, 0.05),
            top=max(memory_budget_gib * 1.12, positive_maximum * 1.4),
        )
        tick_candidates = (0.25, 0.5, 1, 2, 4, 8, 16, 32)
        lower, upper = axis.get_ylim()
        ticks = [value for value in tick_candidates if lower <= value <= upper]
        axis.set_yticks(ticks, [f"{value:g}" for value in ticks])


def plot_scaling_figure(
    path_prefix: Path,
    *,
    workload_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    workloads: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Write the latency/memory by batch/resolution four-panel figure."""

    summaries = {
        (str(row["workload_id"]), str(row["variant"])): row for row in workload_rows
    }
    variants = [str(value) for value in manifest["variants"]]
    memory_budget_gib = int(manifest["memory_budget_bytes"]) / 1024**3
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    figure.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.10,
        top=0.81,
        hspace=0.48,
        wspace=0.30,
    )
    specifications = (
        (axes[0, 0], "batch", "latency", "A  Update latency by batch size"),
        (axes[0, 1], "resolution", "latency", "B  Update latency by resolution"),
        (axes[1, 0], "batch", "memory", "C  Peak allocation by batch size"),
        (axes[1, 1], "resolution", "memory", "D  Peak allocation by resolution"),
    )
    for axis, panel, metric, title in specifications:
        _plot_panel(
            axis,
            panel=panel,
            metric=metric,
            workloads=workloads,
            summaries=summaries,
            variants=variants,
            memory_budget_gib=memory_budget_gib,
        )
        axis.set_title(title, loc="left", fontweight="bold")
    handles = [
        Line2D(
            [],
            [],
            color=PLOT_COLORS.get(variant, "black"),
            marker=PLOT_MARKERS.get(variant, "o"),
            label=variant,
        )
        for variant in variants
    ]
    handles.extend(
        [
            Line2D([], [], color="black", marker="X", linestyle="none", label="OOM"),
            Line2D(
                [], [], color="black", marker="P", linestyle="none", label="Failed/partial"
            ),
            Line2D(
                [],
                [],
                color="#555555",
                linestyle=(0, (3, 3)),
                label="VRAM budget",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(handles),
        frameon=False,
    )
    figure.suptitle(
        "Poincaré U-Net complete-update scaling",
        fontsize=11,
        y=0.985,
    )
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path_prefix.with_suffix(".pdf")
    png_path = path_prefix.with_suffix(".png")
    figure.savefig(
        pdf_path,
        metadata=dict(FIGURE_PDF_METADATA),
        facecolor="white",
    )
    figure.savefig(
        png_path,
        dpi=220,
        metadata={"Software": "poincare-execution-experiments"},
        facecolor="white",
    )
    plt.close(figure)
    return pdf_path, png_path


def export_scaling_campaign(
    raw_paths: Sequence[str | Path],
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit raw records and export all scaling tables and the figure."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = load_yaml(manifest_file)
    workloads = _validate_manifest(manifest)
    provenance = _provenance_contract(manifest, workloads)
    records, sources = load_raw_records(raw_paths)
    identity = _validate_records(records, manifest, workloads, provenance)
    session_rows = build_session_outcomes(records, manifest, workloads)
    workload_rows = summarize_workloads(session_rows, manifest)
    paired_rows = summarize_paired_ratios(session_rows, manifest)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_names = manifest["outputs"]
    required_output_keys = (
        "session_outcomes",
        "workload_summary",
        "paired_ratios",
        "figure_stem",
        "audit_summary",
    )
    missing_outputs = [key for key in required_output_keys if key not in output_names]
    if missing_outputs:
        raise ValueError(f"manifest outputs missing: {', '.join(missing_outputs)}")

    def output_path(name: str) -> Path:
        relative = Path(str(output_names[name]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"outputs.{name} must stay below --output-dir")
        resolved = (destination / relative).resolve()
        if not resolved.is_relative_to(destination):
            raise ValueError(f"outputs.{name} escapes --output-dir")
        return resolved

    paths = {
        "session_outcomes": output_path("session_outcomes"),
        "workload_summary": output_path("workload_summary"),
        "paired_ratios": output_path("paired_ratios"),
        "audit_summary": output_path("audit_summary"),
    }
    figure_prefix = output_path("figure_stem")
    prospective = [
        *paths.values(),
        figure_prefix.with_suffix(".pdf"),
        figure_prefix.with_suffix(".png"),
    ]
    existing = [str(path) for path in prospective if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite analysis outputs: " + ", ".join(existing)
        )

    _atomic_csv(paths["session_outcomes"], session_rows)
    _atomic_csv(paths["workload_summary"], workload_rows)
    _atomic_csv(paths["paired_ratios"], paired_rows)
    figure_pdf, figure_png = plot_scaling_figure(
        figure_prefix,
        workload_rows=workload_rows,
        manifest=manifest,
        workloads=workloads,
    )
    outcome_counts: dict[str, int] = defaultdict(int)
    for row in session_rows:
        outcome_counts[str(row["outcome"])] += 1
    summary = {
        "schema_version": manifest["schema_version"],
        "analysis_id": manifest["analysis_id"],
        "experiment_id": manifest["experiment_id"],
        "derivation": "CPU-only validation and aggregation; no benchmark was launched.",
        "analysis_manifest": {
            "path": str(manifest_file),
            "sha256": _sha256(manifest_file),
        },
        "raw_sources": sources,
        "validated_identity": identity,
        "provenance_contract": provenance,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "session_rows": len(session_rows),
        "workload_rows": len(workload_rows),
        "paired_rows": len(paired_rows),
        "statistical_method": manifest.get("statistical_method"),
        "memory_scope": manifest.get("memory_scope"),
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                **paths,
                "figure_pdf": figure_pdf,
                "figure_png": figure_png,
            }.items()
            if name != "audit_summary"
        },
        "all_planned_groups_accounted_for": len(session_rows)
        == len(workloads) * int(manifest["sessions"]) * len(manifest["variants"]),
    }
    _atomic_json(paths["audit_summary"], summary)
    return summary
