#!/usr/bin/env python3
"""Validate E24/E25/E27 evidence and export manuscript-facing tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

from poincare_exec.artifacts.provenance import git_provenance, precision_policy
from poincare_exec.config.loader import canonical_json, config_hash, load_yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "df3f6f556a314f74a6966701b38ac5d135a10a62"
EXPECTED_PROTOCOL_VERSION = "1.17"
EXPORT_SCHEMA_VERSION = "1.1"
EXPECTED_DEVICE = {
    "hostname": "aibox2a6000",
    "physical_index": 1,
    "uuid": "GPU-6daddab9-4b59-fdde-bf0b-09ecd8f31723",
    "name": "NVIDIA RTX A6000",
    "logical_id": 0,
}
EXPECTED_REFERENCE_COMMIT = canonical_json(
    {
        "hyperbolic_unet": "bec504e2f4d198ee4c438579b86d72f9d8186d78",
        "poincare_resnet": "fbbbbf890571115444b89b114f9ed52d128d5aa3",
        "hnnpp": "28737e22822562ac18d5ea03f8c3e3929e945a83",
    }
)

COHORTS: dict[str, dict[str, Any]] = {
    "E24": {
        "config": "configs/benchmarks/e24_unet_mechanism_ablation.yaml",
        "raw": "e24_unet_mechanism_ablation.csv",
        "runtime_gate": "e24_e25_runtime_gate.json",
        "environment": "e24_e25_environment.json",
        "runtime_role": "pinned_jax_0_4_25_cuda_11_8",
        "jax_version": "0.4.25",
        "jaxlib_version": "0.4.25",
        "ld_library_policy": "pinned_cuda_11_8",
        "variant_roles": {
            "R": "reference_materialized_ball_geometry",
            "M": "materialized_patch_exact_score_control",
            "F": "factorized_primary_method",
        },
        "contrasts": (
            (
                "R",
                "M",
                "geometry_removal_with_patch_materialization_fixed",
            ),
            (
                "M",
                "F",
                "patch_materialization_and_backend_with_geometry_removed",
            ),
        ),
    },
    "E25": {
        "config": "configs/benchmarks/e25_euclidean_context.yaml",
        "raw": "e25_euclidean_context.csv",
        "runtime_gate": "e24_e25_runtime_gate.json",
        "environment": "e24_e25_environment.json",
        "runtime_role": "pinned_jax_0_4_25_cuda_11_8",
        "jax_version": "0.4.25",
        "jaxlib_version": "0.4.25",
        "ld_library_policy": "pinned_cuda_11_8",
        "variant_roles": {
            "R": "reference_poincare_execution",
            "F": "factorized_poincare_execution",
            "E": "euclidean_same_topology_context",
        },
        "contrasts": (
            ("R", "F", "factorized_vs_reference_poincare"),
            ("R", "E", "reference_poincare_cost_relative_to_euclidean"),
            ("F", "E", "factorized_poincare_cost_relative_to_euclidean"),
        ),
    },
    "E27": {
        "config": "configs/benchmarks/e27_current_jax_unet_update.yaml",
        "raw": "e27_current_jax_unet_update.csv",
        "runtime_gate": "e27_runtime_gate.json",
        "environment": "e27_environment.json",
        "runtime_role": "current_jax_0_11_2_cuda_12_portability",
        "jax_version": "0.11.2",
        "jaxlib_version": "0.11.2",
        "ld_library_policy": "bundled_cuda_12",
        "variant_roles": {
            "R": "reference_poincare_execution",
            "F": "factorized_poincare_execution",
        },
        "contrasts": (("R", "F", "current_jax_factorized_sanity"),),
    },
}

OUTPUT_FILES = {
    "sessions": "e24_e25_e27_session_outcomes.csv",
    "variants": "e24_e25_e27_variant_summary.csv",
    "contrasts": "e24_e25_e27_contrasts.csv",
    "audit": "e24_e25_e27_audit.json",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPOSITORY_ROOT / "protocol.yaml")
    parser.add_argument("--expected-code-commit", default=EXPECTED_COMMIT)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {"name": path.name, "exists": False}
    return {
        "name": path.name,
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON mapping")
            rows.append(value)
    return rows


def _truth(value: Any, *, context: str) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"{context}: expected a boolean value, received {value!r}")


def _finite(value: Any, *, context: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected a numeric value") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{context}: expected a finite {qualifier}value")
    return result


def _integer(value: Any, *, context: str, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected an integer") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{context}: expected an integer >= {minimum}")
    return result


def _validated_batch_size(workload: Mapping[str, Any], *, context: str) -> int:
    shape = workload.get("shape")
    if not isinstance(shape, dict):
        raise ValueError(f"{context}: workload shape must be a mapping")
    batch_size = shape.get("batch")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"{context}: batch size must be a positive integer")
    return batch_size


def _one(values: Iterable[Any], *, context: str) -> Any:
    materialized = list(values)
    if not materialized:
        raise ValueError(f"{context}: no values")
    first = materialized[0]
    if any(value != first for value in materialized[1:]):
        raise ValueError(f"{context}: inconsistent values")
    return first


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _log_t_interval(values: Sequence[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    logs = [math.log(value) for value in values]
    mean = statistics.fmean(logs)
    standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
    half_width = float(student_t.ppf(0.975, df=len(logs) - 1) * standard_error)
    return math.exp(mean - half_width), math.exp(mean + half_width)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    fields = list(materialized[0])
    if any(list(row) != fields for row in materialized):
        raise ValueError(f"inconsistent fields for {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(materialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_runtime_capture(
    artifact_dir: Path,
    cohort: Mapping[str, Any],
    *,
    expected_commit: str,
) -> dict[str, Any]:
    gate_path = artifact_dir / str(cohort["runtime_gate"])
    environment_path = artifact_dir / str(cohort["environment"])
    gate = _read_json(gate_path)
    environment = _read_json(environment_path)
    expected_jax = str(cohort["jax_version"])
    expected_jaxlib = str(cohort["jaxlib_version"])

    gate_identity = {
        "jax_version": expected_jax,
        "jaxlib_version": expected_jaxlib,
        "default_backend": "gpu",
        "compiled_gpu_smoke": "passed",
    }
    for field, expected in gate_identity.items():
        if gate.get(field) != expected:
            raise ValueError(
                f"{gate_path.name}: {field} is {gate.get(field)!r}, expected {expected!r}"
            )
    devices = gate.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError(f"{gate_path.name}: expected exactly one device")
    device = devices[0]
    expected_device_identity = {
        "id": EXPECTED_DEVICE["logical_id"],
        "platform": "gpu",
        "device_kind": EXPECTED_DEVICE["name"],
    }
    if device != expected_device_identity:
        raise ValueError(f"{gate_path.name}: unexpected logical device {device!r}")

    git = environment.get("git", {})
    if git.get("commit") != expected_commit or git.get("dirty") is not False:
        raise ValueError(
            f"{environment_path.name}: expected a clean {expected_commit} checkout"
        )
    if git.get("status_porcelain") not in ([], None):
        raise ValueError(f"{environment_path.name}: nonempty git status")
    if environment.get("hostname") != EXPECTED_DEVICE["hostname"]:
        raise ValueError(f"{environment_path.name}: unexpected hostname")
    jax = environment.get("jax", {})
    expected_jax_identity = {
        "version": expected_jax,
        "jaxlib_version": expected_jaxlib,
        "backend": "gpu",
    }
    for field, expected in expected_jax_identity.items():
        if jax.get(field) != expected:
            raise ValueError(
                f"{environment_path.name}: jax.{field} is {jax.get(field)!r}, "
                f"expected {expected!r}"
            )
    environment_devices = jax.get("devices")
    if not isinstance(environment_devices, list) or len(environment_devices) != 1:
        raise ValueError(f"{environment_path.name}: expected exactly one JAX device")
    environment_device = environment_devices[0]
    if (
        environment_device.get("id") != EXPECTED_DEVICE["logical_id"]
        or environment_device.get("platform") != "gpu"
        or environment_device.get("device_kind") != EXPECTED_DEVICE["name"]
    ):
        raise ValueError(f"{environment_path.name}: unexpected JAX device")

    for source_name, source in (
        (gate_path.name, gate),
        (environment_path.name, environment),
    ):
        variables = source.get("environment", {})
        expected_variables = {
            "CUDA_VISIBLE_DEVICES": str(EXPECTED_DEVICE["physical_index"]),
            "JAX_PLATFORMS": None,
            "NVIDIA_TF32_OVERRIDE": None,
            "XLA_FLAGS": None,
            "XLA_PYTHON_CLIENT_ALLOCATOR": None,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        for field, expected in expected_variables.items():
            if variables.get(field) != expected:
                raise ValueError(
                    f"{source_name}: environment {field} is {variables.get(field)!r}, "
                    f"expected {expected!r}"
                )
        library_path = variables.get("LD_LIBRARY_PATH")
        if cohort["ld_library_policy"] == "pinned_cuda_11_8":
            if not isinstance(library_path, str) or "cuda-11.8" not in library_path:
                raise ValueError(f"{source_name}: CUDA-11.8 library path is not pinned")
        elif library_path is not None:
            raise ValueError(f"{source_name}: CUDA-12 cohort inherited LD_LIBRARY_PATH")

    packages = environment.get("packages", {})
    for package, expected in (("jax", expected_jax), ("jaxlib", expected_jaxlib)):
        observed = packages.get(package)
        distribution_base = (
            observed.split("+", maxsplit=1)[0] if isinstance(observed, str) else None
        )
        if distribution_base != expected:
            raise ValueError(
                f"{environment_path.name}: package {package} is "
                f"{observed!r}, expected base version {expected!r}"
            )
    return {
        "runtime_role": cohort["runtime_role"],
        "jax_version": expected_jax,
        "jaxlib_version": expected_jaxlib,
        "runtime_gate": _evidence(gate_path),
        "environment": _evidence(environment_path),
    }


def _load_registration(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_yaml(protocol_path)
    if protocol.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise ValueError(
            f"protocol version is {protocol.get('protocol_version')!r}, "
            f"expected {EXPECTED_PROTOCOL_VERSION!r}"
        )
    if canonical_json(protocol.get("reference_commits", {})) != EXPECTED_REFERENCE_COMMIT:
        raise ValueError("protocol reference commits do not match the frozen registration")
    registrations: dict[str, Any] = {}
    for experiment_id, cohort in COHORTS.items():
        config_path = REPOSITORY_ROOT / str(cohort["config"])
        config = load_yaml(config_path)
        registered = protocol["revision_benchmarks"][experiment_id]
        if config.get("experiment_id") != experiment_id:
            raise ValueError(f"{experiment_id}: config experiment_id mismatch")
        identity_checks = {
            "config": registered.get("config") == cohort["config"],
            "variants": registered.get("variants") == config.get("variants"),
            "workloads": registered.get("workloads")
            == [workload["id"] for workload in config.get("workloads", [])],
            "sessions": registered.get("sessions") == config.get("sessions"),
            "warmup_calls": registered.get("warmup_calls") == config.get("warmup_calls"),
            "timed_calls_per_session": registered.get("timed_calls_per_session")
            == config.get("timed_calls_per_session"),
            "allocator_fraction": float(registered.get("allocator_fraction")) == 0.95,
            "preallocate": registered.get("preallocate") is False,
            "variant_roles": set(cohort["variant_roles"]) == set(config["variants"]),
        }
        failed = [field for field, passed in identity_checks.items() if not passed]
        if failed:
            raise ValueError(f"{experiment_id}: frozen registration mismatch: {failed}")
        workloads = {}
        for workload in config["workloads"]:
            batch_size = _validated_batch_size(
                workload, context=f"{experiment_id}/{workload['id']}"
            )
            workloads[workload["id"]] = {
                "spec": workload,
                "config_hash": config_hash(protocol, config, workload),
                "batch_size": batch_size,
            }
        registrations[experiment_id] = {
            "config": config,
            "config_path": config_path,
            "workloads": workloads,
        }
    return protocol, registrations


def _snapshot(value: Any, *, context: str, before: bool) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            snapshot = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: invalid GPU snapshot JSON") from error
    else:
        snapshot = value
    if not isinstance(snapshot, dict):
        raise ValueError(f"{context}: expected a GPU snapshot mapping")
    checks = {
        "status": snapshot.get("status") == "ok",
        "process_query_status": snapshot.get("process_query_status") == "ok",
        "physical_index": snapshot.get("index") == EXPECTED_DEVICE["physical_index"],
        "uuid": snapshot.get("uuid") == EXPECTED_DEVICE["uuid"],
        "name": snapshot.get("name") == EXPECTED_DEVICE["name"],
        "no_compute_processes": snapshot.get("compute_processes") == [],
    }
    if before:
        checks.update(
            {
                "idle_memory": int(snapshot.get("memory.used", 10**9)) <= 512,
                "idle_utilization": int(snapshot.get("utilization.gpu", 10**9)) <= 5,
            }
        )
    failed = [field for field, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{context}: GPU snapshot failed {failed}")
    return snapshot


def _expected_groups(
    experiment_id: str, registration: Mapping[str, Any]
) -> set[tuple[str, int, str]]:
    config = registration["config"]
    return {
        (workload_id, session, variant)
        for workload_id in registration["workloads"]
        for session in range(int(config["sessions"]))
        for variant in config["variants"]
    }


def _validate_common_identity(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    key: tuple[str, int, str],
    registration: Mapping[str, Any],
    expected_commit: str,
    context: str,
) -> None:
    workload_id, session, variant = key
    workload = registration["workloads"][workload_id]
    expected = {
        "experiment_id": experiment_id,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "variant": variant,
        "config_hash": workload["config_hash"],
        "code_commit": expected_commit,
        "reference_commit": EXPECTED_REFERENCE_COMMIT,
        "device_id": (f"{EXPECTED_DEVICE['hostname']}::gpu:0::{EXPECTED_DEVICE['name']}"),
        "seed_or_session": session,
        "workload_id": workload_id,
    }
    for field, wanted in expected.items():
        observed: Any = row.get(field)
        if field == "seed_or_session":
            observed = _integer(observed, context=f"{context} {field}")
        if observed != wanted:
            raise ValueError(f"{context}: {field} is {observed!r}, expected {wanted!r}")
    if _truth(row.get("code_dirty"), context=f"{context} code_dirty"):
        raise ValueError(f"{context}: code_dirty must be false")
    if row.get("precision_policy") != precision_policy(str(workload["spec"]["dtype"])):
        raise ValueError(f"{context}: precision_policy mismatch")


def _validate_success_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    key: tuple[str, int, str],
    registration: Mapping[str, Any],
    cohort: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    workload_id, session, variant = key
    context = f"{experiment_id}/{workload_id}/session-{session}/{variant}"
    config = registration["config"]
    workload_registration = registration["workloads"][workload_id]
    workload = workload_registration["spec"]
    batch_size = int(workload_registration["batch_size"])
    timed = int(config["timed_calls_per_session"])
    if len(rows) != timed + 3:
        raise ValueError(f"{context}: row count is {len(rows)}, expected {timed + 3}")
    for row in rows:
        _validate_common_identity(
            row,
            experiment_id=experiment_id,
            key=key,
            registration=registration,
            expected_commit=expected_commit,
            context=context,
        )
        if int(row.get("warmup_count", -1)) != int(config["warmup_calls"]):
            raise ValueError(f"{context}: warmup_count mismatch")
        if int(row.get("repeat_count", -1)) != timed:
            raise ValueError(f"{context}: repeat_count mismatch")
        if _truth(
            row.get("state_evolves_across_calls"), context=f"{context} state evolution"
        ) != (workload["mode"] == "update"):
            raise ValueError(f"{context}: state-evolution mismatch")
        expected_coordinate = (
            "euclidean_to_euclidean" if variant == "E" else "tangent_to_tangent"
        )
        if row.get("coordinate_boundary") != expected_coordinate:
            raise ValueError(f"{context}: coordinate boundary mismatch")
        expected_operation = f"{workload['kind']}:{workload['mode']}:depth=1"
        if row.get("operation_boundary") != expected_operation:
            raise ValueError(f"{context}: operation boundary mismatch")
        if row.get("shape") != canonical_json(workload["shape"]):
            raise ValueError(f"{context}: shape mismatch")
        _snapshot(row.get("gpu_snapshot_before"), context=context, before=True)
        _snapshot(row.get("gpu_snapshot_after"), context=context, before=False)

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("phase"))].append(row)
    expected_phases = {
        "compilation": {-2},
        "first_compiled_call": {-1},
        "sustained_sequence": {-3},
        "steady_state": set(range(timed)),
    }
    if set(grouped) != set(expected_phases):
        raise ValueError(f"{context}: phase inventory mismatch")
    for phase, observations in expected_phases.items():
        observed = {
            _integer(row.get("observation"), context=f"{context} observation")
            for row in grouped[phase]
        }
        if observed != observations or len(grouped[phase]) != len(observations):
            raise ValueError(f"{context}: invalid {phase} observations")

    result_status = _one(
        (row.get("status") for row in grouped["steady_state"]),
        context=f"{context} result status",
    )
    if result_status not in {"ok", "nonfinite"}:
        raise ValueError(f"{context}: invalid result status {result_status!r}")
    for phase in ("compilation", "first_compiled_call"):
        if grouped[phase][0].get("status") != result_status:
            raise ValueError(f"{context}: inconsistent {phase} status")
    sustained = grouped["sustained_sequence"][0]
    if sustained.get("status") != "not_applicable" or sustained.get("latency_ms") not in (
        "",
        None,
    ):
        raise ValueError(f"{context}: sustained sequence must be not applicable")

    latency_values = [
        _finite(row.get("latency_ms"), context=f"{context} latency", positive=True)
        for row in grouped["steady_state"]
    ]
    compilation_ms = _finite(
        grouped["compilation"][0].get("latency_ms"),
        context=f"{context} compilation latency",
        positive=True,
    )
    first_call_ms = _finite(
        grouped["first_compiled_call"][0].get("latency_ms"),
        context=f"{context} first-call latency",
        positive=True,
    )
    exemplar = rows[0]
    repeated_fields = (
        "measured_peak_bytes_in_use",
        "measured_peak_pool_bytes",
        "process_reservation_mib",
        "output_nonfinite_count",
        "state_summary_before",
        "state_summary_after",
    )
    for field in repeated_fields:
        _one((row.get(field) for row in rows), context=f"{context} {field}")
    peak_live = _integer(
        exemplar.get("measured_peak_bytes_in_use"),
        context=f"{context} peak live bytes",
        minimum=1,
    )
    peak_pool = _integer(
        exemplar.get("measured_peak_pool_bytes"),
        context=f"{context} peak pool bytes",
        minimum=1,
    )
    if peak_pool < peak_live:
        raise ValueError(f"{context}: peak pool bytes are below peak live bytes")
    reservation = _integer(
        exemplar.get("process_reservation_mib"),
        context=f"{context} process reservation",
        minimum=0,
    )
    output_nonfinite = _integer(
        exemplar.get("output_nonfinite_count"),
        context=f"{context} output_nonfinite_count",
        minimum=0,
    )
    states = [
        json.loads(str(exemplar[field]))
        for field in ("state_summary_before", "state_summary_after")
    ]
    state_nonfinite = sum(int(state["nonfinite_count"]) for state in states)
    if result_status == "ok" and (output_nonfinite or state_nonfinite):
        raise ValueError(f"{context}: ok group contains nonfinite values")
    if result_status == "nonfinite" and not (output_nonfinite or state_nonfinite):
        raise ValueError(f"{context}: nonfinite group has no retained nonfinite count")

    session_median_latency_ms = statistics.median(latency_values)
    session_median_throughput_samples_per_s = _finite(
        1000.0 * batch_size / session_median_latency_ms,
        context=f"{context} derived throughput",
        positive=True,
    )
    return {
        "experiment_id": experiment_id,
        "runtime_role": cohort["runtime_role"],
        "jax_version": cohort["jax_version"],
        "jaxlib_version": cohort["jaxlib_version"],
        "workload_id": workload_id,
        "workload_kind": workload["kind"],
        "mode": workload["mode"],
        "shape": canonical_json(workload["shape"]),
        "batch_size": batch_size,
        "session": session,
        "variant": variant,
        "variant_role": cohort["variant_roles"][variant],
        "outcome": result_status,
        "steady_state_observations": timed,
        "session_median_latency_ms": session_median_latency_ms,
        "session_median_throughput_samples_per_s": (
            session_median_throughput_samples_per_s
        ),
        "compilation_ms": compilation_ms,
        "first_compiled_call_ms": first_call_ms,
        "peak_live_gib": peak_live / 1024**3,
        "peak_pool_gib": peak_pool / 1024**3,
        "process_reservation_mib": reservation,
        "failure_kind": None,
        "failure_phase": None,
        "failure_returncode": None,
        "failure_stdout": None,
        "failure_stderr": None,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "code_commit": expected_commit,
        "config_hash": registration["workloads"][workload_id]["config_hash"],
        "device_id": exemplar["device_id"],
        "physical_gpu_index": EXPECTED_DEVICE["physical_index"],
        "physical_gpu_uuid": EXPECTED_DEVICE["uuid"],
        "physical_gpu_name": EXPECTED_DEVICE["name"],
    }


def _validate_failure(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    key: tuple[str, int, str],
    registration: Mapping[str, Any],
    cohort: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    workload_id, session, variant = key
    context = f"{experiment_id}/{workload_id}/session-{session}/{variant}/failure"
    _validate_common_identity(
        row,
        experiment_id=experiment_id,
        key=key,
        registration=registration,
        expected_commit=expected_commit,
        context=context,
    )
    if row.get("status") != "failed":
        raise ValueError(f"{context}: status must be failed")
    failure_kind = str(row.get("failure_kind"))
    if failure_kind not in {"oom", "failed", "terminated_by_signal"}:
        raise ValueError(f"{context}: unrecognized failure kind {failure_kind!r}")
    returncode = _integer(row.get("returncode"), context=f"{context} returncode")
    _snapshot(row.get("gpu_snapshot_before"), context=context, before=True)
    _snapshot(row.get("gpu_snapshot_after"), context=context, before=False)
    workload_registration = registration["workloads"][workload_id]
    workload = workload_registration["spec"]
    batch_size = int(workload_registration["batch_size"])
    if row.get("shape") != workload["shape"]:
        raise ValueError(f"{context}: shape mismatch")
    if row.get("operation_boundary") != f"{workload['kind']}:{workload['mode']}":
        raise ValueError(f"{context}: operation boundary mismatch")
    return {
        "experiment_id": experiment_id,
        "runtime_role": cohort["runtime_role"],
        "jax_version": cohort["jax_version"],
        "jaxlib_version": cohort["jaxlib_version"],
        "workload_id": workload_id,
        "workload_kind": workload["kind"],
        "mode": workload["mode"],
        "shape": canonical_json(workload["shape"]),
        "batch_size": batch_size,
        "session": session,
        "variant": variant,
        "variant_role": cohort["variant_roles"][variant],
        "outcome": "failed",
        "steady_state_observations": 0,
        "session_median_latency_ms": None,
        "session_median_throughput_samples_per_s": None,
        "compilation_ms": None,
        "first_compiled_call_ms": None,
        "peak_live_gib": None,
        "peak_pool_gib": None,
        "process_reservation_mib": None,
        "failure_kind": failure_kind,
        "failure_phase": str(row.get("failure_phase", "unknown")),
        "failure_returncode": returncode,
        "failure_stdout": str(row.get("stdout", "")),
        "failure_stderr": str(row.get("stderr", "")),
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "code_commit": expected_commit,
        "config_hash": registration["workloads"][workload_id]["config_hash"],
        "device_id": row["device_id"],
        "physical_gpu_index": EXPECTED_DEVICE["physical_index"],
        "physical_gpu_uuid": EXPECTED_DEVICE["uuid"],
        "physical_gpu_name": EXPECTED_DEVICE["name"],
    }


def _validate_cohort(
    artifact_dir: Path,
    experiment_id: str,
    cohort: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    expected_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path = artifact_dir / str(cohort["raw"])
    failure_path = raw_path.with_name(f"{raw_path.name}.failures.jsonl")
    raw_rows = _read_csv(raw_path)
    failure_rows = _read_jsonl(failure_path)
    expected = _expected_groups(experiment_id, registration)
    success_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(raw_rows, start=2):
        try:
            key = (
                str(row["workload_id"]),
                int(row["seed_or_session"]),
                str(row["variant"]),
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"{raw_path.name}:{row_number}: invalid group key") from error
        if key not in expected:
            raise ValueError(f"{raw_path.name}:{row_number}: unexpected group {key}")
        success_groups[key].append(row)
    failures: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row_number, row in enumerate(failure_rows, start=1):
        try:
            key = (
                str(row["workload_id"]),
                int(row["seed_or_session"]),
                str(row["variant"]),
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"{failure_path.name}:{row_number}: invalid group key"
            ) from error
        if key not in expected:
            raise ValueError(f"{failure_path.name}:{row_number}: unexpected group {key}")
        if key in failures:
            raise ValueError(f"{failure_path.name}: duplicate failure group {key}")
        failures[key] = row
    overlap = set(success_groups) & set(failures)
    if overlap:
        raise ValueError(
            f"{experiment_id}: groups have both success and failure: {overlap}"
        )
    observed = set(success_groups) | set(failures)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"{experiment_id}: outcome matrix mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    ordered_keys = [
        (workload_id, session, variant)
        for workload_id in registration["workloads"]
        for session in range(int(registration["config"]["sessions"]))
        for variant in registration["config"]["variants"]
    ]
    outcomes = []
    for key in ordered_keys:
        if key in success_groups:
            outcomes.append(
                _validate_success_group(
                    success_groups[key],
                    experiment_id=experiment_id,
                    key=key,
                    registration=registration,
                    cohort=cohort,
                    expected_commit=expected_commit,
                )
            )
        else:
            outcomes.append(
                _validate_failure(
                    failures[key],
                    experiment_id=experiment_id,
                    key=key,
                    registration=registration,
                    cohort=cohort,
                    expected_commit=expected_commit,
                )
            )
    return outcomes, {
        "experiment_id": experiment_id,
        "raw": _evidence(raw_path, required=False),
        "failures": _evidence(failure_path, required=False),
        "raw_rows": len(raw_rows),
        "planned_groups": len(expected),
        "successful_worker_groups": len(success_groups),
        "failed_worker_groups": len(failures),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in outcomes).items())),
        "config": _evidence(registration["config_path"]),
        "expected_config_hashes": {
            workload_id: value["config_hash"]
            for workload_id, value in registration["workloads"].items()
        },
        "expected_batch_sizes": {
            workload_id: value["batch_size"]
            for workload_id, value in registration["workloads"].items()
        },
    }


def _variant_summaries(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[(row["experiment_id"], row["workload_id"], row["variant"])].append(row)
    output = []
    for key, selected in grouped.items():
        selected = sorted(selected, key=lambda row: int(row["session"]))
        exemplar = selected[0]
        accepted = [row for row in selected if row["outcome"] == "ok"]
        latencies = [float(row["session_median_latency_ms"]) for row in accepted]
        throughputs = [
            float(row["session_median_throughput_samples_per_s"]) for row in accepted
        ]
        live = [float(row["peak_live_gib"]) for row in accepted]
        pool = [float(row["peak_pool_gib"]) for row in accepted]
        reservation = [float(row["process_reservation_mib"]) for row in accepted]
        output.append(
            {
                "experiment_id": key[0],
                "runtime_role": exemplar["runtime_role"],
                "jax_version": exemplar["jax_version"],
                "jaxlib_version": exemplar["jaxlib_version"],
                "workload_id": key[1],
                "workload_kind": exemplar["workload_kind"],
                "mode": exemplar["mode"],
                "shape": exemplar["shape"],
                "batch_size": exemplar["batch_size"],
                "variant": key[2],
                "variant_role": exemplar["variant_role"],
                "planned_sessions": len(selected),
                "accepted_finite_sessions": len(accepted),
                "nonfinite_sessions": sum(
                    row["outcome"] == "nonfinite" for row in selected
                ),
                "failed_sessions": sum(row["outcome"] == "failed" for row in selected),
                "all_sessions_accepted": len(accepted) == len(selected),
                "latency_estimand": (
                    "median of accepted session-level medians; each session median uses "
                    "200 synchronized steady-state calls"
                ),
                "median_latency_ms": statistics.median(latencies) if latencies else None,
                "q1_session_latency_ms": _quantile(latencies, 0.25),
                "q3_session_latency_ms": _quantile(latencies, 0.75),
                "session_medians_ms": json.dumps(latencies, separators=(",", ":")),
                "throughput_estimand": (
                    "median of accepted per-session throughputs; each session is "
                    "1000 times validated batch size divided by its synchronized "
                    "steady-state median latency in milliseconds"
                ),
                "median_throughput_samples_per_s": (
                    statistics.median(throughputs) if throughputs else None
                ),
                "min_throughput_samples_per_s": (min(throughputs) if throughputs else None),
                "max_throughput_samples_per_s": (max(throughputs) if throughputs else None),
                "session_throughputs_samples_per_s": json.dumps(
                    throughputs, separators=(",", ":")
                ),
                "median_peak_live_gib": statistics.median(live) if live else None,
                "min_peak_live_gib": min(live) if live else None,
                "max_peak_live_gib": max(live) if live else None,
                "median_peak_pool_gib": statistics.median(pool) if pool else None,
                "median_process_reservation_mib": (
                    statistics.median(reservation) if reservation else None
                ),
                "protocol_version": exemplar["protocol_version"],
                "code_commit": exemplar["code_commit"],
                "config_hash": exemplar["config_hash"],
                "device_id": exemplar["device_id"],
                "physical_gpu_uuid": exemplar["physical_gpu_uuid"],
            }
        )
    return output


def _contrasts(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["experiment_id"], row["workload_id"], int(row["session"]), row["variant"]): row
        for row in outcomes
    }
    workload_ids = {
        experiment_id: list(
            dict.fromkeys(
                row["workload_id"]
                for row in outcomes
                if row["experiment_id"] == experiment_id
            )
        )
        for experiment_id in COHORTS
    }
    output = []
    for experiment_id, cohort in COHORTS.items():
        for workload_id in workload_ids[experiment_id]:
            sessions = sorted(
                {
                    int(row["session"])
                    for row in outcomes
                    if row["experiment_id"] == experiment_id
                    and row["workload_id"] == workload_id
                }
            )
            for reference, candidate, role in cohort["contrasts"]:
                paired = []
                exclusions = {}
                for session in sessions:
                    reference_row = indexed[
                        (experiment_id, workload_id, session, reference)
                    ]
                    candidate_row = indexed[
                        (experiment_id, workload_id, session, candidate)
                    ]
                    if reference_row["outcome"] == candidate_row["outcome"] == "ok":
                        paired.append((session, reference_row, candidate_row))
                    else:
                        exclusions[str(session)] = {
                            reference: reference_row["outcome"],
                            candidate: candidate_row["outcome"],
                        }
                latency_ratios = [
                    float(reference_row["session_median_latency_ms"])
                    / float(candidate_row["session_median_latency_ms"])
                    for _, reference_row, candidate_row in paired
                ]
                throughput_ratios = [
                    float(candidate_row["session_median_throughput_samples_per_s"])
                    / float(reference_row["session_median_throughput_samples_per_s"])
                    for _, reference_row, candidate_row in paired
                ]
                live_ratios = [
                    float(reference_row["peak_live_gib"])
                    / float(candidate_row["peak_live_gib"])
                    for _, reference_row, candidate_row in paired
                ]
                pool_ratios = [
                    float(reference_row["peak_pool_gib"])
                    / float(candidate_row["peak_pool_gib"])
                    for _, reference_row, candidate_row in paired
                ]
                latency_ratio = _geometric_mean(latency_ratios)
                throughput_ratio = _geometric_mean(throughput_ratios)
                live_ratio = _geometric_mean(live_ratios)
                pool_ratio = _geometric_mean(pool_ratios)
                lower, upper = _log_t_interval(latency_ratios)
                throughput_lower, throughput_upper = _log_t_interval(throughput_ratios)
                exemplar = indexed[(experiment_id, workload_id, sessions[0], reference)]
                output.append(
                    {
                        "experiment_id": experiment_id,
                        "runtime_role": cohort["runtime_role"],
                        "jax_version": cohort["jax_version"],
                        "jaxlib_version": cohort["jaxlib_version"],
                        "workload_id": workload_id,
                        "workload_kind": exemplar["workload_kind"],
                        "mode": exemplar["mode"],
                        "shape": exemplar["shape"],
                        "batch_size": exemplar["batch_size"],
                        "contrast_role": role,
                        "reference_variant": reference,
                        "reference_variant_role": cohort["variant_roles"][reference],
                        "candidate_variant": candidate,
                        "candidate_variant_role": cohort["variant_roles"][candidate],
                        "planned_paired_sessions": len(sessions),
                        "accepted_paired_sessions": len(paired),
                        "all_planned_sessions_paired": len(paired) == len(sessions),
                        "excluded_session_outcomes": json.dumps(
                            exclusions, sort_keys=True, separators=(",", ":")
                        ),
                        "latency_ratio_estimand": (
                            "geometric mean of paired session-median reference/candidate "
                            "ratios; values above one favor the candidate"
                        ),
                        "geometric_mean_latency_ratio": latency_ratio,
                        "log_t_95_lower": lower,
                        "log_t_95_upper": upper,
                        "paired_session_ids": json.dumps(
                            [session for session, _, _ in paired], separators=(",", ":")
                        ),
                        "paired_latency_ratios": json.dumps(
                            latency_ratios, separators=(",", ":")
                        ),
                        "throughput_ratio_estimand": (
                            "geometric mean of paired candidate/reference "
                            "session-median throughput ratios; values above one favor "
                            "the candidate"
                        ),
                        "geometric_mean_throughput_ratio": throughput_ratio,
                        "throughput_log_t_95_lower": throughput_lower,
                        "throughput_log_t_95_upper": throughput_upper,
                        "paired_throughput_ratios": json.dumps(
                            throughput_ratios, separators=(",", ":")
                        ),
                        "geometric_mean_peak_live_ratio": live_ratio,
                        "candidate_peak_live_reduction_percent": (
                            100.0 * (1.0 - 1.0 / live_ratio) if live_ratio else None
                        ),
                        "paired_peak_live_ratios": json.dumps(
                            live_ratios, separators=(",", ":")
                        ),
                        "geometric_mean_peak_pool_ratio": pool_ratio,
                        "candidate_peak_pool_reduction_percent": (
                            100.0 * (1.0 - 1.0 / pool_ratio) if pool_ratio else None
                        ),
                        "protocol_version": exemplar["protocol_version"],
                        "code_commit": exemplar["code_commit"],
                        "config_hash": exemplar["config_hash"],
                        "device_id": exemplar["device_id"],
                        "physical_gpu_uuid": exemplar["physical_gpu_uuid"],
                    }
                )
    return output


def export(
    artifact_dir: Path,
    output_dir: Path,
    *,
    protocol_path: Path = REPOSITORY_ROOT / "protocol.yaml",
    expected_commit: str = EXPECTED_COMMIT,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    analysis_code = git_provenance(REPOSITORY_ROOT)
    protocol, registrations = _load_registration(protocol_path)

    runtime_audits = {}
    for experiment_id, cohort in COHORTS.items():
        key = str(cohort["runtime_role"])
        audit = _validate_runtime_capture(
            artifact_dir, cohort, expected_commit=expected_commit
        )
        if key in runtime_audits and runtime_audits[key] != audit:
            raise ValueError(
                f"{experiment_id}: shared runtime capture audit is inconsistent"
            )
        runtime_audits[key] = audit

    outcomes = []
    cohort_audits = []
    for experiment_id, cohort in COHORTS.items():
        selected, audit = _validate_cohort(
            artifact_dir,
            experiment_id,
            cohort,
            registrations[experiment_id],
            expected_commit=expected_commit,
        )
        outcomes.extend(selected)
        cohort_audits.append(audit)
    if {row["physical_gpu_uuid"] for row in outcomes} != {EXPECTED_DEVICE["uuid"]}:
        raise ValueError("cohorts do not bind to the registered physical GPU")

    variants = _variant_summaries(outcomes)
    contrasts = _contrasts(outcomes)
    paths = {name: output_dir / filename for name, filename in OUTPUT_FILES.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(paths["sessions"], outcomes)
    _atomic_csv(paths["variants"], variants)
    _atomic_csv(paths["contrasts"], contrasts)
    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "analysis_id": "e24_e25_e27_revision_benchmark_export",
        "derivation": (
            "CPU-only validation and deterministic aggregation of existing raw GPU "
            "artifacts; no benchmark is launched by this exporter. Failed groups are "
            "retained as planned outcomes and excluded explicitly from paired ratios."
        ),
        "protocol": _evidence(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "expected_run_commit": expected_commit,
        "analysis_code": analysis_code,
        "physical_device": EXPECTED_DEVICE,
        "runtime_captures": runtime_audits,
        "cohorts": cohort_audits,
        "output_rows": {
            OUTPUT_FILES["sessions"]: len(outcomes),
            OUTPUT_FILES["variants"]: len(variants),
            OUTPUT_FILES["contrasts"]: len(contrasts),
        },
        "output_evidence": {
            OUTPUT_FILES[name]: _evidence(paths[name])
            for name in ("sessions", "variants", "contrasts")
        },
        "all_planned_groups_accounted_for": True,
    }
    _atomic_json(paths["audit"], summary)
    return summary


def main() -> None:
    arguments = _parser().parse_args()
    summary = export(
        arguments.artifact_dir,
        arguments.output_dir,
        protocol_path=arguments.protocol,
        expected_commit=arguments.expected_code_commit,
    )
    print(json.dumps(summary["output_rows"], sort_keys=True))


if __name__ == "__main__":
    main()
