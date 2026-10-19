#!/usr/bin/env python3
"""Validate V100 E29/E30 evidence and export reviewer-requested tables.

E29 times the Euclidean path E on the E08 and E13 U-Net workloads; its ratios
against the published E08/E13 R/F medians are unpaired and descriptive. E30
times paired R/F complete updates at U-Net initial widths 16 and 32; its
paired contrasts are placed beside the published E13 width-8 pivot. The
exporter is CPU-only and never launches a benchmark. Every planned group is
retained, including failures, and every input and output is SHA-256 recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from poincare_exec.artifacts.provenance import git_provenance, precision_policy
from poincare_exec.config.loader import canonical_json, config_hash, load_yaml
from scripts.export_e24_e25_e27_tables import (
    _atomic_csv,
    _atomic_json,
    _evidence,
    _finite,
    _geometric_mean,
    _integer,
    _log_t_interval,
    _one,
    _quantile,
    _read_csv,
    _read_json,
    _read_jsonl,
    _truth,
    _validated_batch_size,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPOSITORY_ROOT / "results"
# Protocol 1.19 registered E31 after these runs; the 1.18 snapshot is byte-identical
# to the protocol.yaml that the E29/E30 configuration hashes bind.
PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "protocols" / "protocol_1_18.yaml"
EXPECTED_COMMIT = "20a886e431de9cc5e38264f4bc4781f0434fe944"
EXPECTED_PROTOCOL_VERSION = "1.18"
EXPORT_SCHEMA_VERSION = "1.0"
EXPECTED_HOSTNAME = "argexpr3.vsb.cz"
EXPECTED_GPU_NAME = "Tesla V100-PCIE-32GB"
EXPECTED_DEVICE_ID = f"{EXPECTED_HOSTNAME}::gpu:0::{EXPECTED_GPU_NAME}"
PHYSICAL_GPU_UUIDS = {
    0: "GPU-a37e9fd0-2d52-1766-149e-64287a10103d",
    1: "GPU-4e0547c3-d65d-7d8b-9f6a-d40ad61a0f0f",
}
EXPECTED_REFERENCE_COMMIT = canonical_json(
    {
        "hyperbolic_unet": "bec504e2f4d198ee4c438579b86d72f9d8186d78",
        "poincare_resnet": "fbbbbf890571115444b89b114f9ed52d128d5aa3",
        "hnnpp": "28737e22822562ac18d5ea03f8c3e3929e945a83",
    }
)
JAX_VERSION = "0.4.25"
REGISTERED_UNET_WIDTH = 8
# The E08/E13 V100 allocator ceiling captured at the default fraction (0.75).
ALLOCATOR_LIMIT_EVIDENCE = "environment.json"
ALLOCATOR_ENVIRONMENT = {
    "JAX_PLATFORMS": None,
    "NVIDIA_TF32_OVERRIDE": None,
    "XLA_FLAGS": None,
    "XLA_PYTHON_CLIENT_ALLOCATOR": None,
    "XLA_PYTHON_CLIENT_MEM_FRACTION": None,
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
OOM_PATTERN = re.compile(r"out of memory|resource_exhausted", re.IGNORECASE)
REQUESTED_BYTES_PATTERN = re.compile(
    r"out of memory while trying to allocate\s+(\d+)\s+bytes", re.IGNORECASE
)
SOFT_LOCKUP_PATTERN = re.compile(r"soft lockup - CPU#\d+ stuck for (\d+)s!")
STALL_MARGIN_SECONDS = 180
TIMEOUT_CRASH_SECONDS = 60
STALL_RULE = "logs/driver_stall_rule.txt"
KERNEL_LOCKUP_LOG = "logs/kernel_driver_lockups.log"
EUCLIDEAN_ROLE = "euclidean_same_topology_reference"
COHORTS: dict[str, dict[str, Any]] = {
    "E29": {
        "config": "configs/benchmarks/e29_v100_euclidean_reference.yaml",
        "launcher": "scripts/run_e29.sh",
        "launcher_sha256": (
            "b0588ec286b3c2b155c7456d33cc4fefc692327ac86741d801f90b0860f506cc"
        ),
        "raw": "e29_v100_euclidean_reference.csv",
        "runtime_gate": "e29_runtime_gate.json",
        "environment": "e29_environment.json",
        "physical_index": 0,
        "variant_roles": {"E": EUCLIDEAN_ROLE},
        "contrasts": (),
    },
    "E30": {
        "config": "configs/benchmarks/e30_v100_unet_width.yaml",
        "launcher": "scripts/run_e30.sh",
        "launcher_sha256": (
            "08168e5eede8b76af86e95061710911d8a948649a07b6caa047d616d5e568be6"
        ),
        "raw": "e30_v100_unet_width.csv",
        "runtime_gate": "e30_runtime_gate.json",
        "environment": "e30_environment.json",
        "physical_index": 1,
        "variant_roles": {
            "R": "reference_poincare_execution",
            "F": "factorized_poincare_execution",
        },
        "contrasts": (("R", "F", "factorized_vs_reference_poincare"),),
    },
}
OUTPUT_FILES = {
    "E29": {
        "sessions": "e29_session_outcomes.csv",
        "variants": "e29_variant_summary.csv",
        "published": "e29_published_reference_ratios.csv",
        "audit": "e29_audit.json",
    },
    "E30": {
        "sessions": "e30_session_outcomes.csv",
        "variants": "e30_variant_summary.csv",
        "contrasts": "e30_contrasts.csv",
        "width_series": "e30_width_series.csv",
        "audit": "e30_audit.json",
    },
}
# Published R/F evidence; each file is also bound by results/SHA256SUMS.
PUBLISHED_SOURCES = {
    "E08": "unet_timing/workload_latency_memory_summary.csv",
    "E13": "e13_scaling/workload_summary.csv",
    "E13_paired": "e13_scaling/paired_ratio_summary.csv",
}
PUBLISHED_PHYSICAL_GPU_UUID = PHYSICAL_GPU_UUIDS[1]
E08_WORKLOADS = ("unet_b1_forward", "unet_b8_forward", "unet_b8_update")
E13_WIDTH_REFERENCE_WORKLOAD = "unet_update_b8_r256"
FAILURE_PHASES = {
    "initialization",
    "compilation",
    "first_call",
    "first_update",
    "warmup_calls",
    "warmup_updates",
    "measured_calls",
    "measured_updates",
    "post_measurement",
    "unknown",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-code-commit", default=EXPECTED_COMMIT)
    return parser


def _optional_float(value: Any) -> float | None:
    return None if value in ("", None) else float(value)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / denominator


def _checksummed_result(relative: str) -> dict[str, Any]:
    """Return the evidence of a committed result file listed in SHA256SUMS."""

    manifest = RESULTS_DIR / "SHA256SUMS"
    declared = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            declared[line[66:].removeprefix("./")] = line[:64]
    path = RESULTS_DIR / relative
    evidence = _evidence(path)
    if declared.get(relative) != evidence["sha256"]:
        raise ValueError(f"results/{relative} does not match results/SHA256SUMS")
    return {**evidence, "path": f"results/{relative}"}


def _allocator_limit_bytes() -> int:
    environment = _read_json(RESULTS_DIR / ALLOCATOR_LIMIT_EVIDENCE)
    return _integer(
        environment["hardware"]["jax_allocator_bytes_limit"],
        context="results/environment.json allocator limit",
        minimum=1,
    )


def _validate_launcher(cohort: Mapping[str, Any]) -> dict[str, Any]:
    path = REPOSITORY_ROOT / str(cohort["launcher"])
    evidence = _evidence(path)
    if evidence["sha256"] != cohort["launcher_sha256"]:
        raise ValueError(f"{cohort['launcher']}: launcher differs from the frozen run")
    text = path.read_text(encoding="utf-8")
    for required in (
        "unset XLA_PYTHON_CLIENT_MEM_FRACTION",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "--protocol protocol.yaml",
        '--config "$config"',
        f"config={cohort['config']}",
    ):
        if required not in text:
            raise ValueError(f"{cohort['launcher']}: missing {required!r}")
    return {**evidence, "path": cohort["launcher"]}


def _validate_runtime_capture(
    artifact_dir: Path,
    cohort: Mapping[str, Any],
    *,
    expected_commit: str,
    allocator_limit_bytes: int,
) -> dict[str, Any]:
    gate_path = artifact_dir / str(cohort["runtime_gate"])
    environment_path = artifact_dir / str(cohort["environment"])
    gate = _read_json(gate_path)
    environment = _read_json(environment_path)
    expected_device = {"id": 0, "platform": "gpu", "device_kind": EXPECTED_GPU_NAME}
    gate_identity = {
        "jax_version": JAX_VERSION,
        "jaxlib_version": JAX_VERSION,
        "default_backend": "gpu",
        "compiled_gpu_smoke": "passed",
        "devices": [expected_device],
    }
    for field, expected in gate_identity.items():
        if gate.get(field) != expected:
            raise ValueError(f"{gate_path.name}: {field} is {gate.get(field)!r}")

    git = environment.get("git", {})
    if (
        git.get("commit") != expected_commit
        or git.get("dirty") is not False
        or git.get("status_porcelain") not in ([], None)
    ):
        raise ValueError(f"{environment_path.name}: expected a clean {expected_commit}")
    if environment.get("hostname") != EXPECTED_HOSTNAME:
        raise ValueError(f"{environment_path.name}: unexpected hostname")
    jax = environment.get("jax", {})
    for field, expected in (
        ("version", JAX_VERSION),
        ("jaxlib_version", JAX_VERSION),
        ("backend", "gpu"),
    ):
        if jax.get(field) != expected:
            raise ValueError(f"{environment_path.name}: jax.{field} is {jax.get(field)!r}")
    devices = jax.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError(f"{environment_path.name}: expected exactly one JAX device")
    device = devices[0]
    if {key: device.get(key) for key in expected_device} != expected_device:
        raise ValueError(f"{environment_path.name}: unexpected JAX device")
    bytes_limit = (device.get("memory_stats") or {}).get("bytes_limit")
    if bytes_limit != allocator_limit_bytes:
        raise ValueError(
            f"{environment_path.name}: allocator bytes_limit {bytes_limit!r} differs "
            f"from the default-fraction E08/E13 limit {allocator_limit_bytes}"
        )
    packages = environment.get("packages", {})
    for package in ("jax", "jaxlib"):
        observed = str(packages.get(package, ""))
        if observed.split("+", maxsplit=1)[0] != JAX_VERSION:
            raise ValueError(f"{environment_path.name}: package {package} is {observed!r}")

    expected_variables = {
        "CUDA_VISIBLE_DEVICES": str(cohort["physical_index"]),
        **ALLOCATOR_ENVIRONMENT,
    }
    for source_name, source in (
        (gate_path.name, gate),
        (environment_path.name, environment),
    ):
        variables = source.get("environment", {})
        for field, expected in expected_variables.items():
            if variables.get(field) != expected:
                raise ValueError(
                    f"{source_name}: environment {field} is {variables.get(field)!r}, "
                    f"expected {expected!r}"
                )
        library_path = variables.get("LD_LIBRARY_PATH")
        if not isinstance(library_path, str) or "cuda-11.8" not in library_path:
            raise ValueError(f"{source_name}: CUDA-11.8 library path is not pinned")
    return {
        "runtime_gate": _evidence(gate_path),
        "environment": _evidence(environment_path),
        "cuda_visible_devices": str(cohort["physical_index"]),
        "allocator_environment": ALLOCATOR_ENVIRONMENT,
        "allocator_bytes_limit": bytes_limit,
        "allocator_bytes_limit_gib": bytes_limit / 1024**3,
    }


def _load_registrations(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_yaml(protocol_path)
    if protocol.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise ValueError(
            f"protocol version is {protocol.get('protocol_version')!r}, "
            f"expected {EXPECTED_PROTOCOL_VERSION!r}"
        )
    if canonical_json(protocol.get("reference_commits", {})) != EXPECTED_REFERENCE_COMMIT:
        raise ValueError("protocol reference commits do not match the registration")
    task_width = protocol["tasks"]["isic16_poincare_unet"]["model"]["initial_width"]
    if task_width != REGISTERED_UNET_WIDTH:
        raise ValueError("registered U-Net width is not 8")
    registrations: dict[str, Any] = {}
    for experiment_id, cohort in COHORTS.items():
        config_path = REPOSITORY_ROOT / str(cohort["config"])
        config = load_yaml(config_path)
        registered = protocol["revision_benchmarks"][experiment_id]
        checks = {
            "experiment_id": config.get("experiment_id") == experiment_id,
            "config": registered.get("config") == cohort["config"],
            "variants": registered.get("variants") == config.get("variants"),
            "workloads": registered.get("workloads")
            == [workload["id"] for workload in config.get("workloads", [])],
            "sessions": registered.get("sessions") == config.get("sessions"),
            "warmup_calls": registered.get("warmup_calls") == config.get("warmup_calls"),
            "timed_calls_per_session": registered.get("timed_calls_per_session")
            == config.get("timed_calls_per_session"),
            "allocator_fraction": registered.get("allocator_fraction") == "default",
            "preallocate": registered.get("preallocate") is False,
            "variant_roles": set(cohort["variant_roles"]) == set(config["variants"]),
        }
        failed = [field for field, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{experiment_id}: registration mismatch: {failed}")
        workloads = {}
        for workload in config["workloads"]:
            workloads[workload["id"]] = {
                "spec": workload,
                "config_hash": config_hash(protocol, config, workload),
                "batch_size": _validated_batch_size(
                    workload, context=f"{experiment_id}/{workload['id']}"
                ),
                "initial_width": int(workload.get("initial_width", task_width)),
            }
        registrations[experiment_id] = {
            "config": config,
            "config_path": config_path,
            "workloads": workloads,
        }
    return protocol, registrations


def _snapshot(value: Any, *, physical_index: int, context: str, before: bool) -> None:
    snapshot = json.loads(value) if isinstance(value, str) else value
    if not isinstance(snapshot, dict):
        raise ValueError(f"{context}: expected a GPU snapshot mapping")
    checks = {
        "status": snapshot.get("status") == "ok",
        "process_query_status": snapshot.get("process_query_status") == "ok",
        "physical_index": snapshot.get("index") == physical_index,
        "uuid": snapshot.get("uuid") == PHYSICAL_GPU_UUIDS[physical_index],
        "name": snapshot.get("name") == EXPECTED_GPU_NAME,
        "no_compute_processes": snapshot.get("compute_processes") == [],
    }
    if before:
        checks["idle_memory"] = int(snapshot.get("memory.used", 10**9)) <= 512
        checks["idle_utilization"] = int(snapshot.get("utilization.gpu", 10**9)) <= 5
    failed = [field for field, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{context}: GPU snapshot failed {failed}")


def _validate_identity(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    key: tuple[str, int, str],
    workload: Mapping[str, Any],
    expected_commit: str,
    context: str,
) -> None:
    workload_id, session, variant = key
    expected = {
        "experiment_id": experiment_id,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "variant": variant,
        "config_hash": workload["config_hash"],
        "code_commit": expected_commit,
        "reference_commit": EXPECTED_REFERENCE_COMMIT,
        "device_id": EXPECTED_DEVICE_ID,
        "workload_id": workload_id,
    }
    for field, wanted in expected.items():
        if row.get(field) != wanted:
            raise ValueError(
                f"{context}: {field} is {row.get(field)!r}, expected {wanted!r}"
            )
    if _integer(row.get("seed_or_session"), context=f"{context} session") != session:
        raise ValueError(f"{context}: seed_or_session mismatch")
    if _truth(row.get("code_dirty"), context=f"{context} code_dirty"):
        raise ValueError(f"{context}: code_dirty must be false")
    if row.get("precision_policy") != precision_policy(str(workload["spec"]["dtype"])):
        raise ValueError(f"{context}: precision_policy mismatch")


def _outcome_base(
    experiment_id: str,
    key: tuple[str, int, str],
    workload: Mapping[str, Any],
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    workload_id, session, variant = key
    spec = workload["spec"]
    return {
        "experiment_id": experiment_id,
        "workload_id": workload_id,
        "workload_kind": spec["kind"],
        "mode": spec["mode"],
        "shape": canonical_json(spec["shape"]),
        "batch_size": workload["batch_size"],
        "model_initial_width": workload["initial_width"],
        "session": session,
        "variant": variant,
        "variant_role": cohort["variant_roles"][variant],
    }


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
    workload = registration["workloads"][workload_id]
    spec = workload["spec"]
    timed = int(config["timed_calls_per_session"])
    if len(rows) != timed + 3:
        raise ValueError(f"{context}: row count is {len(rows)}, expected {timed + 3}")
    expected_coordinate = (
        "euclidean_to_euclidean" if variant == "E" else "tangent_to_tangent"
    )
    for row in rows:
        _validate_identity(
            row,
            experiment_id=experiment_id,
            key=key,
            workload=workload,
            expected_commit=expected_commit,
            context=context,
        )
        checks = {
            "warmup_count": _integer(row.get("warmup_count"), context=context)
            == int(config["warmup_calls"]),
            "repeat_count": _integer(row.get("repeat_count"), context=context) == timed,
            "state_evolution": _truth(
                row.get("state_evolves_across_calls"), context=context
            )
            == (spec["mode"] == "update"),
            "coordinate_boundary": row.get("coordinate_boundary") == expected_coordinate,
            "operation_boundary": row.get("operation_boundary")
            == f"{spec['kind']}:{spec['mode']}:depth=1",
            "shape": row.get("shape") == canonical_json(spec["shape"]),
        }
        failed = [field for field, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{context}: {failed} mismatch")
        _snapshot(
            row.get("gpu_snapshot_before"),
            physical_index=int(cohort["physical_index"]),
            context=context,
            before=True,
        )
        _snapshot(
            row.get("gpu_snapshot_after"),
            physical_index=int(cohort["physical_index"]),
            context=context,
            before=False,
        )

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
        observed = [
            _integer(row.get("observation"), context=context) for row in grouped[phase]
        ]
        if set(observed) != observations or len(observed) != len(observations):
            raise ValueError(f"{context}: invalid {phase} observations")
    status = _one((row.get("status") for row in grouped["steady_state"]), context=context)
    if status not in {"ok", "nonfinite"}:
        raise ValueError(f"{context}: invalid result status {status!r}")
    for phase in ("compilation", "first_compiled_call"):
        if grouped[phase][0].get("status") != status:
            raise ValueError(f"{context}: inconsistent {phase} status")
    sustained = grouped["sustained_sequence"][0]
    if sustained.get("status") != "not_applicable" or sustained.get("latency_ms") not in (
        "",
        None,
    ):
        raise ValueError(f"{context}: sustained sequence must be not applicable")

    latencies = [
        _finite(row.get("latency_ms"), context=f"{context} latency", positive=True)
        for row in grouped["steady_state"]
    ]
    exemplar = rows[0]
    for field in (
        "measured_peak_bytes_in_use",
        "measured_peak_pool_bytes",
        "process_reservation_mib",
        "output_nonfinite_count",
        "state_summary_before",
        "state_summary_after",
    ):
        _one((row.get(field) for row in rows), context=f"{context} {field}")
    peak_live = _integer(
        exemplar.get("measured_peak_bytes_in_use"), context=context, minimum=1
    )
    peak_pool = _integer(
        exemplar.get("measured_peak_pool_bytes"), context=context, minimum=1
    )
    if peak_pool < peak_live:
        raise ValueError(f"{context}: peak pool bytes are below peak live bytes")
    output_nonfinite = _integer(
        exemplar.get("output_nonfinite_count"), context=context, minimum=0
    )
    state_nonfinite = sum(
        int(json.loads(str(exemplar[field]))["nonfinite_count"])
        for field in ("state_summary_before", "state_summary_after")
    )
    if (status == "ok") == bool(output_nonfinite or state_nonfinite):
        raise ValueError(f"{context}: status {status!r} contradicts nonfinite counts")
    median_latency = statistics.median(latencies)
    return {
        **_outcome_base(experiment_id, key, workload, cohort),
        "outcome": status,
        "steady_state_observations": timed,
        "session_median_latency_ms": median_latency,
        "session_median_throughput_samples_per_s": 1000.0
        * workload["batch_size"]
        / median_latency,
        "compilation_ms": _finite(
            grouped["compilation"][0].get("latency_ms"), context=context, positive=True
        ),
        "first_compiled_call_ms": _finite(
            grouped["first_compiled_call"][0].get("latency_ms"),
            context=context,
            positive=True,
        ),
        "peak_live_gib": peak_live / 1024**3,
        "peak_pool_gib": peak_pool / 1024**3,
        "process_reservation_mib": _integer(
            exemplar.get("process_reservation_mib"), context=context, minimum=0
        ),
        "max_steady_latency_over_session_median": max(latencies) / median_latency,
        "failure_kind": None,
        "failure_phase": None,
        "failure_returncode": None,
        "failure_requested_allocation_bytes": None,
        "failure_diagnostic_excerpt": None,
        "failure_stdout_sha256": None,
        "failure_stderr_sha256": None,
        "completed_utc": max(str(row["timestamp_utc"]) for row in rows),
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "code_commit": expected_commit,
        "config_hash": workload["config_hash"],
        "device_id": EXPECTED_DEVICE_ID,
        "physical_gpu_index": cohort["physical_index"],
        "physical_gpu_uuid": PHYSICAL_GPU_UUIDS[int(cohort["physical_index"])],
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
    workload = registration["workloads"][workload_id]
    spec = workload["spec"]
    _validate_identity(
        row,
        experiment_id=experiment_id,
        key=key,
        workload=workload,
        expected_commit=expected_commit,
        context=context,
    )
    if row.get("status") != "failed":
        raise ValueError(f"{context}: status must be failed")
    if row.get("shape") != spec["shape"]:
        raise ValueError(f"{context}: shape mismatch")
    if row.get("operation_boundary") != f"{spec['kind']}:{spec['mode']}":
        raise ValueError(f"{context}: operation boundary mismatch")
    failure_kind = str(row.get("failure_kind"))
    if failure_kind not in {"oom", "failed", "terminated_by_signal"}:
        raise ValueError(f"{context}: unrecognized failure kind {failure_kind!r}")
    phase = str(row.get("failure_phase", "unknown"))
    if phase not in FAILURE_PHASES:
        raise ValueError(f"{context}: invalid failure phase {phase!r}")
    for before in (True, False):
        _snapshot(
            row.get("gpu_snapshot_before" if before else "gpu_snapshot_after"),
            physical_index=int(cohort["physical_index"]),
            context=context,
            before=before,
        )
    stdout = str(row.get("stdout", ""))
    stderr = str(row.get("stderr", ""))
    requested = {int(value) for value in REQUESTED_BYTES_PATTERN.findall(stderr)}
    if len(requested) > 1:
        raise ValueError(f"{context}: inconsistent requested allocation sizes")
    if failure_kind == "oom" and not OOM_PATTERN.search(f"{stdout}\n{stderr}"):
        raise ValueError(f"{context}: OOM lacks an explicit allocator diagnostic")
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    excerpt = next((line for line in lines if OOM_PATTERN.search(line)), None)
    if excerpt is None:
        excerpt = lines[-1] if lines else ""
    return {
        **_outcome_base(experiment_id, key, workload, cohort),
        "outcome": "oom" if failure_kind == "oom" else "failed",
        "steady_state_observations": 0,
        "session_median_latency_ms": None,
        "session_median_throughput_samples_per_s": None,
        "compilation_ms": None,
        "first_compiled_call_ms": None,
        "peak_live_gib": None,
        "peak_pool_gib": None,
        "process_reservation_mib": None,
        "max_steady_latency_over_session_median": None,
        "failure_kind": failure_kind,
        "failure_phase": phase,
        "failure_returncode": _integer(row.get("returncode"), context=context),
        "failure_requested_allocation_bytes": next(iter(requested)) if requested else None,
        "failure_diagnostic_excerpt": excerpt[:500],
        "failure_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "failure_stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "completed_utc": str(row["timestamp_utc"]),
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "code_commit": expected_commit,
        "config_hash": workload["config_hash"],
        "device_id": EXPECTED_DEVICE_ID,
        "physical_gpu_index": cohort["physical_index"],
        "physical_gpu_uuid": PHYSICAL_GPU_UUIDS[int(cohort["physical_index"])],
    }


def _validate_cohort(
    artifact_dir: Path,
    experiment_id: str,
    registration: Mapping[str, Any],
    *,
    expected_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = COHORTS[experiment_id]
    config = registration["config"]
    raw_path = artifact_dir / str(cohort["raw"])
    failure_path = raw_path.with_name(f"{raw_path.name}.failures.jsonl")
    ordered = [
        (workload_id, session, variant)
        for workload_id in registration["workloads"]
        for session in range(int(config["sessions"]))
        for variant in config["variants"]
    ]
    expected = set(ordered)
    successes: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for line, row in enumerate(_read_csv(raw_path), start=2):
        key = (str(row["workload_id"]), int(row["seed_or_session"]), str(row["variant"]))
        if key not in expected:
            raise ValueError(f"{raw_path.name}:{line}: unexpected group {key}")
        successes[key].append(row)
    failures: dict[tuple[str, int, str], dict[str, Any]] = {}
    for line, row in enumerate(_read_jsonl(failure_path), start=1):
        key = (str(row["workload_id"]), int(row["seed_or_session"]), str(row["variant"]))
        if key not in expected or key in failures:
            raise ValueError(f"{failure_path.name}:{line}: unexpected or duplicate {key}")
        failures[key] = row
    if set(successes) & set(failures):
        raise ValueError(f"{experiment_id}: groups have both success and failure")
    observed = set(successes) | set(failures)
    if observed != expected:
        raise ValueError(
            f"{experiment_id}: outcome matrix mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    outcomes = [
        _validate_success_group(
            successes[key],
            experiment_id=experiment_id,
            key=key,
            registration=registration,
            cohort=cohort,
            expected_commit=expected_commit,
        )
        if key in successes
        else _validate_failure(
            failures[key],
            experiment_id=experiment_id,
            key=key,
            registration=registration,
            cohort=cohort,
            expected_commit=expected_commit,
        )
        for key in ordered
    ]
    return outcomes, {
        "experiment_id": experiment_id,
        "raw": _evidence(raw_path, required=False),
        "failures": _evidence(failure_path, required=False),
        "raw_rows": sum(len(rows) for rows in successes.values()),
        "planned_groups": len(expected),
        "successful_worker_groups": len(successes),
        "failed_worker_groups": len(failures),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in outcomes).items())),
        "config": {**_evidence(registration["config_path"]), "path": cohort["config"]},
        "expected_config_hashes": {
            workload_id: value["config_hash"]
            for workload_id, value in registration["workloads"].items()
        },
    }


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _marker(log_dir: Path, name: str, *, required: bool = True) -> dict[str, Any] | None:
    path = log_dir / name
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    text = path.read_text(encoding="utf-8").strip()
    return {
        "file": f"logs/{name}",
        "sha256": _evidence(path)["sha256"],
        "utc": text.split()[0],
        "text": text,
    }


def _resume_events(log_dir: Path, experiment: str) -> tuple[list[datetime], list[datetime]]:
    """Return resume-attempt starts and harness crashes on nvidia-smi timeouts."""

    path = log_dir / f"{experiment}_resume.log"
    attempts: list[datetime] = []
    crashes: list[datetime] = []
    timed_out = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("=== attempt "):
            attempts.append(_utc(line.split()[3].rstrip(":")))
            timed_out = False
        elif "TimeoutExpired" in line and "nvidia-smi" in line:
            timed_out = True
        elif line.startswith("=== exit ") and timed_out:
            crashes.append(_utc(line.split()[3]))
    return attempts, crashes


def _stall_intervals(log_dir: Path) -> list[tuple[datetime, datetime]]:
    """Merge widened kernel soft-lockup intervals (rule in logs/driver_stall_rule.txt)."""

    intervals = []
    for line in (
        (log_dir.parent / KERNEL_LOCKUP_LOG).read_text(encoding="utf-8").splitlines()
    ):
        match = SOFT_LOCKUP_PATTERN.search(line)
        if match is None:
            continue
        end = _utc(line.split()[0])
        intervals.append((end - timedelta(seconds=int(match.group(1))), end))
    for experiment in ("e29", "e30"):
        for crash in _resume_events(log_dir, experiment)[1]:
            intervals.append((crash - timedelta(seconds=TIMEOUT_CRASH_SECONDS), crash))
    margin = timedelta(seconds=STALL_MARGIN_SECONDS)
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted((start - margin, end + margin) for start, end in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _annotate_schedule(
    outcomes: Sequence[dict[str, Any]],
    *,
    stage_start: str,
    attempt_starts: Sequence[datetime],
    stall_intervals: Sequence[tuple[datetime, datetime]],
    other_windows: Sequence[tuple[datetime, datetime, str]],
) -> None:
    """Flag group windows that overlap the other GPU's experiment or a driver stall.

    Groups run one at a time, so a group's window opens at the later of the
    previous group's completion (or the stage start marker) and the latest
    resume-attempt start before its completion, and closes at its completion.
    """

    previous = _utc(stage_start)
    for row in sorted(outcomes, key=lambda item: _utc(item["completed_utc"])):
        completed = _utc(row["completed_utc"])
        if completed < previous:
            raise ValueError("group completion precedes its experiment start")
        restarts = [start for start in attempt_starts if start <= completed]
        opened = max([previous, *restarts[-1:]])
        labels = [
            label
            for start, end, label in other_windows
            if opened < end and completed > start
        ]
        row["window_start_utc"] = opened.isoformat().replace("+00:00", "Z")
        row["other_gpu_concurrent_activity"] = bool(labels)
        row["other_gpu_activity"] = "; ".join(labels) if labels else None
        row["driver_stall_overlap"] = any(
            opened < end and completed > start for start, end in stall_intervals
        )
        previous = completed


def _variant_summaries(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[(row["workload_id"], row["variant"])].append(row)
    output = []
    for (workload_id, variant), selected in grouped.items():
        exemplar = selected[0]
        finite = [row for row in selected if row["outcome"] == "ok"]
        # Primary sessions: accepted and not overlapping a driver stall (fixed rule).
        accepted = [row for row in finite if not row["driver_stall_overlap"]]
        latencies = [float(row["session_median_latency_ms"]) for row in accepted]
        throughputs = [
            float(row["session_median_throughput_samples_per_s"]) for row in accepted
        ]
        live = [float(row["peak_live_gib"]) for row in accepted]
        pool = [float(row["peak_pool_gib"]) for row in accepted]
        all_latencies = [float(row["session_median_latency_ms"]) for row in finite]
        all_live = [float(row["peak_live_gib"]) for row in finite]
        isolated = [row for row in accepted if not row["other_gpu_concurrent_activity"]]
        isolated_latencies = [float(row["session_median_latency_ms"]) for row in isolated]
        isolated_live = [float(row["peak_live_gib"]) for row in isolated]
        output.append(
            {
                "experiment_id": exemplar["experiment_id"],
                "workload_id": workload_id,
                "workload_kind": exemplar["workload_kind"],
                "mode": exemplar["mode"],
                "shape": exemplar["shape"],
                "batch_size": exemplar["batch_size"],
                "model_initial_width": exemplar["model_initial_width"],
                "variant": variant,
                "variant_role": exemplar["variant_role"],
                "planned_sessions": len(selected),
                "finite_sessions": len(finite),
                "driver_stall_flagged_sessions": sum(
                    bool(row["driver_stall_overlap"]) for row in finite
                ),
                "accepted_finite_sessions": len(accepted),
                "nonfinite_sessions": sum(
                    row["outcome"] == "nonfinite" for row in selected
                ),
                "oom_sessions": sum(row["outcome"] == "oom" for row in selected),
                "failed_sessions": sum(row["outcome"] == "failed" for row in selected),
                "oom_failure_phases": json.dumps(
                    sorted(
                        {
                            str(row["failure_phase"])
                            for row in selected
                            if row["outcome"] == "oom"
                        }
                    ),
                    separators=(",", ":"),
                ),
                "latency_estimand": (
                    "median of accepted session-level medians (finite sessions without a "
                    "driver-stall overlap); each session median uses 200 synchronized "
                    "steady-state calls"
                ),
                "median_latency_ms": statistics.median(latencies) if latencies else None,
                "min_session_latency_ms": min(latencies) if latencies else None,
                "max_session_latency_ms": max(latencies) if latencies else None,
                "q1_session_latency_ms": _quantile(latencies, 0.25),
                "q3_session_latency_ms": _quantile(latencies, 0.75),
                "session_medians_ms": json.dumps(latencies, separators=(",", ":")),
                "throughput_estimand": (
                    "median of accepted per-session throughputs; each is 1000 times the "
                    "batch size divided by the session-median latency in milliseconds"
                ),
                "median_throughput_samples_per_s": (
                    statistics.median(throughputs) if throughputs else None
                ),
                "session_throughputs_samples_per_s": json.dumps(
                    throughputs, separators=(",", ":")
                ),
                "median_peak_live_gib": statistics.median(live) if live else None,
                "min_peak_live_gib": min(live) if live else None,
                "max_peak_live_gib": max(live) if live else None,
                "median_peak_pool_gib": statistics.median(pool) if pool else None,
                "all_finite_sessions_median_latency_ms": (
                    statistics.median(all_latencies) if all_latencies else None
                ),
                "all_finite_sessions_median_peak_live_gib": (
                    statistics.median(all_live) if all_live else None
                ),
                "sessions_with_other_gpu_activity": sum(
                    bool(row["other_gpu_concurrent_activity"]) for row in selected
                ),
                "isolated_accepted_sessions": len(isolated),
                "median_latency_ms_isolated_sessions": (
                    statistics.median(isolated_latencies) if isolated_latencies else None
                ),
                "median_peak_live_gib_isolated_sessions": (
                    statistics.median(isolated_live) if isolated_live else None
                ),
                "protocol_version": exemplar["protocol_version"],
                "code_commit": exemplar["code_commit"],
                "config_hash": exemplar["config_hash"],
                "device_id": exemplar["device_id"],
                "physical_gpu_uuid": exemplar["physical_gpu_uuid"],
            }
        )
    return output


def _paired_contrasts(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["workload_id"], int(row["session"]), row["variant"]): row for row in outcomes
    }
    output = []
    for workload_id in dict.fromkeys(row["workload_id"] for row in outcomes):
        sessions = sorted(
            {int(row["session"]) for row in outcomes if row["workload_id"] == workload_id}
        )
        for reference, candidate, role in COHORTS["E30"]["contrasts"]:
            paired = []
            finite_pairs = []
            exclusions = {}
            for session in sessions:
                left = indexed[(workload_id, session, reference)]
                right = indexed[(workload_id, session, candidate)]
                if left["outcome"] == right["outcome"] == "ok":
                    finite_pairs.append((session, left, right))
                    if not (left["driver_stall_overlap"] or right["driver_stall_overlap"]):
                        paired.append((session, left, right))
                        continue
                exclusions[str(session)] = {
                    variant: (
                        "driver_stall_overlap"
                        if row["outcome"] == "ok" and row["driver_stall_overlap"]
                        else row["outcome"]
                    )
                    for variant, row in ((reference, left), (candidate, right))
                }
            finite_ratios = [
                float(left["session_median_latency_ms"])
                / float(right["session_median_latency_ms"])
                for _, left, right in finite_pairs
            ]
            finite_lower, finite_upper = _log_t_interval(finite_ratios)
            latency_ratios = [
                float(left["session_median_latency_ms"])
                / float(right["session_median_latency_ms"])
                for _, left, right in paired
            ]
            live_ratios = [
                float(left["peak_live_gib"]) / float(right["peak_live_gib"])
                for _, left, right in paired
            ]
            latency_ratio = _geometric_mean(latency_ratios)
            lower, upper = _log_t_interval(latency_ratios)
            live_ratio = _geometric_mean(live_ratios)
            exemplar = indexed[(workload_id, sessions[0], reference)]
            output.append(
                {
                    "experiment_id": exemplar["experiment_id"],
                    "workload_id": workload_id,
                    "mode": exemplar["mode"],
                    "shape": exemplar["shape"],
                    "batch_size": exemplar["batch_size"],
                    "model_initial_width": exemplar["model_initial_width"],
                    "contrast_role": role,
                    "reference_variant": reference,
                    "candidate_variant": candidate,
                    "planned_paired_sessions": len(sessions),
                    "accepted_paired_sessions": len(paired),
                    "all_planned_sessions_paired": len(paired) == len(sessions),
                    "excluded_session_outcomes": json.dumps(
                        exclusions, sort_keys=True, separators=(",", ":")
                    ),
                    "latency_ratio_estimand": (
                        "geometric mean of paired session-median reference/candidate "
                        "latency ratios with a two-sided 95% Student-t interval on log "
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
                    "geometric_mean_peak_live_ratio": live_ratio,
                    "candidate_peak_live_reduction_percent": (
                        100.0 * (1.0 - 1.0 / live_ratio) if live_ratio else None
                    ),
                    "paired_peak_live_ratios": json.dumps(
                        live_ratios, separators=(",", ":")
                    ),
                    "all_finite_pairs": len(finite_pairs),
                    "all_finite_pairs_geometric_mean_latency_ratio": _geometric_mean(
                        finite_ratios
                    ),
                    "all_finite_pairs_log_t_95_lower": finite_lower,
                    "all_finite_pairs_log_t_95_upper": finite_upper,
                    "protocol_version": exemplar["protocol_version"],
                    "code_commit": exemplar["code_commit"],
                    "config_hash": exemplar["config_hash"],
                    "device_id": exemplar["device_id"],
                    "physical_gpu_uuid": exemplar["physical_gpu_uuid"],
                }
            )
    return output


def _published_medians() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Read published E08/E13 R/F medians from checksummed result tables."""

    evidence = {name: _checksummed_result(path) for name, path in PUBLISHED_SOURCES.items()}
    medians: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_csv(RESULTS_DIR / PUBLISHED_SOURCES["E08"]):
        if row["workload_id"] in E08_WORKLOADS and row["variant"] in {"R", "F"}:
            medians[(row["workload_id"], row["variant"])] = {
                "source_experiment": "E08",
                "source": "E08",
                "protocol_version": row["protocol_version"],
                "code_commit": row["code_commit"],
                "median_latency_ms": float(row["median_latency_ms"]),
                "median_peak_live_gib": float(row["median_peak_live_gib"]),
                "outcome": f"ok {row['sessions']}/{row['sessions']}",
            }
    for row in _read_csv(RESULTS_DIR / PUBLISHED_SOURCES["E13"]):
        if row["physical_gpu_uuid"] != PUBLISHED_PHYSICAL_GPU_UUID:
            raise ValueError("E13 published rows do not bind the expected V100")
        expected = int(row["expected_sessions"])
        medians[(row["workload_id"], row["variant"])] = {
            "source_experiment": "E13",
            "source": "E13",
            "protocol_version": row["protocol_version"],
            "code_commit": row["code_commit"],
            "median_latency_ms": _optional_float(
                row["median_latency_ms_successful_sessions"]
            ),
            "median_peak_live_gib": _optional_float(
                row["median_peak_live_gib_successful_sessions"]
            ),
            "outcome": (
                f"ok {row['successful_sessions']}/{expected}"
                if int(row["successful_sessions"]) == expected
                else f"oom {row['oom_sessions']}/{expected}; "
                f"failed {row['failed_sessions']}/{expected}"
            ),
        }
    return medians, evidence


def _published_reference_ratios(
    variants: Sequence[Mapping[str, Any]],
    published: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in variants:
        workload_id = str(row["workload_id"])
        reference = published[(workload_id, "R")]
        factorized = published[(workload_id, "F")]
        e_latency = _optional_float(row["median_latency_ms"])
        e_peak = _optional_float(row["median_peak_live_gib"])
        output.append(
            {
                "workload_id": workload_id,
                "mode": row["mode"],
                "shape": row["shape"],
                "batch_size": row["batch_size"],
                "published_source_experiment": reference["source_experiment"],
                "published_protocol_version": reference["protocol_version"],
                "published_code_commit": reference["code_commit"],
                "published_physical_gpu_uuid": PUBLISHED_PHYSICAL_GPU_UUID,
                "published_R_outcome": reference["outcome"],
                "published_F_outcome": factorized["outcome"],
                "published_R_median_latency_ms": reference["median_latency_ms"],
                "published_F_median_latency_ms": factorized["median_latency_ms"],
                "published_R_median_peak_live_gib": reference["median_peak_live_gib"],
                "published_F_median_peak_live_gib": factorized["median_peak_live_gib"],
                "E29_accepted_sessions": row["accepted_finite_sessions"],
                "E29_planned_sessions": row["planned_sessions"],
                "E29_median_latency_ms": e_latency,
                "E29_median_throughput_samples_per_s": row[
                    "median_throughput_samples_per_s"
                ],
                "E29_median_peak_live_gib": e_peak,
                "E29_physical_gpu_uuid": row["physical_gpu_uuid"],
                "E29_sessions_with_other_gpu_activity": row[
                    "sessions_with_other_gpu_activity"
                ],
                "E29_median_latency_ms_isolated_sessions": row[
                    "median_latency_ms_isolated_sessions"
                ],
                "ratio_estimand": (
                    "descriptive unpaired ratio of medians across separate experiments, "
                    "dates, protocol versions, and physical V100s; no interval"
                ),
                "R_over_E_latency": _ratio(reference["median_latency_ms"], e_latency),
                "F_over_E_latency": _ratio(factorized["median_latency_ms"], e_latency),
                "R_over_E_peak_live": _ratio(reference["median_peak_live_gib"], e_peak),
                "F_over_E_peak_live": _ratio(factorized["median_peak_live_gib"], e_peak),
            }
        )
    return output


def _width_series(
    variants: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    published: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    paired = {
        row["workload_id"]: row
        for row in _read_csv(RESULTS_DIR / PUBLISHED_SOURCES["E13_paired"])
    }[E13_WIDTH_REFERENCE_WORKLOAD]
    reference = published[(E13_WIDTH_REFERENCE_WORKLOAD, "R")]
    factorized = published[(E13_WIDTH_REFERENCE_WORKLOAD, "F")]
    peak_ratio = float(paired["geometric_mean_peak_live_ratio"])
    rows = [
        {
            "model_initial_width": REGISTERED_UNET_WIDTH,
            "source_experiment": "E13",
            "workload_id": E13_WIDTH_REFERENCE_WORKLOAD,
            "R_outcome": reference["outcome"],
            "F_outcome": factorized["outcome"],
            "R_median_latency_ms": reference["median_latency_ms"],
            "F_median_latency_ms": factorized["median_latency_ms"],
            "R_median_peak_live_gib": reference["median_peak_live_gib"],
            "F_median_peak_live_gib": factorized["median_peak_live_gib"],
            "accepted_paired_sessions": int(paired["paired_latency_sessions"]),
            "geometric_mean_latency_ratio": float(paired["geometric_mean_latency_ratio"]),
            "log_t_95_lower": float(paired["log_t_95_interval_lower"]),
            "log_t_95_upper": float(paired["log_t_95_interval_upper"]),
            "geometric_mean_peak_live_ratio": peak_ratio,
            "F_peak_live_reduction_percent": 100.0 * (1.0 - 1.0 / peak_ratio),
            "protocol_version": reference["protocol_version"],
            "code_commit": reference["code_commit"],
            "physical_gpu_uuid": PUBLISHED_PHYSICAL_GPU_UUID,
            "comparison_note": "published width-8 pivot; compare descriptively only",
        }
    ]
    summaries = {(row["workload_id"], row["variant"]): row for row in variants}
    for contrast in contrasts:
        workload_id = contrast["workload_id"]
        left = summaries[(workload_id, "R")]
        right = summaries[(workload_id, "F")]

        def outcome(summary: Mapping[str, Any]) -> str:
            planned = summary["planned_sessions"]
            stalled = summary["driver_stall_flagged_sessions"]
            text = (
                f"ok {summary['finite_sessions']}/{planned}"
                if summary["finite_sessions"] == planned
                else f"ok {summary['finite_sessions']}/{planned}; "
                f"oom {summary['oom_sessions']}/{planned}; "
                f"failed {summary['failed_sessions']}/{planned}"
            )
            return f"{text}; stall-excluded {stalled}" if stalled else text

        live_ratio = contrast["geometric_mean_peak_live_ratio"]
        rows.append(
            {
                "model_initial_width": contrast["model_initial_width"],
                "source_experiment": "E30",
                "workload_id": workload_id,
                "R_outcome": outcome(left),
                "F_outcome": outcome(right),
                "R_median_latency_ms": left["median_latency_ms"],
                "F_median_latency_ms": right["median_latency_ms"],
                "R_median_peak_live_gib": left["median_peak_live_gib"],
                "F_median_peak_live_gib": right["median_peak_live_gib"],
                "accepted_paired_sessions": contrast["accepted_paired_sessions"],
                "geometric_mean_latency_ratio": contrast["geometric_mean_latency_ratio"],
                "log_t_95_lower": contrast["log_t_95_lower"],
                "log_t_95_upper": contrast["log_t_95_upper"],
                "geometric_mean_peak_live_ratio": live_ratio,
                "F_peak_live_reduction_percent": contrast[
                    "candidate_peak_live_reduction_percent"
                ],
                "protocol_version": contrast["protocol_version"],
                "code_commit": contrast["code_commit"],
                "physical_gpu_uuid": contrast["physical_gpu_uuid"],
                "comparison_note": "paired within E30 sessions",
            }
        )
    return rows


def _execution_schedule(
    log_dir: Path, outcomes: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Record the actual GPU schedule and flag each group's window.

    E29 was registered to run alone on GPU 0 with GPU 1 idle, followed by E30.
    E30 was instead started on GPU 1 while E29 was running; ``e30_parallel_note.log``
    marks the start of that GPU-1 activity. The E29 harness was later paused
    between groups (``e29_signals.log``) while E34 used the GPUs; a resume counts
    as a window start because no E29 worker existed during the pause.
    """

    e29_started = _marker(log_dir, "e29.started")
    e30_started = _marker(log_dir, "e30.started")
    e30_finished = _marker(log_dir, "e30.finished")
    parallel_note = _marker(log_dir, "e30_parallel_note.log", required=False)
    e29_end = max((row["completed_utc"] for row in outcomes["E29"]), key=_utc)
    e30_window = (_utc((parallel_note or e30_started)["utc"]), _utc(e30_finished["utc"]))
    e29_window = (_utc(e29_started["utc"]), _utc(e29_end))
    e34_windows = []
    for started in sorted(log_dir.glob("e34_*_gpu[0-9].started")):
        finished = started.with_suffix(".finished")
        end = finished.read_text(encoding="utf-8").strip() if finished.is_file() else None
        if end is None:
            raise ValueError(f"{started.name} has no finished marker")
        e34_windows.append(
            (
                _utc(started.read_text(encoding="utf-8")),
                _utc(end),
                f"E34 {started.name.removesuffix('.started')}",
            )
        )
    signals = log_dir / "e29_signals.log"
    signal_lines = (
        signals.read_text(encoding="utf-8").splitlines() if signals.is_file() else []
    )
    e29_resumes = [_utc(line.split()[0]) for line in signal_lines if "SIGCONT" in line]
    stalls = _stall_intervals(log_dir)
    concurrent = e30_window[0] < e29_window[1] and e30_window[1] > e29_window[0]
    attempts = {name: _resume_events(log_dir, name)[0] for name in ("e29", "e30")}
    _annotate_schedule(
        outcomes["E29"],
        stage_start=e29_started["utc"],
        attempt_starts=sorted([*attempts["e29"], *e29_resumes]),
        stall_intervals=stalls,
        other_windows=[(*e30_window, "E30 on physical GPU 1"), *e34_windows],
    )
    _annotate_schedule(
        outcomes["E30"],
        stage_start=e30_started["utc"],
        attempt_starts=attempts["e30"],
        stall_intervals=stalls,
        other_windows=[(*e29_window, "E29 on physical GPU 0"), *e34_windows],
    )

    def stamp(moment: datetime) -> str:
        return moment.isoformat().replace("+00:00", "Z")

    evidence = [
        {"file": f"logs/{name}", "sha256": _evidence(log_dir / name)["sha256"]}
        for name in (
            "e29_resume.log",
            "e30_resume.log",
            "driver_stall_rule.txt",
            "kernel_driver_lockups.log",
            *(["e29_signals.log"] if signal_lines else []),
        )
    ]
    return {
        "e29_and_e30_ran_concurrently": concurrent,
        "registered_schedule": (
            "E29 alone on physical GPU 0 with GPU 1 idle, then E30 on GPU 1"
        ),
        "e29_window_utc": [stamp(value) for value in e29_window],
        "e30_gpu1_activity_window_utc": [stamp(value) for value in e30_window],
        "e34_windows_utc": [
            [stamp(start), stamp(end), label] for start, end, label in e34_windows
        ],
        "e29_pause_signals": signal_lines,
        "markers": [
            marker
            for marker in (e29_started, e30_started, e30_finished, parallel_note)
            if marker is not None
        ],
        "driver_stall_intervals_utc": [[stamp(start), stamp(end)] for start, end in stalls],
        "evidence": evidence,
        "group_window_rule": (
            "a group window opens at the later of the previous group completion of "
            "the same experiment (or its start marker) and the latest resume-attempt "
            "or SIGCONT time before its completion, and closes at its own completion"
        ),
        "driver_stall_rule": (
            "groups whose window intersects a merged kernel soft-lockup or harness "
            "nvidia-smi-timeout interval, widened by 180 s, are excluded from primary "
            "medians and paired contrasts; all-session values are reported beside them"
        ),
        "other_gpu_activity_use": (
            "descriptive only; isolated-session medians are a sensitivity check"
        ),
    }


def export(
    artifact_dir: Path,
    output_dir: Path,
    *,
    protocol_path: Path = PROTOCOL_PATH,
    expected_commit: str = EXPECTED_COMMIT,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    analysis_code = git_provenance(REPOSITORY_ROOT)
    protocol, registrations = _load_registrations(protocol_path.expanduser().resolve())
    allocator_limit = _allocator_limit_bytes()
    published, published_evidence = _published_medians()

    paths = {
        experiment_id: {name: output_dir / filename for name, filename in files.items()}
        for experiment_id, files in OUTPUT_FILES.items()
    }
    existing = [
        str(path) for files in paths.values() for path in files.values() if path.exists()
    ]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    outcomes: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for experiment_id, cohort in COHORTS.items():
        runtime = _validate_runtime_capture(
            artifact_dir,
            cohort,
            expected_commit=expected_commit,
            allocator_limit_bytes=allocator_limit,
        )
        launcher = _validate_launcher(cohort)
        outcomes[experiment_id], cohort_audit = _validate_cohort(
            artifact_dir,
            experiment_id,
            registrations[experiment_id],
            expected_commit=expected_commit,
        )
        audits[experiment_id] = {
            "runtime_capture": runtime,
            "launcher": launcher,
            "cohort": cohort_audit,
        }
    schedule = _execution_schedule(artifact_dir / "logs", outcomes)

    tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for experiment_id, selected in outcomes.items():
        variants = _variant_summaries(selected)
        tables[experiment_id] = {"sessions": selected, "variants": variants}
        if experiment_id == "E29":
            tables[experiment_id]["published"] = _published_reference_ratios(
                variants, published
            )
        else:
            contrasts = _paired_contrasts(selected)
            tables[experiment_id]["contrasts"] = contrasts
            tables[experiment_id]["width_series"] = _width_series(
                variants, contrasts, published
            )
        audits[experiment_id]["execution_schedule"] = {
            **schedule,
            "groups_with_other_gpu_activity": sum(
                bool(row["other_gpu_concurrent_activity"]) for row in selected
            ),
            "planned_groups": len(selected),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for experiment_id, experiment_tables in tables.items():
        for name, rows in experiment_tables.items():
            _atomic_csv(paths[experiment_id][name], rows)
        summary = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "analysis_id": f"{experiment_id.lower()}_v100_reviewer_requested_export",
            "experiment_id": experiment_id,
            "derivation": (
                "CPU-only validation and deterministic aggregation of existing raw GPU "
                "artifacts; no benchmark is launched by this exporter. Failed groups are "
                "retained as planned outcomes and excluded explicitly from paired ratios."
            ),
            "protocol": {**_evidence(protocol_path), "path": "protocol.yaml"},
            "protocol_version": protocol["protocol_version"],
            "registration": protocol["revision_benchmarks"][experiment_id],
            "expected_run_commit": expected_commit,
            "analysis_code": analysis_code,
            "physical_device": {
                "hostname": EXPECTED_HOSTNAME,
                "physical_index": COHORTS[experiment_id]["physical_index"],
                "uuid": PHYSICAL_GPU_UUIDS[int(COHORTS[experiment_id]["physical_index"])],
                "name": EXPECTED_GPU_NAME,
            },
            "allocator_limit_reference": {
                **_checksummed_result(ALLOCATOR_LIMIT_EVIDENCE),
                "field": "hardware.jax_allocator_bytes_limit",
                "bytes": allocator_limit,
            },
            "published_reference_sources": published_evidence,
            **audits[experiment_id],
            "output_rows": {
                OUTPUT_FILES[experiment_id][name]: len(rows)
                for name, rows in experiment_tables.items()
            },
            "output_evidence": {
                OUTPUT_FILES[experiment_id][name]: _evidence(paths[experiment_id][name])
                for name in experiment_tables
            },
            "all_planned_groups_accounted_for": True,
        }
        _atomic_json(paths[experiment_id]["audit"], summary)
        summaries[experiment_id] = summary
    return summaries


def main() -> None:
    arguments = _parser().parse_args()
    summaries = export(
        arguments.artifact_dir,
        arguments.output_dir,
        expected_commit=arguments.expected_code_commit,
    )
    print(
        json.dumps(
            {name: summary["output_rows"] for name, summary in summaries.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
