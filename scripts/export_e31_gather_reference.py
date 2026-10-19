#!/usr/bin/env python3
"""Validate E31 gather-reference evidence and export reviewer-facing tables.

E31 times R, Rg, Mg, and F in paired fresh-process sessions on one RTX A6000.
This exporter is CPU-only: it validates the raw benchmark rows, the runtime
captures, the compile-only memory capture, and the GPU equivalence capture,
then writes deterministic session, variant, contrast, and equivalence tables.
It launches nothing. The compiled buffer-assignment byte fields of this run are
zero, both in the raw rows and in the compile-only capture, so they are
validated and hashed but not reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from poincare_exec.artifacts.provenance import git_provenance, precision_policy
from poincare_exec.config.loader import canonical_json, config_hash, load_yaml
from scripts import export_e24_e25_e27_tables as shared

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E31"
# The clean E31 run commit; the analysis was frozen before results.
EXPECTED_COMMIT = "486b5bd274a4c384d1f754f77f37dd6c43c3c46b"
EXPECTED_PROTOCOL_VERSION = "1.19"
# The E31 configuration hashes bind protocol 1.19; this snapshot is byte-identical
# to the protocol.yaml of the run commit and keeps them verifiable after amendments.
PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "protocols" / "protocol_1_19.yaml"
EXPORT_SCHEMA_VERSION = "1.0"
# E31 is registered on the same physical GPU as E24/E25.
EXPECTED_DEVICE = shared.EXPECTED_DEVICE
EXPECTED_REFERENCE_COMMIT = shared.EXPECTED_REFERENCE_COMMIT
CONFIG = "configs/benchmarks/e31_gather_reference.yaml"
RAW = "e31_gather_reference.csv"
MEMORY_CAPTURE = "e31_compiled_memory.json"
EQUIVALENCE_CAPTURE = "e31_gpu_equivalence.json"
COHORT: dict[str, Any] = {
    "runtime_gate": "e31_runtime_gate.json",
    "environment": "e31_environment.json",
    "runtime_role": "pinned_jax_0_4_25_cuda_11_8",
    "jax_version": "0.4.25",
    "jaxlib_version": "0.4.25",
    "ld_library_policy": "pinned_cuda_11_8",
}
VARIANT_ROLES = {
    "R": "reference_convolution_patch_ball_geometry",
    "Rg": "reference_gathered_patch_ball_geometry",
    "Mg": "gathered_patch_exact_score_control",
    "F": "factorized_primary_method",
}
CONTRAST_ROLES = {
    ("R", "Rg"): "patch_construction_convolution_to_gather_with_geometry_fixed",
    ("Rg", "F"): "factorized_vs_gathered_patch_reference",
    ("R", "F"): "factorized_vs_convolution_patch_reference",
    ("Rg", "Mg"): "geometry_removal_with_gathered_patch_fixed",
    ("Mg", "F"): "gathered_patch_materialization_and_backend_with_geometry_removed",
}
# Reported in addition to the registered contrasts and labelled post hoc.
POST_HOC_CONTRAST_ROLES = {
    ("R", "Mg"): "convolution_patch_reference_vs_gathered_patch_without_geometry",
}
COMPILER_FIELDS = (
    "compiler_argument_bytes",
    "compiler_output_bytes",
    "compiler_temporary_bytes",
    "compiler_alias_bytes",
)
MEMORY_WORKLOAD = "unet_b8_update"
MEMORY_VARIANTS = ("R", "Rg", "F")
EQUIVALENCE_PAIRS = (("R", "Rg"), ("M", "Mg"))
RUNTIME_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": str(EXPECTED_DEVICE["physical_index"]),
    "JAX_PLATFORMS": None,
    "NVIDIA_TF32_OVERRIDE": None,
    "XLA_FLAGS": None,
    "XLA_PYTHON_CLIENT_ALLOCATOR": None,
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
OUTPUT_FILES = {
    "sessions": "e31_session_outcomes.csv",
    "variants": "e31_variant_summary.csv",
    "contrasts": "e31_contrasts.csv",
    "gpu_equivalence": "e31_gpu_equivalence.csv",
    "audit": "e31_audit.json",
}
GIB = 1024**3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--expected-code-commit", default=EXPECTED_COMMIT)
    return parser


def _load_registration(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_yaml(protocol_path)
    if protocol.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise ValueError(
            f"protocol version is {protocol.get('protocol_version')!r}, "
            f"expected {EXPECTED_PROTOCOL_VERSION!r}"
        )
    if canonical_json(protocol.get("reference_commits", {})) != EXPECTED_REFERENCE_COMMIT:
        raise ValueError("protocol reference commits do not match the frozen registration")
    registered = protocol["revision_benchmarks"][EXPERIMENT_ID]
    config_path = REPOSITORY_ROOT / CONFIG
    config = load_yaml(config_path)
    contrasts = [tuple(pair) for pair in registered.get("contrasts", [])]
    memory = registered.get("compiled_memory_analysis", {})
    equivalence = registered.get("gpu_equivalence_check", {})
    checks = {
        "experiment_id": config.get("experiment_id") == EXPERIMENT_ID,
        "config": registered.get("config") == CONFIG,
        "variants": registered.get("variants") == config.get("variants"),
        "variant_roles": list(VARIANT_ROLES) == config.get("variants"),
        "workloads": registered.get("workloads")
        == [workload["id"] for workload in config.get("workloads", [])],
        "sessions": registered.get("sessions") == config.get("sessions"),
        "warmup_calls": registered.get("warmup_calls") == config.get("warmup_calls"),
        "timed_calls_per_session": registered.get("timed_calls_per_session")
        == config.get("timed_calls_per_session"),
        "allocator_fraction": float(registered.get("allocator_fraction")) == 0.95,
        "preallocate": registered.get("preallocate") is False,
        "contrasts": contrasts == list(CONTRAST_ROLES),
        "memory_workload": memory.get("workload") == MEMORY_WORKLOAD,
        "memory_variants": tuple(memory.get("variants", ())) == MEMORY_VARIANTS,
        "equivalence_pairs": [tuple(pair) for pair in equivalence.get("pairs", [])]
        == list(EQUIVALENCE_PAIRS),
    }
    failed = [field for field, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{EXPERIMENT_ID}: frozen registration mismatch: {failed}")
    workloads = {}
    for workload in config["workloads"]:
        workloads[workload["id"]] = {
            "spec": workload,
            "config_hash": config_hash(protocol, config, workload),
            "batch_size": shared._validated_batch_size(
                workload, context=f"{EXPERIMENT_ID}/{workload['id']}"
            ),
        }
    return protocol, {"config": config, "config_path": config_path, "workloads": workloads}


def _validate_identity(
    row: Mapping[str, Any],
    *,
    key: tuple[str, int, str],
    registration: Mapping[str, Any],
    expected_commit: str,
    context: str,
) -> None:
    workload_id, session, variant = key
    workload = registration["workloads"][workload_id]
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "variant": variant,
        "config_hash": workload["config_hash"],
        "code_commit": expected_commit,
        "reference_commit": EXPECTED_REFERENCE_COMMIT,
        "device_id": f"{EXPECTED_DEVICE['hostname']}::gpu:0::{EXPECTED_DEVICE['name']}",
        "seed_or_session": session,
        "workload_id": workload_id,
    }
    for field, wanted in expected.items():
        observed: Any = row.get(field)
        if field == "seed_or_session":
            observed = shared._integer(observed, context=f"{context} {field}")
        if observed != wanted:
            raise ValueError(f"{context}: {field} is {observed!r}, expected {wanted!r}")
    if shared._truth(row.get("code_dirty"), context=f"{context} code_dirty"):
        raise ValueError(f"{context}: code_dirty must be false")
    if row.get("precision_policy") != precision_policy(str(workload["spec"]["dtype"])):
        raise ValueError(f"{context}: precision_policy mismatch")


def _session_base(
    key: tuple[str, int, str], registration: Mapping[str, Any]
) -> dict[str, Any]:
    workload_id, session, variant = key
    workload = registration["workloads"][workload_id]
    return {
        "experiment_id": EXPERIMENT_ID,
        "runtime_role": COHORT["runtime_role"],
        "jax_version": COHORT["jax_version"],
        "jaxlib_version": COHORT["jaxlib_version"],
        "workload_id": workload_id,
        "workload_kind": workload["spec"]["kind"],
        "mode": workload["spec"]["mode"],
        "shape": canonical_json(workload["spec"]["shape"]),
        "batch_size": workload["batch_size"],
        "session": session,
        "variant": variant,
        "variant_role": VARIANT_ROLES[variant],
    }


def _session_provenance(
    workload_id: str, registration: Mapping[str, Any], expected_commit: str, device_id: str
) -> dict[str, Any]:
    return {
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "code_commit": expected_commit,
        "config_hash": registration["workloads"][workload_id]["config_hash"],
        "device_id": device_id,
        "physical_gpu_index": EXPECTED_DEVICE["physical_index"],
        "physical_gpu_uuid": EXPECTED_DEVICE["uuid"],
        "physical_gpu_name": EXPECTED_DEVICE["name"],
    }


def _validate_success_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: tuple[str, int, str],
    registration: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    workload_id, session, variant = key
    context = f"{EXPERIMENT_ID}/{workload_id}/session-{session}/{variant}"
    config = registration["config"]
    workload = registration["workloads"][workload_id]["spec"]
    batch_size = int(registration["workloads"][workload_id]["batch_size"])
    timed = int(config["timed_calls_per_session"])
    if len(rows) != timed + 3:
        raise ValueError(f"{context}: row count is {len(rows)}, expected {timed + 3}")
    for row in rows:
        _validate_identity(
            row,
            key=key,
            registration=registration,
            expected_commit=expected_commit,
            context=context,
        )
        if int(row.get("warmup_count", -1)) != int(config["warmup_calls"]):
            raise ValueError(f"{context}: warmup_count mismatch")
        if int(row.get("repeat_count", -1)) != timed:
            raise ValueError(f"{context}: repeat_count mismatch")
        if shared._truth(
            row.get("state_evolves_across_calls"), context=f"{context} state evolution"
        ) != (workload["mode"] == "update"):
            raise ValueError(f"{context}: state-evolution mismatch")
        if row.get("coordinate_boundary") != "tangent_to_tangent":
            raise ValueError(f"{context}: coordinate boundary mismatch")
        if (
            row.get("operation_boundary")
            != f"{workload['kind']}:{workload['mode']}:depth=1"
        ):
            raise ValueError(f"{context}: operation boundary mismatch")
        if row.get("shape") != canonical_json(workload["shape"]):
            raise ValueError(f"{context}: shape mismatch")
        shared._snapshot(row.get("gpu_snapshot_before"), context=context, before=True)
        shared._snapshot(row.get("gpu_snapshot_after"), context=context, before=False)

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
            shared._integer(row.get("observation"), context=f"{context} observation")
            for row in grouped[phase]
        }
        if observed != observations or len(grouped[phase]) != len(observations):
            raise ValueError(f"{context}: invalid {phase} observations")

    result_status = shared._one(
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

    latencies = [
        shared._finite(row.get("latency_ms"), context=f"{context} latency", positive=True)
        for row in grouped["steady_state"]
    ]
    compilation_ms = shared._finite(
        grouped["compilation"][0].get("latency_ms"),
        context=f"{context} compilation latency",
        positive=True,
    )
    first_call_ms = shared._finite(
        grouped["first_compiled_call"][0].get("latency_ms"),
        context=f"{context} first-call latency",
        positive=True,
    )
    repeated_fields = (
        "measured_peak_bytes_in_use",
        "measured_peak_pool_bytes",
        "process_reservation_mib",
        "output_nonfinite_count",
        "state_summary_before",
        "state_summary_after",
        *COMPILER_FIELDS,
    )
    for field in repeated_fields:
        shared._one((row.get(field) for row in rows), context=f"{context} {field}")
    exemplar = rows[0]
    peak_live = shared._integer(
        exemplar.get("measured_peak_bytes_in_use"),
        context=f"{context} peak live bytes",
        minimum=1,
    )
    peak_pool = shared._integer(
        exemplar.get("measured_peak_pool_bytes"),
        context=f"{context} peak pool bytes",
        minimum=1,
    )
    if peak_pool < peak_live:
        raise ValueError(f"{context}: peak pool bytes are below peak live bytes")
    reservation = shared._integer(
        exemplar.get("process_reservation_mib"),
        context=f"{context} process reservation",
        minimum=0,
    )
    for field in COMPILER_FIELDS:
        shared._integer(exemplar.get(field), context=f"{context} {field}", minimum=0)
    output_nonfinite = shared._integer(
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

    session_median = statistics.median(latencies)
    return {
        **_session_base(key, registration),
        "outcome": result_status,
        "steady_state_observations": timed,
        "session_median_latency_ms": session_median,
        "session_median_throughput_samples_per_s": shared._finite(
            1000.0 * batch_size / session_median,
            context=f"{context} derived throughput",
            positive=True,
        ),
        "compilation_ms": compilation_ms,
        "first_compiled_call_ms": first_call_ms,
        "peak_live_gib": peak_live / GIB,
        "peak_pool_gib": peak_pool / GIB,
        "process_reservation_mib": reservation,
        "failure_kind": None,
        "failure_phase": None,
        "failure_returncode": None,
        "failure_stdout": None,
        "failure_stderr": None,
        **_session_provenance(
            workload_id, registration, expected_commit, exemplar["device_id"]
        ),
    }


def _validate_failure(
    row: Mapping[str, Any],
    *,
    key: tuple[str, int, str],
    registration: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    workload_id, session, variant = key
    context = f"{EXPERIMENT_ID}/{workload_id}/session-{session}/{variant}/failure"
    _validate_identity(
        row,
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
    returncode = shared._integer(row.get("returncode"), context=f"{context} returncode")
    shared._snapshot(row.get("gpu_snapshot_before"), context=context, before=True)
    shared._snapshot(row.get("gpu_snapshot_after"), context=context, before=False)
    workload = registration["workloads"][workload_id]["spec"]
    if row.get("shape") != workload["shape"]:
        raise ValueError(f"{context}: shape mismatch")
    if row.get("operation_boundary") != f"{workload['kind']}:{workload['mode']}":
        raise ValueError(f"{context}: operation boundary mismatch")
    return {
        **_session_base(key, registration),
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
        **_session_provenance(workload_id, registration, expected_commit, row["device_id"]),
    }


def _validate_cohort(
    artifact_dir: Path, registration: Mapping[str, Any], *, expected_commit: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path = artifact_dir / RAW
    failure_path = raw_path.with_name(f"{raw_path.name}.failures.jsonl")
    raw_rows = shared._read_csv(raw_path)
    failure_rows = shared._read_jsonl(failure_path)
    config = registration["config"]
    ordered_keys = [
        (workload_id, session, variant)
        for workload_id in registration["workloads"]
        for session in range(int(config["sessions"]))
        for variant in config["variants"]
    ]
    expected = set(ordered_keys)
    successes: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
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
        successes[key].append(row)
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
    overlap = set(successes) & set(failures)
    if overlap:
        raise ValueError(
            f"{EXPERIMENT_ID}: groups have both success and failure: {overlap}"
        )
    observed = set(successes) | set(failures)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"{EXPERIMENT_ID}: outcome matrix mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    outcomes = [
        _validate_success_group(
            successes[key],
            key=key,
            registration=registration,
            expected_commit=expected_commit,
        )
        if key in successes
        else _validate_failure(
            failures[key],
            key=key,
            registration=registration,
            expected_commit=expected_commit,
        )
        for key in ordered_keys
    ]
    return outcomes, {
        "experiment_id": EXPERIMENT_ID,
        "raw": shared._evidence(raw_path, required=False),
        "failures": shared._evidence(failure_path, required=False),
        "raw_rows": len(raw_rows),
        "planned_groups": len(expected),
        "successful_worker_groups": len(successes),
        "failed_worker_groups": len(failures),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in outcomes).items())),
        "config": shared._evidence(registration["config_path"]),
        "expected_config_hashes": {
            workload_id: value["config_hash"]
            for workload_id, value in registration["workloads"].items()
        },
    }


def _spread(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


def _values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def _paired_ratios(
    paired: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    field: str,
    *,
    candidate_over_reference: bool = False,
) -> list[float]:
    return [
        float(candidate[field]) / float(reference[field])
        if candidate_over_reference
        else float(reference[field]) / float(candidate[field])
        for _, reference, candidate in paired
    ]


def _variant_summaries(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[(row["workload_id"], row["variant"])].append(row)
    output = []
    for (workload_id, variant), selected in grouped.items():
        selected = sorted(selected, key=lambda row: int(row["session"]))
        exemplar = selected[0]
        accepted = [row for row in selected if row["outcome"] == "ok"]
        latencies = _values(accepted, "session_median_latency_ms")
        throughputs = _values(accepted, "session_median_throughput_samples_per_s")
        live = _spread(_values(accepted, "peak_live_gib"))
        output.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "runtime_role": exemplar["runtime_role"],
                "jax_version": exemplar["jax_version"],
                "jaxlib_version": exemplar["jaxlib_version"],
                "workload_id": workload_id,
                "workload_kind": exemplar["workload_kind"],
                "mode": exemplar["mode"],
                "shape": exemplar["shape"],
                "batch_size": exemplar["batch_size"],
                "variant": variant,
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
                "q1_session_latency_ms": shared._quantile(latencies, 0.25),
                "q3_session_latency_ms": shared._quantile(latencies, 0.75),
                "session_medians_ms": json.dumps(latencies, separators=(",", ":")),
                "median_throughput_samples_per_s": (
                    statistics.median(throughputs) if throughputs else None
                ),
                "median_peak_live_gib": live[0],
                "min_peak_live_gib": live[1],
                "max_peak_live_gib": live[2],
                "median_peak_pool_gib": _spread(_values(accepted, "peak_pool_gib"))[0],
                "median_process_reservation_mib": _spread(
                    _values(accepted, "process_reservation_mib")
                )[0],
                "protocol_version": exemplar["protocol_version"],
                "code_commit": exemplar["code_commit"],
                "config_hash": exemplar["config_hash"],
                "device_id": exemplar["device_id"],
                "physical_gpu_uuid": exemplar["physical_gpu_uuid"],
            }
        )
    return output


def _ratio_summary(ratios: Sequence[float]) -> tuple[Any, Any, Any]:
    lower, upper = shared._log_t_interval(ratios)
    return shared._geometric_mean(ratios), lower, upper


def _contrasts(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["workload_id"], int(row["session"]), row["variant"]): row for row in outcomes
    }
    workload_ids = list(dict.fromkeys(row["workload_id"] for row in outcomes))
    output = []
    for workload_id in workload_ids:
        sessions = sorted(
            {int(row["session"]) for row in outcomes if row["workload_id"] == workload_id}
        )
        roles = [(pair, role, "registered") for pair, role in CONTRAST_ROLES.items()] + [
            (pair, role, "post_hoc") for pair, role in POST_HOC_CONTRAST_ROLES.items()
        ]
        for (reference, candidate), role, registration in roles:
            paired = []
            exclusions = {}
            for session in sessions:
                reference_row = indexed[(workload_id, session, reference)]
                candidate_row = indexed[(workload_id, session, candidate)]
                if reference_row["outcome"] == candidate_row["outcome"] == "ok":
                    paired.append((session, reference_row, candidate_row))
                else:
                    exclusions[str(session)] = {
                        reference: reference_row["outcome"],
                        candidate: candidate_row["outcome"],
                    }
            latency_ratios = _paired_ratios(paired, "session_median_latency_ms")
            throughput_ratios = _paired_ratios(
                paired,
                "session_median_throughput_samples_per_s",
                candidate_over_reference=True,
            )
            live_ratios = _paired_ratios(paired, "peak_live_gib")
            latency, latency_lower, latency_upper = _ratio_summary(latency_ratios)
            throughput, throughput_lower, throughput_upper = _ratio_summary(
                throughput_ratios
            )
            live, live_lower, live_upper = _ratio_summary(live_ratios)
            exemplar = indexed[(workload_id, sessions[0], reference)]
            output.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "runtime_role": exemplar["runtime_role"],
                    "jax_version": exemplar["jax_version"],
                    "jaxlib_version": exemplar["jaxlib_version"],
                    "workload_id": workload_id,
                    "workload_kind": exemplar["workload_kind"],
                    "mode": exemplar["mode"],
                    "shape": exemplar["shape"],
                    "batch_size": exemplar["batch_size"],
                    "contrast_role": role,
                    "contrast_registration": registration,
                    "reference_variant": reference,
                    "reference_variant_role": VARIANT_ROLES[reference],
                    "candidate_variant": candidate,
                    "candidate_variant_role": VARIANT_ROLES[candidate],
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
                    "geometric_mean_latency_ratio": latency,
                    "log_t_95_lower": latency_lower,
                    "log_t_95_upper": latency_upper,
                    "paired_session_ids": json.dumps(
                        [session for session, _, _ in paired], separators=(",", ":")
                    ),
                    "paired_latency_ratios": json.dumps(
                        latency_ratios, separators=(",", ":")
                    ),
                    "geometric_mean_throughput_ratio": throughput,
                    "throughput_log_t_95_lower": throughput_lower,
                    "throughput_log_t_95_upper": throughput_upper,
                    "geometric_mean_peak_live_ratio": live,
                    "peak_live_log_t_95_lower": live_lower,
                    "peak_live_log_t_95_upper": live_upper,
                    "candidate_peak_live_reduction_percent": (
                        100.0 * (1.0 - 1.0 / live) if live else None
                    ),
                    "paired_peak_live_ratios": json.dumps(
                        live_ratios, separators=(",", ":")
                    ),
                    "protocol_version": exemplar["protocol_version"],
                    "code_commit": exemplar["code_commit"],
                    "config_hash": exemplar["config_hash"],
                    "device_id": exemplar["device_id"],
                    "physical_gpu_uuid": exemplar["physical_gpu_uuid"],
                }
            )
    return output


def _validate_capture_provenance(
    capture: Mapping[str, Any], *, name: str, expected_commit: str
) -> None:
    git = capture.get("git", {})
    if git.get("commit") != expected_commit or git.get("dirty") is not False:
        raise ValueError(f"{name}: expected a clean {expected_commit} checkout")
    if capture.get("jax_version") != COHORT["jax_version"]:
        raise ValueError(f"{name}: unexpected JAX version")
    if capture.get("jaxlib_version") != COHORT["jaxlib_version"]:
        raise ValueError(f"{name}: unexpected jaxlib version")
    environment = capture.get("environment", {})
    for field, expected in RUNTIME_ENVIRONMENT.items():
        if environment.get(field) != expected:
            raise ValueError(f"{name}: environment {field} is {environment.get(field)!r}")
    library_path = environment.get("LD_LIBRARY_PATH")
    if not isinstance(library_path, str) or "cuda-11.8" not in library_path:
        raise ValueError(f"{name}: CUDA-11.8 library path is not pinned")


def _compiled_memory(
    artifact_dir: Path,
    registration: Mapping[str, Any],
    *,
    expected_commit: str,
) -> dict[str, Any]:
    """Validate and hash the compile-only capture; its byte fields are not reported."""

    path = artifact_dir / MEMORY_CAPTURE
    capture = shared._read_json(path)
    _validate_capture_provenance(capture, name=path.name, expected_commit=expected_commit)
    if capture.get("workload_id") != MEMORY_WORKLOAD:
        raise ValueError(f"{path.name}: unexpected workload")
    captures = capture.get("captures", [])
    if tuple(item.get("variant") for item in captures) != MEMORY_VARIANTS:
        raise ValueError(f"{path.name}: expected variants {MEMORY_VARIANTS}")
    expected_hash = registration["workloads"][MEMORY_WORKLOAD]["config_hash"]
    buffer_bytes = []
    for item in captures:
        context = f"{path.name}/{item['variant']}"
        if item.get("config_hash") != expected_hash:
            raise ValueError(f"{context}: configuration hash mismatch")
        if item.get("device_kind") != EXPECTED_DEVICE["name"]:
            raise ValueError(f"{context}: unexpected device")
        buffer_bytes.extend(
            shared._integer(
                item["compiled_bytes"].get(field), context=f"{context} {field}", minimum=0
            )
            for field in ("argument", "output", "alias", "temp")
        )
    return {
        "capture": shared._evidence(path),
        "captured_at_utc": capture.get("captured_at_utc"),
        "buffer_assignment_bytes_all_zero": not any(buffer_bytes),
        "reported": False,
        "note": (
            "The compiled buffer-assignment byte fields are zero in this capture and "
            "in every raw benchmark row, so no compiled-memory estimate is reported."
        ),
    }


def _gpu_equivalence(
    artifact_dir: Path, registration: Mapping[str, Any], *, expected_commit: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = artifact_dir / EQUIVALENCE_CAPTURE
    capture = shared._read_json(path)
    _validate_capture_provenance(capture, name=path.name, expected_commit=expected_commit)
    if capture.get("device_kind") != EXPECTED_DEVICE["name"]:
        raise ValueError(f"{path.name}: unexpected device")
    workload_ids = list(registration["workloads"])
    model_rows = capture.get("model_comparisons", [])
    expected_models = [
        (workload_id, reference, candidate)
        for workload_id in workload_ids
        for reference, candidate in EQUIVALENCE_PAIRS
    ]
    observed_models = [
        (row["workload_id"], row["reference_variant"], row["candidate_variant"])
        for row in model_rows
    ]
    if observed_models != expected_models:
        raise ValueError(f"{path.name}: model comparison inventory mismatch")
    patch_rows = capture.get("patch_comparisons", [])
    if {row["workload_id"] for row in patch_rows} != set(workload_ids):
        raise ValueError(f"{path.name}: patch comparisons do not cover every workload")
    output = []
    for row in patch_rows:
        output.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "comparison": "patch_tensor",
                "workload_id": row["workload_id"],
                "reference": "R one-hot convolution patch",
                "candidate": "Rg gathered patch",
                "detail": canonical_json(
                    {
                        field: row[field]
                        for field in (
                            "input_shape",
                            "dtype",
                            "kernel_shape",
                            "stride",
                            "padding",
                            "dilation",
                        )
                    }
                ),
                "bitwise_equal": bool(row["bitwise_equal"]),
                "differing_elements": int(row["differing_elements"]),
                "elements": int(row["elements"]),
                "max_abs_difference": float(row["max_abs_difference"]),
                "max_abs_difference_over_reference_max_abs": float(
                    row["max_abs_difference_over_reference_max_abs"]
                ),
            }
        )
    for row in model_rows:
        output.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "comparison": f"one_compiled_{row['mode']}_call",
                "workload_id": row["workload_id"],
                "reference": row["reference_variant"],
                "candidate": row["candidate_variant"],
                "detail": "all output leaves",
                "bitwise_equal": bool(row["bitwise_equal"]),
                "differing_elements": int(row["differing_elements"]),
                "elements": int(row["elements"]),
                "max_abs_difference": float(row["max_abs_difference"]),
                "max_abs_difference_over_reference_max_abs": float(
                    row["max_abs_difference_over_reference_max_abs"]
                ),
            }
        )
    return output, {
        "capture": shared._evidence(path),
        "captured_at_utc": capture.get("captured_at_utc"),
        "patch_comparisons": len(patch_rows),
        "patch_comparisons_bitwise_equal": sum(
            bool(row["bitwise_equal"]) for row in patch_rows
        ),
        "model_comparisons": len(model_rows),
        "model_comparisons_bitwise_equal": sum(
            bool(row["bitwise_equal"]) for row in model_rows
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
    protocol_path = protocol_path.expanduser().resolve()
    analysis_code = git_provenance(REPOSITORY_ROOT)
    protocol, registration = _load_registration(protocol_path)
    runtime = shared._validate_runtime_capture(
        artifact_dir, COHORT, expected_commit=expected_commit
    )
    outcomes, cohort_audit = _validate_cohort(
        artifact_dir, registration, expected_commit=expected_commit
    )
    if {row["physical_gpu_uuid"] for row in outcomes} != {EXPECTED_DEVICE["uuid"]}:
        raise ValueError("E31 does not bind to the registered physical GPU")
    variants = _variant_summaries(outcomes)
    contrasts = _contrasts(outcomes)
    memory_audit = _compiled_memory(
        artifact_dir, registration, expected_commit=expected_commit
    )
    equivalence, equivalence_audit = _gpu_equivalence(
        artifact_dir, registration, expected_commit=expected_commit
    )
    tables = {
        "sessions": outcomes,
        "variants": variants,
        "contrasts": contrasts,
        "gpu_equivalence": equivalence,
    }
    paths = {name: output_dir / filename for name, filename in OUTPUT_FILES.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        shared._atomic_csv(paths[name], rows)
    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "analysis_id": "e31_gather_reference_export",
        "derivation": (
            "CPU-only validation and deterministic aggregation of existing raw GPU "
            "artifacts; no benchmark is launched by this exporter. Failed groups are "
            "retained as planned outcomes and excluded explicitly from paired ratios."
        ),
        "protocol": shared._evidence(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "expected_run_commit": expected_commit,
        "analysis_code": analysis_code,
        "physical_device": EXPECTED_DEVICE,
        "runtime_capture": runtime,
        "cohort": cohort_audit,
        "compiled_memory": memory_audit,
        "gpu_equivalence": equivalence_audit,
        "output_rows": {OUTPUT_FILES[name]: len(rows) for name, rows in tables.items()},
        "output_evidence": {
            OUTPUT_FILES[name]: shared._evidence(paths[name]) for name in tables
        },
        "all_planned_groups_accounted_for": True,
    }
    shared._atomic_json(paths["audit"], summary)
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
