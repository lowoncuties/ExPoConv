#!/usr/bin/env python3
"""Audit accepted E08 U-Net evidence and export publication-facing tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

from poincare_exec.artifacts import append_csv_rows, write_json
from poincare_exec.artifacts.provenance import git_provenance
from poincare_exec.config.constants import REPOSITORY_ROOT
from poincare_exec.config.loader import load_yaml


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    evidence: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        evidence.update({"sha256": _sha256(path), "byte_size": path.stat().st_size})
    return evidence


def _truth(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected a boolean CSV value, received {value!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _audit_attempts(source: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for attempt in source.get("wrapper_attempts", []):
        log_path = _resolve(attempt["log_path"])
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        status_value = None
        status_path = None
        if attempt.get("status_path") is not None:
            status_path = _resolve(attempt["status_path"])
            if not status_path.exists():
                raise FileNotFoundError(status_path)
            status_value = status_path.read_text(encoding="utf-8").strip()
            if status_value != str(attempt["expected_status"]):
                raise ValueError(
                    f"unexpected wrapper status in {status_path}: {status_value!r}"
                )
        output.append(
            {
                "log": _evidence(log_path),
                "status": _evidence(status_path),
                "status_value": status_value,
                "disposition": attempt["disposition"],
            }
        )
    return output


def _audit_failure_manifests(source: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for manifest in source.get("benchmark_failure_manifests", []):
        path = _resolve(manifest["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected = manifest["expected_rows"]
        if len(rows) != len(expected):
            raise ValueError(f"unexpected failure row count in {path}: {len(rows)}")
        unmatched = list(rows)
        for wanted in expected:
            for index, row in enumerate(unmatched):
                if all(row.get(field) == value for field, value in wanted.items()):
                    unmatched.pop(index)
                    break
            else:
                raise ValueError(f"missing expected failure row in {path}: {wanted}")
        output.append({"evidence": _evidence(path), "rows": rows})
    return output


def _audit_source(
    source: dict[str, Any], analysis: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    raw_path = _resolve(source["raw_path"])
    config_path = _resolve(source["config_path"])
    environment_path = _resolve(source["environment_path"])
    for path in (raw_path, config_path, environment_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = _read_csv(raw_path)
    workloads = {item["id"]: item for item in source["workloads"]}
    variants = {str(value) for value in analysis["variants"]}
    sessions = set(range(int(analysis["sessions"])))
    expected_groups = {
        (workload_id, session, variant)
        for workload_id in workloads
        for session in sessions
        for variant in variants
    }
    expected_rows_per_group = int(analysis["timed_calls_per_session"]) + 3
    expected_phases = {
        "compilation": 1,
        "first_compiled_call": 1,
        "sustained_sequence": 1,
        "steady_state": int(analysis["timed_calls_per_session"]),
    }
    group_counts: Counter[tuple[str, int, str]] = Counter()
    phase_counts: Counter[tuple[tuple[str, int, str], str]] = Counter()
    statuses: Counter[tuple[str, str]] = Counter()
    devices: set[str] = set()
    snapshot_indices: set[int] = set()
    snapshot_uuids: set[str] = set()
    timestamps: list[str] = []
    nonfinite_rows = 0
    contaminated_snapshots = 0
    expected_commit = str(source["required_code_commit"])
    expected_protocol = str(source["required_protocol_version"])
    expected_gpu = analysis["physical_gpu"]

    for row in rows:
        key = (row["workload_id"], int(row["seed_or_session"]), row["variant"])
        group_counts[key] += 1
        phase_counts[(key, row["phase"])] += 1
        statuses[(row["phase"], row["status"])] += 1
        devices.add(row["device_id"])
        timestamps.append(row["timestamp_utc"])
        if key not in expected_groups:
            raise ValueError(f"unexpected E08 group in {raw_path}: {key}")
        workload = workloads[row["workload_id"]]
        checks = {
            "experiment_id": row["experiment_id"] == analysis["experiment_id"],
            "protocol_version": row["protocol_version"] == expected_protocol,
            "code_commit": row["code_commit"] == expected_commit,
            "code_clean": not _truth(row["code_dirty"]),
            "config_hash": row["config_hash"] == workload["config_hash"],
            "warmup_count": int(row["warmup_count"]) == int(analysis["warmup_calls"]),
            "repeat_count": int(row["repeat_count"])
            == int(analysis["timed_calls_per_session"]),
            "state_evolution": _truth(row["state_evolves_across_calls"])
            == (workload["mode"] == "update"),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"row identity failure in {raw_path} {key}: {failed}")
        state_before = json.loads(row["state_summary_before"])
        state_after = json.loads(row["state_summary_after"])
        if (
            int(row["output_nonfinite_count"] or 0)
            or int(row["sustained_output_nonfinite_count"] or 0)
            or int(state_before["nonfinite_count"])
            or int(state_after["nonfinite_count"])
        ):
            nonfinite_rows += 1
        for field in ("gpu_snapshot_before", "gpu_snapshot_after"):
            snapshot = json.loads(row[field])
            if (
                snapshot.get("status") != "ok"
                or snapshot.get("process_query_status") != "ok"
            ):
                raise ValueError(f"invalid {field} for {key}: {snapshot}")
            snapshot_indices.add(int(snapshot["index"]))
            snapshot_uuids.add(str(snapshot["uuid"]))
            contaminated_snapshots += int(bool(snapshot["compute_processes"]))

    missing_groups = sorted(expected_groups - set(group_counts))
    bad_group_counts = {
        str(key): count
        for key, count in group_counts.items()
        if count != expected_rows_per_group
    }
    bad_phase_counts = {
        str((key, phase)): phase_counts[(key, phase)]
        for key in expected_groups
        for phase, count in expected_phases.items()
        if phase_counts[(key, phase)] != count
    }
    status_ok = all(
        status == "ok"
        if phase != "sustained_sequence"
        else status in {"ok", "not_applicable"}
        for phase, status in statuses
    )
    checks = {
        "complete_group_set": not missing_groups and set(group_counts) == expected_groups,
        "group_row_counts": not bad_group_counts,
        "phase_row_counts": not bad_phase_counts,
        "required_statuses": status_ok,
        "all_outputs_and_states_finite": nonfinite_rows == 0,
        "one_device_identity": len(devices) == 1,
        "physical_gpu_index": snapshot_indices == {int(expected_gpu["index"])},
        "physical_gpu_uuid": snapshot_uuids == {str(expected_gpu["uuid"])},
        "no_competing_snapshot_process": contaminated_snapshots == 0,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(f"source audit failed for {raw_path}: {failed_checks}")

    environment = _read_json(environment_path)
    environment_checks = {
        "commit": environment["git"]["commit"] == expected_commit,
        "clean": environment["git"]["dirty"] is False,
        "single_gpu_mask": str(environment["environment"]["CUDA_VISIBLE_DEVICES"])
        == str(expected_gpu["index"]),
        "jax_gpu_backend": environment["jax"]["backend"] == "gpu",
        "one_logical_jax_device": len(environment["jax"]["devices"]) == 1,
        "device_kind": environment["jax"]["devices"][0]["device_kind"]
        == expected_gpu["name"],
    }
    failed_environment = [name for name, passed in environment_checks.items() if not passed]
    if failed_environment:
        raise ValueError(
            f"environment audit failed for {environment_path}: {failed_environment}"
        )

    return rows, {
        "source_id": source["id"],
        "raw_evidence": _evidence(raw_path),
        "config_evidence": _evidence(config_path),
        "environment_evidence": _evidence(environment_path),
        "wrapper_attempts": _audit_attempts(source),
        "benchmark_failure_manifests": _audit_failure_manifests(source),
        "accepted_workloads": sorted(workloads),
        "code_commit": expected_commit,
        "protocol_version": expected_protocol,
        "device_identities": sorted(devices),
        "physical_gpu_indices": sorted(snapshot_indices),
        "physical_gpu_uuids": sorted(snapshot_uuids),
        "rows": len(rows),
        "groups": len(group_counts),
        "expected_rows_per_group": expected_rows_per_group,
        "statuses": {
            f"{phase}:{status}": count
            for (phase, status), count in sorted(statuses.items())
        },
        "missing_groups": missing_groups,
        "bad_group_counts": bad_group_counts,
        "bad_phase_counts": bad_phase_counts,
        "nonfinite_rows": nonfinite_rows,
        "contaminated_snapshots": contaminated_snapshots,
        "first_timestamp_utc": min(timestamps),
        "last_timestamp_utc": max(timestamps),
        "checks": checks,
        "environment_checks": environment_checks,
        "all_checks_pass": all(checks.values()) and all(environment_checks.values()),
    }


def _session_table(
    rows_by_source: list[tuple[dict[str, Any], list[dict[str, str]]]],
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for source, rows in rows_by_source:
        for workload in source["workloads"]:
            for session in range(int(analysis["sessions"])):
                for variant in analysis["variants"]:
                    selected = [
                        row
                        for row in rows
                        if row["phase"] == "steady_state"
                        and row["workload_id"] == workload["id"]
                        and int(row["seed_or_session"]) == session
                        and row["variant"] == variant
                    ]
                    if len(selected) != int(analysis["timed_calls_per_session"]):
                        group = (workload["id"], session, variant)
                        raise ValueError(f"invalid steady-state call count for {group}")
                    exemplar = selected[0]
                    output.append(
                        {
                            "experiment_id": analysis["experiment_id"],
                            "workload_id": workload["id"],
                            "workload_label": workload["label"],
                            "mode": workload["mode"],
                            "variant": variant,
                            "session": session,
                            "median_latency_ms": statistics.median(
                                float(row["latency_ms"]) for row in selected
                            ),
                            "observation_count": len(selected),
                            "measured_peak_live_gib": int(
                                exemplar["measured_peak_bytes_in_use"]
                            )
                            / 1024**3,
                            "measured_peak_pool_gib": int(
                                exemplar["measured_peak_pool_bytes"]
                            )
                            / 1024**3,
                            "process_reservation_mib": int(
                                exemplar["process_reservation_mib"]
                            ),
                            "compilation_ms": float(
                                next(
                                    row["latency_ms"]
                                    for row in rows
                                    if row["phase"] == "compilation"
                                    and row["workload_id"] == workload["id"]
                                    and int(row["seed_or_session"]) == session
                                    and row["variant"] == variant
                                )
                            ),
                            "first_compiled_call_ms": float(
                                next(
                                    row["latency_ms"]
                                    for row in rows
                                    if row["phase"] == "first_compiled_call"
                                    and row["workload_id"] == workload["id"]
                                    and int(row["seed_or_session"]) == session
                                    and row["variant"] == variant
                                )
                            ),
                            "source_id": source["id"],
                            "protocol_version": exemplar["protocol_version"],
                            "code_commit": exemplar["code_commit"],
                            "config_hash": exemplar["config_hash"],
                            "device_id": exemplar["device_id"],
                            "coordinate_boundary": exemplar["coordinate_boundary"],
                            "operation_boundary": exemplar["operation_boundary"],
                        }
                    )
    return output


def _latency_summary(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        grouped[(row["workload_id"], row["variant"])].append(row)
    output = []
    for (_, _), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: row["session"])
        exemplar = selected[0]
        latencies = [float(row["median_latency_ms"]) for row in selected]
        live = [float(row["measured_peak_live_gib"]) for row in selected]
        pool = [float(row["measured_peak_pool_gib"]) for row in selected]
        reservations = [int(row["process_reservation_mib"]) for row in selected]
        compilation = [float(row["compilation_ms"]) for row in selected]
        first = [float(row["first_compiled_call_ms"]) for row in selected]
        output.append(
            {
                "experiment_id": exemplar["experiment_id"],
                "workload_id": exemplar["workload_id"],
                "workload_label": exemplar["workload_label"],
                "mode": exemplar["mode"],
                "variant": exemplar["variant"],
                "sessions": len(selected),
                "latency_estimand": "median of session-level steady-state medians",
                "median_latency_ms": statistics.median(latencies),
                "q1_latency_ms": _quantile(latencies, 0.25),
                "q3_latency_ms": _quantile(latencies, 0.75),
                "min_session_median_ms": min(latencies),
                "max_session_median_ms": max(latencies),
                "session_medians_ms": json.dumps(latencies),
                "median_peak_live_gib": statistics.median(live),
                "min_peak_live_gib": min(live),
                "max_peak_live_gib": max(live),
                "median_peak_pool_gib": statistics.median(pool),
                "min_peak_pool_gib": min(pool),
                "max_peak_pool_gib": max(pool),
                "median_process_reservation_mib": statistics.median(reservations),
                "median_compilation_ms": statistics.median(compilation),
                "min_compilation_ms": min(compilation),
                "max_compilation_ms": max(compilation),
                "median_first_compiled_call_ms": statistics.median(first),
                "min_first_compiled_call_ms": min(first),
                "max_first_compiled_call_ms": max(first),
                "source_id": exemplar["source_id"],
                "protocol_version": exemplar["protocol_version"],
                "code_commit": exemplar["code_commit"],
                "config_hash": exemplar["config_hash"],
                "device_id": exemplar["device_id"],
                "coordinate_boundary": exemplar["coordinate_boundary"],
                "operation_boundary": exemplar["operation_boundary"],
            }
        )
    return output


def _speedup_summary(
    sessions: list[dict[str, Any]], analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    indexed = {
        (row["workload_id"], int(row["session"]), row["variant"]): row for row in sessions
    }
    output = []
    for workload_id in dict.fromkeys(row["workload_id"] for row in sessions):
        for pair in analysis["speedup_pairs"]:
            baseline = pair["baseline"]
            candidate = pair["candidate"]
            ratios = [
                float(indexed[(workload_id, session, baseline)]["median_latency_ms"])
                / float(indexed[(workload_id, session, candidate)]["median_latency_ms"])
                for session in range(int(analysis["sessions"]))
            ]
            logs = [math.log(value) for value in ratios]
            mean_log = statistics.mean(logs)
            standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
            half_width = float(student_t.ppf(0.975, df=len(logs) - 1) * standard_error)
            lower = math.exp(mean_log - half_width)
            upper = math.exp(mean_log + half_width)
            exemplar = indexed[(workload_id, 0, baseline)]
            output.append(
                {
                    "experiment_id": exemplar["experiment_id"],
                    "workload_id": workload_id,
                    "workload_label": exemplar["workload_label"],
                    "mode": exemplar["mode"],
                    "baseline_variant": baseline,
                    "candidate_variant": candidate,
                    "paired_sessions": len(ratios),
                    "estimand": (
                        "geometric mean of paired session-median "
                        "baseline/candidate latency ratios; values above one "
                        "favor candidate"
                    ),
                    "geometric_mean_speedup": math.exp(mean_log),
                    "log_t_95_interval_lower": lower,
                    "log_t_95_interval_upper": upper,
                    "classification": (
                        "candidate_faster"
                        if lower > 1.0
                        else "candidate_slower"
                        if upper < 1.0
                        else "inconclusive"
                    ),
                    "session_ids": json.dumps(list(range(len(ratios)))),
                    "session_speedups": json.dumps(ratios),
                    "source_id": exemplar["source_id"],
                    "protocol_version": exemplar["protocol_version"],
                    "code_commit": exemplar["code_commit"],
                    "config_hash": exemplar["config_hash"],
                    "device_id": exemplar["device_id"],
                }
            )
    return output


def _paper_table(
    latency: list[dict[str, Any]], speedups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    latency_index = {(row["workload_id"], row["variant"]): row for row in latency}
    speedup_index = {
        (row["workload_id"], row["baseline_variant"], row["candidate_variant"]): row
        for row in speedups
    }
    output = []
    for workload_id in dict.fromkeys(row["workload_id"] for row in latency):
        reference = latency_index[(workload_id, "R")]
        factorized = latency_index[(workload_id, "F")]
        tangent = latency_index[(workload_id, "T")]
        rf = speedup_index[(workload_id, "R", "F")]
        rt = speedup_index[(workload_id, "R", "T")]
        ft = speedup_index[(workload_id, "F", "T")]
        output.append(
            {
                "experiment_id": reference["experiment_id"],
                "workload_id": workload_id,
                "workload_label": reference["workload_label"],
                "mode": reference["mode"],
                "R_median_latency_ms": reference["median_latency_ms"],
                "F_median_latency_ms": factorized["median_latency_ms"],
                "T_median_latency_ms": tangent["median_latency_ms"],
                "R_over_F_speedup": rf["geometric_mean_speedup"],
                "R_over_F_ci95_lower": rf["log_t_95_interval_lower"],
                "R_over_F_ci95_upper": rf["log_t_95_interval_upper"],
                "R_over_T_speedup": rt["geometric_mean_speedup"],
                "R_over_T_ci95_lower": rt["log_t_95_interval_lower"],
                "R_over_T_ci95_upper": rt["log_t_95_interval_upper"],
                "F_over_T_ratio": ft["geometric_mean_speedup"],
                "F_over_T_ci95_lower": ft["log_t_95_interval_lower"],
                "F_over_T_ci95_upper": ft["log_t_95_interval_upper"],
                "R_peak_live_gib": reference["median_peak_live_gib"],
                "F_peak_live_gib": factorized["median_peak_live_gib"],
                "T_peak_live_gib": tangent["median_peak_live_gib"],
                "F_peak_live_reduction_vs_R_percent": 100.0
                * (
                    1.0
                    - factorized["median_peak_live_gib"] / reference["median_peak_live_gib"]
                ),
                "T_peak_live_reduction_vs_R_percent": 100.0
                * (
                    1.0
                    - tangent["median_peak_live_gib"] / reference["median_peak_live_gib"]
                ),
                "protocol_version": reference["protocol_version"],
                "code_commit": reference["code_commit"],
                "config_hash": reference["config_hash"],
                "device_id": reference["device_id"],
                "operation_boundary": reference["operation_boundary"],
            }
        )
    return output


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    if path.exists():
        raise FileExistsError(path)
    append_csv_rows(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    manifest_path = _resolve(arguments.manifest)
    analysis = load_yaml(manifest_path)
    rows_by_source = []
    audits = []
    for source in analysis["sources"]:
        rows, audit = _audit_source(source, analysis)
        rows_by_source.append((source, rows))
        audits.append(audit)
    if any(not audit["all_checks_pass"] for audit in audits):
        raise ValueError("one or more E08 evidence sources failed acceptance")

    sessions = _session_table(rows_by_source, analysis)
    latency = _latency_summary(sessions)
    speedups = _speedup_summary(sessions, analysis)
    paper = _paper_table(latency, speedups)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = analysis["outputs"]
    paths = {
        "session_medians": output_dir / outputs["session_medians"],
        "latency_summary": output_dir / outputs["latency_summary"],
        "speedup_summary": output_dir / outputs["speedup_summary"],
        "paper_table": output_dir / outputs["paper_table"],
        "audit_summary": output_dir / outputs["audit_summary"],
    }
    if any(path.exists() for path in paths.values()):
        existing = [str(path) for path in paths.values() if path.exists()]
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    _write_table(paths["session_medians"], sessions)
    _write_table(paths["latency_summary"], latency)
    _write_table(paths["speedup_summary"], speedups)
    _write_table(paths["paper_table"], paper)

    summary = {
        "schema_version": analysis["schema_version"],
        "analysis_id": analysis["analysis_id"],
        "experiment_id": analysis["experiment_id"],
        "description": analysis["description"],
        "derivation": (
            "CPU-only audit and aggregation of existing synchronized benchmark "
            "rows; no experiment was run by this exporter."
        ),
        "analysis_code": git_provenance(),
        "analysis_manifest": _evidence(manifest_path),
        "source_audits": audits,
        "statistical_method": analysis["statistical_method"],
        "memory_evidence_policy": analysis["memory_evidence_policy"],
        "table_evidence": {
            name: _evidence(path) for name, path in paths.items() if name != "audit_summary"
        },
        "session_rows": len(sessions),
        "latency_rows": len(latency),
        "speedup_rows": len(speedups),
        "paper_rows": len(paper),
        "all_checks_pass": True,
    }
    write_json(paths["audit_summary"], summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
