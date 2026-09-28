#!/usr/bin/env python3
"""Validate corrected anchor evidence and export publication-facing tables."""

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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

from poincare_exec.artifacts import write_json
from poincare_exec.config.loader import load_yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write an empty table: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _truth(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected a boolean CSV value, received {value!r}")


def _audit_raw(
    path: Path, config: dict[str, Any], visible_gpu: int
) -> dict[str, Any]:
    workloads = {item["id"] for item in config["workloads"]}
    variants = {str(item) for item in config["variants"]}
    sessions = set(range(int(config["sessions"])))
    expected_keys = {
        (workload, session, variant)
        for workload in workloads
        for session in sessions
        for variant in variants
    }
    expected_rows_per_group = int(config["timed_calls_per_session"]) + 3
    group_counts: Counter[tuple[str, int, str]] = Counter()
    phase_counts: Counter[tuple[tuple[str, int, str], str]] = Counter()
    statuses: Counter[tuple[str, str]] = Counter()
    commits: set[str] = set()
    dirty_values: set[bool] = set()
    devices: set[str] = set()
    configuration_hashes: dict[str, set[str]] = defaultdict(set)
    snapshot_indices: set[int] = set()
    snapshot_uuids: set[str] = set()
    timestamps: list[str] = []
    nonfinite_rows = 0
    contaminated_snapshots = 0
    raw_rows = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            raw_rows += 1
            key = (
                row["workload_id"],
                int(row["seed_or_session"]),
                row["variant"],
            )
            group_counts[key] += 1
            phase_counts[(key, row["phase"])] += 1
            statuses[(row["phase"], row["status"])] += 1
            commits.add(row["code_commit"])
            dirty_values.add(_truth(row["code_dirty"]))
            devices.add(row["device_id"])
            configuration_hashes[row["workload_id"]].add(row["config_hash"])
            timestamps.append(row["timestamp_utc"])
            state = json.loads(row["state_summary_after"])
            if int(row["output_nonfinite_count"]) or int(state["nonfinite_count"]):
                nonfinite_rows += 1
            for field in ("gpu_snapshot_before", "gpu_snapshot_after"):
                snapshot = json.loads(row[field])
                if snapshot.get("status") != "ok":
                    raise ValueError(f"non-ok {field} in {key}: {snapshot}")
                snapshot_indices.add(int(snapshot["index"]))
                snapshot_uuids.add(str(snapshot["uuid"]))
                contaminated_snapshots += int(bool(snapshot["compute_processes"]))

    expected_phase_counts = {
        "compilation": 1,
        "first_compiled_call": 1,
        "sustained_sequence": 1,
        "steady_state": int(config["timed_calls_per_session"]),
    }
    bad_groups = {
        key: count
        for key, count in group_counts.items()
        if count != expected_rows_per_group
    }
    bad_phases = {
        (key, phase): phase_counts[(key, phase)]
        for key in expected_keys
        for phase, count in expected_phase_counts.items()
        if phase_counts[(key, phase)] != count
    }
    missing_groups = sorted(expected_keys - set(group_counts))
    unexpected_groups = sorted(set(group_counts) - expected_keys)
    checks = {
        "expected_group_set": not missing_groups and not unexpected_groups,
        "group_row_counts": not bad_groups,
        "phase_row_counts": not bad_phases,
        "one_clean_commit": len(commits) == 1 and dirty_values == {False},
        "one_device_identity": len(devices) == 1,
        "one_config_hash_per_workload": all(
            len(configuration_hashes[workload]) == 1 for workload in workloads
        ),
        "all_outputs_and_states_finite": nonfinite_rows == 0,
        "physical_gpu_matches_mask": snapshot_indices == {visible_gpu},
        "one_physical_gpu_uuid": len(snapshot_uuids) == 1,
        "no_competing_snapshot_process": contaminated_snapshots == 0,
        "required_statuses": all(
            status == "ok"
            for (phase, status), count in statuses.items()
            if phase != "sustained_sequence" and count
        )
        and all(
            status in {"ok", "not_applicable"}
            for (phase, status), count in statuses.items()
            if phase == "sustained_sequence" and count
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("raw anchor evidence failed checks: " + ", ".join(failed))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_size": path.stat().st_size,
        "rows": raw_rows,
        "groups": len(group_counts),
        "expected_rows_per_group": expected_rows_per_group,
        "missing_groups": missing_groups,
        "unexpected_groups": unexpected_groups,
        "bad_group_counts": {str(key): value for key, value in bad_groups.items()},
        "bad_phase_counts": {str(key): value for key, value in bad_phases.items()},
        "statuses": {
            f"{phase}:{status}": count
            for (phase, status), count in sorted(statuses.items())
        },
        "code_commits": sorted(commits),
        "code_dirty_values": sorted(dirty_values),
        "device_identities": sorted(devices),
        "physical_gpu_indices": sorted(snapshot_indices),
        "physical_gpu_uuids": sorted(snapshot_uuids),
        "nonfinite_rows": nonfinite_rows,
        "contaminated_snapshots": contaminated_snapshots,
        "first_timestamp_utc": min(timestamps),
        "last_timestamp_utc": max(timestamps),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _session_table(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    workloads = [item["id"] for item in config["workloads"]]
    variants = [str(item) for item in config["variants"]]
    sessions = range(int(config["sessions"]))
    selected = {
        (row["workload_id"], int(row["seed_or_session"]), row["variant"]): row
        for row in rows
        if row["phase"] == "steady_state"
    }
    expected = {
        (workload, session, variant)
        for workload in workloads
        for session in sessions
        for variant in variants
    }
    if set(selected) != expected:
        raise ValueError("steady-state session-median key set is incomplete or unexpected")
    output = []
    for key in sorted(selected):
        row = selected[key]
        if not _truth(row["complete"]) or _truth(row["mixed_identity"]):
            raise ValueError(f"invalid median group: {key}")
        if json.loads(row["statuses"]) != ["ok"]:
            raise ValueError(f"non-ok steady-state median group: {key}")
        output.append(
            {
                "experiment_id": row["experiment_id"],
                "workload_id": row["workload_id"],
                "variant": row["variant"],
                "session": int(row["seed_or_session"]),
                "median_latency_ms": float(row["median_latency_ms"]),
                "observation_count": int(row["observation_count"]),
                "code_commit": row["code_commit"],
                "config_hash": row["config_hash"],
                "device_id": row["device_id"],
                "coordinate_boundary": row["coordinate_boundary"],
                "operation_boundary": row["operation_boundary"],
            }
        )
    return output


def _latency_summary(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        grouped[(row["workload_id"], row["variant"])].append(row)
    output = []
    for (workload, variant), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: row["session"])
        values = [float(row["median_latency_ms"]) for row in selected]
        output.append(
            {
                "experiment_id": selected[0]["experiment_id"],
                "workload_id": workload,
                "variant": variant,
                "sessions": len(selected),
                "estimand": "median of five session-level steady-state medians",
                "median_latency_ms": statistics.median(values),
                "min_session_median_ms": min(values),
                "max_session_median_ms": max(values),
                "session_ids": json.dumps([row["session"] for row in selected]),
                "session_medians_ms": json.dumps(values),
                "code_commit": selected[0]["code_commit"],
                "config_hash": selected[0]["config_hash"],
                "device_id": selected[0]["device_id"],
                "coordinate_boundary": selected[0]["coordinate_boundary"],
                "operation_boundary": selected[0]["operation_boundary"],
            }
        )
    return output


def _speedup_summary(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["workload_id"], row["session"], row["variant"]): row
        for row in sessions
    }
    workloads = sorted({row["workload_id"] for row in sessions})
    session_ids = sorted({int(row["session"]) for row in sessions})
    pairs = (("R", "F"), ("R", "T"), ("F", "T"))
    output = []
    for workload in workloads:
        for baseline, candidate in pairs:
            ratios = [
                float(indexed[(workload, session, baseline)]["median_latency_ms"])
                / float(indexed[(workload, session, candidate)]["median_latency_ms"])
                for session in session_ids
            ]
            log_ratios = [math.log(value) for value in ratios]
            mean_log = statistics.mean(log_ratios)
            standard_error = statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
            half_width = float(
                student_t.ppf(0.975, df=len(log_ratios) - 1) * standard_error
            )
            estimate = math.exp(mean_log)
            lower = math.exp(mean_log - half_width)
            upper = math.exp(mean_log + half_width)
            exemplar = indexed[(workload, session_ids[0], baseline)]
            output.append(
                {
                    "experiment_id": exemplar["experiment_id"],
                    "workload_id": workload,
                    "baseline_variant": baseline,
                    "candidate_variant": candidate,
                    "paired_sessions": len(ratios),
                    "estimand": (
                        "geometric mean of paired session-median baseline/candidate "
                        "latency ratios; values above one favor candidate"
                    ),
                    "geometric_mean_speedup": estimate,
                    "log_t_95_interval_lower": lower,
                    "log_t_95_interval_upper": upper,
                    "classification": (
                        "candidate_faster"
                        if lower > 1.0
                        else "candidate_slower"
                        if upper < 1.0
                        else "inconclusive"
                    ),
                    "session_ids": json.dumps(session_ids),
                    "session_speedups": json.dumps(ratios),
                    "code_commit": exemplar["code_commit"],
                    "config_hash": exemplar["config_hash"],
                    "device_id": exemplar["device_id"],
                }
            )
    return output


def _file_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    evidence = {"path": str(path.resolve()), "exists": path.exists()}
    if path.exists():
        evidence.update(
            {
                "sha256": _sha256(path),
                "byte_size": path.stat().st_size,
            }
        )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--session-medians", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    config = load_yaml(arguments.config)
    environment = json.loads(arguments.environment.read_text(encoding="utf-8"))
    visible = str(environment["environment"]["CUDA_VISIBLE_DEVICES"])
    if not visible.isdigit():
        raise ValueError("environment must record exactly one numeric visible GPU")
    raw_audit = _audit_raw(arguments.raw, config, int(visible))
    sessions = _session_table(_read_csv(arguments.session_medians), config)
    latency = _latency_summary(sessions)
    speedups = _speedup_summary(sessions)

    output_dir = arguments.output_dir.resolve()
    session_path = output_dir / "steady_state_session_medians.csv"
    latency_path = output_dir / "workload_latency_summary.csv"
    speedup_path = output_dir / "paired_speedup_summary.csv"
    summary_path = output_dir / "summary.json"
    for path in (session_path, latency_path, speedup_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    _write_csv(session_path, sessions)
    _write_csv(latency_path, latency)
    _write_csv(speedup_path, speedups)

    status = "unknown"
    if arguments.status_file is not None and arguments.status_file.exists():
        status = arguments.status_file.read_text(encoding="utf-8").strip()
    summary = {
        "schema_version": "1.0",
        "derivation": (
            "CPU-only validation and aggregation of existing synchronized timing "
            "rows; no benchmark was run by this exporter."
        ),
        "raw_evidence": raw_audit,
        "session_median_evidence": _file_evidence(arguments.session_medians),
        "config_evidence": _file_evidence(arguments.config),
        "environment_evidence": _file_evidence(arguments.environment),
        "log_evidence": _file_evidence(arguments.log),
        "wrapper_exit_status": status,
        "wrapper_status_evidence": _file_evidence(arguments.status_file),
        "wrapper_status_note": (
            None
            if status != "unknown"
            else (
                "Unknown: the parent terminal ended after the complete CSV was written; "
                "completion is established by the validated complete matrix and absence of "
                "a failure sidecar."
            )
        ),
        "failure_sidecar": _file_evidence(
            arguments.raw.with_name(f"{arguments.raw.name}.failures.jsonl")
        ),
        "memory_evidence_policy": config["memory_evidence_policy"],
        "session_table": _file_evidence(session_path),
        "latency_table": _file_evidence(latency_path),
        "speedup_table": _file_evidence(speedup_path),
        "session_rows": len(sessions),
        "latency_rows": len(latency),
        "speedup_rows": len(speedups),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
