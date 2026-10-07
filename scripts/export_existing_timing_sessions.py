#!/usr/bin/env python3
"""Validate and export per-session medians from an existing benchmark CSV.

This exporter performs no measurements.  It binds the raw rows to the clean
release commit and environment capture supplied by an E19 reproducer before
deriving session-level medians.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

GROUP_FIELDS = (
    "experiment_id",
    "workload_id",
    "config_hash",
    "code_commit",
    "code_dirty",
    "reference_commit",
    "device_id",
    "precision_policy",
    "variant",
    "seed_or_session",
    "phase",
    "coordinate_boundary",
    "operation_boundary",
)
STRUCTURAL_FIELDS = (
    "experiment_id",
    "workload_id",
    "variant",
    "seed_or_session",
    "phase",
)
AUDIT_FIELDS = (
    "gpu_snapshot_before",
    "gpu_snapshot_after",
    "latency_ms",
    "observation",
    "repeat_count",
    "status",
    "timestamp_utc",
)
ALLOWED_PHASE_STATUSES = {
    "compilation": {"ok"},
    "first_compiled_call": {"ok"},
    "sustained_sequence": {"not_applicable", "ok"},
    "steady_state": {"ok"},
}
LATENCY_PHASES = {"compilation", "first_compiled_call", "steady_state"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected a boolean CSV value, received {value!r}")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_environment(path: Path, expected_code_commit: str) -> tuple[dict[str, Any], int]:
    environment = json.loads(path.read_text(encoding="utf-8"))
    git = environment.get("git")
    if not isinstance(git, dict):
        raise ValueError("environment capture is missing the git identity")
    if git.get("commit") != expected_code_commit:
        raise ValueError("environment commit does not match the checked-out release commit")
    if git.get("dirty") is not False or git.get("status_porcelain") != []:
        raise ValueError("environment capture does not describe a clean checkout")
    variables = environment.get("environment")
    visible = variables.get("CUDA_VISIBLE_DEVICES") if isinstance(variables, dict) else None
    if not isinstance(visible, str) or not visible.isdigit():
        raise ValueError(
            "environment must record exactly one decimal CUDA_VISIBLE_DEVICES index"
        )
    return environment, int(visible)


def _audit_snapshot(
    serialized: str,
    *,
    field: str,
    physical_gpu_index: int,
) -> tuple[str, int]:
    try:
        snapshot = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON") from error
    if snapshot.get("status") != "ok":
        raise ValueError(f"{field} has non-ok GPU-query status")
    if snapshot.get("process_query_status") != "ok":
        raise ValueError(f"{field} has non-ok compute-process query status")
    if int(snapshot.get("index", -1)) != physical_gpu_index:
        raise ValueError(f"{field} does not match CUDA_VISIBLE_DEVICES")
    processes = snapshot.get("compute_processes")
    if processes != []:
        raise ValueError(f"{field} records a competing compute process")
    uuid = snapshot.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        raise ValueError(f"{field} is missing the physical GPU UUID")
    return uuid, len(processes)


def export(
    input_path: Path,
    output_path: Path,
    *,
    environment_path: Path,
    expected_code_commit: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_code_commit):
        raise ValueError("expected code commit must be a full 40-character Git SHA")
    expected_code_commit = expected_code_commit.lower()
    _, physical_gpu_index = _load_environment(environment_path, expected_code_commit)

    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    structural_identities: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    raw_rows = 0
    raw_commits: set[str] = set()
    dirty_values: set[bool] = set()
    snapshot_uuids: set[str] = set()
    snapshot_count = 0
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = (*GROUP_FIELDS, *AUDIT_FIELDS)
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("input CSV is missing fields: " + ", ".join(missing))
        for row in reader:
            raw_rows += 1
            phase = row["phase"]
            allowed_statuses = ALLOWED_PHASE_STATUSES.get(phase)
            if allowed_statuses is None:
                raise ValueError(f"raw row has an unexpected phase: {phase!r}")
            if row["status"] not in allowed_statuses:
                raise ValueError(
                    f"raw row has invalid status {row['status']!r} for phase {phase!r}"
                )
            raw_commits.add(row["code_commit"].lower())
            dirty_values.add(_truth(row["code_dirty"]))
            for field in ("gpu_snapshot_before", "gpu_snapshot_after"):
                uuid, _ = _audit_snapshot(
                    row[field],
                    field=field,
                    physical_gpu_index=physical_gpu_index,
                )
                snapshot_uuids.add(uuid)
                snapshot_count += 1

            key = tuple(row[field] for field in GROUP_FIELDS)
            structural = tuple(row[field] for field in STRUCTURAL_FIELDS)
            identity = tuple(
                row[field] for field in GROUP_FIELDS if field not in STRUCTURAL_FIELDS
            )
            structural_identities[structural].add(identity)
            group = groups.setdefault(
                key,
                {
                    "latencies": [],
                    "statuses": set(),
                    "observations": set(),
                    "repeat_counts": set(),
                    "timestamps": [],
                    "row_count": 0,
                },
            )
            group["row_count"] += 1
            if row.get("latency_ms"):
                group["latencies"].append(float(row["latency_ms"]))
            group["statuses"].add(row["status"])
            group["observations"].add(row["observation"])
            group["repeat_counts"].add(row["repeat_count"])
            group["timestamps"].append(row["timestamp_utc"])

    if not groups:
        raise ValueError("input CSV contains no rows")
    if raw_commits != {expected_code_commit}:
        raise ValueError(
            "raw-row code commits do not match the checked-out release commit: "
            f"{sorted(raw_commits)}"
        )
    if dirty_values != {False}:
        raise ValueError("raw rows do not all record a clean checkout")
    if len(snapshot_uuids) != 1:
        raise ValueError("raw rows do not bind to exactly one physical GPU UUID")

    output_rows = []
    incomplete = 0
    mixed_identity = 0
    invalid_latency_counts = 0
    invalid_row_counts = 0
    for key, values in sorted(groups.items()):
        row = dict(zip(GROUP_FIELDS, key, strict=True))
        structural = tuple(row[field] for field in STRUCTURAL_FIELDS)
        phase = row["phase"]
        repeat_counts = {int(value) for value in values["repeat_counts"] if value}
        if len(repeat_counts) > 1:
            raise ValueError(f"group has inconsistent repeat counts: {key}")
        expected = next(iter(repeat_counts), 1) if phase == "steady_state" else 1
        observation_count = len(values["observations"])
        latency_count = len(values["latencies"])
        expected_latency_count = expected if phase in LATENCY_PHASES else 0
        valid_latency_count = latency_count == expected_latency_count
        valid_row_count = values["row_count"] == expected
        complete = observation_count == expected and valid_latency_count and valid_row_count
        identities = len(structural_identities[structural])
        incomplete += int(not complete)
        mixed_identity += int(identities > 1)
        invalid_latency_counts += int(not valid_latency_count)
        invalid_row_counts += int(not valid_row_count)
        output_rows.append(
            {
                **row,
                "statuses": json.dumps(sorted(values["statuses"])),
                "observation_count": observation_count,
                "expected_observation_count": expected,
                "complete": complete,
                "identity_count_for_structural_group": identities,
                "mixed_identity": identities > 1,
                "median_latency_ms": (
                    statistics.median(values["latencies"]) if values["latencies"] else None
                ),
                "first_timestamp_utc": min(values["timestamps"]),
                "last_timestamp_utc": max(values["timestamps"]),
            }
        )

    invalid_conditions = []
    if incomplete:
        invalid_conditions.append(f"{incomplete} incomplete groups")
    if invalid_row_counts:
        invalid_conditions.append(f"{invalid_row_counts} groups with invalid row counts")
    if invalid_latency_counts:
        invalid_conditions.append(
            f"{invalid_latency_counts} groups with invalid latency counts"
        )
    if mixed_identity:
        invalid_conditions.append(f"{mixed_identity} groups with mixed structural identity")
    if invalid_conditions:
        raise ValueError("raw timing evidence is invalid: " + "; ".join(invalid_conditions))

    _atomic_csv(output_path, output_rows)
    metadata = {
        "schema_version": "1.1",
        "source_path": str(input_path.resolve()),
        "source_byte_size": input_path.stat().st_size,
        "source_sha256": _sha256(input_path),
        "source_rows": raw_rows,
        "environment_path": str(environment_path.resolve()),
        "environment_sha256": _sha256(environment_path),
        "expected_code_commit": expected_code_commit,
        "raw_code_commits": sorted(raw_commits),
        "raw_code_dirty_values": sorted(dirty_values),
        "physical_gpu_index": physical_gpu_index,
        "physical_gpu_uuids": sorted(snapshot_uuids),
        "audited_snapshots": snapshot_count,
        "snapshot_status_required": "status == ok and process_query_status == ok",
        "competing_processes_allowed": False,
        "output_path": str(output_path.resolve()),
        "output_rows": len(output_rows),
        "incomplete_groups": incomplete,
        "invalid_row_count_groups": invalid_row_counts,
        "invalid_latency_count_groups": invalid_latency_counts,
        "mixed_identity_groups": mixed_identity,
        "phase_status_policy": {
            phase: sorted(statuses)
            for phase, statuses in sorted(ALLOWED_PHASE_STATUSES.items())
        },
        "derivation": (
            "Median latency by workload/config/code/variant/session/phase; no "
            "new timing measurements and no pooling across code identities."
        ),
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".metadata.json"), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite: {arguments.output}")
    metadata_path = arguments.output.with_suffix(arguments.output.suffix + ".metadata.json")
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite: {metadata_path}")
    print(
        json.dumps(
            export(
                arguments.input,
                arguments.output,
                environment_path=arguments.environment,
                expected_code_commit=arguments.expected_code_commit,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
