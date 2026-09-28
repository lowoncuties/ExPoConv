#!/usr/bin/env python3
"""Export auditable tables for the accepted bounded and structured E02 repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write an empty table: {path.name}")
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _accepted(row: dict[str, Any], protocol: dict[str, Any]) -> bool:
    tolerance = protocol["numerics"]["tolerances"][row["dtype"]]
    threshold = float(tolerance["absolute"]) + float(tolerance["relative"]) * float(
        row["reference_l2_norm"]
    )
    return bool(row["finite"] and float(row["max_absolute_error"]) <= threshold)


def _bounded_check_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["check_kind"]].append(row)
    output = []
    for check_kind, selected in sorted(grouped.items()):
        tolerance_rows = [row for row in selected if "within_tolerance" in row]
        output.append(
            {
                "check_kind": check_kind,
                "rows": len(selected),
                "finite_false_rows": sum(row.get("finite") is False for row in selected),
                "tolerance_rows": len(tolerance_rows),
                "within_tolerance_rows": sum(
                    bool(row["within_tolerance"]) for row in tolerance_rows
                ),
                "code_commit": selected[0]["code_commit"],
                "config_hash": selected[0]["config_hash"],
            }
        )
    return output


def _agreement_tables(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agreements = [row for row in rows if row["record_type"] == "agreement"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    failures = []
    for row in agreements:
        grouped[(row["case_id"], row["dtype"], row["variant"])].append(row)
        if not _accepted(row, protocol):
            tolerance = protocol["numerics"]["tolerances"][row["dtype"]]
            failures.append(
                {
                    "case_id": row["case_id"],
                    "dtype": row["dtype"],
                    "realization": row["realization"],
                    "variant": row["variant"],
                    "variable_group": row["variable_group"],
                    "finite": row["finite"],
                    "max_absolute_error": row["max_absolute_error"],
                    "relative_l2_error": row["relative_l2_error"],
                    "reference_l2_norm": row["reference_l2_norm"],
                    "absolute_tolerance": tolerance["absolute"],
                    "relative_tolerance": tolerance["relative"],
                }
            )
    overview = []
    for key, selected in sorted(grouped.items()):
        absolute = [float(row["max_absolute_error"]) for row in selected if row["finite"]]
        relative = [float(row["relative_l2_error"]) for row in selected if row["finite"]]
        overview.append(
            {
                "case_id": key[0],
                "dtype": key[1],
                "variant": key[2],
                "rows": len(selected),
                "finite_rows": sum(bool(row["finite"]) for row in selected),
                "accepted_rows": sum(_accepted(row, protocol) for row in selected),
                "acceptance_fraction": sum(
                    _accepted(row, protocol) for row in selected
                )
                / len(selected),
                "median_absolute_error": statistics.median(absolute),
                "p95_absolute_error": _quantile(absolute, 0.95),
                "max_absolute_error": max(absolute),
                "median_relative_l2_error": statistics.median(relative),
                "p95_relative_l2_error": _quantile(relative, 0.95),
                "max_relative_l2_error": max(relative),
            }
        )
    return overview, failures


def _finite_difference_tables(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finite_difference = [
        row for row in rows if row["record_type"] == "finite_difference"
    ]
    step_groups: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    sweep_groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in finite_difference:
        step_groups[
            (
                row["case_id"],
                row["dtype"],
                row["variant"],
                row["finite_difference_group"],
                float(row["relative_step"]),
            )
        ].append(row)
        sweep_groups[
            (
                row["case_id"],
                row["dtype"],
                int(row["realization"]),
                row["variant"],
                row["finite_difference_group"],
            )
        ].append(row)

    step_summary = []
    for key, selected in sorted(step_groups.items()):
        absolute = [float(row["absolute_error"]) for row in selected if row["finite"]]
        relative = [float(row["relative_error"]) for row in selected if row["finite"]]
        step_summary.append(
            {
                "case_id": key[0],
                "dtype": key[1],
                "variant": key[2],
                "finite_difference_group": key[3],
                "relative_step": key[4],
                "rows": len(selected),
                "finite_rows": sum(bool(row["finite"]) for row in selected),
                "median_absolute_error": statistics.median(absolute),
                "max_absolute_error": max(absolute),
                "median_relative_error": statistics.median(relative),
                "max_relative_error": max(relative),
            }
        )

    best_rows = []
    for key, selected in sorted(sweep_groups.items()):
        finite = [row for row in selected if row["finite"]]
        if not finite:
            continue
        best = min(finite, key=lambda row: float(row["relative_error"]))
        best_rows.append(
            {
                "case_id": key[0],
                "dtype": key[1],
                "realization": key[2],
                "variant": key[3],
                "finite_difference_group": key[4],
                "best_relative_step": best["relative_step"],
                "best_absolute_error": best["absolute_error"],
                "best_relative_error": best["relative_error"],
            }
        )
    return step_summary, best_rows


def _completeness(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        (case["id"], dtype, realization)
        for case in config["cases"]
        for dtype in config["precisions"]
        for realization in range(int(config["realizations"]))
    }
    actual = {
        (row["case_id"], row["dtype"], int(row["realization"])) for row in rows
    }
    counts: dict[tuple[str, str], set[int]] = defaultdict(set)
    per_realization: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in rows:
        key = (row["case_id"], row["dtype"], int(row["realization"]))
        per_realization[key] += 1
    for key, count in per_realization.items():
        counts[key[:2]].add(count)
    return {
        "expected_realization_groups": len(expected),
        "observed_realization_groups": len(actual),
        "missing_realization_groups": [list(key) for key in sorted(expected - actual)],
        "unexpected_realization_groups": [list(key) for key in sorted(actual - expected)],
        "uniform_rows_per_case_dtype": all(len(values) == 1 for values in counts.values()),
        "rows_per_realization_by_case_dtype": {
            f"{key[0]}::{key[1]}": sorted(values) for key, values in sorted(counts.items())
        },
    }


def export(
    protocol_path: Path,
    bounded_summary_path: Path,
    bounded_results_path: Path,
    structured_config_path: Path,
    structured_results_path: Path,
    environment_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = _yaml(protocol_path)
    bounded_summary = _json(bounded_summary_path)
    bounded_rows = _jsonl(bounded_results_path)
    structured_config = _yaml(structured_config_path)
    structured_rows = _jsonl(structured_results_path)
    environment = _json(environment_path)
    output_dir.mkdir(parents=True, exist_ok=False)

    bounded_table = _bounded_check_table(bounded_rows)
    agreement_overview, agreement_failures = _agreement_tables(
        structured_rows, protocol
    )
    step_summary, best_rows = _finite_difference_tables(structured_rows)
    _atomic_csv(output_dir / "bounded_check_summary.csv", bounded_table)
    _atomic_csv(output_dir / "structured_agreement_overview.csv", agreement_overview)
    _atomic_csv(output_dir / "structured_agreement_failures.csv", agreement_failures)
    _atomic_csv(output_dir / "finite_difference_step_summary.csv", step_summary)
    _atomic_csv(output_dir / "finite_difference_best_steps.csv", best_rows)

    agreements = [
        row for row in structured_rows if row["record_type"] == "agreement"
    ]
    finite_differences = [
        row for row in structured_rows if row["record_type"] == "finite_difference"
    ]
    summary = {
        "schema_version": "1.0",
        "bounded": bounded_summary,
        "structured": {
            "result_rows": len(structured_rows),
            "status_non_ok_rows": sum(
                row["status"] != "ok" for row in structured_rows
            ),
            "agreement_rows": len(agreements),
            "agreement_finite_rows": sum(bool(row["finite"]) for row in agreements),
            "agreement_accepted_rows": sum(
                _accepted(row, protocol) for row in agreements
            ),
            "agreement_acceptance_rule": (
                "finite and max_absolute_error <= absolute_tolerance + "
                "relative_tolerance * reference_l2_norm"
            ),
            "finite_difference_rows": len(finite_differences),
            "finite_difference_finite_rows": sum(
                bool(row["finite"]) for row in finite_differences
            ),
            "finite_difference_interpretation": (
                "descriptive registered step sweep; best steps are selected by "
                "minimum relative error and are not an acceptance gate"
            ),
            "completeness": _completeness(structured_rows, structured_config),
            "code_commits": sorted({row["code_commit"] for row in structured_rows}),
            "code_dirty_values": sorted(
                {bool(row["code_dirty"]) for row in structured_rows}
            ),
            "config_hashes": sorted({row["config_hash"] for row in structured_rows}),
            "device_ids": sorted({row["device_id"] for row in structured_rows}),
        },
        "execution_environment": {
            "hostname": environment["hostname"],
            "cuda_visible_devices": environment["environment"][
                "CUDA_VISIBLE_DEVICES"
            ],
            "jax_backend": environment["jax"]["backend"],
            "jax_devices": environment["jax"]["devices"],
            "git": environment["git"],
        },
        "source_files": {
            str(path.resolve()): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (
                protocol_path,
                bounded_summary_path,
                bounded_results_path,
                structured_config_path,
                structured_results_path,
                environment_path,
            )
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bounded-summary", type=Path, required=True)
    parser.add_argument("--bounded-results", type=Path, required=True)
    parser.add_argument("--structured-config", type=Path, required=True)
    parser.add_argument("--structured-results", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summary = export(
        arguments.protocol,
        arguments.bounded_summary,
        arguments.bounded_results,
        arguments.structured_config,
        arguments.structured_results,
        arguments.environment,
        arguments.output_dir,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
