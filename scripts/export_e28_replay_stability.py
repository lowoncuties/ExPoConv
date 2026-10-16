#!/usr/bin/env python3
"""Validate three E28 process replays and export replay-stability summaries."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.export_e26_elementwise_tables import (
    EXPECTED_CASE_SPECS,
    EXPECTED_DEVICE,
    EXPECTED_OUTPUT_SHAPES,
    EXPECTED_REFERENCE_COMMIT,
    EXPECTED_VARIABLE_GROUPS,
    QUANTILE_FIELDS,
    _atomic_csv,
    _atomic_json,
    _sha256,
)

EXPECTED_COMMIT = "df3f6f556a314f74a6966701b38ac5d135a10a62"
EXPECTED_PROTOCOL = "1.17"
EXPECTED_CONFIG_HASH = "83ae5c2243f631bef869e368a43920c60b9581485643284c2425a83b5c3ed503"
REGISTERED_WRAPPER_SHA256 = (
    "15d8907f712ee567920ea147b8a6c318b262680960957947b26def9eb01d76f3"
)
REPLAYS = tuple(range(3))
REALIZATIONS = tuple(range(3))
VARIANTS = ("F", "T")
DTYPE = "float32"
ABSOLUTE_TOLERANCE = 2.0e-5
RELATIVE_TOLERANCE = 2.0e-5
EXPECTED_ROWS_PER_REPLAY = (
    sum(len(groups) for groups in EXPECTED_VARIABLE_GROUPS.values())
    * len(REALIZATIONS)
    * len(VARIANTS)
)
REPLAY_PROVENANCE = (
    "Replay index and separate-process status are assigned from explicit CLI file "
    "roles and the registered run_e28.sh contract; neither is embedded in E28 rows."
)
METRICS = (
    "reference_l2_norm",
    "max_absolute_error",
    "relative_l2_error",
    "denominator_floored_fraction",
    "p50",
    "p90",
    "p95",
    "p99",
    "p99_9",
    "maximum",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-0", required=True, type=Path)
    parser.add_argument("--replay-1", required=True, type=Path)
    parser.add_argument("--replay-2", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/reviewer_requested")
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty E28 input: {path}")
    return rows


def _one(values: Sequence[Any] | Any, *, field: str) -> Any:
    materialized = list(values)
    unique = {json.dumps(value, sort_keys=True) for value in materialized}
    if len(unique) != 1:
        raise ValueError(f"E28 has inconsistent {field}")
    return materialized[0]


def _finite_nonnegative(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected numeric value") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{context}: expected finite nonnegative value")
    return number


def _null(value: Any) -> bool:
    return value is None


def _validate_replay(
    rows: Sequence[dict[str, Any]],
    *,
    replay: int,
    source_path: Path,
    source_hash: str,
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS_PER_REPLAY:
        raise ValueError(
            f"E28 replay {replay} row count is {len(rows)}, "
            f"expected {EXPECTED_ROWS_PER_REPLAY}"
        )
    identity = {
        "experiment_id": "E28",
        "protocol_version": EXPECTED_PROTOCOL,
        "code_commit": EXPECTED_COMMIT,
        "code_dirty": False,
        "status": "ok",
        "record_type": "agreement",
        "comparison_reference": "R_same_precision",
        "elementwise_relative_error_definition": (
            "abs(candidate-reference)/max(abs(reference),denominator_floor)"
        ),
        "elementwise_relative_error_quantile_method": "linear",
        "elementwise_relative_error_denominator_floor": 1.0e-12,
        "config_hash": EXPECTED_CONFIG_HASH,
        "device_id": EXPECTED_DEVICE,
        "reference_commit": EXPECTED_REFERENCE_COMMIT,
        "state_semantics": "current_batch_no_running_state",
    }
    for field, expected in identity.items():
        observed = _one([row.get(field) for row in rows], field=field)
        matches = (
            observed is expected if isinstance(expected, bool) else observed == expected
        )
        if not matches:
            raise ValueError(
                f"E28 replay {replay} {field} is {observed!r}, expected {expected!r}"
            )

    keys: set[tuple[str, str, int, str, str]] = set()
    cells: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    prepared = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id"))
        dtype = str(row.get("dtype"))
        variant = str(row.get("variant"))
        try:
            realization = int(row.get("realization"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"replay {replay} row {index}: invalid realization") from error
        variable_group = str(row.get("variable_group"))
        if case_id not in EXPECTED_VARIABLE_GROUPS:
            raise ValueError(f"replay {replay} row {index}: unexpected case")
        if dtype != DTYPE or variant not in VARIANTS:
            raise ValueError(f"replay {replay} row {index}: unexpected dtype/variant")
        if realization not in REALIZATIONS:
            raise ValueError(f"replay {replay} row {index}: unexpected realization")
        key = (case_id, dtype, realization, variant, variable_group)
        if key in keys:
            raise ValueError(f"replay {replay} row {index}: duplicate E28 key {key}")
        keys.add(key)
        cells[(case_id, realization, variant)].add(variable_group)
        if row.get("case_spec") != EXPECTED_CASE_SPECS[case_id]:
            raise ValueError(f"replay {replay} row {index}: case_spec mismatch")
        if row.get("output_shape") != EXPECTED_OUTPUT_SHAPES[case_id]:
            raise ValueError(f"replay {replay} row {index}: output_shape mismatch")
        if row.get("seed_or_session") != f"{case_id}:{realization}":
            raise ValueError(f"replay {replay} row {index}: seed mismatch")
        expected_precision = json.dumps(
            {
                "matmul_precision": "highest",
                "reduced_precision_reduction": False,
                "storage_dtype": DTYPE,
                "tf32": False,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        if row.get("precision_policy") != expected_precision:
            raise ValueError(f"replay {replay} row {index}: precision mismatch")
        try:
            elements = int(row.get("elementwise_relative_error_elements"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"replay {replay} row {index}: invalid element count"
            ) from error
        if elements <= 0:
            raise ValueError(f"replay {replay} row {index}: invalid element count")
        finite = row.get("finite")
        if not isinstance(finite, bool):
            raise ValueError(f"replay {replay} row {index}: invalid finite flag")
        if finite:
            try:
                floored = int(
                    row.get("elementwise_relative_error_denominator_floored_elements")
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"replay {replay} row {index}: invalid floor count"
                ) from error
            if not 0 <= floored <= elements:
                raise ValueError(f"replay {replay} row {index}: invalid floor count")
            floor_fraction = _finite_nonnegative(
                row.get("elementwise_relative_error_denominator_floored_fraction"),
                context=f"replay {replay} row {index} floor fraction",
            )
            if not math.isclose(floor_fraction, floored / elements, abs_tol=1.0e-15):
                raise ValueError(f"replay {replay} row {index}: floor fraction mismatch")
            quantiles = [
                _finite_nonnegative(
                    row.get(field), context=f"replay {replay} row {index} {field}"
                )
                for field in QUANTILE_FIELDS
            ]
            maximum = _finite_nonnegative(
                row.get("elementwise_relative_error_max"),
                context=f"replay {replay} row {index} elementwise max",
            )
            if quantiles != sorted(quantiles) or quantiles[-1] > maximum:
                raise ValueError(
                    f"replay {replay} row {index}: nonmonotone elementwise quantiles"
                )
            relative_l2 = _finite_nonnegative(
                row.get("relative_l2_error"),
                context=f"replay {replay} row {index} relative L2",
            )
            absolute = _finite_nonnegative(
                row.get("max_absolute_error"),
                context=f"replay {replay} row {index} absolute error",
            )
            reference_l2 = _finite_nonnegative(
                row.get("reference_l2_norm"),
                context=f"replay {replay} row {index} reference L2",
            )
            threshold = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference_l2
            passed = absolute <= threshold
        else:
            null_fields = (
                "elementwise_relative_error_denominator_floored_elements",
                "elementwise_relative_error_denominator_floored_fraction",
                *QUANTILE_FIELDS,
                "elementwise_relative_error_max",
                "relative_l2_error",
                "max_absolute_error",
                "reference_l2_norm",
            )
            if any(not _null(row.get(field)) for field in null_fields):
                raise ValueError(
                    f"replay {replay} row {index}: nonfinite metrics must be null"
                )
            floored = None
            floor_fraction = None
            quantiles = [None] * len(QUANTILE_FIELDS)
            maximum = None
            relative_l2 = None
            absolute = None
            reference_l2 = None
            threshold = None
            passed = False
        prepared.append(
            {
                "experiment_id": "E28",
                "protocol_version": EXPECTED_PROTOCOL,
                "code_commit": EXPECTED_COMMIT,
                "config_hash": EXPECTED_CONFIG_HASH,
                "device_id": EXPECTED_DEVICE,
                "replay": replay,
                "case_id": case_id,
                "dtype": dtype,
                "realization": realization,
                "variant": variant,
                "scope": "output" if variable_group == "output" else "gradient",
                "variable_group": variable_group,
                "finite": finite,
                "elements": elements,
                "denominator_floored_elements": floored,
                "denominator_floored_fraction": floor_fraction,
                "p50": quantiles[0],
                "p90": quantiles[1],
                "p95": quantiles[2],
                "p99": quantiles[3],
                "p99_9": quantiles[4],
                "maximum": maximum,
                "relative_l2_error": relative_l2,
                "max_absolute_error": absolute,
                "reference_l2_norm": reference_l2,
                "mixed_absolute_tolerance": ABSOLUTE_TOLERANCE,
                "mixed_relative_tolerance": RELATIVE_TOLERANCE,
                "mixed_criterion_threshold": threshold,
                "mixed_criterion_pass": passed,
                "source_file": source_path.name,
                "source_sha256": source_hash,
                "replay_provenance": "explicit_cli_file_role",
            }
        )

    expected_cells = {
        (case_id, realization, variant)
        for case_id in EXPECTED_VARIABLE_GROUPS
        for realization in REALIZATIONS
        for variant in VARIANTS
    }
    if set(cells) != expected_cells:
        raise ValueError(f"E28 replay {replay} matrix is incomplete")
    for cell, groups in cells.items():
        expected_groups = EXPECTED_VARIABLE_GROUPS[cell[0]]
        if groups != expected_groups:
            missing = sorted(expected_groups - groups)
            unexpected = sorted(groups - expected_groups)
            raise ValueError(
                f"E28 replay {replay} variable-group inventory mismatch for {cell}; "
                f"missing={missing}, unexpected={unexpected}"
            )
    indexed = {
        (
            str(row["case_id"]),
            int(row["realization"]),
            str(row["variant"]),
            str(row["variable_group"]),
        ): row
        for row in prepared
    }
    for case_id in EXPECTED_VARIABLE_GROUPS:
        for realization in REALIZATIONS:
            for group in EXPECTED_VARIABLE_GROUPS[case_id]:
                factorized = indexed[(case_id, realization, "F", group)]
                tangent = indexed[(case_id, realization, "T", group)]
                if factorized["elements"] != tangent["elements"]:
                    raise ValueError(
                        f"E28 replay {replay} F/T element counts differ for "
                        f"{(case_id, realization, group)}"
                    )
                if factorized["finite"] and tangent["finite"]:
                    for field in (
                        "reference_l2_norm",
                        "denominator_floored_elements",
                        "denominator_floored_fraction",
                    ):
                        if factorized[field] != tangent[field]:
                            raise ValueError(
                                f"E28 replay {replay} F/T {field} differs for "
                                f"{(case_id, realization, group)}"
                            )
    return sorted(
        prepared,
        key=lambda row: (
            row["case_id"],
            row["realization"],
            row["variant"],
            row["variable_group"],
        ),
    )


def _range(values: Sequence[Any]) -> tuple[Any, Any, Any, bool]:
    exact = all(value == values[0] for value in values[1:])
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None, None, None, exact
    minimum = min(numeric)
    maximum = max(numeric)
    return minimum, maximum, maximum - minimum, exact


def _stability_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["dtype"]),
            int(row["realization"]),
            str(row["variant"]),
            str(row["variable_group"]),
        )
        grouped[key].append(row)
    if len(grouped) != EXPECTED_ROWS_PER_REPLAY:
        raise ValueError("E28 cross-replay key matrix is incomplete")
    output = []
    for key in sorted(grouped):
        selected = sorted(grouped[key], key=lambda row: int(row["replay"]))
        if [int(row["replay"]) for row in selected] != list(REPLAYS):
            raise ValueError(f"E28 replay roles are incomplete for {key}")
        if len({int(row["elements"]) for row in selected}) != 1:
            raise ValueError(f"E28 element count changes across replays for {key}")
        pass_pattern = "".join(
            "N"
            if not bool(row["finite"])
            else "P"
            if bool(row["mixed_criterion_pass"])
            else "F"
            for row in selected
        )
        passed = sum(bool(row["mixed_criterion_pass"]) for row in selected)
        nonfinite = sum(not bool(row["finite"]) for row in selected)
        result: dict[str, Any] = {
            "experiment_id": "E28",
            "protocol_version": EXPECTED_PROTOCOL,
            "code_commit": EXPECTED_COMMIT,
            "config_hash": EXPECTED_CONFIG_HASH,
            "device_id": EXPECTED_DEVICE,
            "case_id": key[0],
            "dtype": key[1],
            "realization": key[2],
            "variant": key[3],
            "scope": selected[0]["scope"],
            "variable_group": key[4],
            "elements": selected[0]["elements"],
        }
        for replay, row in zip(REPLAYS, selected, strict=True):
            result.update(
                {
                    f"replay_{replay}_source_sha256": row["source_sha256"],
                    f"replay_{replay}_finite": row["finite"],
                    f"replay_{replay}_mixed_criterion_pass": row["mixed_criterion_pass"],
                }
            )
            for metric in METRICS:
                result[f"replay_{replay}_{metric}"] = row[metric]
        result.update(
            {
                "pass_pattern": pass_pattern,
                "pass_count": passed,
                "nonfinite_replay_count": nonfinite,
                "stability_classification": (
                    "stable_pass"
                    if passed == len(REPLAYS)
                    else "stable_fail"
                    if passed == 0
                    else "mixed"
                ),
            }
        )
        exact_all = True
        for metric in METRICS:
            minimum, maximum, span, exact = _range([row[metric] for row in selected])
            result[f"{metric}_minimum"] = minimum
            result[f"{metric}_maximum"] = maximum
            result[f"{metric}_range"] = span
            result[f"{metric}_exact_across_replays"] = exact
            exact_all = exact_all and exact
        result["all_metrics_exact_across_replays"] = exact_all
        output.append(result)
    return output


def _stability_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["case_id"]),
                int(row["realization"]),
                str(row["variant"]),
                str(row["scope"]),
            )
        ].append(row)
    output = []
    for key in sorted(grouped):
        selected = grouped[key]
        patterns = Counter(str(row["pass_pattern"]) for row in selected)
        result: dict[str, Any] = {
            "experiment_id": "E28",
            "protocol_version": EXPECTED_PROTOCOL,
            "case_id": key[0],
            "realization": key[1],
            "variant": key[2],
            "scope": key[3],
            "tensor_count": len(selected),
        }
        for replay in REPLAYS:
            result[f"replay_{replay}_passed_tensors"] = sum(
                bool(row[f"replay_{replay}_mixed_criterion_pass"]) for row in selected
            )
            result[f"replay_{replay}_failed_tensors"] = len(selected) - int(
                result[f"replay_{replay}_passed_tensors"]
            )
            result[f"replay_{replay}_nonfinite_tensors"] = sum(
                not bool(row[f"replay_{replay}_finite"]) for row in selected
            )
        result.update(
            {
                "stable_pass_tensors": sum(
                    row["stability_classification"] == "stable_pass" for row in selected
                ),
                "stable_fail_tensors": sum(
                    row["stability_classification"] == "stable_fail" for row in selected
                ),
                "mixed_tensors": sum(
                    row["stability_classification"] == "mixed" for row in selected
                ),
                "pass_pattern_counts": json.dumps(
                    dict(sorted(patterns.items())), separators=(",", ":")
                ),
                "all_metrics_exact_tensors": sum(
                    bool(row["all_metrics_exact_across_replays"]) for row in selected
                ),
            }
        )
        for metric in METRICS:
            result[f"{metric}_exact_tensors"] = sum(
                bool(row[f"{metric}_exact_across_replays"]) for row in selected
            )
            spans = [
                float(row[f"{metric}_range"])
                for row in selected
                if row[f"{metric}_range"] is not None
            ]
            result[f"largest_{metric}_range"] = max(spans) if spans else None
        output.append(result)
    if len(output) != 24:
        raise ValueError(f"E28 scope summary has {len(output)} rows, expected 24")
    return output


def export(replay_paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if len(replay_paths) != len(REPLAYS):
        raise ValueError("E28 requires exactly three replay inputs")
    paths = [path.resolve() for path in replay_paths]
    if len(set(paths)) != len(REPLAYS):
        raise ValueError("E28 replay inputs must be distinct files")
    raw_by_replay = [_read_jsonl(path) for path in paths]
    hashes = [_sha256(path) for path in paths]
    tensors = []
    for replay in REPLAYS:
        tensors.extend(
            _validate_replay(
                raw_by_replay[replay],
                replay=replay,
                source_path=paths[replay],
                source_hash=hashes[replay],
            )
        )
    tensors.sort(
        key=lambda row: (
            int(row["replay"]),
            row["case_id"],
            int(row["realization"]),
            row["variant"],
            row["variable_group"],
        )
    )
    stability = _stability_rows(tensors)
    summaries = _stability_summary(stability)
    if len(tensors) != 1224 or len(stability) != 408:
        raise ValueError("E28 exported table matrix is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "e28_replay_tensor_statistics.csv", tensors)
    _atomic_csv(output_dir / "e28_replay_tensor_stability.csv", stability)
    _atomic_csv(output_dir / "e28_replay_stability_summary.csv", summaries)
    summary = {
        "schema_version": "1.0",
        "analysis_status": "post_hoc_descriptive_replay_stability_diagnostic",
        "scientific_behavior_change": False,
        "identity": {
            "experiment_id": "E28",
            "protocol_version": EXPECTED_PROTOCOL,
            "code_commit": EXPECTED_COMMIT,
            "code_dirty": False,
            "config_hash": EXPECTED_CONFIG_HASH,
            "device_id": EXPECTED_DEVICE,
        },
        "sources": [
            {
                "replay": replay,
                "file": paths[replay].name,
                "bytes": paths[replay].stat().st_size,
                "sha256": hashes[replay],
                "rows": len(raw_by_replay[replay]),
            }
            for replay in REPLAYS
        ],
        "replay_provenance": {
            "status": "wrapper_and_source_file_role_not_raw_embedded",
            "note": REPLAY_PROVENANCE,
            "registered_wrapper": "scripts/run_e28.sh",
            "registered_wrapper_sha256": REGISTERED_WRAPPER_SHA256,
        },
        "mixed_criterion": {
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "comparison": "max_absolute_error <= absolute + relative * reference_l2_norm",
        },
        "overall": {
            **{
                f"replay_{replay}_passed_tensors": sum(
                    bool(row["mixed_criterion_pass"])
                    for row in tensors
                    if int(row["replay"]) == replay
                )
                for replay in REPLAYS
            },
            **{
                f"replay_{replay}_nonfinite_tensors": sum(
                    not bool(row["finite"])
                    for row in tensors
                    if int(row["replay"]) == replay
                )
                for replay in REPLAYS
            },
            "stable_pass_tensors": sum(
                row["stability_classification"] == "stable_pass" for row in stability
            ),
            "stable_fail_tensors": sum(
                row["stability_classification"] == "stable_fail" for row in stability
            ),
            "mixed_tensors": sum(
                row["stability_classification"] == "mixed" for row in stability
            ),
        },
        "interpretation": (
            "Post-hoc descriptive process replay only. E28 does not replace or modify "
            "the accepted E02/E26 numerical-agreement gate."
        ),
        "output_rows": {
            "e28_replay_tensor_statistics.csv": len(tensors),
            "e28_replay_tensor_stability.csv": len(stability),
            "e28_replay_stability_summary.csv": len(summaries),
        },
    }
    _atomic_json(output_dir / "e28_replay_summary.json", summary)
    return summary


def main() -> int:
    arguments = _parser().parse_args()
    summary = export(
        [arguments.replay_0, arguments.replay_1, arguments.replay_2],
        arguments.output_dir.resolve(),
    )
    print(json.dumps(summary["output_rows"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
