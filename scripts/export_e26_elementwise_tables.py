#!/usr/bin/env python3
"""Validate E26 and export manuscript-sized elementwise agreement tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "30142ce2c9f60a115fa77cf844cac14166446b27"
EXPECTED_PROTOCOL = "1.15"
EXPECTED_CONFIG_HASH = "92b66e55c0d268f6f5d0a5463f28e1484eb5008d4d5c99ad13ba0450e2d63047"
EXPECTED_DEVICE = "argexpr3.vsb.cz::gpu:0::Tesla V100-PCIE-32GB"
EXPECTED_REFERENCE_COMMIT = (
    '{"hnnpp":"28737e22822562ac18d5ea03f8c3e3929e945a83",'
    '"hyperbolic_unet":"bec504e2f4d198ee4c438579b86d72f9d8186d78",'
    '"poincare_resnet":"fbbbbf890571115444b89b114f9ed52d128d5aa3"}'
)
EXPECTED_CASE_SPECS = {
    "convolution_stack_8": (
        '{"activation":"relu","depth":8,"id":"convolution_stack_8",'
        '"input_channels":4,"input_shape":[2,7,7,4],"kernel_size":3,'
        '"kind":"convolution_stack","output_channels":4,"padding":"SAME",'
        '"stride":1}'
    ),
    "small_unet": (
        '{"classes":2,"depth":2,"id":"small_unet","initial_curvature":0.1,'
        '"initial_width":2,"input_channels":3,"input_shape":[2,16,16,3],'
        '"kind":"small_unet"}'
    ),
}
EXPECTED_OUTPUT_SHAPES = {
    "convolution_stack_8": [2, 7, 7, 4],
    "small_unet": [2, 16, 16, 2],
}
DTYPES = ("float32", "float64")
VARIANTS = ("F", "T")
REALIZATIONS = tuple(range(10))
QUANTILE_FIELDS = (
    "elementwise_relative_error_p50",
    "elementwise_relative_error_p90",
    "elementwise_relative_error_p95",
    "elementwise_relative_error_p99",
    "elementwise_relative_error_p99_9",
)
MIXED_TOLERANCES = {
    "float32": (2.0e-5, 2.0e-5),
    "float64": (1.0e-10, 1.0e-10),
}


def _convolution_groups() -> frozenset[str]:
    groups = {"output", "input_gradient", "curvature_gradient"}
    for index in range(8):
        for field in ("offset", "weight"):
            groups.add(f"parameter_gradient:[{index}].{field}")
    return frozenset(groups)


def _small_unet_groups() -> frozenset[str]:
    groups = {
        "output",
        "input_gradient",
        "parameter_gradient:['curvature_raw']",
        "parameter_gradient:['output_conv'].offset",
        "parameter_gradient:['output_conv'].weight",
    }

    def add_double_convolution(prefix: str) -> None:
        for layer in ("bn1", "bn2"):
            for field in ("bias", "weight"):
                groups.add(f"parameter_gradient:{prefix}['{layer}'].{field}")
        for layer in ("conv1", "conv2"):
            for field in ("offset", "weight"):
                groups.add(f"parameter_gradient:{prefix}['{layer}'].{field}")

    add_double_convolution("['input_block']")
    for index in range(2):
        add_double_convolution(f"['down_blocks'][{index}]")
        add_double_convolution(f"['up_blocks'][{index}]['conv']")
        for field in ("offset", "weight"):
            groups.add(f"parameter_gradient:['up_blocks'][{index}]['transpose'].{field}")
    return frozenset(groups)


EXPECTED_VARIABLE_GROUPS = {
    "convolution_stack_8": _convolution_groups(),
    "small_unet": _small_unet_groups(),
}
EXPECTED_GROUPS = {
    case_id: len(groups) for case_id, groups in EXPECTED_VARIABLE_GROUPS.items()
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/reviewer_requested")
    )
    parser.add_argument("--expected-code-commit", default=EXPECTED_COMMIT)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = list(materialized[0])
    if any(list(row) != fields for row in materialized):
        raise ValueError(f"inconsistent fields for {path}")
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty E26 input: {path}")
    return rows


def _finite_nonnegative(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected numeric value") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{context}: expected finite nonnegative value")
    return number


def _one(values: Iterable[Any], *, field: str) -> Any:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"E26 has inconsistent {field}: {sorted(map(str, unique))}")
    return next(iter(unique))


def _validate_rows(
    rows: Sequence[dict[str, Any]], *, expected_commit: str
) -> list[dict[str, Any]]:
    expected_rows = (
        sum(EXPECTED_GROUPS.values()) * len(DTYPES) * len(VARIANTS) * len(REALIZATIONS)
    )
    if len(rows) != expected_rows:
        raise ValueError(f"E26 row count is {len(rows)}, expected {expected_rows}")
    identity = {
        "experiment_id": "E26",
        "protocol_version": EXPECTED_PROTOCOL,
        "code_commit": expected_commit,
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
        observed = _one((row.get(field) for row in rows), field=field)
        if observed != expected:
            raise ValueError(f"E26 {field} is {observed!r}, expected {expected!r}")
    if _one((row.get("finite") for row in rows), field="finite") is not True:
        raise ValueError("E26 contains a nonfinite agreement row")
    keys: set[tuple[Any, ...]] = set()
    group_counts: Counter[tuple[str, str, int, str]] = Counter()
    variable_groups: dict[tuple[str, str, int, str], set[str]] = defaultdict(set)
    prepared = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id"))
        dtype = str(row.get("dtype"))
        variant = str(row.get("variant"))
        realization = int(row.get("realization"))
        variable_group = str(row.get("variable_group"))
        if case_id not in EXPECTED_GROUPS:
            raise ValueError(f"row {index}: unexpected case {case_id}")
        if dtype not in DTYPES or variant not in VARIANTS:
            raise ValueError(f"row {index}: unexpected dtype/variant")
        if realization not in REALIZATIONS:
            raise ValueError(f"row {index}: unexpected realization {realization}")
        key = (case_id, dtype, realization, variant, variable_group)
        if key in keys:
            raise ValueError(f"row {index}: duplicate E26 key {key}")
        keys.add(key)
        cell = (case_id, dtype, realization, variant)
        group_counts[cell] += 1
        variable_groups[cell].add(variable_group)
        if row.get("case_spec") != EXPECTED_CASE_SPECS[case_id]:
            raise ValueError(f"row {index}: case_spec does not match {case_id}")
        if row.get("output_shape") != EXPECTED_OUTPUT_SHAPES[case_id]:
            raise ValueError(f"row {index}: output_shape does not match {case_id}")
        if row.get("seed_or_session") != f"{case_id}:{realization}":
            raise ValueError(f"row {index}: seed_or_session does not match its cell")
        expected_precision = json.dumps(
            {
                "matmul_precision": "highest",
                "reduced_precision_reduction": False,
                "storage_dtype": dtype,
                "tf32": False,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        if row.get("precision_policy") != expected_precision:
            raise ValueError(f"row {index}: precision_policy does not match dtype")

        elements = int(row.get("elementwise_relative_error_elements"))
        floored = int(row.get("elementwise_relative_error_denominator_floored_elements"))
        if elements <= 0 or not 0 <= floored <= elements:
            raise ValueError(f"row {index}: invalid element/floor counts")
        floored_fraction = _finite_nonnegative(
            row.get("elementwise_relative_error_denominator_floored_fraction"),
            context=f"row {index} floored fraction",
        )
        if not math.isclose(floored_fraction, floored / elements, abs_tol=1.0e-15):
            raise ValueError(f"row {index}: floored fraction does not match counts")
        quantiles = [
            _finite_nonnegative(row.get(field), context=f"row {index} {field}")
            for field in QUANTILE_FIELDS
        ]
        maximum = _finite_nonnegative(
            row.get("elementwise_relative_error_max"),
            context=f"row {index} elementwise max",
        )
        if quantiles != sorted(quantiles) or quantiles[-1] > maximum:
            raise ValueError(f"row {index}: elementwise quantiles are not monotone")
        relative_l2 = _finite_nonnegative(
            row.get("relative_l2_error"), context=f"row {index} relative L2"
        )
        absolute = _finite_nonnegative(
            row.get("max_absolute_error"), context=f"row {index} absolute error"
        )
        reference_l2 = _finite_nonnegative(
            row.get("reference_l2_norm"), context=f"row {index} reference L2"
        )
        absolute_tolerance, relative_tolerance = MIXED_TOLERANCES[dtype]
        mixed_threshold = absolute_tolerance + relative_tolerance * reference_l2
        scope = "output" if variable_group == "output" else "gradient"
        prepared.append(
            {
                "experiment_id": "E26",
                "protocol_version": EXPECTED_PROTOCOL,
                "code_commit": expected_commit,
                "case_id": case_id,
                "dtype": dtype,
                "realization": realization,
                "variant": variant,
                "scope": scope,
                "variable_group": variable_group,
                "elements": elements,
                "denominator_floored_elements": floored,
                "denominator_floored_fraction": floored_fraction,
                "p50": quantiles[0],
                "p90": quantiles[1],
                "p95": quantiles[2],
                "p99": quantiles[3],
                "p99_9": quantiles[4],
                "maximum": maximum,
                "relative_l2_error": relative_l2,
                "max_absolute_error": absolute,
                "reference_l2_norm": reference_l2,
                "mixed_absolute_tolerance": absolute_tolerance,
                "mixed_relative_tolerance": relative_tolerance,
                "mixed_criterion_threshold": mixed_threshold,
                "mixed_criterion_pass": absolute <= mixed_threshold,
            }
        )
    expected_counts = {
        (case_id, dtype, realization, variant): groups
        for case_id, groups in EXPECTED_GROUPS.items()
        for dtype in DTYPES
        for realization in REALIZATIONS
        for variant in VARIANTS
    }
    if dict(group_counts) != expected_counts:
        raise ValueError("E26 case/dtype/realization/variant matrix is incomplete")
    for cell, observed_groups in variable_groups.items():
        case_id = cell[0]
        if observed_groups != EXPECTED_VARIABLE_GROUPS[case_id]:
            missing = sorted(EXPECTED_VARIABLE_GROUPS[case_id] - observed_groups)
            unexpected = sorted(observed_groups - EXPECTED_VARIABLE_GROUPS[case_id])
            raise ValueError(
                f"E26 variable-group inventory mismatch for {cell}; "
                f"missing={missing}, unexpected={unexpected}"
            )
    return prepared


def _summaries(
    rows: Sequence[dict[str, Any]], *, include_case: bool
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            (str(row["case_id"]), str(row["dtype"]), str(row["variant"]), str(row["scope"]))
            if include_case
            else (str(row["dtype"]), str(row["variant"]), str(row["scope"]))
        )
        groups[key].append(row)
    output = []
    for key in sorted(groups):
        selected = groups[key]
        case_scope, dtype, variant, scope = key if include_case else ("all_cases", *key)
        elements = sum(int(row["elements"]) for row in selected)
        floored = sum(int(row["denominator_floored_elements"]) for row in selected)
        passed = sum(bool(row["mixed_criterion_pass"]) for row in selected)
        output.append(
            {
                "experiment_id": "E26",
                "protocol_version": EXPECTED_PROTOCOL,
                "code_commit": selected[0]["code_commit"],
                "config_hash": selected[0]["config_hash"],
                "device_id": selected[0]["device_id"],
                "source_sha256": selected[0]["source_sha256"],
                "case_scope": case_scope,
                "dtype": dtype,
                "variant": variant,
                "scope": scope,
                "tensor_count": len(selected),
                "mixed_criterion_passed_tensors": passed,
                "mixed_criterion_failed_tensors": len(selected) - passed,
                "total_elements": elements,
                "denominator_floored_elements": floored,
                "pooled_denominator_floored_fraction": floored / elements,
                "largest_tensor_p50": max(float(row["p50"]) for row in selected),
                "largest_tensor_p90": max(float(row["p90"]) for row in selected),
                "largest_tensor_p95": max(float(row["p95"]) for row in selected),
                "largest_tensor_p99": max(float(row["p99"]) for row in selected),
                "largest_tensor_p99_9": max(float(row["p99_9"]) for row in selected),
                "largest_elementwise_relative_error": max(
                    float(row["maximum"]) for row in selected
                ),
                "largest_tensor_relative_l2_error": max(
                    float(row["relative_l2_error"]) for row in selected
                ),
                "largest_absolute_error": max(
                    float(row["max_absolute_error"]) for row in selected
                ),
                "aggregation": (
                    "maxima across per-tensor elementwise quantiles; only the "
                    "denominator-floored fraction is pooled by element count"
                ),
            }
        )
    return output


def export(input_path: Path, output_dir: Path, *, expected_commit: str) -> dict[str, Any]:
    raw = _read_jsonl(input_path)
    tensors = _validate_rows(raw, expected_commit=expected_commit)
    source_hash = _sha256(input_path)
    for row in tensors:
        row["config_hash"] = EXPECTED_CONFIG_HASH
        row["device_id"] = EXPECTED_DEVICE
        row["source_sha256"] = source_hash
    case_summary = _summaries(tensors, include_case=True)
    scope_summary = _summaries(tensors, include_case=False)
    if len(case_summary) != 16 or len(scope_summary) != 8:
        raise ValueError("E26 summary matrix does not contain 16 case and 8 scope rows")
    for row in scope_summary:
        expected_tensors = 20 if row["scope"] == "output" else 660
        if row["tensor_count"] != expected_tensors:
            raise ValueError("E26 all-case summary has an unexpected tensor count")
    for row in case_summary:
        if row["scope"] == "output":
            expected_tensors = 10
        elif row["case_scope"] == "convolution_stack_8":
            expected_tensors = 180
        else:
            expected_tensors = 480
        if row["tensor_count"] != expected_tensors:
            raise ValueError("E26 case summary has an unexpected tensor count")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "e26_elementwise_tensor_statistics.csv", tensors)
    _atomic_csv(output_dir / "e26_elementwise_case_summary.csv", case_summary)
    _atomic_csv(output_dir / "e26_elementwise_scope_summary.csv", scope_summary)
    summary = {
        "schema_version": "1.0",
        "analysis_status": "prospective_reviewer_requested_diagnostic",
        "scientific_behavior_change": False,
        "source": {
            "file": input_path.name,
            "bytes": input_path.stat().st_size,
            "sha256": source_hash,
            "rows": len(raw),
        },
        "identity": {
            "experiment_id": "E26",
            "protocol_version": EXPECTED_PROTOCOL,
            "code_commit": expected_commit,
            "code_dirty": False,
            "config_hash": EXPECTED_CONFIG_HASH,
            "device_id": EXPECTED_DEVICE,
        },
        "definition": raw[0]["elementwise_relative_error_definition"],
        "denominator_floor": 1.0e-12,
        "quantile_method": "linear",
        "mixed_criterion": {
            dtype: {
                "absolute_tolerance": tolerances[0],
                "relative_tolerance": tolerances[1],
            }
            for dtype, tolerances in MIXED_TOLERANCES.items()
        },
        "interpretation": (
            "Fresh descriptive replay; pass/failure counts are reported to expose "
            "replay behavior but do not replace the accepted E02 gate."
        ),
        "aggregation_note": (
            "Summary quantile columns are conservative maxima of the named "
            "per-tensor elementwise quantile across all included tensor rows; "
            "they are not pooled elementwise quantiles. The floor fraction alone "
            "is pooled from element counts."
        ),
        "output_rows": {
            "e26_elementwise_tensor_statistics.csv": len(tensors),
            "e26_elementwise_case_summary.csv": len(case_summary),
            "e26_elementwise_scope_summary.csv": len(scope_summary),
        },
    }
    _atomic_json(output_dir / "e26_elementwise_summary.json", summary)
    return summary


def main() -> int:
    arguments = _parser().parse_args()
    summary = export(
        arguments.input.resolve(),
        arguments.output_dir.resolve(),
        expected_commit=str(arguments.expected_code_commit),
    )
    print(json.dumps(summary["output_rows"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
