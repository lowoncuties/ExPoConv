#!/usr/bin/env python3
"""Export the exact numerical rows cited in the manuscript appendix.

This is a read-only analysis of completed E02 JSONL files. It does not execute
JAX or a numerical experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

FIELDNAMES = (
    "selection",
    "source_file",
    "source_sha256",
    "case_id",
    "dtype",
    "realization",
    "variant",
    "check_kind",
    "variable_group",
    "comparison_reference",
    "max_absolute_error",
    "relative_l2_error",
    "reference_l2_norm",
    "finite",
    "within_tolerance",
    "code_commit",
    "config_hash",
)
VARIANT_ORDER = {"R": 0, "F": 1, "T": 2}
AGREEMENT_TOLERANCE = {
    "float32": (2e-5, 2e-5),
    "float64": (1e-10, 1e-10),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_dtype(row: dict[str, Any]) -> str:
    if row.get("dtype"):
        return str(row["dtype"])
    policy = json.loads(row["precision_policy"])
    return str(policy["storage_dtype"])


def _normalize(
    row: dict[str, Any],
    *,
    selection: str,
    source_file: str,
    source_sha256: str,
) -> dict[str, Any]:
    dtype = _storage_dtype(row)
    within_tolerance = row.get("within_tolerance")
    if within_tolerance is None and row.get("record_type") == "agreement":
        absolute_tolerance, relative_tolerance = AGREEMENT_TOLERANCE[dtype]
        threshold = absolute_tolerance + relative_tolerance * float(
            row["reference_l2_norm"]
        )
        within_tolerance = bool(
            row.get("finite") is True
            and float(row["max_absolute_error"]) <= threshold
        )
    return {
        "selection": selection,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "case_id": row.get("case_id"),
        "dtype": dtype,
        "realization": row.get("realization"),
        "variant": row.get("variant"),
        "check_kind": row.get("check_kind") or row.get("record_type"),
        "variable_group": row.get("variable_group"),
        "comparison_reference": row.get("comparison_reference"),
        "max_absolute_error": row.get("max_absolute_error"),
        "relative_l2_error": row.get("relative_l2_error"),
        "reference_l2_norm": row.get("reference_l2_norm"),
        "finite": row.get("finite"),
        "within_tolerance": within_tolerance,
        "code_commit": row.get("code_commit"),
        "config_hash": row.get("config_hash"),
    }


def _select_rows(
    bounded_path: Path,
    structured_path: Path,
) -> list[dict[str, Any]]:
    bounded = _load_jsonl(bounded_path)
    structured = _load_jsonl(structured_path)
    bounded_hash = _sha256(bounded_path)
    structured_hash = _sha256(structured_path)

    factorized = [
        row
        for row in structured
        if row.get("record_type") == "agreement"
        and row.get("variant") == "F"
        and row.get("finite") is True
    ]
    if len(factorized) != 1360:
        raise ValueError(f"expected 1360 finite structured F rows, found {len(factorized)}")
    largest_absolute = max(factorized, key=lambda row: float(row["max_absolute_error"]))
    largest_relative = max(factorized, key=lambda row: float(row["relative_l2_error"]))

    selected = [
        _normalize(
            largest_absolute,
            selection="structured_f_largest_absolute_error",
            source_file=structured_path.name,
            source_sha256=structured_hash,
        ),
        _normalize(
            largest_relative,
            selection="structured_f_largest_relative_l2_error",
            source_file=structured_path.name,
            source_sha256=structured_hash,
        ),
    ]

    analytic_zero = [
        row for row in bounded if row.get("check_kind") == "n084_zero_curvature_oracle"
    ]
    if len(analytic_zero) != 6:
        raise ValueError(f"expected six analytic-zero rows, found {len(analytic_zero)}")
    analytic_zero.sort(
        key=lambda row: (_storage_dtype(row), VARIANT_ORDER[str(row["variant"])])
    )
    selected.extend(
        _normalize(
            row,
            selection="analytic_zero_curvature_diagnostic",
            source_file=bounded_path.name,
            source_sha256=bounded_hash,
        )
        for row in analytic_zero
    )

    exact_cap_failures = [
        row
        for row in bounded
        if row.get("check_kind") == "projection_jvp"
        and row.get("exact_cap_case") is True
        and row.get("within_tolerance") is False
    ]
    if len(exact_cap_failures) != 1:
        raise ValueError(
            "expected one constructed exact-cap diagnostic discrepancy, "
            f"found {len(exact_cap_failures)}"
        )
    selected.append(
        _normalize(
            exact_cap_failures[0],
            selection="constructed_exact_cap_jvp_discrepancy",
            source_file=bounded_path.name,
            source_sha256=bounded_hash,
        )
    )
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded-jsonl", type=Path, required=True)
    parser.add_argument("--structured-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    rows = _select_rows(arguments.bounded_jsonl, arguments.structured_jsonl)
    _write_csv(arguments.output, rows)
    print(f"wrote {arguments.output} with {len(rows)} claim-level rows")


if __name__ == "__main__":
    main()
