from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.export_e26_elementwise_tables import (
    EXPECTED_CASE_SPECS,
    EXPECTED_COMMIT,
    EXPECTED_CONFIG_HASH,
    EXPECTED_DEVICE,
    EXPECTED_OUTPUT_SHAPES,
    EXPECTED_REFERENCE_COMMIT,
    EXPECTED_VARIABLE_GROUPS,
    export,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    for case_id, groups in EXPECTED_VARIABLE_GROUPS.items():
        for dtype in ("float32", "float64"):
            for realization in range(10):
                for variant in ("F", "T"):
                    for group_index, group in enumerate(sorted(groups)):
                        scale = 1.0e-7 if dtype == "float32" else 1.0e-14
                        value = scale * (group_index + 1)
                        rows.append(
                            {
                                "experiment_id": "E26",
                                "protocol_version": "1.15",
                                "code_commit": EXPECTED_COMMIT,
                                "code_dirty": False,
                                "status": "ok",
                                "record_type": "agreement",
                                "comparison_reference": "R_same_precision",
                                "config_hash": EXPECTED_CONFIG_HASH,
                                "device_id": EXPECTED_DEVICE,
                                "reference_commit": EXPECTED_REFERENCE_COMMIT,
                                "case_id": case_id,
                                "case_spec": EXPECTED_CASE_SPECS[case_id],
                                "output_shape": EXPECTED_OUTPUT_SHAPES[case_id],
                                "dtype": dtype,
                                "variant": variant,
                                "realization": realization,
                                "seed_or_session": f"{case_id}:{realization}",
                                "state_semantics": "current_batch_no_running_state",
                                "precision_policy": json.dumps(
                                    {
                                        "matmul_precision": "highest",
                                        "reduced_precision_reduction": False,
                                        "storage_dtype": dtype,
                                        "tf32": False,
                                    },
                                    separators=(",", ":"),
                                    sort_keys=True,
                                ),
                                "variable_group": group,
                                "finite": True,
                                "elementwise_relative_error_definition": (
                                    "abs(candidate-reference)/max(abs(reference),"
                                    "denominator_floor)"
                                ),
                                "elementwise_relative_error_quantile_method": "linear",
                                "elementwise_relative_error_denominator_floor": 1.0e-12,
                                "elementwise_relative_error_elements": 10,
                                (
                                    "elementwise_relative_error_denominator_"
                                    "floored_elements"
                                ): 2,
                                (
                                    "elementwise_relative_error_denominator_"
                                    "floored_fraction"
                                ): 0.2,
                                "elementwise_relative_error_p50": value,
                                "elementwise_relative_error_p90": 2 * value,
                                "elementwise_relative_error_p95": 3 * value,
                                "elementwise_relative_error_p99": 4 * value,
                                "elementwise_relative_error_p99_9": 5 * value,
                                "elementwise_relative_error_max": 6 * value,
                                "relative_l2_error": 7 * value,
                                "max_absolute_error": 8 * value,
                                "reference_l2_norm": 10.0,
                            }
                        )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_export_validates_complete_matrix_and_writes_three_tables(tmp_path: Path) -> None:
    source = tmp_path / "e26.jsonl"
    output = tmp_path / "output"
    _write(source, _rows())
    summary = export(source, output, expected_commit=EXPECTED_COMMIT)
    assert summary["output_rows"] == {
        "e26_elementwise_tensor_statistics.csv": 2720,
        "e26_elementwise_case_summary.csv": 16,
        "e26_elementwise_scope_summary.csv": 8,
    }
    with (output / "e26_elementwise_scope_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["scope"] for row in rows} == {"output", "gradient"}
    assert {row["case_scope"] for row in rows} == {"all_cases"}
    assert all(float(row["pooled_denominator_floored_fraction"]) == 0.2 for row in rows)
    assert {(row["scope"], int(row["tensor_count"])) for row in rows} == {
        ("output", 20),
        ("gradient", 660),
    }
    assert all(
        int(row["mixed_criterion_passed_tensors"]) == int(row["tensor_count"])
        for row in rows
    )
    assert all(int(row["mixed_criterion_failed_tensors"]) == 0 for row in rows)


def test_export_rejects_dirty_or_incomplete_artifact(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["code_dirty"] = True
    source = tmp_path / "dirty.jsonl"
    _write(source, rows)
    with pytest.raises(ValueError, match="code_dirty"):
        export(source, tmp_path / "dirty-output", expected_commit=EXPECTED_COMMIT)

    source = tmp_path / "incomplete.jsonl"
    _write(source, _rows()[:-1])
    with pytest.raises(ValueError, match="row count"):
        export(source, tmp_path / "incomplete-output", expected_commit=EXPECTED_COMMIT)


def test_export_rejects_wrong_config_or_variable_inventory(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["config_hash"] = "wrong"
    source = tmp_path / "wrong-config.jsonl"
    _write(source, rows)
    with pytest.raises(ValueError, match="config_hash"):
        export(source, tmp_path / "wrong-config-output", expected_commit=EXPECTED_COMMIT)

    rows = _rows()
    output_index = next(
        index
        for index, row in enumerate(rows)
        if row["case_id"] == "convolution_stack_8"
        and row["dtype"] == "float32"
        and row["realization"] == 0
        and row["variant"] == "F"
        and row["variable_group"] == "output"
    )
    rows[output_index]["variable_group"] = "invented_gradient"
    source = tmp_path / "wrong-groups.jsonl"
    _write(source, rows)
    with pytest.raises(ValueError, match="variable-group inventory"):
        export(source, tmp_path / "wrong-groups-output", expected_commit=EXPECTED_COMMIT)
