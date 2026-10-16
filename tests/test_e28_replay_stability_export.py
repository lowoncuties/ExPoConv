from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.export_e26_elementwise_tables import (
    EXPECTED_CASE_SPECS,
    EXPECTED_DEVICE,
    EXPECTED_OUTPUT_SHAPES,
    EXPECTED_REFERENCE_COMMIT,
    EXPECTED_VARIABLE_GROUPS,
)
from scripts.export_e28_replay_stability import (
    EXPECTED_COMMIT,
    EXPECTED_CONFIG_HASH,
    export,
)


def _rows(replay: int) -> list[dict[str, object]]:
    rows = []
    for case_id, groups in EXPECTED_VARIABLE_GROUPS.items():
        for realization in range(3):
            for variant in ("F", "T"):
                for group_index, group in enumerate(sorted(groups)):
                    value = 1.0e-8 * (group_index + 1)
                    absolute = 1.0e-6
                    # A replay-sensitive P/F/P tensor.
                    if (
                        case_id == "convolution_stack_8"
                        and realization == 0
                        and variant == "F"
                        and group == "output"
                        and replay == 1
                    ):
                        absolute = 1.0e-3
                    # A stable mixed-criterion failure.
                    if (
                        case_id == "small_unet"
                        and realization == 0
                        and variant == "F"
                        and group == "output"
                    ):
                        absolute = 1.0e-3
                    # Exercise the inclusive boundary: 2e-5 + 2e-5 * 1.
                    if (
                        case_id == "small_unet"
                        and realization == 2
                        and variant == "T"
                        and group == "output"
                    ):
                        absolute = 4.0e-5
                    finite = not (
                        case_id == "convolution_stack_8"
                        and realization == 1
                        and variant == "T"
                        and group == "output"
                        and replay == 2
                    )
                    row: dict[str, object] = {
                        "experiment_id": "E28",
                        "protocol_version": "1.17",
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
                        "dtype": "float32",
                        "variant": variant,
                        "realization": realization,
                        "seed_or_session": f"{case_id}:{realization}",
                        "state_semantics": "current_batch_no_running_state",
                        "precision_policy": json.dumps(
                            {
                                "matmul_precision": "highest",
                                "reduced_precision_reduction": False,
                                "storage_dtype": "float32",
                                "tf32": False,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "variable_group": group,
                        "finite": finite,
                        "elementwise_relative_error_definition": (
                            "abs(candidate-reference)/max(abs(reference),denominator_floor)"
                        ),
                        "elementwise_relative_error_quantile_method": "linear",
                        "elementwise_relative_error_denominator_floor": 1.0e-12,
                        "elementwise_relative_error_elements": 10,
                    }
                    if finite:
                        row.update(
                            {
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
                                "max_absolute_error": absolute,
                                "reference_l2_norm": 1.0,
                            }
                        )
                    else:
                        row.update(
                            {
                                (
                                    "elementwise_relative_error_denominator_"
                                    "floored_elements"
                                ): None,
                                (
                                    "elementwise_relative_error_denominator_"
                                    "floored_fraction"
                                ): None,
                                "elementwise_relative_error_p50": None,
                                "elementwise_relative_error_p90": None,
                                "elementwise_relative_error_p95": None,
                                "elementwise_relative_error_p99": None,
                                "elementwise_relative_error_p99_9": None,
                                "elementwise_relative_error_max": None,
                                "relative_l2_error": None,
                                "max_absolute_error": None,
                                "reference_l2_norm": None,
                            }
                        )
                    rows.append(row)
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sources(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / f"replay_{replay}.jsonl" for replay in range(3)]
    for replay, path in enumerate(paths):
        _write(path, _rows(replay))
    return paths


def _find(rows: list[dict[str, str]], **wanted: object) -> dict[str, str]:
    return next(
        row
        for row in rows
        if all(row[field] == str(value) for field, value in wanted.items())
    )


def test_export_writes_complete_replay_stability_tables(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    output = tmp_path / "output"
    summary = export(sources, output)
    assert summary["output_rows"] == {
        "e28_replay_tensor_statistics.csv": 1224,
        "e28_replay_tensor_stability.csv": 408,
        "e28_replay_stability_summary.csv": 24,
    }
    assert summary["replay_provenance"]["status"] == (
        "wrapper_and_source_file_role_not_raw_embedded"
    )
    assert summary["overall"]["mixed_tensors"] == 2
    assert summary["overall"]["stable_fail_tensors"] == 1
    with (output / "e28_replay_tensor_stability.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        stability = list(csv.DictReader(stream))
    mixed = _find(
        stability,
        case_id="convolution_stack_8",
        realization=0,
        variant="F",
        variable_group="output",
    )
    assert mixed["pass_pattern"] == "PFP"
    assert mixed["stability_classification"] == "mixed"
    nonfinite = _find(
        stability,
        case_id="convolution_stack_8",
        realization=1,
        variant="T",
        variable_group="output",
    )
    assert nonfinite["pass_pattern"] == "PPN"
    assert nonfinite["nonfinite_replay_count"] == "1"
    boundary = _find(
        stability,
        case_id="small_unet",
        realization=2,
        variant="T",
        variable_group="output",
    )
    assert boundary["pass_pattern"] == "PPP"
    assert boundary["stability_classification"] == "stable_pass"
    assert boundary["max_absolute_error_range"] == "0.0"
    with (output / "e28_replay_stability_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        summaries = list(csv.DictReader(stream))
    nonfinite_summary = _find(
        summaries,
        case_id="convolution_stack_8",
        realization=1,
        variant="T",
        scope="output",
    )
    assert nonfinite_summary["replay_2_nonfinite_tensors"] == "1"


def test_export_rejects_incomplete_duplicate_and_duplicate_paths(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    _write(sources[0], _rows(0)[:-1])
    with pytest.raises(ValueError, match="row count"):
        export(sources, tmp_path / "incomplete")

    sources = _sources(tmp_path)
    rows = _rows(0)
    rows[-1] = deepcopy(rows[-2])
    _write(sources[0], rows)
    with pytest.raises(ValueError, match="duplicate E28 key"):
        export(sources, tmp_path / "duplicate")

    sources = _sources(tmp_path)
    with pytest.raises(ValueError, match="distinct files"):
        export([sources[0], sources[0], sources[2]], tmp_path / "same")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_hash", "wrong", "config_hash"),
        ("code_dirty", True, "code_dirty"),
        ("dtype", "float64", "dtype/variant"),
        ("realization", 4, "unexpected realization"),
        ("case_spec", "{}", "case_spec"),
        ("seed_or_session", "wrong", "seed"),
        ("precision_policy", "{}", "precision"),
    ],
)
def test_export_rejects_identity_and_cell_errors(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    sources = _sources(tmp_path)
    rows = _rows(0)
    rows[0][field] = value
    _write(sources[0], rows)
    with pytest.raises(ValueError, match=message):
        export(sources, tmp_path / field)


def test_export_rejects_wrong_inventory_and_ft_reference_mismatch(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    rows = _rows(0)
    rows[0]["variable_group"] = "invented_gradient"
    _write(sources[0], rows)
    with pytest.raises(ValueError, match="variable-group inventory"):
        export(sources, tmp_path / "inventory")

    sources = _sources(tmp_path)
    rows = _rows(0)
    target = next(
        row
        for row in rows
        if row["case_id"] == "convolution_stack_8"
        and row["realization"] == 0
        and row["variant"] == "F"
        and row["variable_group"] == "output"
    )
    target["reference_l2_norm"] = 2.0
    _write(sources[0], rows)
    with pytest.raises(ValueError, match="F/T reference_l2_norm differs"):
        export(sources, tmp_path / "reference")


def test_export_accepts_nonfinite_null_schema_but_rejects_partial_metrics(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    export(sources, tmp_path / "valid")

    rows = _rows(2)
    target = next(row for row in rows if row["finite"] is False)
    target["max_absolute_error"] = 1.0
    _write(sources[2], rows)
    with pytest.raises(ValueError, match="nonfinite metrics must be null"):
        export(sources, tmp_path / "invalid")
