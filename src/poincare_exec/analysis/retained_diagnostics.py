"""Deterministic completeness checks and tables for retained-state diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from poincare_exec.config.constants import (
    DIAGNOSTIC_PLOT_COLORS,
    DIAGNOSTIC_PLOT_MARKERS,
    DIAGNOSTIC_SOURCE_LINESTYLES,
    DIAGNOSTIC_SYMLOG_LINEAR_THRESHOLD,
    FIGURE_PDF_METADATA,
    TASK_DISPLAY_NAMES,
)
from poincare_exec.config.loader import load_yaml, validate_hashed_document


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a complete CSV atomically so partial exports are never promoted."""

    if not rows:
        raise ValueError(f"refusing to write empty diagnostic table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata=dict(FIGURE_PDF_METADATA),
    )
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def aggregate_comparison_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate leaf comparisons without averaging away large coordinates."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = (
        "source_experiment_id",
        "source_task",
        "source_variant",
        "source_seed",
        "run_variant",
        "comparison_type",
        "update",
    )
    for row in rows:
        if row.get("parameter_path") is None:
            continue
        grouped[tuple(row.get(field) for field in key_fields)].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        reference_sq = sum(
            float(row["reference_l2_norm"]) ** 2
            for row in selected
            if row.get("reference_l2_norm") is not None
        )
        difference_sq = sum(
            float(row["difference_l2_norm"]) ** 2
            for row in selected
            if row.get("difference_l2_norm") is not None
        )
        comparison_nonfinite = sum(
            int(row["comparison_nonfinite_count"]) for row in selected
        )
        maximum_error = None
        relative_error = None
        difference_norm = None
        reference_norm = None
        if comparison_nonfinite == 0:
            reference_norm = math.sqrt(reference_sq)
            difference_norm = math.sqrt(difference_sq)
            maximum_error = max(float(row["max_absolute_error"]) for row in selected)
            relative_error = difference_norm / max(reference_norm, 1.0e-12)
        output.append(
            {
                **dict(zip(key_fields, key, strict=True)),
                "parameter_paths": len(selected),
                "element_count": sum(int(row["element_count"]) for row in selected),
                "reference_l2_norm": reference_norm,
                "difference_l2_norm": difference_norm,
                "max_absolute_error": maximum_error,
                "relative_l2_error": relative_error,
                "comparison_nonfinite_count": comparison_nonfinite,
                "exactly_equal": all(bool(row["exactly_equal"]) for row in selected),
            }
        )
    return output


def diagnostic_key_metrics(
    initial_rows: list[dict[str, Any]],
    first_update_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one review-facing row per source checkpoint and execution path."""

    identity_fields = ("source_experiment_id", "source_task", "source_variant")
    available_variants = {
        str(row["run_variant"])
        for row in initial_rows
        if row.get("comparison_type") == "initial_scalar"
    }
    execution_variants = tuple(
        variant
        for variant in ("R", "F", "T", "R_repeat", "R_ulp")
        if variant in available_variants
    )
    gradient_rows = aggregate_comparison_rows(
        row for row in initial_rows if row.get("comparison_type") == "loss_gradient"
    )
    parameter_update_rows = aggregate_comparison_rows(
        row for row in first_update_rows if row.get("comparison_type") == "parameter_update"
    )

    def relative_l2(row: dict[str, Any]) -> float | None:
        value = row.get("relative_l2_error")
        if value is not None:
            return float(value)
        difference = row.get("difference_l2_norm")
        reference = row.get("reference_l2_norm")
        if difference is None or reference is None:
            return None
        return float(difference) / max(float(reference), 1.0e-12)

    def keyed(
        rows: Iterable[dict[str, Any]], comparison_type: str | None = None
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        selected = (
            row
            for row in rows
            if comparison_type is None or row.get("comparison_type") == comparison_type
        )
        return {
            (*tuple(row.get(field) for field in identity_fields), row["run_variant"]): row
            for row in selected
        }

    scalar_by_key = keyed(initial_rows, "initial_scalar")
    logits_by_key = keyed(initial_rows, "logits")
    probabilities_by_key = keyed(initial_rows, "probabilities")
    gradients_by_key = keyed(gradient_rows)
    parameter_updates_by_key = keyed(parameter_update_rows)
    final_update = max(int(row["update"]) for row in trajectory_rows)
    final_by_key = keyed(
        row for row in trajectory_rows if int(row["update"]) == final_update
    )
    sources = sorted(
        {tuple(row.get(field) for field in identity_fields) for row in trajectory_rows}
    )
    output = []
    for source in sources:
        reference = final_by_key[(*source, "R")]
        for run_variant in execution_variants:
            key = (*source, run_variant)
            scalar = scalar_by_key[key]
            logits = logits_by_key[key]
            probabilities = probabilities_by_key[key]
            gradient = gradients_by_key[key]
            parameter_update = parameter_updates_by_key[key]
            final = final_by_key[key]
            output.append(
                {
                    **dict(zip(identity_fields, source, strict=True)),
                    "source_seed": int(scalar["source_seed"]),
                    "run_variant": run_variant,
                    "initial_loss_absolute_difference": abs(
                        float(scalar["loss_difference_from_R"])
                    ),
                    "logits_max_absolute_error": float(logits["max_absolute_error"]),
                    "logits_relative_l2_error": relative_l2(logits),
                    "probabilities_max_absolute_error": float(
                        probabilities["max_absolute_error"]
                    ),
                    "probabilities_relative_l2_error": relative_l2(probabilities),
                    "gradient_max_absolute_error": gradient["max_absolute_error"],
                    "gradient_relative_l2_error": relative_l2(gradient),
                    "first_parameter_update_max_absolute_error": parameter_update[
                        "max_absolute_error"
                    ],
                    "first_parameter_update_relative_l2_error": relative_l2(
                        parameter_update
                    ),
                    "final_observation_update": final_update,
                    "final_loss_absolute_difference": abs(
                        float(final["loss"]) - float(reference["loss"])
                    ),
                    "final_parameter_l2_difference": float(
                        final["parameter_l2_difference_from_R"]
                    ),
                    "final_optimizer_first_moment_l2_difference": float(
                        final["optimizer_first_moment_difference_from_R"]
                    ),
                    "final_optimizer_second_moment_l2_difference": float(
                        final["optimizer_second_moment_difference_from_R"]
                    ),
                    "initial_nonfinite_count": int(scalar["output_nonfinite_count"])
                    + int(scalar["gradient_nonfinite_count"]),
                    "final_nonfinite_count": int(final["parameter_nonfinite_count"])
                    + int(final["optimizer_nonfinite_count"]),
                }
            )
    return output


def plot_retained_parameter_trajectory(
    trajectory_rows: list[dict[str, Any]], destination: Path
) -> None:
    """Plot short-horizon parameter drift against repeat and one-ULP controls."""

    tasks = list(dict.fromkeys(str(row["source_task"]) for row in trajectory_rows))
    if not tasks:
        raise ValueError("cannot plot an empty retained-state trajectory")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    figure, axes = plt.subplots(1, len(tasks), figsize=(7.1, 2.75), squeeze=False)
    for axis, task in zip(axes[0], tasks, strict=True):
        selected = [
            row
            for row in trajectory_rows
            if row["source_task"] == task
            and row["run_variant"] in DIAGNOSTIC_PLOT_COLORS
            and int(row["update"]) > 0
        ]
        source_variants = sorted(
            {str(row["source_variant"]) for row in selected},
            key=("R", "F", "T").index,
        )
        for run_variant, color in DIAGNOSTIC_PLOT_COLORS.items():
            for source_variant in source_variants:
                series = sorted(
                    (
                        row
                        for row in selected
                        if row["run_variant"] == run_variant
                        and row["source_variant"] == source_variant
                    ),
                    key=lambda row: int(row["update"]),
                )
                axis.plot(
                    [int(row["update"]) for row in series],
                    [float(row["parameter_l2_difference_from_R"]) for row in series],
                    color=color,
                    linestyle=DIAGNOSTIC_SOURCE_LINESTYLES[source_variant],
                    marker=DIAGNOSTIC_PLOT_MARKERS[run_variant],
                    markersize=3.5,
                    linewidth=1.2,
                    alpha=0.9,
                )
        axis.set_yscale(
            "symlog",
            linthresh=DIAGNOSTIC_SYMLOG_LINEAR_THRESHOLD,
            linscale=0.5,
        )
        axis.set_xticks([1, 10, 50, 100])
        axis.set_xlabel("Post-budget update")
        axis.set_title(TASK_DISPLAY_NAMES.get(task, task))
        axis.grid(axis="y", which="both", alpha=0.2)
    axes[0, 0].set_ylabel(r"Parameter $L_2$ distance from R")
    path_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=DIAGNOSTIC_PLOT_MARKERS[variant],
            label={
                "R_repeat": "R repeat",
                "R_ulp": "R + one ULP",
            }.get(variant, variant),
        )
        for variant, color in DIAGNOSTIC_PLOT_COLORS.items()
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=linestyle,
            label=f"{variant}-trained source",
        )
        for variant, linestyle in DIAGNOSTIC_SOURCE_LINESTYLES.items()
    ]
    figure.legend(
        handles=[*path_handles, *source_handles],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.subplots_adjust(bottom=0.34, wspace=0.3)
    _save_figure(figure, destination)


def _expected_sources(config: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    return [
        (task, str(source_variant))
        for task in config["tasks"]
        for source_variant in task["source_variants"]
    ]


def _rows_match_identity(rows: Iterable[dict[str, Any]], summary: dict[str, Any]) -> bool:
    expected = {
        "experiment_id": summary.get("experiment_id"),
        "protocol_version": summary.get("protocol_version"),
        "variant": summary.get("variant"),
        "config_hash": summary.get("config_hash"),
        "code_commit": summary.get("code_commit"),
        "code_dirty": summary.get("code_dirty"),
        "reference_commit": summary.get("reference_commit"),
        "device_id": summary.get("device_id"),
        "precision_policy": summary.get("precision_policy"),
        "seed_or_session": summary.get("seed_or_session"),
        "design_document_path": summary.get("design_document_path"),
        "source_experiment_id": summary.get("source_experiment_id"),
        "source_task": summary.get("source_task"),
        "source_variant": summary.get("source_variant"),
        "source_seed": summary.get("source_seed"),
        "source_update": summary.get("source_update"),
        "source_checkpoint_sha256": summary.get("source_checkpoint_sha256"),
        "source_summary_sha256": summary.get("source_summary_sha256"),
        "source_code_commit": summary.get("source_code_commit"),
        "source_config_hash": summary.get("source_config_hash"),
        "source_protocol_version": summary.get("source_protocol_version"),
        "design_document_sha256": summary.get("design_document_sha256"),
    }
    return all(
        all(row.get(field) == value for field, value in expected.items()) for row in rows
    )


def _initial_structure_is_complete(
    rows: list[dict[str, Any]], run_variants: list[str]
) -> bool:
    reference_paths = None
    for run_variant in run_variants:
        selected = [row for row in rows if row.get("run_variant") == run_variant]
        if sum(row.get("comparison_type") == "initial_scalar" for row in selected) != 1:
            return False
        if sum(row.get("comparison_type") == "logits" for row in selected) != 1:
            return False
        if sum(row.get("comparison_type") == "probabilities" for row in selected) != 1:
            return False
        paths = [
            row.get("parameter_path")
            for row in selected
            if row.get("comparison_type") == "loss_gradient"
        ]
        if not paths or len(paths) != len(set(paths)):
            return False
        if reference_paths is None:
            reference_paths = set(paths)
        elif set(paths) != reference_paths:
            return False
    return True


def _first_update_structure_is_complete(
    rows: list[dict[str, Any]], run_variants: list[str]
) -> bool:
    comparison_types = (
        "parameter_update",
        "updated_parameter",
        "updated_first_moment",
        "updated_second_moment",
    )
    for comparison_type in comparison_types:
        reference_paths = None
        for run_variant in run_variants:
            paths = [
                row.get("parameter_path")
                for row in rows
                if row.get("run_variant") == run_variant
                and row.get("comparison_type") == comparison_type
            ]
            if not paths or len(paths) != len(set(paths)):
                return False
            if reference_paths is None:
                reference_paths = set(paths)
            elif set(paths) != reference_paths:
                return False
    return True


def _first_batch_outputs_are_complete(path: Path, run_variants: list[str]) -> bool:
    required = {
        "inputs",
        "targets",
        "batch_indices",
        *(f"{run_variant}__logits" for run_variant in run_variants),
        *(f"{run_variant}__probabilities" for run_variant in run_variants),
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                return False
            batch_size = int(archive["batch_indices"].shape[0])
            arrays = [archive[name] for name in required if name != "batch_indices"]
            return (
                batch_size > 0
                and all(array.ndim > 0 and array.shape[0] == batch_size for array in arrays)
                and all(np.all(np.isfinite(array)) for array in arrays)
            )
    except (OSError, TypeError, ValueError):
        return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def export_retained_diagnostic_tables(
    artifact_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Require the complete matrix and export publication-review tables."""

    root = Path(artifact_dir).expanduser().resolve()
    config = load_yaml(config_path)
    design_path, design_sha256 = validate_hashed_document(config, field="design_document")
    destination = Path(output_dir).expanduser().resolve()
    audit_rows = []
    initial_rows = []
    first_update_rows = []
    trajectory_rows = []
    curvature_rows = []
    for task_entry, source_variant in _expected_sources(config):
        experiment_id = str(task_entry["training_experiment_id"])
        source_seed = int(task_entry["source_seed"])
        run_dir = root / experiment_id.lower() / source_variant / f"seed_{source_seed}"
        required = {
            "summary": run_dir / "summary.json",
            "initial": run_dir / "initial_comparisons.jsonl",
            "first_update": run_dir / "first_update_comparisons.jsonl",
            "trajectory": run_dir / "trajectory.jsonl",
            "outputs": run_dir / "first_batch_outputs.npz",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"incomplete retained diagnostic {experiment_id}/{source_variant}: "
                + ", ".join(missing)
            )
        summary = _json(required["summary"])
        selected_initial = _jsonl(required["initial"])
        selected_first_update = _jsonl(required["first_update"])
        selected_trajectory = _jsonl(required["trajectory"])
        expected_trajectory = len(summary["run_variants"]) * len(
            summary["observation_updates"]
        )
        expected_run_variants = [
            *[str(value) for value in config["execution_variants"]],
            "R_repeat",
            "R_ulp",
        ]
        repeat_policy = str(config["controls"].get("unperturbed_repeat_policy", "bitwise"))
        all_rows = selected_initial + selected_first_update + selected_trajectory
        expected_trajectory_keys = {
            (run_variant, int(update))
            for run_variant in expected_run_variants
            for update in config["observation_updates"]
        }
        actual_trajectory_keys = [
            (str(row.get("run_variant")), int(row.get("update", -1)))
            for row in selected_trajectory
        ]
        curvature_applicable = task_entry["task"] == "isic16_poincare_unet"
        selected_curvature = [
            row
            for row in selected_initial
            if row.get("comparison_type") == "loss_gradient"
            and row.get("is_curvature_parameter_path") is True
        ]
        hashes = summary.get("artifact_sha256", {})
        checks = {
            "status_ok": summary.get("status") == "ok",
            "clean_code": summary.get("code_dirty") is False,
            "code_commit_recorded": _is_git_commit(summary.get("code_commit")),
            "source_code_commit_recorded": _is_git_commit(
                summary.get("source_code_commit")
            ),
            "source_hashes_recorded": _is_sha256(summary.get("source_checkpoint_sha256"))
            and _is_sha256(summary.get("source_summary_sha256")),
            "experiment_matches": summary.get("experiment_id") == config["experiment_id"],
            "design_document_matches": summary.get("design_document_path") == design_path
            and summary.get("design_document_sha256") == design_sha256,
            "training_experiment_matches": summary.get("source_experiment_id")
            == experiment_id,
            "task_matches": summary.get("task") == task_entry["task"]
            and summary.get("source_task") == task_entry["task"],
            "source_variant_matches": summary.get("source_variant") == source_variant,
            "source_seed_matches": int(summary.get("source_seed", -1)) == source_seed,
            "source_config_matches": summary.get("source_config_hash")
            == task_entry["source_config_hash"],
            "source_protocol_matches": summary.get("source_protocol_version")
            == task_entry["source_protocol_version"],
            "curvature_applicability_matches": summary.get("curvature_gradient_applicable")
            is curvature_applicable,
            "repeat_policy_matches": summary.get("unperturbed_repeat_policy")
            == repeat_policy,
            "repeat_acceptance_met": (
                summary.get("unperturbed_repeat_bitwise_exact") is True
                if repeat_policy == "bitwise"
                else isinstance(summary.get("unperturbed_repeat_bitwise_exact"), bool)
            ),
            "repeat_schedule_matches": summary.get("unperturbed_repeat_check_updates")
            == [int(value) for value in config["observation_updates"]],
            "run_variants_match": summary.get("run_variants") == expected_run_variants,
            "observation_schedule_matches": summary.get("observation_updates")
            == [int(value) for value in config["observation_updates"]]
            and int(summary.get("continuation_updates", -1))
            == int(config["continuation_updates"]),
            "initial_rows_match": len(selected_initial)
            == int(summary["initial_comparison_rows"]),
            "first_update_rows_match": len(selected_first_update)
            == int(summary["first_update_comparison_rows"]),
            "trajectory_rows_match": len(selected_trajectory)
            == int(summary["trajectory_rows"])
            == expected_trajectory,
            "trajectory_finite": all(
                row.get("status") == "ok" for row in selected_trajectory
            ),
            "all_rows_finite": all(row.get("status") == "ok" for row in all_rows),
            "row_identities_match": _rows_match_identity(all_rows, summary),
            "initial_structure_complete": _initial_structure_is_complete(
                selected_initial, expected_run_variants
            )
            and all(int(row.get("update", -1)) == 0 for row in selected_initial),
            "first_update_structure_complete": _first_update_structure_is_complete(
                selected_first_update, expected_run_variants
            )
            and all(int(row.get("update", -1)) == 1 for row in selected_first_update),
            "trajectory_matrix_complete": len(actual_trajectory_keys)
            == len(set(actual_trajectory_keys))
            and set(actual_trajectory_keys) == expected_trajectory_keys,
            "optimizer_steps_match": all(
                int(row.get("optimizer_step", -1))
                == int(summary.get("source_update", -1)) + int(row.get("update", -1))
                for row in selected_trajectory
            ),
            "artifact_hashes_match": isinstance(hashes, dict)
            and all(
                hashes.get(path.name) == _file_sha256(path)
                for name, path in required.items()
                if name != "summary"
            ),
            "continuation_schedule_recorded": _is_sha256(
                summary.get("continuation_index_schedule_sha256")
            )
            and len(summary.get("continuation_batch_sizes", []))
            == int(config["continuation_updates"])
            and all(int(size) > 0 for size in summary.get("continuation_batch_sizes", [])),
            "first_batch_outputs_complete": _first_batch_outputs_are_complete(
                required["outputs"], expected_run_variants
            ),
            "curvature_gradient_matrix_complete": (
                len(selected_curvature) == len(expected_run_variants)
                and {row.get("run_variant") for row in selected_curvature}
                == set(expected_run_variants)
            )
            if curvature_applicable
            else not selected_curvature,
        }
        complete = all(checks.values())
        audit_rows.append(
            {
                "training_experiment_id": experiment_id,
                "task": task_entry["task"],
                "source_variant": source_variant,
                "source_seed": source_seed,
                "status": "complete" if complete else "invalid",
                **checks,
                "source_checkpoint_sha256": summary.get("source_checkpoint_sha256"),
                "diagnostic_code_commit": summary.get("code_commit"),
                "diagnostic_config_hash": summary.get("config_hash"),
                "design_document_sha256": design_sha256,
                "curvature_gradient_applicable": curvature_applicable,
                "summary_path": str(required["summary"]),
            }
        )
        if not complete:
            raise RuntimeError(
                f"invalid retained diagnostic {experiment_id}/{source_variant}"
            )
        initial_rows.extend(selected_initial)
        first_update_rows.extend(selected_first_update)
        trajectory_rows.extend(selected_trajectory)
        curvature_rows.extend(selected_curvature)

    initial_scalars = [
        row for row in initial_rows if row.get("comparison_type") == "initial_scalar"
    ]
    logit_rows = [row for row in initial_rows if row.get("comparison_type") == "logits"]
    probability_rows = [
        row for row in initial_rows if row.get("comparison_type") == "probabilities"
    ]
    if not initial_scalars:
        raise RuntimeError("complete diagnostic matrix has no initial scalar rows")
    if not logit_rows:
        raise RuntimeError("complete diagnostic matrix has no logits rows")
    if not probability_rows:
        raise RuntimeError("complete diagnostic matrix has no probability rows")
    _write_csv(destination / "diagnostic_source_audit.csv", audit_rows)
    _write_csv(destination / "initial_scalar_summary.csv", initial_scalars)
    _write_csv(destination / "logits_agreement_summary.csv", logit_rows)
    _write_csv(destination / "probability_agreement_summary.csv", probability_rows)
    _write_csv(
        destination / "gradient_agreement_summary.csv",
        aggregate_comparison_rows(
            row for row in initial_rows if row.get("comparison_type") == "loss_gradient"
        ),
    )
    _write_csv(
        destination / "first_update_agreement_summary.csv",
        aggregate_comparison_rows(first_update_rows),
    )
    _write_csv(destination / "trajectory.csv", trajectory_rows)
    key_metrics = diagnostic_key_metrics(initial_rows, first_update_rows, trajectory_rows)
    _write_csv(destination / "diagnostic_key_metrics.csv", key_metrics)
    if curvature_rows:
        _write_csv(destination / "curvature_gradient_agreement.csv", curvature_rows)
    figure_stem = destination.parent / "figures" / "figure_retained_state_trajectory"
    plot_retained_parameter_trajectory(trajectory_rows, figure_stem)
    return {
        "status": "complete",
        "sources": len(audit_rows),
        "initial_rows": len(initial_rows),
        "logit_rows": len(logit_rows),
        "probability_rows": len(probability_rows),
        "first_update_rows": len(first_update_rows),
        "trajectory_rows": len(trajectory_rows),
        "curvature_gradient_rows": len(curvature_rows),
        "key_metric_rows": len(key_metrics),
        "figure_pdf": str(figure_stem.with_suffix(".pdf")),
        "figure_png": str(figure_stem.with_suffix(".png")),
    }
