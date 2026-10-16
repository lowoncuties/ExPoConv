"""Render the E09/E10/E20/E21 quality-strata figure from its compact table."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TASK_ORDER = ("isic16_poincare_unet", "cifar100_poincare_resnet20")
TASK_LABELS = {
    "isic16_poincare_unet": "ISIC16 Poincare U-Net",
    "cifar100_poincare_resnet20": "CIFAR-100 Poincare ResNet-20",
}
EXPECTED_TASKS = {
    "E09": "isic16_poincare_unet",
    "E10": "cifar100_poincare_resnet20",
    "E20": "isic16_poincare_unet",
    "E21": "cifar100_poincare_resnet20",
}
EXPECTED_OUTCOMES = {
    "E09": "mean_per_image_foreground_dice",
    "E10": "top1_accuracy",
    "E20": "mean_per_image_foreground_dice",
    "E21": "top1_accuracy",
}
REQUIRED_COLUMNS = frozenset(
    {
        "experiment_id",
        "task",
        "physical_gpu_name",
        "pairs",
        "estimate",
        "interval_95_lower",
        "interval_95_upper",
        "endpoint",
        "test_partition",
        "comparison",
        "interval_method",
        "analysis_scope",
        "grand_pool_performed",
        "result_family",
        "outcome",
    }
)
GPU_LABELS = {
    "Tesla V100-PCIE-32GB": "V100",
    "NVIDIA RTX A6000": "A6000",
}
FIGURE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def read_quality_strata_csv(path: str | Path) -> list[dict[str, str]]:
    """Read only the compact columns needed to reproduce the strata figure."""

    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"{source}: missing required quality-strata columns: {missing}"
            )
        return [dict(row) for row in reader]


def _cohort_label(row: Mapping[str, Any]) -> str:
    gpu_name = str(row["physical_gpu_name"])
    gpu_label = GPU_LABELS.get(gpu_name, gpu_name)
    return f"{row['experiment_id']} | {gpu_label} | n={row['pairs']}"


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(EXPECTED_TASKS):
        raise ValueError("expected exactly four E09/E10/E20/E21 strata")
    by_experiment = {str(row["experiment_id"]): row for row in rows}
    if set(by_experiment) != set(EXPECTED_TASKS) or len(by_experiment) != len(rows):
        raise ValueError("expected one row for each of E09, E10, E20, and E21")
    for experiment_id, expected_task in EXPECTED_TASKS.items():
        row = by_experiment[experiment_id]
        if row["task"] != expected_task:
            raise ValueError(f"{experiment_id}: unexpected task {row['task']!r}")
        expected_fields = {
            "endpoint": "fixed_final_update",
            "test_partition": "official_test",
            "comparison": "F_minus_R",
            "interval_method": "paired_t",
            "analysis_scope": "within_platform_no_grand_pool",
            "result_family": "quality",
            "outcome": EXPECTED_OUTCOMES[experiment_id],
        }
        for field, expected in expected_fields.items():
            if row[field] != expected:
                raise ValueError(f"{experiment_id}: unexpected {field} {row[field]!r}")
        pool_value = row["grand_pool_performed"]
        if not (pool_value is False or pool_value == "False"):
            raise ValueError(f"{experiment_id}: pooled rows are not valid strata")
        try:
            pairs = int(row["pairs"])
            estimate = float(row["estimate"])
            lower = float(row["interval_95_lower"])
            upper = float(row["interval_95_upper"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{experiment_id}: invalid numeric stratum values") from error
        if pairs <= 0:
            raise ValueError(f"{experiment_id}: pairs must be a positive integer")
        if not all(math.isfinite(value) for value in (estimate, lower, upper)):
            raise ValueError(f"{experiment_id}: stratum values must be finite")
        if not lower <= estimate <= upper:
            raise ValueError(f"{experiment_id}: estimate must lie inside its interval")


def build_quality_strata_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Build the four-stratum figure without recomputing or pooling estimates."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    _validate_rows(rows)
    with matplotlib.rc_context():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams.update(FIGURE_RC)
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), squeeze=False)
        for axis, task in zip(axes[0], TASK_ORDER, strict=True):
            selected = sorted(
                (row for row in rows if row["task"] == task),
                key=lambda row: str(row["experiment_id"]),
            )
            y_positions = [1, 0]
            for y_position, row in zip(y_positions, selected, strict=True):
                estimate = float(row["estimate"])
                lower = float(row["interval_95_lower"])
                upper = float(row["interval_95_upper"])
                axis.errorbar(
                    estimate,
                    y_position,
                    xerr=[[estimate - lower], [upper - estimate]],
                    fmt="D",
                    color="#842C46",
                    capsize=3,
                    linewidth=1.3,
                    markersize=4.5,
                )
            axis.axvline(0.0, color="#8C8C8C", linestyle="--", linewidth=0.9)
            axis.set_yticks(y_positions, [_cohort_label(row) for row in selected])
            axis.set_title(TASK_LABELS[task])
            axis.set_xlabel("F minus R (native score scale)")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
            axis.grid(axis="x", alpha=0.2)
        figure.suptitle("Matched-seed quality differences by cohort/platform")
        figure.tight_layout()
    return figure


def write_quality_strata_figure(
    rows: Sequence[Mapping[str, Any]], destination: str | Path
) -> tuple[Path, Path]:
    """Write deterministic PDF and PNG renderings for compact strata rows."""

    stem = Path(destination).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    with matplotlib.rc_context():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams.update(FIGURE_RC)
        figure = build_quality_strata_figure(rows)
        import matplotlib.pyplot as plt

        pdf = stem.with_suffix(".pdf")
        png = stem.with_suffix(".png")
        try:
            figure.savefig(
                pdf,
                bbox_inches="tight",
                metadata={
                    "Creator": "poincare-execution-experiments",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
            figure.savefig(
                png,
                dpi=300,
                bbox_inches="tight",
                metadata={"Software": "poincare-execution-experiments"},
            )
        finally:
            plt.close(figure)
    return pdf, png


def write_quality_strata_figure_from_csv(
    source: str | Path, destination: str | Path
) -> tuple[Path, Path]:
    """Render a figure directly from the tracked compact strata CSV."""

    return write_quality_strata_figure(read_quality_strata_csv(source), destination)
