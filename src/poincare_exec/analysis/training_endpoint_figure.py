"""Render the original V100 training endpoints from the compact result table."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPERIMENTS = {
    "cifar100_poincare_resnet20": "E10",
    "isic16_poincare_unet": "E09",
}
TASK_ORDER = tuple(EXPERIMENTS)
TASK_LABELS = {
    "cifar100_poincare_resnet20": "CIFAR-100 Poincare ResNet-20",
    "isic16_poincare_unet": "ISIC 2016 Poincare U-Net",
}
METRIC_NAMES = {
    "cifar100_poincare_resnet20": "top1_accuracy",
    "isic16_poincare_unet": "mean_foreground_dice",
}
METRIC_LABELS = {
    "cifar100_poincare_resnet20": "Top-1 accuracy",
    "isic16_poincare_unet": "Foreground Dice",
}
ENDPOINTS = (
    ("fixed_final_update", "Fixed final update"),
    ("validation_best", "Validation-selected checkpoint"),
)
VARIANTS = ("R", "F", "T")
SEEDS = (11, 29, 47, 71, 101)
PROTOCOL_VERSION = "1.10"
DEVICE_ID = "argexpr3.vsb.cz::gpu:0::Tesla V100-PCIE-32GB"
EXPECTED_CONFIG_HASHES = {
    "E09": "e658aa862e4fe2f9f925cdfa954e78e98ef0f3a32c0f548b1002cc9af2831b14",
    "E10": "b851c30049080019f48f6298b441c8d1502f8c5734f27f12f45a4e6845962a40",
}
REQUIRED_COLUMNS = frozenset(
    {
        "experiment_id",
        "protocol_version",
        "config_hash",
        "device_id",
        "task",
        "variant",
        "seed",
        "status",
        "failure_reason",
        "endpoint",
        "official_metric_name",
        "official_metric",
    }
)
PLOT_COLORS = {"R": "#8C8C8C", "F": "#0072B2", "T": "#D55E00"}
PLOT_MARKERS = {"R": "o", "F": "s", "T": "^"}
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


def source_sha256(path: str | Path) -> str:
    """Return the SHA-256 identity of a compact input table."""

    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def read_training_runs_csv(path: str | Path) -> list[dict[str, str]]:
    """Read the compact E09/E10 table needed for the endpoint figure."""

    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"{source}: missing required training columns: {missing}")
        return [dict(row) for row in reader]


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    try:
        seed = int(row["seed"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid training seed {row['seed']!r}") from error
    return str(row["task"]), str(row["variant"]), seed, str(row["endpoint"])


def validate_training_endpoint_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require the exact accepted E09/E10 endpoint cohort before plotting."""

    expected_keys = {
        (task, variant, seed, endpoint)
        for task in TASK_ORDER
        for variant in VARIANTS
        for seed in SEEDS
        for endpoint, _label in ENDPOINTS
    }
    if len(rows) != len(expected_keys):
        raise ValueError(
            f"expected {len(expected_keys)} accepted E09/E10 endpoint rows, "
            f"found {len(rows)}"
        )

    keyed: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in keyed:
            raise ValueError(f"duplicate training endpoint row for {key!r}")
        keyed[key] = row
    if set(keyed) != expected_keys:
        missing = sorted(expected_keys - set(keyed))
        extra = sorted(set(keyed) - expected_keys)
        raise ValueError(f"unexpected training cohort; missing={missing}, extra={extra}")

    for (task, _variant, _seed, _endpoint), row in keyed.items():
        experiment_id = EXPERIMENTS[task]
        expected_fields = {
            "experiment_id": experiment_id,
            "protocol_version": PROTOCOL_VERSION,
            "config_hash": EXPECTED_CONFIG_HASHES[experiment_id],
            "device_id": DEVICE_ID,
            "status": "ok",
            "failure_reason": "",
            "official_metric_name": METRIC_NAMES[task],
        }
        for field, expected in expected_fields.items():
            if str(row[field]) != expected:
                raise ValueError(
                    f"{experiment_id}: unexpected {field} {row[field]!r}; "
                    f"expected {expected!r}"
                )
        try:
            metric = float(row["official_metric"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{experiment_id}: invalid official metric") from error
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise ValueError(
                f"{experiment_id}: official metric must be finite and in [0, 1]"
            )


def _values_by_seed(
    rows: Sequence[Mapping[str, Any]], task: str, endpoint: str
) -> dict[int, dict[str, float]]:
    selected: dict[int, dict[str, float]] = {seed: {} for seed in SEEDS}
    for row in rows:
        if row["task"] == task and row["endpoint"] == endpoint:
            selected[int(row["seed"])][str(row["variant"])] = float(
                row["official_metric"]
            )
    return selected


def build_training_endpoint_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Build the four-panel matched-seed endpoint figure without recomputation."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validate_training_endpoint_rows(rows)
    with matplotlib.rc_context():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams.update(FIGURE_RC)
        figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), squeeze=False)
        figure.subplots_adjust(hspace=0.60, wspace=0.24)
        for task_index, task in enumerate(TASK_ORDER):
            task_values = [
                float(row["official_metric"]) for row in rows if row["task"] == task
            ]
            span = max(task_values) - min(task_values)
            padding = max(0.002, 0.08 * span)
            limits = (min(task_values) - padding, max(task_values) + padding)
            for endpoint_index, (endpoint, endpoint_label) in enumerate(ENDPOINTS):
                axis = axes[task_index, endpoint_index]
                by_seed = _values_by_seed(rows, task, endpoint)
                for seed in SEEDS:
                    values = by_seed[seed]
                    axis.plot(
                        range(len(VARIANTS)),
                        [values[variant] for variant in VARIANTS],
                        color="#A0A0A0",
                        linewidth=0.8,
                        alpha=0.62,
                        zorder=1,
                    )
                    for variant_index, variant in enumerate(VARIANTS):
                        axis.scatter(
                            variant_index,
                            values[variant],
                            color=PLOT_COLORS[variant],
                            marker=PLOT_MARKERS[variant],
                            edgecolor="white",
                            linewidth=0.35,
                            s=31,
                            zorder=2,
                        )
                for variant_index, variant in enumerate(VARIANTS):
                    mean = sum(by_seed[seed][variant] for seed in SEEDS) / len(SEEDS)
                    axis.scatter(
                        variant_index,
                        mean,
                        color="black",
                        marker="D",
                        edgecolor="white",
                        linewidth=0.45,
                        s=38,
                        zorder=3,
                    )
                axis.set_xticks(range(len(VARIANTS)), VARIANTS)
                axis.set_ylabel(METRIC_LABELS[task] if endpoint_index == 0 else "")
                axis.set_title(f"{TASK_LABELS[task]}\n{endpoint_label}", fontsize=9)
                axis.set_ylim(*limits)
                axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
    return figure


def write_training_endpoint_figure(
    rows: Sequence[Mapping[str, Any]],
    destination: str | Path,
    *,
    input_sha256: str,
) -> tuple[Path, Path]:
    """Write deterministic PDF and PNG renderings for validated endpoint rows."""

    if len(input_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in input_sha256
    ):
        raise ValueError("input_sha256 must be a lowercase SHA-256 digest")
    stem = Path(destination).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    with matplotlib.rc_context():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams.update(FIGURE_RC)
        figure = build_training_endpoint_figure(rows)
        import matplotlib.pyplot as plt

        pdf = stem.with_suffix(".pdf")
        png = stem.with_suffix(".png")
        source_identity = f"training_runs.csv sha256:{input_sha256}"
        try:
            figure.savefig(
                pdf,
                metadata={
                    "Creator": "poincare-execution-experiments",
                    "Subject": source_identity,
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
            figure.savefig(
                png,
                dpi=300,
                metadata={
                    "Software": "poincare-execution-experiments",
                    "Source": source_identity,
                },
            )
        finally:
            plt.close(figure)
    return pdf, png


def write_training_endpoint_figure_from_csv(
    source: str | Path, destination: str | Path
) -> tuple[Path, Path]:
    """Validate and render a figure directly from a compact training CSV."""

    digest = source_sha256(source)
    rows = read_training_runs_csv(source)
    return write_training_endpoint_figure(rows, destination, input_sha256=digest)
