"""Build the training and checkpoint-replay tables and figures from raw run artifacts."""

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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poincare_exec.analysis.checkpoint_tables import (
    checkpoint_attempts,
    checkpoint_compatibility_summary,
    checkpoint_rows,
    trained_checkpoint_campaign_audit,
    trained_checkpoint_compatibility_aggregate,
    trained_checkpoint_task_agreement,
)
from poincare_exec.config.constants import (
    FIGURE_PDF_METADATA,
    PLOT_COLORS,
    PLOT_LINESTYLES,
    PLOT_MARKERS,
    TASK_DISPLAY_NAMES,
)
from poincare_exec.config.loader import load_yaml


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _bootstrap_generator(
    protocol: dict[str, Any], *identity: str
) -> tuple[np.random.Generator, str]:
    """Return a stable RNG stream scoped to one declared estimand.

    A single mutable generator makes an interval depend on which unrelated
    tasks or comparisons happened to be exported before it. Deriving the stream
    from the registered base seed and the full estimand identity makes exports
    invariant to row order and to adding another task.
    """

    payload = json.dumps(
        {
            "base_seed": int(protocol["seeds"]["bootstrap"]),
            "identity": list(identity),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    seed = int(digest[:16], 16)
    return np.random.default_rng(seed), digest


def _paired_t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return sample SD, standard error, and a two-sided paired-t 95% interval."""

    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired-t interval requires at least two one-dimensional values")
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(len(values))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    mean = float(np.mean(values))
    half_width = critical * standard_error
    return sample_sd, standard_error, mean - half_width, mean + half_width


def _exact_two_sided_sign_pvalue(values: np.ndarray) -> float:
    """Return the exact two-sided sign-test p-value after omitting exact ties."""

    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 1.0
    minority = min(int(np.sum(nonzero > 0)), int(np.sum(nonzero < 0)))
    probability = (
        2.0
        * sum(math.comb(len(nonzero), count) for count in range(minority + 1))
        / (2 ** len(nonzero))
    )
    return min(1.0, float(probability))


def _training_summaries(artifact_dir: Path) -> list[dict[str, Any]]:
    rows = []
    resolved: set[tuple[str, str, int]] = set()
    for path in sorted((artifact_dir / "training").glob("**/summary.json")):
        if "rejected" in path.relative_to(artifact_dir).parts:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        resolved.add(
            (
                summary["experiment_id"],
                summary["variant"],
                int(summary["seed_or_session"]),
            )
        )
        status = summary.get("status", "ok")
        if status != "ok":
            rows.append(
                {
                    "experiment_id": summary["experiment_id"],
                    "protocol_version": summary["protocol_version"],
                    "config_hash": summary["config_hash"],
                    "code_commit": summary["code_commit"],
                    "code_dirty": summary["code_dirty"],
                    "device_id": summary["device_id"],
                    "task": summary["task"],
                    "variant": summary["variant"],
                    "seed": int(summary["seed_or_session"]),
                    "status": status,
                    "failure_reason": (summary.get("failure") or {}).get("reason"),
                    "endpoint": "terminal_failure",
                    "checkpoint_update": int(summary["updates"]),
                    "official_metric_name": None,
                    "official_metric": None,
                    "official_mean_foreground_iou": None,
                    "official_loss": None,
                    "deduplicated_metric": None,
                    "deduplicated_mean_foreground_iou": None,
                    "deduplicated_loss": None,
                    "compute_seconds": summary["compute_seconds"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "best_development_metric": summary["best_development_metric"],
                    "best_update": summary["best_update"],
                    "summary_path": str(path),
                    "attempt_path": None,
                }
            )
            continue
        endpoints = {
            "fixed_final_update": {
                "update": summary["updates"],
                "official_test": summary["official_test"],
                "deduplicated_test": summary["deduplicated_test"],
            },
            "validation_best": summary["validation_best"],
        }
        for endpoint, values in endpoints.items():
            official = values["official_test"]
            deduplicated = values["deduplicated_test"]
            metric_name = (
                "mean_foreground_dice"
                if "mean_foreground_dice" in official
                else "top1_accuracy"
            )
            rows.append(
                {
                    "experiment_id": summary["experiment_id"],
                    "protocol_version": summary["protocol_version"],
                    "config_hash": summary["config_hash"],
                    "code_commit": summary["code_commit"],
                    "code_dirty": summary["code_dirty"],
                    "device_id": summary["device_id"],
                    "task": summary["task"],
                    "variant": summary["variant"],
                    "seed": int(summary["seed_or_session"]),
                    "status": status,
                    "failure_reason": None,
                    "endpoint": endpoint,
                    "checkpoint_update": int(values["update"]),
                    "official_metric_name": metric_name,
                    "official_metric": official[metric_name],
                    "official_mean_foreground_iou": official.get("mean_foreground_iou"),
                    "official_loss": official["loss"],
                    "deduplicated_metric": deduplicated[metric_name],
                    "deduplicated_mean_foreground_iou": deduplicated.get(
                        "mean_foreground_iou"
                    ),
                    "deduplicated_loss": deduplicated["loss"],
                    "compute_seconds": summary["compute_seconds"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "best_development_metric": summary["best_development_metric"],
                    "best_update": summary["best_update"],
                    "summary_path": str(path),
                    "attempt_path": None,
                }
            )
    latest_failures: dict[tuple[str, str, int], tuple[dict[str, Any], Path]] = {}
    for path in sorted((artifact_dir / "training").glob("**/campaign_attempts.jsonl")):
        if "rejected" in path.relative_to(artifact_dir).parts:
            continue
        for attempt in _jsonl(path):
            if attempt["status"] != "failed":
                continue
            key = (
                attempt["experiment_id"],
                attempt["variant"],
                int(attempt["seed"]),
            )
            latest_failures[key] = (attempt, path)
    for key, (attempt, path) in sorted(latest_failures.items()):
        if key in resolved:
            continue
        rows.append(
            {
                "experiment_id": attempt["experiment_id"],
                "protocol_version": attempt["protocol_version"],
                "config_hash": attempt["config_hash"],
                "code_commit": attempt["code_commit"],
                "code_dirty": attempt["code_dirty"],
                "device_id": None,
                "task": attempt["task"],
                "variant": attempt["variant"],
                "seed": int(attempt["seed"]),
                "status": "failed",
                "failure_reason": (
                    f"worker_exit_{attempt['returncode']}: {attempt['stderr']}"
                ),
                "endpoint": "terminal_failure",
                "checkpoint_update": None,
                "official_metric_name": None,
                "official_metric": None,
                "official_mean_foreground_iou": None,
                "official_loss": None,
                "deduplicated_metric": None,
                "deduplicated_mean_foreground_iou": None,
                "deduplicated_loss": None,
                "compute_seconds": None,
                "elapsed_seconds": None,
                "best_development_metric": None,
                "best_update": None,
                "summary_path": None,
                "attempt_path": str(path),
            }
        )
    return rows


def _development_curves(artifact_dir: Path) -> list[dict[str, Any]]:
    """Read only prespecified development checkpoints from training logs."""

    rows = []
    for path in sorted((artifact_dir / "training").glob("**/training_metrics.jsonl")):
        if "rejected" in path.relative_to(artifact_dir).parts:
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] == "development":
                    rows.append(row)
    return rows


def _training_comparisons(
    protocol: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    roles = protocol["variants"]["training_roles"]
    baseline = str(roles["baseline"])
    primary = str(roles["primary_method"])
    ablation = str(roles["ablation"])
    return (
        ("primary_vs_baseline", baseline, primary),
        ("ablation_vs_baseline", baseline, ablation),
        ("ablation_vs_primary", primary, ablation),
    )


def _training_variant_metric_summary(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    metric_name: str | None,
    official_field: str,
    deduplicated_field: str,
) -> list[dict[str, Any]]:
    expected_seeds = {int(value) for value in protocol["seeds"]["training"]}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row.get(official_field) is not None:
            grouped[(row["task"], row["endpoint"], row["variant"])].append(row)

    output = []
    for (task, endpoint, variant), selected in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in selected}
        if seeds != expected_seeds:
            continue
        for partition, metric_field, loss_field in (
            ("official_test", official_field, "official_loss"),
            (
                "official_test_excluding_exact_training_duplicates",
                deduplicated_field,
                "deduplicated_loss",
            ),
        ):
            if any(row.get(metric_field) is None for row in selected):
                continue
            metrics = [float(row[metric_field]) for row in selected]
            losses = [float(row[loss_field]) for row in selected]
            output.append(
                {
                    "task": task,
                    "endpoint": endpoint,
                    "test_partition": partition,
                    "variant": variant,
                    "seeds": len(selected),
                    "metric_name": metric_name or selected[0]["official_metric_name"],
                    "metric_mean": statistics.mean(metrics),
                    "metric_sample_sd": statistics.stdev(metrics),
                    "metric_median": statistics.median(metrics),
                    "metric_minimum": min(metrics),
                    "metric_maximum": max(metrics),
                    "loss_mean": statistics.mean(losses),
                    "loss_sample_sd": statistics.stdev(losses),
                    "raw_metrics_by_seed": json.dumps(
                        {
                            str(row["seed"]): float(row[metric_field])
                            for row in sorted(selected, key=lambda value: value["seed"])
                        },
                        sort_keys=True,
                    ),
                }
            )
    return output


def training_variant_summary(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    return _training_variant_metric_summary(
        rows,
        protocol,
        metric_name=None,
        official_field="official_metric",
        deduplicated_field="deduplicated_metric",
    )


def training_iou_variant_summary(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    return _training_variant_metric_summary(
        rows,
        protocol,
        metric_name="mean_foreground_iou",
        official_field="official_mean_foreground_iou",
        deduplicated_field="deduplicated_mean_foreground_iou",
    )


def _paired_training_metric_summary(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    metric_name: str | None,
    official_field: str,
    deduplicated_field: str,
) -> list[dict[str, Any]]:
    by_endpoint: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["status"] != "ok":
            continue
        for partition, field in (
            ("official_test", official_field),
            (
                "official_test_excluding_exact_training_duplicates",
                deduplicated_field,
            ),
        ):
            if row.get(field) is None:
                continue
            key = (row["task"], row["endpoint"], partition)
            by_endpoint[key][int(row["seed"])][row["variant"]] = float(row[field])
    replicates = int(protocol["quality"]["evaluation_bootstrap_replicates"])
    output = []
    expected_variants = set(protocol["variants"]["core_training"])
    expected_seeds = {int(value) for value in protocol["seeds"]["training"]}
    for (task, endpoint, partition), seeds in sorted(by_endpoint.items()):
        if set(seeds) != expected_seeds:
            continue
        if any(set(values) != expected_variants for values in seeds.values()):
            continue
        resolved_metric_name = metric_name or protocol["tasks"][task]["primary_metric"]
        for comparison, reference, candidate in _training_comparisons(protocol):
            random, bootstrap_stream = _bootstrap_generator(
                protocol,
                "paired_training_metric",
                task,
                endpoint,
                partition,
                resolved_metric_name,
                comparison,
                reference,
                candidate,
            )
            differences = np.asarray(
                [
                    values[candidate] - values[reference]
                    for _, values in sorted(seeds.items())
                ],
                dtype=np.float64,
            )
            indices = random.integers(
                0, len(differences), size=(replicates, len(differences))
            )
            bootstrap = np.mean(differences[indices], axis=1)
            sample_sd, standard_error, t_lower, t_upper = _paired_t_interval(differences)
            output.append(
                {
                    "task": task,
                    "endpoint": endpoint,
                    "test_partition": partition,
                    "metric_name": resolved_metric_name,
                    "comparison": comparison,
                    "reference_variant": reference,
                    "candidate_variant": candidate,
                    "pairs": len(differences),
                    "estimand": (
                        f"mean paired {candidate}-minus-{reference} {resolved_metric_name}"
                    ),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "difference_sample_sd": sample_sd,
                    "difference_standard_error": standard_error,
                    "paired_t_95_lower": t_lower,
                    "paired_t_95_upper": t_upper,
                    "paired_t_degrees_of_freedom": len(differences) - 1,
                    "exact_two_sided_sign_pvalue": _exact_two_sided_sign_pvalue(
                        differences
                    ),
                    "bootstrap_2.5_percentile": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_97.5_percentile": float(np.quantile(bootstrap, 0.975)),
                    "bootstrap_base_seed": int(protocol["seeds"]["bootstrap"]),
                    "bootstrap_stream_sha256": bootstrap_stream,
                    "bootstrap_replicates": replicates,
                    "interval_adjustment": "none",
                    "raw_differences_by_seed": json.dumps(
                        {
                            str(seed): float(difference)
                            for seed, difference in zip(
                                sorted(seeds), differences, strict=True
                            )
                        },
                        sort_keys=True,
                    ),
                }
            )
    return output


def paired_quality_summary(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    return _paired_training_metric_summary(
        rows,
        protocol,
        metric_name=None,
        official_field="official_metric",
        deduplicated_field="deduplicated_metric",
    )


def paired_iou_summary(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    return _paired_training_metric_summary(
        rows,
        protocol,
        metric_name="mean_foreground_iou",
        official_field="official_mean_foreground_iou",
        deduplicated_field="deduplicated_mean_foreground_iou",
    )


def paired_training_efficiency_summary(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_variants = set(protocol["variants"]["core_training"])
    expected_seeds = {int(value) for value in protocol["seeds"]["training"]}
    grouped: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["status"] != "ok" or row["endpoint"] != "fixed_final_update":
            continue
        grouped[row["task"]][int(row["seed"])][row["variant"]] = {
            "compute_seconds": float(row["compute_seconds"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
        }

    replicates = int(protocol["quality"]["evaluation_bootstrap_replicates"])
    output = []
    for task, seeds in sorted(grouped.items()):
        if set(seeds) != expected_seeds:
            continue
        if any(set(values) != expected_variants for values in seeds.values()):
            continue
        for comparison, reference, candidate in _training_comparisons(protocol):
            for measurement in ("compute_seconds", "elapsed_seconds"):
                random, bootstrap_stream = _bootstrap_generator(
                    protocol,
                    "paired_training_efficiency",
                    task,
                    comparison,
                    reference,
                    candidate,
                    measurement,
                )
                ratios = np.asarray(
                    [
                        values[reference][measurement] / values[candidate][measurement]
                        for _, values in sorted(seeds.items())
                    ],
                    dtype=np.float64,
                )
                log_ratios = np.log(ratios)
                indices = random.integers(0, len(ratios), size=(replicates, len(ratios)))
                bootstrap = np.exp(np.mean(log_ratios[indices], axis=1))
                log_sd, log_standard_error, log_t_lower, log_t_upper = _paired_t_interval(
                    log_ratios
                )
                output.append(
                    {
                        "task": task,
                        "comparison": comparison,
                        "reference_variant": reference,
                        "candidate_variant": candidate,
                        "measurement": measurement,
                        "pairs": len(ratios),
                        "estimand": (
                            "geometric mean paired reference/candidate ratio; "
                            "values above one favor the candidate"
                        ),
                        "geometric_mean_speedup": float(np.exp(np.mean(log_ratios))),
                        "median_speedup": float(np.median(ratios)),
                        "log_speedup_sample_sd": log_sd,
                        "log_speedup_standard_error": log_standard_error,
                        "log_t_95_lower": float(np.exp(log_t_lower)),
                        "log_t_95_upper": float(np.exp(log_t_upper)),
                        "log_t_degrees_of_freedom": len(ratios) - 1,
                        "bootstrap_2.5_percentile": float(np.quantile(bootstrap, 0.025)),
                        "bootstrap_97.5_percentile": float(np.quantile(bootstrap, 0.975)),
                        "bootstrap_base_seed": int(protocol["seeds"]["bootstrap"]),
                        "bootstrap_stream_sha256": bootstrap_stream,
                        "bootstrap_replicates": replicates,
                        "interval_adjustment": "none",
                        "raw_ratios_by_seed": json.dumps(
                            {
                                str(seed): float(ratio)
                                for seed, ratio in zip(sorted(seeds), ratios, strict=True)
                            },
                            sort_keys=True,
                        ),
                    }
                )
    return output


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata=dict(FIGURE_PDF_METADATA),
    )
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _style() -> None:
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


def plot_training_quality(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    protocol: dict[str, Any],
) -> None:
    """Plot every seed's registered development quality trajectory."""

    tasks = sorted({row["task"] for row in rows})
    if not tasks:
        return
    figure, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(7.1, 2.55 * len(tasks)),
        squeeze=False,
    )
    figure.subplots_adjust(hspace=0.72, wspace=0.22)
    for task_index, task in enumerate(tasks):
        metric = (
            "mean_foreground_dice" if task == "isic16_poincare_unet" else "top1_accuracy"
        )
        task_rows = [
            row
            for row in rows
            if row["task"] == task and row["status"] == "ok" and row.get(metric) is not None
        ]
        for variant in protocol["variants"]["core_training"]:
            seeds = sorted(
                {
                    int(row["seed_or_session"])
                    for row in task_rows
                    if row["variant"] == variant
                }
            )
            for seed_index, seed in enumerate(seeds):
                selected = sorted(
                    (
                        row
                        for row in task_rows
                        if row["variant"] == variant and int(row["seed_or_session"]) == seed
                    ),
                    key=lambda row: int(row["update"]),
                )
                label = variant if seed_index == 0 else None
                axes[task_index, 0].plot(
                    [int(row["update"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    color=PLOT_COLORS[variant],
                    linestyle=PLOT_LINESTYLES[variant],
                    linewidth=0.9,
                    alpha=0.65,
                    label=label,
                )
                axes[task_index, 1].plot(
                    [float(row["elapsed_seconds"]) / 3600.0 for row in selected],
                    [float(row[metric]) for row in selected],
                    color=PLOT_COLORS[variant],
                    linestyle=PLOT_LINESTYLES[variant],
                    linewidth=0.9,
                    alpha=0.65,
                    label=label,
                )
        metric_label = (
            "Foreground Dice" if metric == "mean_foreground_dice" else "Top-1 accuracy"
        )
        axes[task_index, 0].set_ylabel(f"Development {metric_label}")
        task_label = TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").title())
        axes[task_index, 0].set_title(f"{task_label}\nCompleted updates", fontsize=9)
        axes[task_index, 1].set_title("Worker elapsed time", fontsize=9)
        axes[task_index, 0].set_xlabel("Update")
        axes[task_index, 1].set_xlabel("Elapsed time (h)")
        for axis in axes[task_index]:
            axis.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False)
    _save_figure(figure, figure_dir / "figure4_training_quality")


def plot_paired_training_endpoints(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    protocol: dict[str, Any],
) -> None:
    """Show every matched seed at the two declared official-test endpoints."""

    endpoints = (
        ("fixed_final_update", "Fixed final update"),
        ("validation_best", "Validation-selected checkpoint"),
    )
    tasks = sorted(
        {
            row["task"]
            for row in rows
            if row["status"] == "ok" and row["endpoint"] in dict(endpoints)
        }
    )
    if not tasks:
        return
    variants = tuple(protocol["variants"]["core_training"])
    figure, axes = plt.subplots(
        len(tasks), 2, figsize=(7.1, 2.5 * len(tasks)), squeeze=False
    )
    figure.subplots_adjust(hspace=0.60, wspace=0.24)
    for task_index, task in enumerate(tasks):
        metric_name = protocol["tasks"][task]["primary_metric"]
        metric_label = "Foreground Dice" if "dice" in metric_name else "Top-1 accuracy"
        task_values = [
            float(row["official_metric"])
            for row in rows
            if row["task"] == task
            and row["endpoint"] in dict(endpoints)
            and row["status"] == "ok"
        ]
        task_min = min(task_values)
        task_max = max(task_values)
        task_span = task_max - task_min
        task_padding = max(0.002, 0.08 * task_span)
        task_limits = (task_min - task_padding, task_max + task_padding)
        for endpoint_index, (endpoint, endpoint_label) in enumerate(endpoints):
            axis = axes[task_index, endpoint_index]
            selected = [
                row
                for row in rows
                if row["task"] == task
                and row["endpoint"] == endpoint
                and row["status"] == "ok"
            ]
            by_seed: dict[int, dict[str, float]] = defaultdict(dict)
            for row in selected:
                by_seed[int(row["seed"])][row["variant"]] = float(row["official_metric"])
            for _seed, values in sorted(by_seed.items()):
                if set(values) != set(variants):
                    continue
                axis.plot(
                    range(len(variants)),
                    [values[variant] for variant in variants],
                    color="#8C8C8C",
                    linewidth=0.7,
                    alpha=0.55,
                    zorder=1,
                )
                for variant_index, variant in enumerate(variants):
                    axis.scatter(
                        variant_index,
                        values[variant],
                        color=PLOT_COLORS[variant],
                        marker=PLOT_MARKERS[variant],
                        edgecolor="white",
                        linewidth=0.35,
                        s=27,
                        zorder=2,
                    )
            for variant_index, variant in enumerate(variants):
                values = [
                    seed_values[variant]
                    for seed_values in by_seed.values()
                    if variant in seed_values
                ]
                if values:
                    axis.scatter(
                        variant_index,
                        float(np.mean(values)),
                        color="black",
                        marker="D",
                        edgecolor="white",
                        linewidth=0.45,
                        s=34,
                        zorder=3,
                    )
            axis.set_xticks(range(len(variants)), variants)
            axis.set_ylabel(metric_label if endpoint_index == 0 else "")
            task_label = TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").title())
            axis.set_title(f"{task_label}\n{endpoint_label}", fontsize=9)
            axis.set_ylim(*task_limits)
            axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, figure_dir / "figure_training_paired_endpoints")


def build_training_outputs(
    artifact_dir: Path,
    table_dir: Path,
    figure_dir: Path,
    protocol: dict[str, Any],
) -> None:
    """Build the E09/E10 training tables and development-curve/endpoint figures."""

    training = _training_summaries(artifact_dir)
    if not training:
        raise FileNotFoundError(f"no training summaries found below {artifact_dir}")
    _write_csv(table_dir / "training_runs.csv", training)
    _write_csv(
        table_dir / "training_variant_summary.csv",
        training_variant_summary(training, protocol),
    )
    _write_csv(
        table_dir / "paired_quality_summary.csv",
        paired_quality_summary(training, protocol),
    )
    _write_csv(
        table_dir / "training_iou_variant_summary.csv",
        training_iou_variant_summary(training, protocol),
    )
    _write_csv(
        table_dir / "paired_iou_summary.csv",
        paired_iou_summary(training, protocol),
    )
    _write_csv(
        table_dir / "paired_training_efficiency_summary.csv",
        paired_training_efficiency_summary(training, protocol),
    )
    development_curves = _development_curves(artifact_dir)
    if development_curves:
        plot_training_quality(development_curves, figure_dir, protocol)
    plot_paired_training_endpoints(training, figure_dir, protocol)


def build_checkpoint_outputs(
    artifact_dir: Path,
    table_dir: Path,
    protocol: dict[str, Any],
) -> None:
    """Build provenance-complete E06 tables without running model code."""

    rows = checkpoint_rows(artifact_dir, "e06_compatibility.jsonl")
    if not rows:
        raise FileNotFoundError(
            f"no checkpoint compatibility rows found below {artifact_dir}"
        )
    repository_root = Path(__file__).resolve().parents[3]
    checkpoint_config = load_yaml(
        repository_root / "configs" / "checkpoints" / "e06_e07.yaml"
    )
    attempts = checkpoint_attempts(artifact_dir)
    audit = trained_checkpoint_campaign_audit(rows, attempts, protocol, checkpoint_config)
    _write_csv(
        table_dir / "checkpoint_compatibility_summary.csv",
        checkpoint_compatibility_summary(rows),
    )
    if audit:
        _write_csv(table_dir / "trained_checkpoint_campaign_audit.csv", audit)
    complete_groups = {
        (row["source_checkpoint_experiment"], row["task"])
        for row in audit
        if row["status"] == "complete"
    }
    complete_rows = [
        row
        for row in rows
        if (row.get("source_checkpoint_experiment"), row["task"]) in complete_groups
    ]
    if not complete_rows:
        return
    task_agreement = trained_checkpoint_task_agreement(complete_rows, protocol)
    _write_csv(table_dir / "trained_checkpoint_task_agreement.csv", task_agreement)
    _write_csv(
        table_dir / "trained_checkpoint_compatibility_aggregate.csv",
        trained_checkpoint_compatibility_aggregate(task_agreement, complete_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--training-only",
        action="store_true",
        help="build only the E09/E10 training tables and figures",
    )
    mode.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="build only the E06 checkpoint-replay tables and completeness audit",
    )
    arguments = parser.parse_args()
    protocol = load_yaml(Path(__file__).resolve().parents[3] / "protocol.yaml")
    if not arguments.checkpoint_only:
        _style()
        build_training_outputs(
            arguments.artifact_dir,
            arguments.table_dir,
            arguments.figure_dir,
            protocol,
        )
    if not arguments.training_only:
        build_checkpoint_outputs(arguments.artifact_dir, arguments.table_dir, protocol)


if __name__ == "__main__":
    main()
