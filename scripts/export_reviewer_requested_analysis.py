#!/usr/bin/env python3
"""Export reviewer-requested curvature, equivalence, and runtime evidence."""

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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from poincare_exec.analysis.reviewer_requested import (
    paired_mean_interval,
    paired_tost,
)

ANALYSIS_STATUS = "post_hoc_exploratory_not_prespecified"
VARIANTS = ("R", "F")
TOST_MARGINS = {
    "isic16_poincare_unet": ("mean_foreground_dice", 0.01, 1.0),
    "cifar100_poincare_resnet20": ("top1_accuracy", 0.005, 0.5),
}
OUTPUT_FILES = (
    "isic16_curvature_trajectories.csv",
    "isic16_curvature_finals_by_seed.csv",
    "isic16_curvature_final_summary.csv",
    "quality_tost.csv",
    "training_runtime_minutes_by_seed.csv",
    "training_runtime_minutes_summary.csv",
    "elementwise_relative_error_status.json",
    "analysis_summary.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-training-runs",
        type=Path,
        default=Path("results/training/training_runs.csv"),
    )
    parser.add_argument(
        "--replication-training-runs",
        type=Path,
        default=Path("results/e20_e21_training_replication/e20_e21_training_runs.csv"),
    )
    parser.add_argument(
        "--replication-run-audit",
        type=Path,
        default=Path("results/e20_e21_training_replication/e20_e21_run_audit.csv"),
    )
    parser.add_argument(
        "--e09-training-root",
        type=Path,
        required=True,
        help="Root containing accepted E09 R/ and F/ seed directories.",
    )
    parser.add_argument(
        "--e20-training-root",
        type=Path,
        required=True,
        help="Root containing accepted E20 R/ and F/ seed directories.",
    )
    parser.add_argument(
        "--accepted-structured-results",
        type=Path,
        required=True,
        help="Accepted post-repair E02 structured JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reviewer_requested"),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _finite(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected a numeric value") from error
    if not math.isfinite(result):
        raise ValueError(f"{context}: nonfinite value")
    return result


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    fields = list(materialized[0])
    if any(list(row) != fields for row in materialized):
        raise ValueError(f"inconsistent fields for {path.name}")
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


def _consistent(values: Sequence[Any], *, context: str) -> Any:
    unique = {str(value) for value in values}
    if len(unique) != 1:
        raise ValueError(f"{context}: inconsistent values {sorted(unique)}")
    return values[0]


def _run_index(
    legacy_rows: Sequence[Mapping[str, str]],
    replication_rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in [*legacy_rows, *replication_rows]:
        variant = row["variant"]
        if variant not in VARIANTS:
            continue
        grouped[(row["experiment_id"], variant, int(row["seed"]))].append(row)
    index = {}
    for key, selected in sorted(grouped.items()):
        experiment_id, variant, seed = key
        task = str(_consistent([row["task"] for row in selected], context=str(key)))
        compute = _finite(
            _consistent([row["compute_seconds"] for row in selected], context=str(key)),
            context=f"{key} compute_seconds",
        )
        elapsed = _finite(
            _consistent([row["elapsed_seconds"] for row in selected], context=str(key)),
            context=f"{key} elapsed_seconds",
        )
        if not 0.0 < compute <= elapsed:
            raise ValueError(f"{key}: invalid compute/elapsed timing")
        if any(row.get("status", "ok") != "ok" for row in selected):
            raise ValueError(f"{key}: non-ok run in accepted table")
        platform = selected[0].get("platform_stratum") or selected[0].get("device_id") or ""
        index[key] = {
            "experiment_id": experiment_id,
            "task": task,
            "platform_stratum": platform,
            "variant": variant,
            "seed": seed,
            "protocol_version": _consistent(
                [row["protocol_version"] for row in selected], context=str(key)
            ),
            "config_hash": _consistent(
                [row["config_hash"] for row in selected], context=str(key)
            ),
            "code_commit": _consistent(
                [row["code_commit"] for row in selected], context=str(key)
            ),
            "device_id": _consistent(
                [row["device_id"] for row in selected], context=str(key)
            ),
            "compute_seconds": compute,
            "elapsed_seconds": elapsed,
        }
    expected = {
        (experiment, variant)
        for experiment in ("E09", "E10", "E20", "E21")
        for variant in VARIANTS
    }
    actual = {(key[0], key[1]) for key in index}
    if actual != expected:
        raise ValueError(f"run-index cohort mismatch: missing={sorted(expected - actual)}")
    return index


def _quality_observations(
    legacy_rows: Sequence[Mapping[str, str]],
    replication_rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, str, str, str, int], float]:
    values: dict[tuple[str, str, str, str, str, int], float] = {}
    for row in legacy_rows:
        if row["variant"] not in VARIANTS or row.get("status") != "ok":
            continue
        metric = row["official_metric_name"]
        for partition, field in (
            ("official_test", "official_metric"),
            ("official_test_excluding_exact_training_duplicates", "deduplicated_metric"),
        ):
            key = (
                row["experiment_id"],
                row["endpoint"],
                partition,
                metric,
                row["variant"],
                int(row["seed"]),
            )
            values[key] = _finite(row[field], context=str(key))
    for row in replication_rows:
        if row["variant"] not in VARIANTS:
            continue
        key = (
            row["experiment_id"],
            row["endpoint"],
            row["test_partition"],
            row["primary_metric_name"],
            row["variant"],
            int(row["seed"]),
        )
        values[key] = _finite(row["primary_metric"], context=str(key))
    return values


def _tost_rows(
    legacy_rows: Sequence[Mapping[str, str]],
    replication_rows: Sequence[Mapping[str, str]],
    run_index: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations = _quality_observations(legacy_rows, replication_rows)
    groups = sorted({key[:4] for key in observations})
    output = []
    for experiment_id, endpoint, partition, observed_metric in groups:
        task = next(
            value["task"] for key, value in run_index.items() if key[0] == experiment_id
        )
        metric, margin, percentage_points = TOST_MARGINS[str(task)]
        aliases = {"mean_foreground_dice", "mean_per_image_foreground_dice"}
        if observed_metric != metric and not (
            metric == "mean_foreground_dice" and observed_metric in aliases
        ):
            continue
        seeds_by_variant = {
            variant: sorted(
                key[5]
                for key in observations
                if key[:4] == (experiment_id, endpoint, partition, observed_metric)
                and key[4] == variant
            )
            for variant in VARIANTS
        }
        if seeds_by_variant["R"] != seeds_by_variant["F"]:
            raise ValueError(f"unpaired quality seeds for {groups}")
        seeds = seeds_by_variant["R"]
        differences = [
            observations[(experiment_id, endpoint, partition, observed_metric, "F", seed)]
            - observations[(experiment_id, endpoint, partition, observed_metric, "R", seed)]
            for seed in seeds
        ]
        representative = run_index[(experiment_id, "R", seeds[0])]
        output.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "experiment_id": experiment_id,
                "platform_stratum": representative["platform_stratum"],
                "task": task,
                "endpoint": endpoint,
                "test_partition": partition,
                "metric_name": observed_metric,
                "comparison": "F_minus_R",
                "margin_native_scale": margin,
                "margin_percentage_points": percentage_points,
                "margin_symmetry": "[-margin,+margin]",
                **paired_tost(differences, margin),
                "raw_differences_by_seed": json.dumps(
                    dict(zip(map(str, seeds), differences, strict=True)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return output


def _runtime_rows(
    run_index: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired = []
    for experiment_id in ("E09", "E10", "E20", "E21"):
        seeds = sorted(
            key[2] for key in run_index if key[0] == experiment_id and key[1] == "R"
        )
        if seeds != sorted(
            key[2] for key in run_index if key[0] == experiment_id and key[1] == "F"
        ):
            raise ValueError(f"unpaired runtime seeds for {experiment_id}")
        for seed in seeds:
            reference = run_index[(experiment_id, "R", seed)]
            candidate = run_index[(experiment_id, "F", seed)]
            paired.append(
                {
                    "experiment_id": experiment_id,
                    "platform_stratum": reference["platform_stratum"],
                    "task": reference["task"],
                    "seed": seed,
                    "reference_variant": "R",
                    "candidate_variant": "F",
                    "reference_compute_minutes": reference["compute_seconds"] / 60.0,
                    "candidate_compute_minutes": candidate["compute_seconds"] / 60.0,
                    "reference_elapsed_wall_clock_minutes": (
                        reference["elapsed_seconds"] / 60.0
                    ),
                    "candidate_elapsed_wall_clock_minutes": (
                        candidate["elapsed_seconds"] / 60.0
                    ),
                    "compute_speedup_R_over_F": (
                        reference["compute_seconds"] / candidate["compute_seconds"]
                    ),
                    "elapsed_speedup_R_over_F": (
                        reference["elapsed_seconds"] / candidate["elapsed_seconds"]
                    ),
                    "reference_code_commit": reference["code_commit"],
                    "candidate_code_commit": candidate["code_commit"],
                    "reference_device_id": reference["device_id"],
                    "candidate_device_id": candidate["device_id"],
                }
            )
    summaries = []
    for experiment_id in ("E09", "E10", "E20", "E21"):
        for variant in VARIANTS:
            selected = [
                value
                for key, value in run_index.items()
                if key[0] == experiment_id and key[1] == variant
            ]
            compute = [value["compute_seconds"] / 60.0 for value in selected]
            elapsed = [value["elapsed_seconds"] / 60.0 for value in selected]
            summaries.append(
                {
                    "experiment_id": experiment_id,
                    "platform_stratum": selected[0]["platform_stratum"],
                    "task": selected[0]["task"],
                    "variant": variant,
                    "runs": len(selected),
                    **_descriptive_minutes("compute", compute),
                    **_descriptive_minutes("elapsed_wall_clock", elapsed),
                    "code_commits": json.dumps(
                        sorted({value["code_commit"] for value in selected}),
                        separators=(",", ":"),
                    ),
                    "device_ids": json.dumps(
                        sorted({value["device_id"] for value in selected}),
                        separators=(",", ":"),
                    ),
                }
            )
    return paired, summaries


def _descriptive_minutes(prefix: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean_minutes": statistics.fmean(values),
        f"{prefix}_sample_sd_minutes": statistics.stdev(values),
        f"{prefix}_median_minutes": statistics.median(values),
        f"{prefix}_min_minutes": min(values),
        f"{prefix}_max_minutes": max(values),
    }


def _audit_metric_hashes(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int], str]:
    output = {}
    for row in rows:
        if (
            row.get("experiment_id") != "E20"
            or row.get("variant") not in VARIANTS
            or row.get("accepted_training_run") != "True"
        ):
            continue
        hashes = json.loads(row["supporting_sha256"])
        output[(row["variant"], int(row["seed"]))] = hashes["training_metrics"]
    return output


def _curvature_rows(
    run_index: Mapping[tuple[str, str, int], Mapping[str, Any]],
    e09_root: Path,
    e20_root: Path,
    replication_audit: Sequence[Mapping[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    audit_hashes = _audit_metric_hashes(replication_audit)
    trajectories = []
    source_hashes = {}
    final_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for experiment_id, root in (("E09", e09_root), ("E20", e20_root)):
        keys = sorted(key for key in run_index if key[0] == experiment_id)
        for key in keys:
            _, variant, seed = key
            metadata = run_index[key]
            path = root / variant / f"seed_{seed}" / "training_metrics.jsonl"
            source_hash = _sha256(path)
            locator = f"{experiment_id}/{variant}/seed_{seed}/training_metrics.jsonl"
            source_hashes[locator] = source_hash
            if experiment_id == "E20" and audit_hashes.get((variant, seed)) != source_hash:
                raise ValueError(f"{locator}: SHA-256 disagrees with accepted run audit")
            rows = _jsonl(path)
            development = []
            final_training = None
            for row_number, row in enumerate(rows, start=1):
                context = f"{locator}:{row_number}"
                expected = {
                    "experiment_id": experiment_id,
                    "task": metadata["task"],
                    "variant": variant,
                    "seed_or_session": seed,
                    "protocol_version": metadata["protocol_version"],
                    "config_hash": metadata["config_hash"],
                    "code_commit": metadata["code_commit"],
                    "device_id": metadata["device_id"],
                    "status": "ok",
                }
                mismatch = {
                    field: (row.get(field), value)
                    for field, value in expected.items()
                    if row.get(field) != value
                }
                if mismatch:
                    raise ValueError(f"{context}: identity mismatch {mismatch}")
                curvature = _finite(row.get("curvature"), context=f"{context} curvature")
                if curvature <= 0.0:
                    raise ValueError(f"{context}: curvature must be positive")
                if row.get("split") == "development":
                    development.append((int(row["epoch"]), int(row["update"]), curvature))
                elif row.get("split") == "train":
                    final_training = (int(row["update"]), curvature)
            if [epoch for epoch, _, _ in development] != list(range(1, 51)):
                raise ValueError(f"{locator}: expected one development record per epoch")
            if len({update for _, update, _ in development}) != 50:
                raise ValueError(f"{locator}: duplicate development updates")
            if final_training is None or final_training[0] != development[-1][1]:
                raise ValueError(f"{locator}: missing final training update")
            if final_training[1] != development[-1][2]:
                raise ValueError(f"{locator}: final train/development curvature mismatch")
            for epoch, update, curvature in development:
                trajectories.append(
                    {
                        "experiment_id": experiment_id,
                        "platform_stratum": metadata["platform_stratum"],
                        "variant": variant,
                        "seed": seed,
                        "epoch": epoch,
                        "update": update,
                        "curvature": curvature,
                        "trajectory_sampling": "end_of_epoch_development_record",
                        "source_locator": locator,
                        "source_sha256": source_hash,
                    }
                )
            final_lookup[key] = {
                "curvature": development[-1][2],
                "update": development[-1][1],
                "source_locator": locator,
                "source_sha256": source_hash,
            }

    finals = []
    summaries = []
    for experiment_id in ("E09", "E20"):
        seeds = sorted(
            key[2] for key in final_lookup if key[0] == experiment_id and key[1] == "R"
        )
        differences = []
        reference_values = []
        candidate_values = []
        for seed in seeds:
            reference = final_lookup[(experiment_id, "R", seed)]
            candidate = final_lookup[(experiment_id, "F", seed)]
            difference = candidate["curvature"] - reference["curvature"]
            differences.append(difference)
            reference_values.append(reference["curvature"])
            candidate_values.append(candidate["curvature"])
            metadata = run_index[(experiment_id, "R", seed)]
            finals.append(
                {
                    "experiment_id": experiment_id,
                    "platform_stratum": metadata["platform_stratum"],
                    "seed": seed,
                    "reference_variant": "R",
                    "candidate_variant": "F",
                    "final_update": reference["update"],
                    "reference_final_curvature": reference["curvature"],
                    "candidate_final_curvature": candidate["curvature"],
                    "candidate_minus_reference": difference,
                    "absolute_difference": abs(difference),
                    "reference_source_locator": reference["source_locator"],
                    "candidate_source_locator": candidate["source_locator"],
                    "reference_source_sha256": reference["source_sha256"],
                    "candidate_source_sha256": candidate["source_sha256"],
                }
            )
        metadata = run_index[(experiment_id, "R", seeds[0])]
        summaries.append(
            {
                "experiment_id": experiment_id,
                "platform_stratum": metadata["platform_stratum"],
                "pairs": len(seeds),
                "reference_mean_final_curvature": statistics.fmean(reference_values),
                "reference_sample_sd_final_curvature": statistics.stdev(reference_values),
                "reference_min_final_curvature": min(reference_values),
                "reference_max_final_curvature": max(reference_values),
                "candidate_mean_final_curvature": statistics.fmean(candidate_values),
                "candidate_sample_sd_final_curvature": statistics.stdev(candidate_values),
                "candidate_min_final_curvature": min(candidate_values),
                "candidate_max_final_curvature": max(candidate_values),
                **paired_mean_interval(differences),
                "max_absolute_paired_difference": max(map(abs, differences)),
                "raw_differences_by_seed": json.dumps(
                    dict(zip(map(str, seeds), differences, strict=True)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return trajectories, finals, summaries, source_hashes


def _elementwise_status(path: Path) -> dict[str, Any]:
    rows = _jsonl(path)
    agreements = [row for row in rows if row.get("record_type") == "agreement"]
    expected = {
        "elementwise_relative_error_elements",
        "elementwise_relative_error_p50",
        "elementwise_relative_error_p90",
        "elementwise_relative_error_p95",
        "elementwise_relative_error_p99",
        "elementwise_relative_error_p99_9",
        "elementwise_relative_error_max",
    }
    observed = set().union(*(row.keys() for row in agreements)) if agreements else set()
    available = expected <= observed
    return {
        "status": "available" if available else "rerun_required",
        "accepted_source_file": path.name,
        "accepted_source_sha256": _sha256(path),
        "accepted_source_rows": len(rows),
        "accepted_agreement_rows": len(agreements),
        "missing_elementwise_fields": sorted(expected - observed),
        "reason": (
            None
            if available
            else (
                "The accepted post-repair JSONL stores tensor-level relative L2, "
                "reference L2, and maximum absolute errors, but not reference and "
                "candidate elements. Quantiles cannot be reconstructed from those "
                "aggregates. The archived per-element NPZ is pre-repair and excluded."
            )
        ),
        "prospective_definition": (
            "abs(candidate-reference)/max(abs(reference),1e-12); report the "
            "fraction of denominators floored, linear p50/p90/p95/p99/p99.9, and max"
        ),
        "quantile_justification": (
            "p50 describes the center; p90/p95 the broad tail; p99/p99.9 the rare "
            "tail; max retains the worst element. Reporting the floored-denominator "
            "fraction exposes sensitivity to exact and near zeros."
        ),
        "required_fresh_command": (
            "./.venv/bin/poincare-exec structured-numerics --protocol protocol.yaml "
            "--config configs/numerics/e26_elementwise_agreement.yaml --output "
            "/fresh/path/e26_elementwise_agreement.jsonl"
        ),
        "overwrite_warning": (
            "E26 requires a fresh output path. Do not append E26 rows to the accepted "
            "E02 ce009c1 artifact or replace the accepted raw source."
        ),
    }


def export(
    *,
    legacy_training_runs: Path,
    replication_training_runs: Path,
    replication_run_audit: Path,
    e09_training_root: Path,
    e20_training_root: Path,
    accepted_structured_results: Path,
    output_dir: Path,
) -> dict[str, Any]:
    inputs = [
        legacy_training_runs,
        replication_training_runs,
        replication_run_audit,
        accepted_structured_results,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (e09_training_root, e20_training_root):
        if not path.is_dir():
            raise FileNotFoundError(path)

    legacy = _csv(legacy_training_runs)
    replication = _csv(replication_training_runs)
    audit = _csv(replication_run_audit)
    run_index = _run_index(legacy, replication)
    tost = _tost_rows(legacy, replication, run_index)
    runtime, runtime_summary = _runtime_rows(run_index)
    trajectories, finals, curvature_summary, curvature_hashes = _curvature_rows(
        run_index, e09_training_root, e20_training_root, audit
    )
    elementwise = _elementwise_status(accepted_structured_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / OUTPUT_FILES[0], trajectories)
    _atomic_csv(output_dir / OUTPUT_FILES[1], finals)
    _atomic_csv(output_dir / OUTPUT_FILES[2], curvature_summary)
    _atomic_csv(output_dir / OUTPUT_FILES[3], tost)
    _atomic_csv(output_dir / OUTPUT_FILES[4], runtime)
    _atomic_csv(output_dir / OUTPUT_FILES[5], runtime_summary)
    _atomic_json(output_dir / OUTPUT_FILES[6], elementwise)

    summary = {
        "schema_version": "1.0",
        "analysis_status": ANALYSIS_STATUS,
        "scope": {
            "curvature": "accepted E09 and E20 ISIC16 R/F runs; cohorts not pooled",
            "quality_tost": (
                "accepted E09/E10 and E20/E21 R/F runs; each cohort, endpoint, and "
                "test partition analyzed separately"
            ),
            "runtime": (
                "accepted E09/E10 and E20/E21 R/F runs; compute and complete worker "
                "elapsed wall-clock minutes retained separately"
            ),
            "elementwise_relative_error": elementwise["status"],
        },
        "tost_margins": {
            "isic16_mean_foreground_dice": {
                "native_scale": 0.01,
                "percentage_points": 1.0,
            },
            "cifar100_top1_accuracy": {
                "native_scale": 0.005,
                "percentage_points": 0.5,
            },
        },
        "source_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs
        },
        "curvature_source_sha256": dict(sorted(curvature_hashes.items())),
        "output_rows": {
            "isic16_curvature_trajectories.csv": len(trajectories),
            "isic16_curvature_finals_by_seed.csv": len(finals),
            "isic16_curvature_final_summary.csv": len(curvature_summary),
            "quality_tost.csv": len(tost),
            "training_runtime_minutes_by_seed.csv": len(runtime),
            "training_runtime_minutes_summary.csv": len(runtime_summary),
        },
        "scientific_behavior_change": False,
        "results_requiring_regeneration": (
            ["accepted post-repair E02 structured elementwise error table"]
            if elementwise["status"] == "rerun_required"
            else []
        ),
    }
    _atomic_json(output_dir / OUTPUT_FILES[7], summary)
    return summary


def main() -> int:
    arguments = _parser().parse_args()
    summary = export(
        legacy_training_runs=arguments.legacy_training_runs.resolve(),
        replication_training_runs=arguments.replication_training_runs.resolve(),
        replication_run_audit=arguments.replication_run_audit.resolve(),
        e09_training_root=arguments.e09_training_root.resolve(),
        e20_training_root=arguments.e20_training_root.resolve(),
        accepted_structured_results=arguments.accepted_structured_results.resolve(),
        output_dir=arguments.output_dir.resolve(),
    )
    print(json.dumps(summary["output_rows"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
