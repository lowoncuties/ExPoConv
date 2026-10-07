"""Deterministic cohort-keyed statistics for the E20/E21 extension."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from poincare_exec.analysis.training_extension_types import (
    COMMON_METADATA_FIELDS,
    IOU_METRIC_FIELD,
    METRIC_NAMES,
    PRIMARY_METRIC_FIELDS,
    REGISTERED_ENDPOINTS,
    TEST_PARTITIONS,
    CohortSpec,
    ExtensionAnalysisError,
    ValidatedRun,
)
from poincare_exec.config.loader import canonical_json


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtensionAnalysisError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ExtensionAnalysisError(f"{name} must be a sequence")
    return value


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ExtensionAnalysisError(f"{context}: boolean is not a metric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ExtensionAnalysisError(f"{context}: expected a numeric value") from error
    if not math.isfinite(result):
        raise ExtensionAnalysisError(f"{context}: nonfinite value")
    return result


def _endpoint(run: ValidatedRun, endpoint: str) -> Mapping[str, Any]:
    if endpoint == "fixed_final_update":
        return {
            "update": run.summary["updates"],
            "official_test": run.summary["official_test"],
            "deduplicated_test": run.summary["deduplicated_test"],
        }
    if endpoint == "validation_best":
        return _mapping(
            run.summary["validation_best"], name=f"{run.summary_path} validation_best"
        )
    raise AssertionError(endpoint)


def _bootstrap_generator(base_seed: int, *identity: str) -> tuple[np.random.Generator, str]:
    payload = canonical_json(
        {"base_seed": int(base_seed), "identity": list(identity)}
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return np.random.default_rng(int(digest[:16], 16)), digest


def _paired_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ExtensionAnalysisError("paired inference requires at least two pairs")
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(len(values))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    mean = float(np.mean(values))
    return (
        sample_sd,
        standard_error,
        mean - critical * standard_error,
        mean + critical * standard_error,
    )


def _sign_pvalue(values: np.ndarray) -> float:
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


def _bootstrap_settings(manifest: Mapping[str, Any]) -> tuple[int, int, str]:
    bootstrap = _mapping(manifest.get("bootstrap"), name="bootstrap")
    seed = int(bootstrap.get("base_seed", -1))
    replicates = int(bootstrap.get("replicates", -1))
    quantile_method = str(bootstrap.get("quantile_method", ""))
    if seed < 0 or replicates < 1:
        raise ValueError("bootstrap seed and positive replicate count are required")
    if quantile_method != "linear":
        raise ValueError("bootstrap.quantile_method must be frozen as 'linear'")
    return seed, replicates, quantile_method


def common_metadata(cohort: CohortSpec) -> dict[str, Any]:
    return {
        "analysis_id": cohort.analysis_id,
        "experiment_id": cohort.experiment_id,
        "cohort_role": cohort.role,
        "platform_stratum": cohort.platform_stratum,
        "task": cohort.task,
        "dataset": cohort.dataset,
        "protocol_version": cohort.protocol_version,
        "config_hash": cohort.config_hash,
        "runtime": cohort.runtime,
        "physical_gpu_host": cohort.physical_device.get("host"),
        "physical_gpu_index": cohort.physical_device.get("physical_index"),
        "physical_gpu_uuid": cohort.physical_device.get("uuid"),
        "physical_gpu_name": cohort.physical_device.get("name"),
    }


def _copy_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in COMMON_METADATA_FIELDS}


def _pair_provenance(reference: ValidatedRun, candidate: ValidatedRun) -> dict[str, Any]:
    return {
        "reference_code_commit": reference.summary["code_commit"],
        "candidate_code_commit": candidate.summary["code_commit"],
        "reference_device_id": reference.summary["device_id"],
        "candidate_device_id": candidate.summary["device_id"],
        "reference_summary_path": str(reference.summary_path),
        "candidate_summary_path": str(candidate.summary_path),
    }


def _endpoint_metric(
    run: ValidatedRun, endpoint: str, partition: str, metric: str
) -> tuple[float, int]:
    endpoint_value = _endpoint(run, endpoint)
    partition_value = _mapping(
        endpoint_value[partition], name=f"{run.summary_path} {endpoint}/{partition}"
    )
    value = _finite_float(
        partition_value.get(metric),
        context=f"{run.summary_path} {endpoint}/{partition}/{metric}",
    )
    return value, int(endpoint_value["update"])


def _aggregate_provenance(runs: Sequence[ValidatedRun]) -> dict[str, str]:
    return {
        "code_commits": canonical_json(
            sorted({str(run.summary["code_commit"]) for run in runs})
        ),
        "device_ids": canonical_json(
            sorted({str(run.summary["device_id"]) for run in runs})
        ),
        "summary_paths": canonical_json(sorted(str(run.summary_path) for run in runs)),
    }


def quality_rows(
    manifest: Mapping[str, Any],
    validated: Mapping[str, Mapping[tuple[str, int], ValidatedRun]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bootstrap_seed, replicates, quantile_method = _bootstrap_settings(manifest)
    per_seed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for experiment_id, run_map in sorted(validated.items()):
        cohort = next(iter(run_map.values())).cohort
        reference, candidate = cohort.analysis_variants
        metrics = [PRIMARY_METRIC_FIELDS[cohort.task]]
        if cohort.task == "isic16_poincare_unet":
            metrics.append(IOU_METRIC_FIELD)
        for endpoint in REGISTERED_ENDPOINTS:
            for partition_name, partition_field in TEST_PARTITIONS.items():
                for metric in metrics:
                    registered_metric = METRIC_NAMES[metric]
                    differences = []
                    selected_runs = []
                    for seed in sorted(cohort.seeds):
                        reference_run = run_map[(reference, seed)]
                        candidate_run = run_map[(candidate, seed)]
                        reference_value, reference_update = _endpoint_metric(
                            reference_run, endpoint, partition_field, metric
                        )
                        candidate_value, candidate_update = _endpoint_metric(
                            candidate_run, endpoint, partition_field, metric
                        )
                        difference = candidate_value - reference_value
                        differences.append(difference)
                        selected_runs.extend((reference_run, candidate_run))
                        per_seed.append(
                            {
                                **common_metadata(cohort),
                                "endpoint": endpoint,
                                "test_partition": partition_name,
                                "metric_name": registered_metric,
                                "metric_field": metric,
                                "comparison": "F_minus_R",
                                "reference_variant": reference,
                                "candidate_variant": candidate,
                                "seed": seed,
                                "reference_value": reference_value,
                                "candidate_value": candidate_value,
                                "candidate_minus_reference": difference,
                                "reference_checkpoint_update": reference_update,
                                "candidate_checkpoint_update": candidate_update,
                                **_pair_provenance(reference_run, candidate_run),
                            }
                        )
                    values = np.asarray(differences, dtype=np.float64)
                    sample_sd, standard_error, lower, upper = _paired_interval(values)
                    random, stream = _bootstrap_generator(
                        bootstrap_seed,
                        str(manifest["analysis_id"]),
                        experiment_id,
                        cohort.task,
                        endpoint,
                        partition_name,
                        registered_metric,
                        "F_minus_R",
                        reference,
                        candidate,
                        "not_applicable",
                    )
                    indices = random.integers(
                        0, len(values), size=(replicates, len(values))
                    )
                    bootstraps = np.mean(values[indices], axis=1)
                    aggregate.append(
                        {
                            **common_metadata(cohort),
                            "endpoint": endpoint,
                            "test_partition": partition_name,
                            "metric_name": registered_metric,
                            "metric_field": metric,
                            "comparison": "F_minus_R",
                            "reference_variant": reference,
                            "candidate_variant": candidate,
                            "pairs": len(values),
                            "estimand": (
                                f"mean paired {candidate}-minus-{reference} "
                                f"{registered_metric}"
                            ),
                            "mean_difference": float(np.mean(values)),
                            "median_difference": float(np.median(values)),
                            "difference_sample_sd": sample_sd,
                            "difference_standard_error": standard_error,
                            "paired_t_95_lower": lower,
                            "paired_t_95_upper": upper,
                            "paired_t_degrees_of_freedom": len(values) - 1,
                            "exact_two_sided_sign_pvalue": _sign_pvalue(values),
                            "bootstrap_2.5_percentile": float(
                                np.quantile(bootstraps, 0.025, method=quantile_method)
                            ),
                            "bootstrap_97.5_percentile": float(
                                np.quantile(bootstraps, 0.975, method=quantile_method)
                            ),
                            "bootstrap_base_seed": bootstrap_seed,
                            "bootstrap_stream_sha256": stream,
                            "bootstrap_replicates": replicates,
                            "bootstrap_quantile_method": quantile_method,
                            "interval_adjustment": "none",
                            "raw_differences_by_seed": canonical_json(
                                dict(
                                    zip(
                                        map(str, sorted(cohort.seeds)),
                                        differences,
                                        strict=True,
                                    )
                                )
                            ),
                            **_aggregate_provenance(selected_runs),
                        }
                    )
    return per_seed, aggregate


def efficiency_rows(
    manifest: Mapping[str, Any],
    validated: Mapping[str, Mapping[tuple[str, int], ValidatedRun]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bootstrap_seed, replicates, quantile_method = _bootstrap_settings(manifest)
    per_seed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for experiment_id, run_map in sorted(validated.items()):
        cohort = next(iter(run_map.values())).cohort
        reference, candidate = cohort.analysis_variants
        for measurement in ("compute_seconds", "elapsed_seconds"):
            ratios = []
            selected_runs = []
            for seed in sorted(cohort.seeds):
                reference_run = run_map[(reference, seed)]
                candidate_run = run_map[(candidate, seed)]
                reference_value = _finite_float(
                    reference_run.summary.get(measurement),
                    context=f"{reference_run.summary_path} {measurement}",
                )
                candidate_value = _finite_float(
                    candidate_run.summary.get(measurement),
                    context=f"{candidate_run.summary_path} {measurement}",
                )
                if min(reference_value, candidate_value) <= 0:
                    raise ExtensionAnalysisError(
                        f"{experiment_id} seed {seed}: {measurement} must be positive"
                    )
                ratio = reference_value / candidate_value
                ratios.append(ratio)
                selected_runs.extend((reference_run, candidate_run))
                per_seed.append(
                    {
                        **common_metadata(cohort),
                        "measurement": measurement,
                        "comparison": "R_over_F",
                        "reference_variant": reference,
                        "candidate_variant": candidate,
                        "seed": seed,
                        "reference_seconds": reference_value,
                        "candidate_seconds": candidate_value,
                        "reference_over_candidate": ratio,
                        **_pair_provenance(reference_run, candidate_run),
                    }
                )
            values = np.asarray(ratios, dtype=np.float64)
            log_values = np.log(values)
            log_sd, log_se, log_lower, log_upper = _paired_interval(log_values)
            random, stream = _bootstrap_generator(
                bootstrap_seed,
                str(manifest["analysis_id"]),
                experiment_id,
                cohort.task,
                "fixed_final_update",
                "not_applicable",
                "not_applicable",
                "R_over_F",
                reference,
                candidate,
                measurement,
            )
            indices = random.integers(0, len(values), size=(replicates, len(values)))
            bootstraps = np.exp(np.mean(log_values[indices], axis=1))
            aggregate.append(
                {
                    **common_metadata(cohort),
                    "measurement": measurement,
                    "comparison": "R_over_F",
                    "reference_variant": reference,
                    "candidate_variant": candidate,
                    "pairs": len(values),
                    "estimand": (
                        "geometric mean paired reference/candidate ratio; "
                        "values above one favor F"
                    ),
                    "geometric_mean_speedup": float(np.exp(np.mean(log_values))),
                    "median_speedup": float(np.median(values)),
                    "log_speedup_sample_sd": log_sd,
                    "log_speedup_standard_error": log_se,
                    "log_t_95_lower": float(np.exp(log_lower)),
                    "log_t_95_upper": float(np.exp(log_upper)),
                    "log_t_degrees_of_freedom": len(values) - 1,
                    "exact_two_sided_sign_pvalue": _sign_pvalue(log_values),
                    "bootstrap_2.5_percentile": float(
                        np.quantile(bootstraps, 0.025, method=quantile_method)
                    ),
                    "bootstrap_97.5_percentile": float(
                        np.quantile(bootstraps, 0.975, method=quantile_method)
                    ),
                    "bootstrap_base_seed": bootstrap_seed,
                    "bootstrap_stream_sha256": stream,
                    "bootstrap_replicates": replicates,
                    "bootstrap_quantile_method": quantile_method,
                    "interval_adjustment": "none",
                    "raw_ratios_by_seed": canonical_json(
                        dict(zip(map(str, sorted(cohort.seeds)), ratios, strict=True))
                    ),
                    **_aggregate_provenance(selected_runs),
                }
            )
    return per_seed, aggregate


def endpoint_rows(
    validated: Mapping[str, Mapping[tuple[str, int], ValidatedRun]],
    extension_ids: frozenset[str],
) -> list[dict[str, Any]]:
    rows = []
    for experiment_id, run_map in sorted(validated.items()):
        if experiment_id not in extension_ids:
            continue
        cohort = next(iter(run_map.values())).cohort
        primary_field = PRIMARY_METRIC_FIELDS[cohort.task]
        for (variant, seed), run in sorted(run_map.items()):
            for endpoint in REGISTERED_ENDPOINTS:
                endpoint_value = _endpoint(run, endpoint)
                for partition_name, partition_field in TEST_PARTITIONS.items():
                    partition = _mapping(
                        endpoint_value[partition_field],
                        name=f"{run.summary_path} {endpoint}/{partition_field}",
                    )
                    nonfinite = int(partition.get("output_nonfinite_count", 0))
                    if nonfinite:
                        raise ExtensionAnalysisError(
                            f"{run.summary_path}: nonfinite endpoint output count "
                            f"{nonfinite}"
                        )
                    rows.append(
                        {
                            **common_metadata(cohort),
                            "variant": variant,
                            "seed": seed,
                            "endpoint": endpoint,
                            "test_partition": partition_name,
                            "checkpoint_update": int(endpoint_value["update"]),
                            "primary_metric_name": METRIC_NAMES[primary_field],
                            "primary_metric": _finite_float(
                                partition.get(primary_field),
                                context=(
                                    f"{run.summary_path} {endpoint}/{partition_field}/"
                                    f"{primary_field}"
                                ),
                            ),
                            "mean_per_image_foreground_iou": (
                                _finite_float(
                                    partition.get(IOU_METRIC_FIELD),
                                    context=(
                                        f"{run.summary_path} {endpoint}/{partition_field}/"
                                        f"{IOU_METRIC_FIELD}"
                                    ),
                                )
                                if cohort.task == "isic16_poincare_unet"
                                else None
                            ),
                            "loss": _finite_float(
                                partition.get("loss"),
                                context=(
                                    f"{run.summary_path} {endpoint}/{partition_field}/loss"
                                ),
                            ),
                            "output_nonfinite_count": nonfinite,
                            "compute_seconds": _finite_float(
                                run.summary.get("compute_seconds"),
                                context=f"{run.summary_path} compute_seconds",
                            ),
                            "elapsed_seconds": _finite_float(
                                run.summary.get("elapsed_seconds"),
                                context=f"{run.summary_path} elapsed_seconds",
                            ),
                            "best_development_metric": _finite_float(
                                run.summary.get("best_development_metric"),
                                context=f"{run.summary_path} best_development_metric",
                            ),
                            "best_update": int(run.summary["best_update"]),
                            "code_commit": run.summary["code_commit"],
                            "code_dirty": run.summary["code_dirty"],
                            "device_id": run.summary["device_id"],
                            "summary_path": str(run.summary_path),
                        }
                    )
    return rows


def variant_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        metrics = [str(row["primary_metric_name"])]
        if row["mean_per_image_foreground_iou"] is not None:
            metrics.append("mean_per_image_foreground_iou")
        for metric in metrics:
            key = (
                str(row["experiment_id"]),
                str(row["endpoint"]),
                str(row["test_partition"]),
                str(row["variant"]),
                metric,
            )
            grouped.setdefault(key, []).append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        experiment_id, endpoint, partition, variant, metric = key
        seeds = [int(row["seed"]) for row in selected]
        if len(seeds) != len(set(seeds)):
            raise ExtensionAnalysisError(
                f"{experiment_id}: duplicate seeds in variant summary {key}"
            )
        field = (
            "mean_per_image_foreground_iou"
            if metric == "mean_per_image_foreground_iou"
            else "primary_metric"
        )
        values = [_finite_float(row[field], context=f"{key} metric") for row in selected]
        losses = [_finite_float(row["loss"], context=f"{key} loss") for row in selected]
        output.append(
            {
                **_copy_metadata(selected[0]),
                "endpoint": endpoint,
                "test_partition": partition,
                "variant": variant,
                "seeds": len(seeds),
                "metric_name": metric,
                "metric_mean": statistics.mean(values),
                "metric_sample_sd": statistics.stdev(values),
                "metric_median": statistics.median(values),
                "metric_minimum": min(values),
                "metric_maximum": max(values),
                "loss_mean": statistics.mean(losses),
                "loss_sample_sd": statistics.stdev(losses),
                "raw_metrics_by_seed": canonical_json(
                    {
                        str(row["seed"]): float(row[field])
                        for row in sorted(selected, key=lambda item: int(item["seed"]))
                    }
                ),
                "code_commits": canonical_json(
                    sorted({str(row["code_commit"]) for row in selected})
                ),
                "device_ids": canonical_json(
                    sorted({str(row["device_id"]) for row in selected})
                ),
            }
        )
    return output


def platform_rows(
    quality: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    strata = _mapping(
        _mapping(
            manifest.get("cross_platform_reporting"), name="cross_platform_reporting"
        ).get("task_strata"),
        name="cross_platform_reporting.task_strata",
    )
    expected = {
        str(experiment_id)
        for experiment_ids in strata.values()
        for experiment_id in _sequence(experiment_ids, name="cross-platform task stratum")
    }
    rows = []
    for row in quality:
        primary_metric = METRIC_NAMES[PRIMARY_METRIC_FIELDS[str(row["task"])]]
        if not (
            row["experiment_id"] in expected
            and row["endpoint"] == "fixed_final_update"
            and row["test_partition"] == "official_test"
            and row["metric_name"] == primary_metric
        ):
            continue
        rows.append(
            {
                **_copy_metadata(row),
                "analysis_scope": "within_platform_no_grand_pool",
                "result_family": "quality",
                "code_commits": row["code_commits"],
                "device_ids": row["device_ids"],
                "endpoint": row["endpoint"],
                "test_partition": row["test_partition"],
                "outcome": row["metric_name"],
                "comparison": row["comparison"],
                "pairs": row["pairs"],
                "estimate": row["mean_difference"],
                "interval_method": "paired_t",
                "interval_95_lower": row["paired_t_95_lower"],
                "interval_95_upper": row["paired_t_95_upper"],
                "grand_pool_performed": False,
            }
        )
    if {str(row["experiment_id"]) for row in rows} != expected:
        raise ExtensionAnalysisError("platform strata are incomplete")
    if len(rows) != len(expected):
        raise ExtensionAnalysisError("platform strata contain duplicate experiment rows")
    return rows
