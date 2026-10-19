"""Bounded E02 validation for the audit-identified derivative contract."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from poincare_exec.artifacts import append_jsonl, base_result_row, write_json
from poincare_exec.config.loader import config_hash, load_yaml, require_keys
from poincare_exec.core.geometry import expmap0, logmap0, project_ball
from poincare_exec.core.hnnpp import (
    HNNPPParams,
    _extract_patches,
    _factorized_conv_scores,
    _reconstruct_scores,
    beta_ratio,
    hyperplane_scores_ball,
    scores_to_ball,
    scores_to_projected_tangent,
)
from poincare_exec.core.numerics import safe_norm
from poincare_exec.experiments.numerics import (
    _case_input,
    _case_parameters,
    _cast_tree,
    _evaluate,
    _finite_metrics,
    _groups,
)

Array = jax.Array


def _dtype(name: str) -> Any:
    if name == "float32":
        return jnp.float32
    if name == "float64":
        return jnp.float64
    raise ValueError(f"unsupported precision: {name}")


def _within_tolerance(
    metrics: dict[str, Any],
    tolerance: dict[str, float],
) -> bool:
    return bool(
        metrics["finite"]
        and (
            metrics["max_absolute_error"] <= tolerance["absolute"]
            or metrics["relative_l2_error"] <= tolerance["relative"]
        )
    )


def _score_output(
    scores: Array,
    *,
    curvature: Array,
    margin: float,
    variant: str,
) -> Array:
    if variant == "T":
        return scores_to_projected_tangent(scores, curvature, margin=margin)
    return logmap0(scores_to_ball(scores, curvature, margin=margin), curvature)


def _published_score_output(
    scores: Array,
    *,
    curvature: Array,
    margin: float,
) -> Array:
    """Evaluate the ordinary published materialized reconstruction directly."""

    sqrt_curvature = jnp.sqrt(curvature)
    sinh_scores = jnp.sinh(sqrt_curvature * scores)
    norm_square = jnp.sum(jnp.square(sinh_scores), axis=-1, keepdims=True)
    point = sinh_scores / (sqrt_curvature * (1.0 + jnp.sqrt(1.0 + norm_square)))
    return logmap0(project_ball(point, curvature, margin=margin), curvature)


def _sum_archived_norm(value: Array) -> Array:
    return jnp.sum(_archived_safe_norm(value))


def _sum_safe_norm(value: Array) -> Array:
    return jnp.sum(safe_norm(value, axis=-1, keepdims=True))


def _sum_projected_tangent(
    scores: Array,
    *,
    curvature: Array,
    margin: float,
) -> Array:
    return jnp.sum(scores_to_projected_tangent(scores, curvature, margin=margin))


def _row(
    protocol: dict[str, Any],
    experiment_id: str,
    suite_hash: str,
    *,
    variant: str,
    precision: str,
    case_id: str,
    **values: Any,
) -> dict[str, Any]:
    return {
        **base_result_row(
            protocol,
            experiment_id=experiment_id,
            variant=variant,
            configuration_hash=suite_hash,
            dtype=precision,
            seed_or_session=case_id,
        ),
        "case_id": case_id,
        **values,
    }


def _score_rows(
    protocol: dict[str, Any],
    config: dict[str, Any],
    suite_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    margin = float(config["fixed_float32_projection_margin"])
    for precision in config["score_reconstruction"]["precisions"]:
        dtype = _dtype(precision)
        tolerance = protocol["numerics"]["tolerances"][precision]
        for curvature_value in config["score_reconstruction"]["curvatures"]:
            curvature = jnp.asarray(curvature_value, dtype=dtype)
            for case in config["score_reconstruction"]["cases"]:
                scale = float(case["scale_by_precision"][precision])
                scores = scale * jnp.asarray(case["scores"], dtype=dtype)
                direction = jnp.asarray(case["direction"], dtype=dtype)
                expected = 0.5 * direction
                expected_output = _published_score_output(
                    scores,
                    curvature=curvature,
                    margin=margin,
                )
                for variant in ("R", "F", "T"):
                    function = partial(
                        _score_output,
                        curvature=curvature,
                        margin=margin,
                        variant=variant,
                    )
                    primal, derivative = jax.jvp(function, (scores,), (direction,))
                    jax.block_until_ready((primal, derivative))
                    output_metrics = _finite_metrics(expected_output, primal, 1.0e-12)
                    derivative_metrics = _finite_metrics(expected, derivative, 1.0e-12)
                    rows.append(
                        _row(
                            protocol,
                            config["experiment_id"],
                            suite_hash,
                            variant=variant,
                            precision=precision,
                            case_id=(f"score:{case['id']}:c={curvature_value}:{precision}"),
                            check_kind="score_reconstruction_jvp",
                            curvature=curvature_value,
                            projection_margin=margin,
                            expectation=case["expectation"],
                            score_scale=scale,
                            output_nonfinite_count=int(
                                np.size(primal) - np.isfinite(np.asarray(primal)).sum()
                            ),
                            derivative_nonfinite_count=int(
                                np.size(derivative)
                                - np.isfinite(np.asarray(derivative)).sum()
                            ),
                            output_within_tolerance=_within_tolerance(
                                output_metrics, tolerance
                            ),
                            derivative_within_tolerance=_within_tolerance(
                                derivative_metrics, tolerance
                            ),
                            within_tolerance=(
                                _within_tolerance(output_metrics, tolerance)
                                and _within_tolerance(derivative_metrics, tolerance)
                            ),
                            **{
                                f"output_{name}": value
                                for name, value in output_metrics.items()
                            },
                            **{
                                f"derivative_{name}": value
                                for name, value in derivative_metrics.items()
                            },
                        )
                    )
    return rows


def _projection_expected(
    point: np.ndarray,
    direction: np.ndarray,
    maximum: float,
) -> np.ndarray:
    radius = float(np.linalg.norm(point))
    if radius <= maximum:
        return direction
    unit = point / radius
    return (maximum / radius) * (direction - unit * float(unit @ direction))


def _projection_rows(
    protocol: dict[str, Any],
    config: dict[str, Any],
    suite_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    margin = float(config["fixed_float32_projection_margin"])
    exact_cap = float(config["projection"]["exact_cap_normalized_radius"])
    for precision in config["projection"]["precisions"]:
        dtype = _dtype(precision)
        tolerance = protocol["numerics"]["tolerances"][precision]
        for curvature_value in config["projection"]["curvatures"]:
            curvature = jnp.asarray(curvature_value, dtype=dtype)
            for normalized_radius in config["projection"]["normalized_radii"]:
                normalized = jnp.asarray(normalized_radius, dtype=dtype)
                maximum_array = (1.0 - jnp.asarray(margin, dtype=dtype)) / jnp.sqrt(
                    curvature
                )
                if normalized_radius == exact_cap:
                    first_coordinate = maximum_array
                else:
                    first_coordinate = normalized / jnp.sqrt(curvature)
                point = jnp.stack((first_coordinate, jnp.asarray(0.0, dtype=dtype)))
                point_np = np.asarray(point)
                maximum = float(np.asarray(maximum_array))
                for direction_case in config["projection"]["directions"]:
                    direction_np = np.asarray(
                        direction_case["value"], dtype=np.dtype(precision)
                    )
                    direction = jnp.asarray(direction_np, dtype=dtype)
                    function = partial(
                        project_ball,
                        curvature=curvature,
                        margin=margin,
                    )
                    primal, derivative = jax.jvp(function, (point,), (direction,))
                    jax.block_until_ready((primal, derivative))
                    expected = _projection_expected(point_np, direction_np, maximum)
                    metrics = _finite_metrics(expected, derivative, 1.0e-12)
                    rows.append(
                        _row(
                            protocol,
                            config["experiment_id"],
                            suite_hash,
                            variant="R/F/T-shared",
                            precision=precision,
                            case_id=(
                                f"projection:rho={normalized_radius}:"
                                f"{direction_case['id']}:c={curvature_value}:"
                                f"{precision}"
                            ),
                            check_kind="projection_jvp",
                            curvature=curvature_value,
                            normalized_radius=normalized_radius,
                            exact_cap_case=normalized_radius == exact_cap,
                            projection_margin=margin,
                            cap_normalized_radius=1.0 - margin,
                            direction=direction_case["id"],
                            threshold_convention=config["projection"][
                                "threshold_convention"
                            ],
                            output_nonfinite_count=int(
                                np.size(primal) - np.isfinite(np.asarray(primal)).sum()
                            ),
                            derivative_nonfinite_count=int(
                                np.size(derivative)
                                - np.isfinite(np.asarray(derivative)).sum()
                            ),
                            within_tolerance=_within_tolerance(metrics, tolerance),
                            **metrics,
                        )
                    )
    return rows


def _manifest_selection(
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> list[tuple[int, dict[str, Any], int]]:
    by_id = {case["case_id"]: (index, case) for index, case in enumerate(manifest["cases"])}
    selected = []
    for request in config["selected_manifest_cases"]:
        if request["case_id"] not in by_id:
            raise ValueError(f"unknown frozen case: {request['case_id']}")
        case_index, case = by_id[request["case_id"]]
        for realization in request["realizations"]:
            selected.append((case_index, case, int(realization)))
    return selected


def _selected_case_rows(
    protocol: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    suite_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(manifest["seed"])
    margin = float(config["fixed_float32_projection_margin"])
    norm_floor = float(protocol["numerics"]["norm_floor"])
    for case_index, case, realization in _manifest_selection(manifest, config):
        dtype = _dtype(case["precision"])
        key = jax.random.fold_in(
            jax.random.PRNGKey(seed), case_index * 10_000 + realization
        )
        params_key, input_key, cotangent_key = jax.random.split(key, 3)
        params = _case_parameters(case, params_key, dtype)
        tangent, realized_rho = _case_input(case, params, input_key, dtype)
        curvature = jnp.asarray(case["curvature"], dtype=dtype)
        output_shape = (
            case["batch"],
            case["height"],
            case["width"],
            case["output_channels"],
        )
        cotangent = jax.random.normal(cotangent_key, output_shape, dtype=dtype)
        evaluated = {
            variant: _evaluate(
                variant, params, tangent, curvature, cotangent, margin=margin
            )
            for variant in ("R", "F", "T")
        }
        diagnostics = _variant_numerical_diagnostics(
            case, params, tangent, curvature, margin
        )
        references = {"R_same_precision": evaluated["R"]}
        high_precision_evaluated: dict[str, tuple[Array, tuple[Any, Array, Array]]] = {}
        if case["precision"] == "float32":
            cast_params = _cast_tree(params, jnp.float64)
            cast_tangent = tangent.astype(jnp.float64)
            cast_curvature = curvature.astype(jnp.float64)
            cast_cotangent = cotangent.astype(jnp.float64)
            high_precision_evaluated = {
                variant: _evaluate(
                    variant,
                    cast_params,
                    cast_tangent,
                    cast_curvature,
                    cast_cotangent,
                    margin=margin,
                )
                for variant in ("R", "F", "T")
            }
            references["R_float64_fixed_float32_margin"] = high_precision_evaluated["R"]
            references["R_float64_dtype_default"] = _evaluate(
                "R", cast_params, cast_tangent, cast_curvature, cast_cotangent
            )
        tolerance = protocol["numerics"]["tolerances"][case["precision"]]
        for variant, candidate in evaluated.items():
            for reference_name, reference in references.items():
                if variant == "R" and reference_name == "R_same_precision":
                    continue
                for group_name, candidate_group in _groups(*candidate).items():
                    reference_group = _groups(*reference)[group_name]
                    metrics = _finite_metrics(reference_group, candidate_group, norm_floor)
                    rows.append(
                        _row(
                            protocol,
                            config["experiment_id"],
                            suite_hash,
                            variant=variant,
                            precision=case["precision"],
                            case_id=f"{case['case_id']}:{realization}",
                            check_kind="selected_frozen_case",
                            source_case_id=case["case_id"],
                            realization=realization,
                            variable_group=group_name,
                            comparison_reference=reference_name,
                            curvature=case["curvature"],
                            realized_normalized_radius=realized_rho,
                            candidate_projection_margin=margin,
                            reference_projection_margin=(
                                float(protocol["projection"]["float64_relative_margin"])
                                if reference_name == "R_float64_dtype_default"
                                else margin
                            ),
                            same_projection_contract=(
                                reference_name != "R_float64_dtype_default"
                            ),
                            **diagnostics[variant],
                            within_tolerance=_within_tolerance(metrics, tolerance),
                            **metrics,
                        )
                    )
        if high_precision_evaluated:
            reference_groups = _groups(*high_precision_evaluated["R"])
            high_tolerance = protocol["numerics"]["tolerances"]["float64"]
            for variant in ("F", "T"):
                candidate_groups = _groups(*high_precision_evaluated[variant])
                for group_name, reference_group in reference_groups.items():
                    metrics = _finite_metrics(
                        reference_group, candidate_groups[group_name], norm_floor
                    )
                    rows.append(
                        _row(
                            protocol,
                            config["experiment_id"],
                            suite_hash,
                            variant=variant,
                            precision="float64",
                            case_id=f"{case['case_id']}:{realization}:float64_fixed",
                            check_kind="selected_high_precision_equivalence",
                            source_case_id=case["case_id"],
                            realization=realization,
                            variable_group=group_name,
                            comparison_reference="R_float64_fixed_float32_margin",
                            curvature=case["curvature"],
                            projection_margin=margin,
                            within_tolerance=_within_tolerance(metrics, high_tolerance),
                            **metrics,
                        )
                    )
        if case["case_id"] == "N084":
            precision_evaluations = [(case["precision"], evaluated)]
            if high_precision_evaluated:
                precision_evaluations.append(("float64", high_precision_evaluated))
            for evaluation_precision, precision_values in precision_evaluations:
                analytic_tolerance = protocol["numerics"]["tolerances"][
                    evaluation_precision
                ]
                for variant, candidate in precision_values.items():
                    curvature_gradient = candidate[1][2]
                    metrics = _finite_metrics(
                        jnp.zeros_like(curvature_gradient),
                        curvature_gradient,
                        norm_floor,
                    )
                    rows.append(
                        _row(
                            protocol,
                            config["experiment_id"],
                            suite_hash,
                            variant=variant,
                            precision=evaluation_precision,
                            case_id=(
                                f"N084:{realization}:{evaluation_precision}:"
                                "curvature_oracle"
                            ),
                            check_kind="n084_zero_curvature_oracle",
                            source_case_id="N084",
                            realization=realization,
                            comparison_reference="analytic_zero",
                            curvature=case["curvature"],
                            projection_margin=margin,
                            within_tolerance=_within_tolerance(metrics, analytic_tolerance),
                            **metrics,
                        )
                    )
    return rows


def _archived_safe_norm(value: Array) -> Array:
    scale = jnp.max(jnp.abs(value), axis=-1, keepdims=True)
    safe_scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    scaled = value / safe_scale
    square_sum = jnp.sum(jnp.square(scaled), axis=-1, keepdims=True)
    safe_square_sum = jnp.where(square_sum > 0.0, square_sum, jnp.ones_like(square_sum))
    result = safe_scale * jnp.sqrt(safe_square_sum)
    return jnp.where(scale > 0.0, result, jnp.zeros_like(result))


def _reference_scores(
    case: dict[str, Any],
    params: HNNPPParams,
    tangent: Array,
    curvature: Array,
    margin: float,
) -> Array:
    kernel = params.weight
    kernel_height, kernel_width, input_channels, output_channels = kernel.shape
    patches = _extract_patches(
        tangent,
        (kernel_height, kernel_width),
        stride=(int(case["stride"]), int(case["stride"])),
        padding=case["padding"],
        dilation=(int(case["dilation"]), int(case["dilation"])),
    )
    patch_dimension = kernel_height * kernel_width * input_channels
    patches *= beta_ratio(patch_dimension, input_channels, dtype=tangent.dtype)
    points = expmap0(patches, curvature, margin=margin)
    return hyperplane_scores_ball(
        points,
        jnp.reshape(kernel, (patch_dimension, output_channels)),
        params.offset,
        curvature,
    )


def _variant_numerical_diagnostics(
    case: dict[str, Any],
    params: HNNPPParams,
    tangent: Array,
    curvature: Array,
    margin: float,
) -> dict[str, dict[str, Any]]:
    """Describe conditioning and readout branches for one frozen case."""

    factorized_scores = _factorized_conv_scores(
        params,
        tangent,
        curvature,
        stride=(int(case["stride"]), int(case["stride"])),
        padding=case["padding"],
        dilation=(int(case["dilation"]), int(case["dilation"])),
        margin=margin,
    )
    score_sets = {
        "R": _reference_scores(case, params, tangent, curvature, margin),
        "F": factorized_scores,
        "T": factorized_scores,
    }
    orientation_norms = np.sqrt(
        np.sum(
            np.square(np.asarray(params.weight, dtype=np.float64)),
            axis=(0, 1, 2),
        )
    )
    zero_columns = int(np.count_nonzero(orientation_norms <= 1.0e-15))
    sqrt_curvature = jnp.sqrt(curvature)
    cap = 1.0 - margin
    result: dict[str, dict[str, Any]] = {}
    for variant, scores in score_sets.items():
        point, _ = _reconstruct_scores(scores, curvature)
        normalized_radius = sqrt_curvature * safe_norm(point, axis=-1, keepdims=True)
        maximum_score = jnp.max(jnp.abs(sqrt_curvature * scores), axis=-1, keepdims=True)
        radius_np = np.asarray(normalized_radius, dtype=np.float64)
        maximum_score_np = np.asarray(maximum_score, dtype=np.float64)
        result[variant] = {
            "orientation_zero_column_count": zero_columns,
            "output_vector_count": int(radius_np.size),
            "output_clip_count": int(np.count_nonzero(radius_np > cap)),
            "output_exact_cap_count": int(np.count_nonzero(radius_np == cap)),
            "minimum_output_cap_gap": float(np.min(cap - radius_np)),
            "maximum_output_cap_gap": float(np.max(cap - radius_np)),
            "scaled_reconstruction_count": int(np.count_nonzero(maximum_score_np > 20.0)),
            "maximum_abs_dimensionless_score": float(np.max(maximum_score_np)),
        }
    return result


def _n088_stage_rows(
    protocol: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    suite_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    request = config["n088_stage_diagnostic"]
    case_index, case = next(
        (index, candidate)
        for index, candidate in enumerate(manifest["cases"])
        if candidate["case_id"] == request["case_id"]
    )
    dtype = _dtype(case["precision"])
    seed = int(manifest["seed"])
    margin = float(config["fixed_float32_projection_margin"])
    for realization in request["realizations"]:
        key = jax.random.fold_in(
            jax.random.PRNGKey(seed), case_index * 10_000 + realization
        )
        params_key, input_key, _ = jax.random.split(key, 3)
        params = _case_parameters(case, params_key, dtype)
        tangent, _ = _case_input(case, params, input_key, dtype)
        curvature = jnp.asarray(case["curvature"], dtype=dtype)
        scores = _reference_scores(case, params, tangent, curvature, margin)
        sinh_values = jnp.sinh(jnp.sqrt(curvature) * scores)
        archived_gradient = jax.grad(_sum_archived_norm)(sinh_values)
        corrected_gradient = jax.grad(_sum_safe_norm)(sinh_values)
        corrected_readout_gradient = jax.grad(
            partial(
                _sum_projected_tangent,
                curvature=curvature,
                margin=margin,
            )
        )(scores)
        jax.block_until_ready(
            (archived_gradient, corrected_gradient, corrected_readout_gradient)
        )
        arrays = {
            "archived_safe_norm_gradient": archived_gradient,
            "corrected_safe_norm_gradient": corrected_gradient,
            "corrected_readout_gradient": corrected_readout_gradient,
        }
        nonfinite = {
            name: int(np.size(value) - np.isfinite(np.asarray(value)).sum())
            for name, value in arrays.items()
        }
        expected_archived_nonfinite = realization in request["archived_nonfinite_expected"]
        rows.append(
            _row(
                protocol,
                config["experiment_id"],
                suite_hash,
                variant="diagnostic",
                precision=case["precision"],
                case_id=f"N088-stage:{realization}",
                check_kind="n088_first_failing_stage",
                realization=realization,
                maximum_abs_sinh=float(np.max(np.abs(np.asarray(sinh_values)))),
                expected_archived_nonfinite=expected_archived_nonfinite,
                archived_safe_norm_gradient_nonfinite_count=nonfinite[
                    "archived_safe_norm_gradient"
                ],
                corrected_safe_norm_gradient_nonfinite_count=nonfinite[
                    "corrected_safe_norm_gradient"
                ],
                corrected_readout_gradient_nonfinite_count=nonfinite[
                    "corrected_readout_gradient"
                ],
                mechanism_confirmed=(
                    (nonfinite["archived_safe_norm_gradient"] > 0)
                    == expected_archived_nonfinite
                ),
                corrected_finite=(
                    nonfinite["corrected_safe_norm_gradient"] == 0
                    and nonfinite["corrected_readout_gradient"] == 0
                ),
            )
        )
    return rows


def run_numerical_repair(
    protocol_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the bounded, registered derivative-repair experiment."""

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    require_keys(
        config,
        (
            "experiment_id",
            "fixed_float32_projection_margin",
            "score_reconstruction",
            "projection",
            "selected_manifest_cases",
            "n088_stage_diagnostic",
            "acceptance",
        ),
        name="E02 repair config",
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    suite_hash = config_hash(protocol, config, manifest)

    rows = [
        *_score_rows(protocol, config, suite_hash),
        *_projection_rows(protocol, config, suite_hash),
        *_selected_case_rows(protocol, config, manifest, suite_hash),
        *_n088_stage_rows(protocol, config, manifest, suite_hash),
    ]
    append_jsonl(destination / "e02_repair_results.jsonl", rows)

    analytic = [
        row
        for row in rows
        if row["check_kind"] in {"score_reconstruction_jvp", "projection_jvp"}
    ]
    threshold_diagnostics = [
        row
        for row in analytic
        if row["check_kind"] == "projection_jvp" and row["exact_cap_case"]
    ]
    smooth_analytic = [
        row
        for row in analytic
        if not (row["check_kind"] == "projection_jvp" and row["exact_cap_case"])
    ]
    same_precision = [
        row
        for row in rows
        if row["check_kind"] == "selected_frozen_case"
        and row["comparison_reference"] == "R_same_precision"
    ]
    required_same_precision_ids = set(
        config["acceptance"]["required_same_precision_case_ids"]
    )
    required_same_precision = [
        row
        for row in same_precision
        if row["source_case_id"] in required_same_precision_ids
    ]
    diagnostic_ids = set(config["acceptance"]["diagnostic_case_ids"])
    diagnostic_same_precision = [
        row for row in same_precision if row["source_case_id"] in diagnostic_ids
    ]
    high_precision = [
        row for row in rows if row["check_kind"] == "selected_high_precision_equivalence"
    ]
    curvature_oracle = [
        row for row in rows if row["check_kind"] == "n084_zero_curvature_oracle"
    ]
    stages = [row for row in rows if row["check_kind"] == "n088_first_failing_stage"]
    summary = {
        "status": "ok",
        "experiment_id": config["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "config_hash": suite_hash,
        "result_rows": len(rows),
        "analytic_rows": len(analytic),
        "analytic_passes": sum(bool(row["within_tolerance"]) for row in analytic),
        "smooth_analytic_rows": len(smooth_analytic),
        "smooth_analytic_passes": sum(
            bool(row["within_tolerance"]) for row in smooth_analytic
        ),
        "threshold_diagnostic_rows": len(threshold_diagnostics),
        "threshold_diagnostic_passes": sum(
            bool(row["within_tolerance"]) for row in threshold_diagnostics
        ),
        "exact_cap_kinks_are_diagnostic": bool(
            config["acceptance"]["exact_cap_kinks_are_diagnostic"]
        ),
        "selected_same_precision_rows": len(same_precision),
        "selected_same_precision_passes": sum(
            bool(row["within_tolerance"]) for row in same_precision
        ),
        "selected_same_precision_nonfinite": sum(
            not bool(row["finite"]) for row in same_precision
        ),
        "required_same_precision_case_ids": sorted(required_same_precision_ids),
        "required_same_precision_rows": len(required_same_precision),
        "required_same_precision_passes": sum(
            bool(row["within_tolerance"]) for row in required_same_precision
        ),
        "diagnostic_case_ids": sorted(diagnostic_ids),
        "diagnostic_same_precision_rows": len(diagnostic_same_precision),
        "diagnostic_same_precision_passes": sum(
            bool(row["within_tolerance"]) for row in diagnostic_same_precision
        ),
        "high_precision_diagnostic_rows": len(high_precision),
        "high_precision_diagnostic_passes": sum(
            bool(row["within_tolerance"]) for row in high_precision
        ),
        "n084_curvature_oracle_rows": len(curvature_oracle),
        "n084_curvature_oracle_passes": sum(
            bool(row["within_tolerance"]) for row in curvature_oracle
        ),
        "n088_stage_rows": len(stages),
        "n088_mechanism_confirmed": all(bool(row["mechanism_confirmed"]) for row in stages),
        "n088_corrected_finite": all(bool(row["corrected_finite"]) for row in stages),
        "required_contract_pass": bool(
            smooth_analytic
            and all(bool(row["within_tolerance"]) for row in smooth_analytic)
            and required_same_precision
            and all(bool(row["within_tolerance"]) for row in required_same_precision)
            and stages
            and all(bool(row["mechanism_confirmed"]) for row in stages)
            and all(bool(row["corrected_finite"]) for row in stages)
        ),
    }
    identity = base_result_row(
        protocol,
        experiment_id=config["experiment_id"],
        variant="R/F/T",
        configuration_hash=suite_hash,
        dtype="mixed_float32_float64",
        seed_or_session="bounded_audit_matrix",
    )
    summary.update(
        {
            "code_commit": identity["code_commit"],
            "code_dirty": identity["code_dirty"],
            "reference_commit": identity["reference_commit"],
            "device_id": identity["device_id"],
            "precision_policy": identity["precision_policy"],
            "timestamp_utc": identity["timestamp_utc"],
        }
    )
    write_json(destination / "summary.json", summary)
    return summary
