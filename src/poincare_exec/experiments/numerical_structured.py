"""Structured E02 topology and directional finite-difference experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.analysis.reviewer_requested import (
    elementwise_relative_error_summary,
)
from poincare_exec.artifacts import (
    append_jsonl,
    base_result_row,
    validate_jsonl_identity,
)
from poincare_exec.config.constants import HYPERBOLIC_VARIANTS
from poincare_exec.config.loader import canonical_json, config_hash, load_yaml, require_keys
from poincare_exec.core.hnnpp import (
    beta_concat_tangent,
    conv2d,
    dense,
    init_conv2d,
    init_dense,
)
from poincare_exec.core.operations import (
    init_midpoint_batch_norm,
    midpoint_batch_norm,
    transpose_conv2d,
)
from poincare_exec.experiments.numerics import _finite_metrics
from poincare_exec.models.common import apply_double_conv, init_double_conv, key_stream
from poincare_exec.models.resnet import (
    init_residual_block,
    residual_block_forward,
)
from poincare_exec.models.unet import (
    decoder_block_forward,
    init_unet,
    unet_forward,
)

Array = jax.Array
CaseFunction = Callable[[Any, Any, Array, str], Array]


@dataclass(frozen=True)
class StructuredCase:
    """One initialized topology with its numerical differentiation contract."""

    parameters: Any
    inputs: Any
    function: CaseFunction
    uses_external_curvature: bool = True
    internal_curvature_parameter: str | None = None


def _dtype(name: str):
    if name == "float32":
        return jnp.float32
    if name == "float64":
        return jnp.float64
    raise ValueError(f"unsupported structured numerical dtype: {name}")


def _random_array(key: Array, shape: list[int], dtype: Any, scale: float) -> Array:
    return scale * jax.random.normal(key, tuple(int(value) for value in shape), dtype)


def _dense_stack_case(
    spec: dict[str, Any], key: Array, dtype: Any, input_scale: float
) -> StructuredCase:
    depth = int(spec["depth"])
    input_features = int(spec["input_features"])
    output_features = int(spec["output_features"])
    keys = jax.random.split(key, depth + 1)
    inputs = _random_array(keys[0], spec["input_shape"], dtype, input_scale)
    params = []
    for index in range(depth):
        params.append(
            init_dense(
                keys[index + 1],
                input_features if index == 0 else output_features,
                output_features,
                dtype=dtype,
                initialization="normal",
            )
        )
    activation = spec["activation"]

    def function(selected, tangent, curvature, variant):
        value = tangent
        for layer in selected:
            value = dense(layer, value, curvature, variant=variant)
            if activation == "relu":
                value = jax.nn.relu(value)
        return value

    return StructuredCase(params, inputs, function)


def _convolution_stack_case(
    spec: dict[str, Any], key: Array, dtype: Any, input_scale: float
) -> StructuredCase:
    depth = int(spec["depth"])
    input_channels = int(spec["input_channels"])
    output_channels = int(spec["output_channels"])
    keys = jax.random.split(key, depth + 1)
    inputs = _random_array(keys[0], spec["input_shape"], dtype, input_scale)
    params = []
    for index in range(depth):
        params.append(
            init_conv2d(
                keys[index + 1],
                int(spec["kernel_size"]),
                input_channels if index == 0 else output_channels,
                output_channels,
                dtype=dtype,
                initialization="normal",
            )
        )
    activation = spec["activation"]

    def function(selected, tangent, curvature, variant):
        value = tangent
        for layer in selected:
            value = conv2d(
                layer,
                value,
                curvature,
                variant=variant,
                stride=int(spec["stride"]),
                padding=spec["padding"],
            )
            if activation == "relu":
                value = jax.nn.relu(value)
        return value

    return StructuredCase(params, inputs, function)


def _case(
    spec: dict[str, Any],
    key: Array,
    dtype: Any,
    input_scale: float,
    normalization_epsilon: float,
) -> StructuredCase:
    kind = spec["kind"]
    parameter_key, input_key = jax.random.split(key)
    if kind == "dense_stack":
        return _dense_stack_case(spec, key, dtype, input_scale)
    if kind == "convolution_stack":
        return _convolution_stack_case(spec, key, dtype, input_scale)
    if kind == "conv_bn_relu":
        params = {
            "conv": init_conv2d(
                parameter_key,
                int(spec["kernel_size"]),
                int(spec["input_channels"]),
                int(spec["output_channels"]),
                dtype=dtype,
                initialization="normal",
            ),
            "bn": init_midpoint_batch_norm(int(spec["output_channels"]), dtype=dtype),
        }
        inputs = _random_array(input_key, spec["input_shape"], dtype, input_scale)

        def function(selected, tangent, curvature, variant):
            value = conv2d(
                selected["conv"],
                tangent,
                curvature,
                variant=variant,
                padding=spec["padding"],
            )
            value = midpoint_batch_norm(
                selected["bn"], value, curvature, epsilon=normalization_epsilon
            )
            return jax.nn.relu(value)

    elif kind == "double_convolution":
        params = init_double_conv(
            key_stream(parameter_key),
            int(spec["input_channels"]),
            int(spec["output_channels"]),
            dtype=dtype,
        )
        inputs = _random_array(input_key, spec["input_shape"], dtype, input_scale)

        def function(selected, tangent, curvature, variant):
            return apply_double_conv(
                selected,
                tangent,
                curvature,
                variant=variant,
                epsilon=normalization_epsilon,
                convolution_bias=False,
            )

    elif kind == "beta_skip":
        branch_keys = jax.random.split(input_key, len(spec["branch_shapes"]))
        inputs = {
            "branches": tuple(
                _random_array(branch_key, shape, dtype, input_scale)
                for branch_key, shape in zip(
                    branch_keys, spec["branch_shapes"], strict=True
                )
            )
        }
        total_channels = sum(int(shape[-1]) for shape in spec["branch_shapes"])
        params = init_conv2d(
            parameter_key,
            1,
            total_channels,
            int(spec["output_channels"]),
            dtype=dtype,
            initialization="normal",
        )

        def function(selected, tangent, curvature, variant):
            joined = beta_concat_tangent(*tangent["branches"])
            return conv2d(selected, joined, curvature, variant=variant)

    elif kind == "residual_block":
        keys = key_stream(parameter_key)
        params = init_residual_block(
            keys,
            int(spec["input_channels"]),
            int(spec["output_channels"]),
            stride=int(spec["stride"]),
            dtype=dtype,
        )
        inputs = _random_array(input_key, spec["input_shape"], dtype, input_scale)

        def function(selected, tangent, curvature, variant):
            return residual_block_forward(
                selected,
                tangent,
                curvature,
                variant=variant,
                epsilon=normalization_epsilon,
                residual_order=spec["residual_order"],
            )

    elif kind == "transpose_convolution":
        kernel_size = int(spec["kernel_size"])
        output_channels = int(spec["output_channels"])
        params = init_dense(
            parameter_key,
            int(spec["input_channels"]),
            kernel_size * kernel_size * output_channels,
            dtype=dtype,
            initialization="source",
        )
        inputs = _random_array(input_key, spec["input_shape"], dtype, input_scale)

        def function(selected, tangent, curvature, variant):
            return transpose_conv2d(
                selected,
                tangent,
                curvature,
                output_channels=output_channels,
                kernel_size=kernel_size,
                stride=int(spec["stride"]),
                variant=variant,
            )

    elif kind == "decoder_block":
        model = init_unet(
            parameter_key,
            input_channels=int(spec["skip_channels"]),
            classes=int(spec["classes"]),
            initial_width=int(spec["skip_channels"]),
            depth=1,
            initial_curvature=float(spec["initial_curvature"]),
            dtype=dtype,
        )
        params = model["up_blocks"][0]
        decoder_key, skip_key = jax.random.split(input_key)
        inputs = {
            "decoder": _random_array(
                decoder_key, spec["decoder_shape"], dtype, input_scale
            ),
            "skip": _random_array(skip_key, spec["skip_shape"], dtype, input_scale),
        }

        def function(selected, tangent, curvature, variant):
            return decoder_block_forward(
                selected,
                tangent["decoder"],
                tangent["skip"],
                curvature,
                variant=variant,
                epsilon=normalization_epsilon,
            )

    elif kind == "small_unet":
        params = init_unet(
            parameter_key,
            input_channels=int(spec["input_channels"]),
            classes=int(spec["classes"]),
            initial_width=int(spec["initial_width"]),
            depth=int(spec["depth"]),
            initial_curvature=float(spec["initial_curvature"]),
            dtype=dtype,
        )
        inputs = _random_array(input_key, spec["input_shape"], dtype, input_scale)

        def function(selected, tangent, _curvature, variant):
            return unet_forward(
                selected,
                tangent,
                variant=variant,
                epsilon=normalization_epsilon,
            )

    else:
        raise ValueError(f"unsupported structured case kind: {kind}")
    return StructuredCase(
        params,
        inputs,
        function,
        uses_external_curvature=kind != "small_unet",
        internal_curvature_parameter=("curvature_raw" if kind == "small_unet" else None),
    )


def _evaluate(
    function: CaseFunction,
    variant: str,
    params: Any,
    inputs: Any,
    curvature: Array,
    cotangent: Array,
) -> tuple[Array, tuple[Any, Any, Array]]:
    def scalarized(selected_params, selected_inputs, selected_curvature):
        output = function(selected_params, selected_inputs, selected_curvature, variant)
        return jnp.sum(output * cotangent), output

    (_, output), gradients = jax.value_and_grad(
        scalarized, argnums=(0, 1, 2), has_aux=True
    )(params, inputs, curvature)
    jax.block_until_ready((output, gradients))
    return output, gradients


def _leaf_groups(prefix: str, tree: Any) -> dict[str, Array]:
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    if len(leaves) == 1 and not leaves[0][0]:
        return {prefix: leaves[0][1]}
    return {f"{prefix}:{jax.tree_util.keystr(path)}": value for path, value in leaves}


def _gradient_groups(
    result: tuple[Array, tuple[Any, Any, Array]],
    *,
    include_external_curvature: bool,
) -> dict[str, Array]:
    output, (parameter_gradients, input_gradients, curvature_gradient) = result
    groups = {"output": output}
    if include_external_curvature:
        groups["curvature_gradient"] = curvature_gradient
    groups.update(_leaf_groups("parameter_gradient", parameter_gradients))
    groups.update(_leaf_groups("input_gradient", input_gradients))
    return groups


def _tree_dot(left: Any, right: Any) -> Array:
    products = jax.tree_util.tree_map(
        lambda first, second: jnp.sum(first * second), left, right
    )
    return jnp.sum(jnp.stack(jax.tree_util.tree_leaves(products)))


def _normalized_direction(tree: Any, key: Array) -> Any:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    random_leaves = [
        jax.random.normal(selected_key, leaf.shape, dtype=leaf.dtype)
        for selected_key, leaf in zip(keys, leaves, strict=True)
    ]
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in random_leaves))
    return jax.tree_util.tree_unflatten(structure, [leaf / norm for leaf in random_leaves])


def _tree_scale(tree: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: scale * value, tree)


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_norm(tree: Any) -> float:
    value = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree))
    )
    return float(value)


def _finite_difference_rows(
    function: CaseFunction,
    variant: str,
    params: Any,
    inputs: Any,
    curvature: Array,
    cotangent: Array,
    gradients: tuple[Any, Any, Array],
    relative_steps: list[float],
    key: Array,
    norm_floor: float,
    *,
    include_external_curvature: bool,
    internal_curvature_parameter: str | None,
) -> list[dict[str, Any]]:
    parameter_gradient, input_gradient, curvature_gradient = gradients

    def scalar(selected_params, selected_inputs, selected_curvature):
        return jnp.sum(
            function(selected_params, selected_inputs, selected_curvature, variant)
            * cotangent
        )

    parameter_key, input_key = jax.random.split(key)
    parameter_direction = _normalized_direction(params, parameter_key)
    input_direction = _normalized_direction(inputs, input_key)
    perturbations: list[tuple[str, Array, float, Callable[[float], Array]]] = [
        (
            "parameters",
            _tree_dot(parameter_gradient, parameter_direction),
            max(_tree_norm(params), 1.0),
            lambda delta: scalar(
                _tree_add(params, _tree_scale(parameter_direction, delta)),
                inputs,
                curvature,
            ),
        ),
        (
            "input",
            _tree_dot(input_gradient, input_direction),
            max(_tree_norm(inputs), 1.0),
            lambda delta: scalar(
                params,
                _tree_add(inputs, _tree_scale(input_direction, delta)),
                curvature,
            ),
        ),
    ]
    if include_external_curvature:
        perturbations.append(
            (
                "curvature",
                curvature_gradient,
                max(float(jnp.abs(curvature)), 1.0),
                lambda delta: scalar(params, inputs, curvature + delta),
            )
        )
    if internal_curvature_parameter is not None:
        if not isinstance(params, dict) or internal_curvature_parameter not in params:
            raise ValueError("declared internal curvature parameter is absent")
        raw_curvature = params[internal_curvature_parameter]
        raw_gradient = parameter_gradient[internal_curvature_parameter]

        def evaluate_internal_curvature(delta: float) -> Array:
            selected = dict(params)
            selected[internal_curvature_parameter] = raw_curvature + delta
            return scalar(selected, inputs, curvature)

        perturbations.append(
            (
                "unconstrained_curvature_parameter",
                raw_gradient,
                max(float(jnp.abs(raw_curvature)), 1.0),
                evaluate_internal_curvature,
            )
        )
    rows = []
    for group, automatic, scale, evaluate_perturbation in perturbations:
        for relative_step in relative_steps:
            step = relative_step * scale
            positive = evaluate_perturbation(step)
            negative = evaluate_perturbation(-step)
            finite_difference = (positive - negative) / (2.0 * step)
            automatic_value = float(automatic)
            finite_value = float(finite_difference)
            finite = math.isfinite(automatic_value) and math.isfinite(finite_value)
            absolute_error = abs(automatic_value - finite_value) if finite else None
            relative_error = (
                absolute_error / max(abs(finite_value), norm_floor)
                if absolute_error is not None
                else None
            )
            rows.append(
                {
                    "finite_difference_group": group,
                    "relative_step": relative_step,
                    "automatic_directional_derivative": (
                        automatic_value if finite else None
                    ),
                    "finite_difference_directional_derivative": (
                        finite_value if finite else None
                    ),
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "finite": finite,
                }
            )
    return rows


def _validate_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        (
            "experiment_id",
            "cases",
            "precisions",
            "realizations",
            "finite_difference_relative_steps",
            "curvature",
            "input_scale",
            "normalization_epsilon",
        ),
        name="structured numerical config",
    )
    cases = config["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("structured numerical config requires at least one case")
    identifiers = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(identifiers) != len(cases) or any(not value for value in identifiers):
        raise ValueError("every structured numerical case requires a nonempty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("structured numerical case ids must be unique")
    if int(config["realizations"]) <= 0:
        raise ValueError("structured numerical realizations must be positive")
    if any(float(step) <= 0.0 for step in config["finite_difference_relative_steps"]):
        raise ValueError("finite-difference relative steps must be positive")
    for precision in config["precisions"]:
        _dtype(precision)


def _elementwise_settings(
    config: dict[str, Any], protocol: dict[str, Any]
) -> tuple[float, tuple[float, ...]] | None:
    settings = config.get("elementwise_relative_error")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("elementwise_relative_error must be a mapping")
    required = {
        "denominator_floor",
        "quantiles",
        "quantile_method",
        "report_denominator_floored_fraction",
    }
    if set(settings) != required:
        raise ValueError("elementwise_relative_error fields differ from the frozen schema")
    floor = float(settings["denominator_floor"])
    protocol_floor = float(protocol["numerics"]["norm_floor"])
    if floor != protocol_floor:
        raise ValueError("elementwise denominator floor must equal numerics.norm_floor")
    quantiles = tuple(float(value) for value in settings["quantiles"])
    if quantiles != (0.5, 0.9, 0.95, 0.99, 0.999):
        raise ValueError("elementwise quantiles differ from the frozen E26 set")
    if settings["quantile_method"] != "linear":
        raise ValueError("elementwise quantile method must be linear")
    if settings["report_denominator_floored_fraction"] is not True:
        raise ValueError("E26 must report denominator-floor exposure")
    return floor, quantiles


def _existing_keys(path: Path) -> set[tuple[str, str, int, str, str, str]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            group = row.get("variable_group") or (
                f"{row['finite_difference_group']}:{row['relative_step']}"
            )
            completed.add(
                (
                    row["case_id"],
                    row["dtype"],
                    int(row["realization"]),
                    row["variant"],
                    row["record_type"],
                    group,
                )
            )
    return completed


def run_structured_numerics(
    protocol_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> int:
    """Run or resume the registered topology and finite-difference matrix."""

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    _validate_config(config)
    digest = config_hash(protocol, config)
    output = Path(output_path).expanduser().resolve()
    identity = base_result_row(
        protocol,
        experiment_id=config["experiment_id"],
        variant="R",
        configuration_hash=digest,
        dtype=config["precisions"][0],
        seed_or_session="identity",
    )
    validate_jsonl_identity(
        output,
        identity,
        required_fields=(
            "experiment_id",
            "protocol_version",
            "config_hash",
            "code_commit",
            "code_dirty",
        ),
    )
    completed = _existing_keys(output)
    seed = int(protocol["seeds"]["numerical"])
    norm_floor = float(protocol["numerics"]["norm_floor"])
    elementwise_settings = _elementwise_settings(config, protocol)
    written = 0
    for case_index, spec in enumerate(config["cases"]):
        case_id = spec["id"]
        for dtype_name in config["precisions"]:
            dtype = _dtype(dtype_name)
            for realization in range(int(config["realizations"])):
                key = jax.random.fold_in(
                    jax.random.PRNGKey(seed), case_index * 10_000 + realization
                )
                case_key, cotangent_key, difference_key = jax.random.split(key, 3)
                selected_case = _case(
                    spec,
                    case_key,
                    dtype,
                    float(config["input_scale"]),
                    float(config["normalization_epsilon"]),
                )
                curvature = jnp.asarray(config["curvature"], dtype=dtype)
                output_shape = selected_case.function(
                    selected_case.parameters,
                    selected_case.inputs,
                    curvature,
                    "R",
                ).shape
                cotangent = jax.random.normal(cotangent_key, output_shape, dtype=dtype)
                evaluated = {
                    variant: _evaluate(
                        selected_case.function,
                        variant,
                        selected_case.parameters,
                        selected_case.inputs,
                        curvature,
                        cotangent,
                    )
                    for variant in HYPERBOLIC_VARIANTS
                }
                reference_groups = _gradient_groups(
                    evaluated["R"],
                    include_external_curvature=(selected_case.uses_external_curvature),
                )
                shared = {
                    "case_id": case_id,
                    "case_spec": canonical_json(spec),
                    "dtype": dtype_name,
                    "realization": realization,
                    "output_shape": list(output_shape),
                    "state_semantics": "current_batch_no_running_state",
                }
                for variant in ("F", "T"):
                    base = base_result_row(
                        protocol,
                        experiment_id=config["experiment_id"],
                        variant=variant,
                        configuration_hash=digest,
                        dtype=dtype_name,
                        seed_or_session=f"{case_id}:{realization}",
                    )
                    candidate_groups = _gradient_groups(
                        evaluated[variant],
                        include_external_curvature=(selected_case.uses_external_curvature),
                    )
                    rows = []
                    for group, candidate in candidate_groups.items():
                        row_key = (
                            case_id,
                            dtype_name,
                            realization,
                            variant,
                            "agreement",
                            group,
                        )
                        if row_key in completed:
                            continue
                        elementwise = (
                            elementwise_relative_error_summary(
                                reference_groups[group],
                                candidate,
                                elementwise_settings[0],
                                quantiles=elementwise_settings[1],
                            )
                            if elementwise_settings is not None
                            else {}
                        )
                        rows.append(
                            {
                                **base,
                                **shared,
                                "record_type": "agreement",
                                "comparison_reference": "R_same_precision",
                                "variable_group": group,
                                **_finite_metrics(
                                    reference_groups[group], candidate, norm_floor
                                ),
                                **elementwise,
                            }
                        )
                        completed.add(row_key)
                    append_jsonl(output, rows)
                    written += len(rows)
                for variant in HYPERBOLIC_VARIANTS:
                    base = base_result_row(
                        protocol,
                        experiment_id=config["experiment_id"],
                        variant=variant,
                        configuration_hash=digest,
                        dtype=dtype_name,
                        seed_or_session=f"{case_id}:{realization}",
                    )
                    finite_difference_rows = _finite_difference_rows(
                        selected_case.function,
                        variant,
                        selected_case.parameters,
                        selected_case.inputs,
                        curvature,
                        cotangent,
                        evaluated[variant][1],
                        list(config["finite_difference_relative_steps"]),
                        difference_key,
                        norm_floor,
                        include_external_curvature=(selected_case.uses_external_curvature),
                        internal_curvature_parameter=(
                            selected_case.internal_curvature_parameter
                        ),
                    )
                    rows = []
                    for row in finite_difference_rows:
                        group = f"{row['finite_difference_group']}:{row['relative_step']}"
                        row_key = (
                            case_id,
                            dtype_name,
                            realization,
                            variant,
                            "finite_difference",
                            group,
                        )
                        if row_key in completed:
                            continue
                        rows.append(
                            {
                                **base,
                                **shared,
                                "record_type": "finite_difference",
                                **row,
                            }
                        )
                        completed.add(row_key)
                    append_jsonl(output, rows)
                    written += len(rows)
    return written
