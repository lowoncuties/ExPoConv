"""Bounded E06/E07 diagnostics from accepted retained training state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from poincare_exec.artifacts import append_jsonl, base_result_row, write_json
from poincare_exec.config.loader import (
    config_hash,
    load_yaml,
    require_keys,
    validate_hashed_document,
)
from poincare_exec.data.datasets import load_dataset
from poincare_exec.training.batches import (
    load_training_batch,
    materialize_dataset,
    ordered_batches,
)
from poincare_exec.training.checkpoints import load_checkpoint
from poincare_exec.training.diagnostics import finite_float, tree_nonfinite_count
from poincare_exec.training.optimizers import tree_l2_norm
from poincare_exec.training.tasks import (
    forward_task,
    make_train_step,
    task_loss,
)


def _synchronize(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _index_schedule_sha256(batches: list[np.ndarray]) -> str:
    """Hash an ordered variable-length batch schedule without ambiguity."""

    digest = hashlib.sha256()
    for batch in batches:
        indices = np.asarray(batch, dtype="<i8")
        digest.update(indices.size.to_bytes(8, byteorder="little", signed=False))
        digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


def _path_string(path: tuple[Any, ...]) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        else:
            parts.append(str(entry))
    return "/".join(parts) or "<root>"


def array_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    """Return explicit finite/error statistics for two equally shaped arrays."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("comparison arrays have different shapes")
    reference64 = reference_array.astype(np.float64)
    candidate64 = candidate_array.astype(np.float64)
    reference_finite = np.isfinite(reference64)
    candidate_finite = np.isfinite(candidate64)
    joint_finite = reference_finite & candidate_finite
    reference_nonfinite = int(reference64.size - np.count_nonzero(reference_finite))
    candidate_nonfinite = int(candidate64.size - np.count_nonzero(candidate_finite))
    comparison_nonfinite = int(reference64.size - np.count_nonzero(joint_finite))
    reference_norm = (
        float(np.linalg.norm(reference64.ravel())) if reference_nonfinite == 0 else None
    )
    candidate_norm = (
        float(np.linalg.norm(candidate64.ravel())) if candidate_nonfinite == 0 else None
    )
    if comparison_nonfinite:
        difference_norm = None
        maximum_error = None
        relative_error = None
        exactly_equal = False
    else:
        difference = candidate64 - reference64
        difference_norm = float(np.linalg.norm(difference.ravel()))
        maximum_error = float(np.max(np.abs(difference), initial=0.0))
        assert reference_norm is not None
        relative_error = difference_norm / max(reference_norm, 1.0e-12)
        exactly_equal = bool(np.array_equal(reference_array, candidate_array))
    return {
        "shape": list(reference_array.shape),
        "reference_dtype": str(reference_array.dtype),
        "candidate_dtype": str(candidate_array.dtype),
        "element_count": int(reference_array.size),
        "reference_l2_norm": reference_norm,
        "candidate_l2_norm": candidate_norm,
        "difference_l2_norm": difference_norm,
        "max_absolute_error": maximum_error,
        "relative_l2_error": relative_error,
        "reference_nonfinite_count": reference_nonfinite,
        "candidate_nonfinite_count": candidate_nonfinite,
        "comparison_nonfinite_count": comparison_nonfinite,
        "exactly_equal": exactly_equal,
    }


def arrays_bitwise_equal(left: Any, right: Any) -> bool:
    """Return whether two arrays have identical shape, dtype, and bit pattern."""

    left_array = np.asarray(jax.device_get(left))
    right_array = np.asarray(jax.device_get(right))
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def tree_comparison_rows(reference: Any, candidate: Any) -> list[dict[str, Any]]:
    """Compare corresponding pytree leaves in stable path order."""

    reference_with_paths, reference_definition = jax.tree_util.tree_flatten_with_path(
        reference
    )
    candidate_with_paths, candidate_definition = jax.tree_util.tree_flatten_with_path(
        candidate
    )
    if reference_definition != candidate_definition:
        raise ValueError("comparison trees have different structures")
    rows = []
    for (reference_path, reference_leaf), (candidate_path, candidate_leaf) in zip(
        reference_with_paths, candidate_with_paths, strict=True
    ):
        reference_name = _path_string(reference_path)
        candidate_name = _path_string(candidate_path)
        if reference_name != candidate_name:
            raise ValueError("comparison trees have different paths")
        rows.append(
            {
                "parameter_path": reference_name,
                **array_comparison(reference_leaf, candidate_leaf),
            }
        )
    return rows


def trees_bitwise_equal(left: Any, right: Any) -> bool:
    """Return whether two pytrees have identical structure, paths, and bits."""

    left_with_paths, left_definition = jax.tree_util.tree_flatten_with_path(left)
    right_with_paths, right_definition = jax.tree_util.tree_flatten_with_path(right)
    if left_definition != right_definition:
        return False
    return all(
        _path_string(left_path) == _path_string(right_path)
        and arrays_bitwise_equal(left_leaf, right_leaf)
        for (left_path, left_leaf), (right_path, right_leaf) in zip(
            left_with_paths, right_with_paths, strict=True
        )
    )


def perturb_parameters_one_ulp(params: Any) -> tuple[Any, dict[str, Any]]:
    """Apply a deterministic alternating one-ULP perturbation to float leaves."""

    leaves, definition = jax.tree_util.tree_flatten(params)
    perturbed_leaves = []
    next_direction_positive = True
    changed = 0
    difference_sq = 0.0
    maximum_error = 0.0
    for leaf in leaves:
        values = np.asarray(jax.device_get(leaf))
        if not np.issubdtype(values.dtype, np.floating):
            perturbed_leaves.append(leaf)
            continue
        flattened = values.reshape(-1)
        directions = np.empty_like(flattened)
        for index in range(flattened.size):
            directions[index] = math.inf if next_direction_positive else -math.inf
            next_direction_positive = not next_direction_positive
        perturbed = np.nextafter(flattened, directions).reshape(values.shape)
        difference = perturbed.astype(np.float64) - values.astype(np.float64)
        changed += int(np.count_nonzero(perturbed != values))
        difference_sq += float(np.sum(np.square(difference)))
        maximum_error = max(maximum_error, float(np.max(np.abs(difference), initial=0.0)))
        perturbed_leaves.append(jnp.asarray(perturbed))
    perturbed_tree = jax.tree_util.tree_unflatten(definition, perturbed_leaves)
    return perturbed_tree, {
        "policy": "alternating_nextafter_all_floating_parameter_elements",
        "changed_element_count": changed,
        "perturbation_l2_norm": math.sqrt(difference_sq),
        "perturbation_max_absolute": maximum_error,
    }


def _tree_difference_norm(left: Any, right: Any) -> float:
    difference = jax.tree_util.tree_map(lambda x, y: x - y, left, right)
    value = tree_l2_norm(difference)
    value.block_until_ready()
    return float(value)


def _curvature(task_name: str, task: dict[str, Any], params: Any) -> float:
    if task_name == "isic16_poincare_unet":
        value = jax.nn.softplus(params["curvature_raw"])
        value.block_until_ready()
        return float(value)
    return float(task["curvature"]["initial_magnitude"])


def _make_loss_gradient(task_name: str, task: dict[str, Any], variant: str):
    def objective(selected_params, inputs, targets):
        logits = forward_task(task_name, task, selected_params, inputs, variant)
        loss = task_loss(task_name, task, logits, targets)
        return loss, logits

    return jax.jit(jax.value_and_grad(objective, has_aux=True))


def _load_source_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("source summary must be a JSON mapping")
    return value


def _validate_source(
    source: dict[str, Any],
    source_summary: dict[str, Any],
    *,
    task_entry: dict[str, Any],
    training_config: dict[str, Any],
    task: dict[str, Any],
) -> None:
    source_variant = str(source["variant"])
    source_seed = int(source["seed"])
    expected = {
        "task": training_config["task"],
        "variant": source_variant,
        "seed": int(task_entry["source_seed"]),
        "update": int(task["max_updates"]),
        "epoch": int(task["epochs"]),
        "config_hash": task_entry["source_config_hash"],
        "protocol_version": task_entry["source_protocol_version"],
        "code_dirty": False,
    }
    actual = {
        "task": source.get("task"),
        "variant": source_variant,
        "seed": source_seed,
        "update": int(source.get("update", -1)),
        "epoch": int(source.get("epoch", -1)),
        "config_hash": source.get("config_hash"),
        "protocol_version": source.get("protocol_version"),
        "code_dirty": source.get("code_dirty"),
    }
    if actual != expected:
        raise RuntimeError(f"source checkpoint identity mismatch: {actual!r}")
    if source.get("format") != "poincare-exec-checkpoint-v1":
        raise RuntimeError("source checkpoint format is not supported")
    if source_variant not in task_entry["source_variants"]:
        raise RuntimeError("source variant is outside the frozen diagnostic matrix")
    summary_expected = {
        "status": "ok",
        "experiment_id": training_config["experiment_id"],
        "task": expected["task"],
        "variant": source_variant,
        "seed_or_session": source_seed,
        "updates": expected["update"],
        "epochs": expected["epoch"],
        "config_hash": expected["config_hash"],
        "protocol_version": expected["protocol_version"],
        "code_commit": source.get("code_commit"),
        "code_dirty": False,
    }
    summary_actual = {key: source_summary.get(key) for key in summary_expected}
    if summary_actual != summary_expected:
        raise RuntimeError("source summary and checkpoint identities disagree")
    if not source.get("code_commit"):
        raise RuntimeError("source checkpoint has no code commit")
    if "params" not in source or "optimizer_state" not in source:
        raise RuntimeError("source checkpoint lacks complete training state")
    optimizer_step = int(np.asarray(source["optimizer_state"].step))
    if optimizer_step != expected["update"]:
        raise RuntimeError("source Adam step does not match the final update")


def _comparison_artifact_rows(
    *,
    base: dict[str, Any],
    source_fields: dict[str, Any],
    run_variant: str,
    comparison_type: str,
    update: int,
    reference: Any,
    candidate: Any,
) -> list[dict[str, Any]]:
    return [
        {
            **base,
            **source_fields,
            "run_variant": run_variant,
            "comparison_type": comparison_type,
            "update": update,
            "is_curvature_parameter_path": row["parameter_path"] == "curvature_raw",
            "status": ("ok" if row["comparison_nonfinite_count"] == 0 else "nonfinite"),
            **row,
        }
        for row in tree_comparison_rows(reference, candidate)
    ]


def run_retained_diagnostic(
    protocol_path: str | Path,
    diagnostic_config_path: str | Path,
    training_config_path: str | Path,
    *,
    checkpoint_path: str | Path,
    source_summary_path: str | Path,
    data_root: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Run the frozen matched-gradient and short-continuation diagnostic."""

    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(protocol_path)
    diagnostic_config = load_yaml(diagnostic_config_path)
    training_config = load_yaml(training_config_path)
    design_path, design_sha256 = validate_hashed_document(
        diagnostic_config, field="design_document"
    )
    require_keys(
        diagnostic_config,
        (
            "experiment_id",
            "execution_variants",
            "continuation_updates",
            "observation_updates",
            "controls",
            "tasks",
        ),
        name="retained diagnostic config",
    )
    require_keys(
        training_config,
        ("experiment_id", "task", "dataset", "variants", "seeds"),
        name="training config",
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError("retained training diagnostics require a CUDA backend")
    task_name = str(training_config["task"])
    task = protocol["tasks"][task_name]
    task_entry = next(
        (entry for entry in diagnostic_config["tasks"] if entry["task"] == task_name),
        None,
    )
    if task_entry is None:
        raise ValueError(f"diagnostic config has no task entry for {task_name}")
    core_variants = tuple(protocol["variants"]["core_training"])
    if tuple(str(value) for value in training_config["variants"]) != core_variants:
        raise RuntimeError("diagnostic training variants disagree with the protocol")
    if tuple(str(value) for value in task_entry["source_variants"]) != core_variants:
        raise RuntimeError("diagnostic source variants disagree with the protocol")
    if training_config["experiment_id"] != task_entry["training_experiment_id"]:
        raise RuntimeError("diagnostic task selects the wrong training experiment")
    if int(task_entry["source_seed"]) not in training_config["seeds"]:
        raise RuntimeError("diagnostic source seed is outside the training matrix")
    source_path = Path(checkpoint_path).expanduser().resolve()
    summary_path = Path(source_summary_path).expanduser().resolve()
    source = load_checkpoint(source_path)
    source_summary = _load_source_summary(summary_path)
    _validate_source(
        source,
        source_summary,
        task_entry=task_entry,
        training_config=training_config,
        task=task,
    )
    execution_variants = tuple(
        str(value) for value in diagnostic_config["execution_variants"]
    )
    if execution_variants != core_variants:
        raise RuntimeError("diagnostic execution variants disagree with the protocol")
    controls = diagnostic_config["controls"]
    repeat_policy = str(controls.get("unperturbed_repeat_policy", "bitwise"))
    if (
        controls.get("unperturbed_repeat") is not True
        or controls.get("one_ulp_perturbation") is not True
        or repeat_policy not in {"bitwise", "quantify"}
    ):
        raise RuntimeError("diagnostic controls disagree with the frozen design")
    update_count = int(diagnostic_config["continuation_updates"])
    observations = {int(value) for value in diagnostic_config["observation_updates"]}
    if observations != {0, 1, 10, 50, 100} or update_count != 100:
        raise RuntimeError("diagnostic continuation schedule is not frozen")
    continuation_epoch = int(task_entry["continuation_epoch"])
    if continuation_epoch != int(task["epochs"]) + 1:
        raise RuntimeError("diagnostic continuation epoch is not post-budget")

    config_digest = config_hash(protocol, diagnostic_config, task_entry, training_config)
    source_seed = int(source["seed"])
    run_identity = base_result_row(
        protocol,
        experiment_id=str(diagnostic_config["experiment_id"]),
        variant="shared",
        configuration_hash=config_digest,
        dtype="float32",
        seed_or_session=source_seed,
    )
    run_identity["design_document_path"] = design_path
    run_identity["design_document_sha256"] = design_sha256
    if run_identity["code_dirty"]:
        raise RuntimeError("retained diagnostics require a clean code tree")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        destination / "summary.json",
        destination / "initial_comparisons.jsonl",
        destination / "first_update_comparisons.jsonl",
        destination / "trajectory.jsonl",
        destination / "first_batch_outputs.npz",
    )
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "diagnostic output requires audit before rerun: " + ", ".join(existing)
        )

    source_fields = {
        "source_experiment_id": training_config["experiment_id"],
        "source_task": task_name,
        "source_variant": source["variant"],
        "source_seed": source_seed,
        "source_update": int(source["update"]),
        "source_checkpoint_sha256": _file_sha256(source_path),
        "source_summary_sha256": _file_sha256(summary_path),
        "source_code_commit": source["code_commit"],
        "source_config_hash": source["config_hash"],
        "source_protocol_version": source["protocol_version"],
    }
    training = materialize_dataset(
        load_dataset(training_config["dataset"], data_root, manifest_dir, "train")
    )
    batches = ordered_batches(
        len(training), int(task["batch_size"]), source_seed, continuation_epoch
    )
    if len(batches) < update_count:
        raise RuntimeError("frozen continuation does not fit one deterministic epoch")
    first_images, first_targets = load_training_batch(
        training,
        batches[0],
        task_name=task_name,
        seed=source_seed,
        epoch=continuation_epoch,
    )

    perturbed_params, perturbation = perturb_parameters_one_ulp(source["params"])
    run_to_execution = {
        **{variant: variant for variant in execution_variants},
        "R_repeat": "R",
        "R_ulp": "R",
    }
    states = {
        variant: (source["params"], source["optimizer_state"])
        for variant in execution_variants
    }
    states["R_repeat"] = (source["params"], source["optimizer_state"])
    states["R_ulp"] = (perturbed_params, source["optimizer_state"])
    gradient_functions = {
        variant: _make_loss_gradient(task_name, task, variant)
        for variant in execution_variants
    }
    train_steps = {
        variant: make_train_step(task_name, task, variant) for variant in execution_variants
    }

    initial = {}
    output_arrays: dict[str, Any] = {
        "inputs": np.asarray(first_images),
        "targets": np.asarray(first_targets),
        "batch_indices": np.asarray(batches[0]),
    }
    for run_variant, execution_variant in run_to_execution.items():
        params, _ = states[run_variant]
        (loss, logits), gradients = gradient_functions[execution_variant](
            params, first_images, first_targets
        )
        probabilities = jax.nn.softmax(logits, axis=-1)
        _synchronize((loss, logits, probabilities, gradients))
        initial[run_variant] = {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "gradients": gradients,
            "gradient_norm": tree_l2_norm(gradients),
        }
        _synchronize(initial[run_variant]["gradient_norm"])
        output_arrays[f"{run_variant}__logits"] = np.asarray(logits)
        output_arrays[f"{run_variant}__probabilities"] = np.asarray(probabilities)
    _save_npz(destination / "first_batch_outputs.npz", **output_arrays)

    repeat_exact = bool(
        arrays_bitwise_equal(initial["R"]["loss"], initial["R_repeat"]["loss"])
        and arrays_bitwise_equal(initial["R"]["logits"], initial["R_repeat"]["logits"])
        and arrays_bitwise_equal(
            initial["R"]["probabilities"], initial["R_repeat"]["probabilities"]
        )
        and trees_bitwise_equal(initial["R"]["gradients"], initial["R_repeat"]["gradients"])
    )
    if repeat_policy == "bitwise" and not repeat_exact:
        raise RuntimeError("unperturbed R repeat disagreed at update 0")

    initial_rows = []
    reference_initial = initial["R"]
    for run_variant, values in initial.items():
        loss_value = finite_float(values["loss"])
        gradient_nonfinite = tree_nonfinite_count(values["gradients"])
        output_nonfinite = tree_nonfinite_count((values["logits"], values["probabilities"]))
        initial_rows.append(
            {
                **run_identity,
                **source_fields,
                "run_variant": run_variant,
                "comparison_type": "initial_scalar",
                "update": 0,
                "parameter_path": None,
                "loss": loss_value,
                "loss_difference_from_R": finite_float(
                    values["loss"] - reference_initial["loss"]
                ),
                "gradient_norm": finite_float(values["gradient_norm"]),
                "gradient_nonfinite_count": gradient_nonfinite,
                "output_nonfinite_count": output_nonfinite,
                "status": (
                    "ok"
                    if loss_value is not None
                    and gradient_nonfinite == 0
                    and output_nonfinite == 0
                    else "nonfinite"
                ),
            }
        )
        logits_row = array_comparison(reference_initial["logits"], values["logits"])
        initial_rows.append(
            {
                **run_identity,
                **source_fields,
                "run_variant": run_variant,
                "comparison_type": "logits",
                "update": 0,
                "parameter_path": None,
                "status": (
                    "ok" if logits_row["comparison_nonfinite_count"] == 0 else "nonfinite"
                ),
                **logits_row,
            }
        )
        probability_row = array_comparison(
            reference_initial["probabilities"], values["probabilities"]
        )
        initial_rows.append(
            {
                **run_identity,
                **source_fields,
                "run_variant": run_variant,
                "comparison_type": "probabilities",
                "update": 0,
                "parameter_path": None,
                "status": (
                    "ok"
                    if probability_row["comparison_nonfinite_count"] == 0
                    else "nonfinite"
                ),
                **probability_row,
            }
        )
        initial_rows.extend(
            _comparison_artifact_rows(
                base=run_identity,
                source_fields=source_fields,
                run_variant=run_variant,
                comparison_type="loss_gradient",
                update=0,
                reference=reference_initial["gradients"],
                candidate=values["gradients"],
            )
        )
    append_jsonl(destination / "initial_comparisons.jsonl", initial_rows)

    def trajectory_row(
        run_variant: str,
        update: int,
        loss: Any,
        gradient_norm: Any,
        update_norm: Any,
    ) -> dict[str, Any]:
        params, optimizer_state = states[run_variant]
        reference_params, reference_optimizer = states["R"]
        curvature = finite_float(_curvature(task_name, task, params))
        reference_curvature = finite_float(_curvature(task_name, task, reference_params))
        parameter_nonfinite = tree_nonfinite_count(params)
        optimizer_nonfinite = tree_nonfinite_count(optimizer_state)
        optimizer_step = int(np.asarray(jax.device_get(optimizer_state.step)))
        loss_value = finite_float(loss)
        gradient_value = finite_float(gradient_norm)
        update_value = finite_float(update_norm)
        return {
            **run_identity,
            **source_fields,
            "run_variant": run_variant,
            "execution_variant": run_to_execution[run_variant],
            "update": update,
            "loss": loss_value,
            "gradient_norm": gradient_value,
            "parameter_update_norm": update_value,
            "parameter_l2_difference_from_R": finite_float(
                _tree_difference_norm(params, reference_params)
            ),
            "optimizer_first_moment_difference_from_R": finite_float(
                _tree_difference_norm(
                    optimizer_state.first_moment,
                    reference_optimizer.first_moment,
                )
            ),
            "optimizer_second_moment_difference_from_R": finite_float(
                _tree_difference_norm(
                    optimizer_state.second_moment,
                    reference_optimizer.second_moment,
                )
            ),
            "optimizer_step": optimizer_step,
            "curvature": curvature,
            "curvature_difference_from_R": (
                finite_float(curvature - reference_curvature)
                if curvature is not None and reference_curvature is not None
                else None
            ),
            "curvature_gradient_applicable": task_name == "isic16_poincare_unet",
            "parameter_nonfinite_count": parameter_nonfinite,
            "optimizer_nonfinite_count": optimizer_nonfinite,
            "status": (
                "ok"
                if loss_value is not None
                and gradient_value is not None
                and (update == 0 or update_value is not None)
                and parameter_nonfinite == 0
                and optimizer_nonfinite == 0
                else "nonfinite"
            ),
        }

    trajectory_rows = [
        trajectory_row(
            run_variant,
            0,
            values["loss"],
            values["gradient_norm"],
            None,
        )
        for run_variant, values in initial.items()
    ]
    first_update_rows = []
    for update in range(1, update_count + 1):
        images, targets = load_training_batch(
            training,
            batches[update - 1],
            task_name=task_name,
            seed=source_seed,
            epoch=continuation_epoch,
        )
        previous_params = {name: value[0] for name, value in states.items()}
        losses = {}
        diagnostics = {}
        for run_variant, execution_variant in run_to_execution.items():
            params, optimizer_state = states[run_variant]
            updated_params, updated_optimizer, loss, diagnostic = train_steps[
                execution_variant
            ](params, optimizer_state, images, targets)
            _synchronize((updated_params, updated_optimizer, loss, diagnostic))
            states[run_variant] = (updated_params, updated_optimizer)
            losses[run_variant] = loss
            diagnostics[run_variant] = diagnostic
        if update in observations:
            repeat_exact = bool(
                repeat_exact
                and trees_bitwise_equal(states["R"][0], states["R_repeat"][0])
                and trees_bitwise_equal(
                    states["R"][1].first_moment,
                    states["R_repeat"][1].first_moment,
                )
                and trees_bitwise_equal(
                    states["R"][1].second_moment,
                    states["R_repeat"][1].second_moment,
                )
                and arrays_bitwise_equal(states["R"][1].step, states["R_repeat"][1].step)
                and arrays_bitwise_equal(losses["R"], losses["R_repeat"])
                and arrays_bitwise_equal(
                    diagnostics["R"].gradient_norm,
                    diagnostics["R_repeat"].gradient_norm,
                )
                and arrays_bitwise_equal(
                    diagnostics["R"].update_norm,
                    diagnostics["R_repeat"].update_norm,
                )
            )
            if repeat_policy == "bitwise" and not repeat_exact:
                raise RuntimeError(f"unperturbed R repeat diverged by observation {update}")
        if update == 1:
            reference_update = jax.tree_util.tree_map(
                lambda before, after: before - after,
                previous_params["R"],
                states["R"][0],
            )
            for run_variant in run_to_execution:
                candidate_update = jax.tree_util.tree_map(
                    lambda before, after: before - after,
                    previous_params[run_variant],
                    states[run_variant][0],
                )
                first_update_rows.extend(
                    _comparison_artifact_rows(
                        base=run_identity,
                        source_fields=source_fields,
                        run_variant=run_variant,
                        comparison_type="parameter_update",
                        update=1,
                        reference=reference_update,
                        candidate=candidate_update,
                    )
                )
                for comparison_type, reference_tree, candidate_tree in (
                    (
                        "updated_parameter",
                        states["R"][0],
                        states[run_variant][0],
                    ),
                    (
                        "updated_first_moment",
                        states["R"][1].first_moment,
                        states[run_variant][1].first_moment,
                    ),
                    (
                        "updated_second_moment",
                        states["R"][1].second_moment,
                        states[run_variant][1].second_moment,
                    ),
                ):
                    first_update_rows.extend(
                        _comparison_artifact_rows(
                            base=run_identity,
                            source_fields=source_fields,
                            run_variant=run_variant,
                            comparison_type=comparison_type,
                            update=1,
                            reference=reference_tree,
                            candidate=candidate_tree,
                        )
                    )
        if update in observations:
            trajectory_rows.extend(
                trajectory_row(
                    run_variant,
                    update,
                    losses[run_variant],
                    diagnostics[run_variant].gradient_norm,
                    diagnostics[run_variant].update_norm,
                )
                for run_variant in run_to_execution
            )
    append_jsonl(destination / "first_update_comparisons.jsonl", first_update_rows)
    append_jsonl(destination / "trajectory.jsonl", trajectory_rows)
    status_rows = initial_rows + first_update_rows + trajectory_rows
    status = "ok" if all(row["status"] == "ok" for row in status_rows) else "nonfinite"
    artifact_sha256 = {
        name: _file_sha256(destination / name)
        for name in (
            "first_batch_outputs.npz",
            "first_update_comparisons.jsonl",
            "initial_comparisons.jsonl",
            "trajectory.jsonl",
        )
    }
    write_json(
        destination / "summary.json",
        {
            **run_identity,
            **source_fields,
            "status": status,
            "task": task_name,
            "source_seed": source_seed,
            "execution_variants": list(execution_variants),
            "run_variants": list(run_to_execution),
            "continuation_epoch": continuation_epoch,
            "continuation_updates": update_count,
            "observation_updates": sorted(observations),
            "unperturbed_repeat_check_updates": sorted(observations),
            "unperturbed_repeat_policy": repeat_policy,
            "continuation_index_schedule_sha256": _index_schedule_sha256(
                batches[:update_count]
            ),
            "continuation_batch_sizes": [len(batch) for batch in batches[:update_count]],
            "initial_comparison_rows": len(initial_rows),
            "first_update_comparison_rows": len(first_update_rows),
            "trajectory_rows": len(trajectory_rows),
            "unperturbed_repeat_bitwise_exact": repeat_exact,
            "one_ulp_perturbation": perturbation,
            "artifact_sha256": artifact_sha256,
            "curvature_gradient_applicable": task_name == "isic16_poincare_unet",
        },
    )
