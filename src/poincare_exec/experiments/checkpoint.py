"""E06 replay of retained training checkpoints through the R/F/T execution paths."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np

from poincare_exec.artifacts import append_jsonl, base_result_row, write_json
from poincare_exec.config.loader import config_hash, load_yaml, require_keys
from poincare_exec.core.geometry import projection_margin
from poincare_exec.data.datasets import load_dataset
from poincare_exec.training.batches import (
    materialize_dataset,
)
from poincare_exec.training.checkpoints import load_checkpoint
from poincare_exec.training.tasks import (
    forward_task_with_intermediates,
)


def _synchronize(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _curvature(task_name: str, task: dict[str, Any], params: Any) -> float:
    if task_name == "isic16_poincare_unet":
        value = jax.nn.softplus(params["curvature_raw"])
        value.block_until_ready()
        return float(value)
    return float(task["curvature"]["initial_magnitude"])


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_array_statistics() -> dict[str, float | int]:
    return {
        "element_count": 0,
        "difference_sq": 0.0,
        "reference_sq": 0.0,
        "max_absolute_error": 0.0,
        "nonfinite_count": 0,
        "reference_nonfinite_count": 0,
        "comparison_nonfinite_count": 0,
        "projection_saturation_count": 0,
        "projection_vector_count": 0,
        "maximum_normalized_radius": 0.0,
    }


def _update_array_statistics(
    statistics: dict[str, float | int],
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    curvature: float,
    projection_relative_tolerance: float,
) -> None:
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    reference_finite = np.isfinite(reference64)
    candidate_finite = np.isfinite(candidate64)
    finite = reference_finite & candidate_finite
    statistics["element_count"] += int(candidate64.size)
    statistics["nonfinite_count"] += int(
        candidate64.size - np.count_nonzero(candidate_finite)
    )
    statistics["reference_nonfinite_count"] += int(
        reference64.size - np.count_nonzero(reference_finite)
    )
    statistics["comparison_nonfinite_count"] += int(
        candidate64.size - np.count_nonzero(finite)
    )
    if np.all(finite):
        difference = candidate64 - reference64
        statistics["difference_sq"] += float(np.sum(np.square(difference)))
        statistics["reference_sq"] += float(np.sum(np.square(reference64)))
        statistics["max_absolute_error"] = max(
            float(statistics["max_absolute_error"]),
            float(np.max(np.abs(difference), initial=0.0)),
        )
    finite_vectors = np.all(np.isfinite(candidate64), axis=-1)
    norms = np.linalg.norm(candidate64, axis=-1)
    normalized_radii = np.tanh(math.sqrt(curvature) * norms)
    finite_radii = normalized_radii[finite_vectors]
    threshold = (1.0 - float(projection_margin(candidate.dtype))) * (
        1.0 - projection_relative_tolerance
    )
    statistics["projection_saturation_count"] += int(
        np.count_nonzero(finite_radii >= threshold)
    )
    statistics["projection_vector_count"] += int(finite_radii.size)
    if finite_radii.size:
        statistics["maximum_normalized_radius"] = max(
            float(statistics["maximum_normalized_radius"]),
            float(np.max(finite_radii)),
        )


def _finalize_array_statistics(
    statistics: dict[str, float | int],
) -> dict[str, float | int | None]:
    if int(statistics["comparison_nonfinite_count"]):
        maximum_error = None
        relative_error = None
    else:
        maximum_error = float(statistics["max_absolute_error"])
        relative_error = math.sqrt(float(statistics["difference_sq"])) / max(
            math.sqrt(float(statistics["reference_sq"])), 1.0e-12
        )
    return {
        "max_absolute_error": maximum_error,
        "relative_l2_error": relative_error,
        "element_count": int(statistics["element_count"]),
        "nonfinite_count": int(statistics["nonfinite_count"]),
        "reference_nonfinite_count": int(statistics["reference_nonfinite_count"]),
        "comparison_nonfinite_count": int(statistics["comparison_nonfinite_count"]),
        "projection_saturation_count": int(statistics["projection_saturation_count"]),
        "projection_vector_count": int(statistics["projection_vector_count"]),
        "maximum_normalized_radius": float(statistics["maximum_normalized_radius"]),
    }


def _compatibility_metrics(
    task_name: str,
    task: dict[str, Any],
    params: Any,
    development: Any,
    variants: list[str],
    *,
    activation_groups: tuple[str, ...],
    retained_activation_batches: int,
    projection_relative_tolerance: float,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    batch_size = int(task["evaluation_batch_size"])
    functions = {}
    for variant in variants:
        selected_variant = variant

        def apply(selected, inputs, *, selected_variant=selected_variant):
            logits, activations = forward_task_with_intermediates(
                task_name, task, selected, inputs, selected_variant
            )
            return logits, jax.nn.softmax(logits, axis=-1), activations

        functions[variant] = jax.jit(apply)
    task_sums = {
        variant: {
            "disagreements": 0,
            "decisions": 0,
            "affected_examples": 0,
            "metric_sum": 0.0,
            "examples": 0,
        }
        for variant in variants
    }
    array_sums: dict[str, dict[str, dict[str, float | int]]] = {
        variant: {} for variant in variants
    }
    output_batches: dict[str, list[np.ndarray]] = {
        f"{variant}__logits": [] for variant in variants
    }
    output_batches.update({f"{variant}__probabilities": [] for variant in variants})
    target_batches: list[np.ndarray] = []
    sample_ids: list[str] = []
    retained_activations: dict[str, list[np.ndarray]] = {}
    for batch_index, start in enumerate(range(0, len(development), batch_size)):
        indices = np.arange(start, min(start + batch_size, len(development)))
        images, targets = development.batch(indices)
        device_outputs = {
            variant: functions[variant](params, images) for variant in variants
        }
        _synchronize(device_outputs)
        host = {
            variant: (
                np.asarray(values[0]),
                np.asarray(values[1]),
                {name: np.asarray(value) for name, value in values[2].items()},
            )
            for variant, values in device_outputs.items()
        }
        target_batches.append(targets)
        sample_ids.extend(development.sample_ids[int(index)] for index in indices)
        for variant, (logits, probabilities, activations) in host.items():
            output_batches[f"{variant}__logits"].append(logits)
            output_batches[f"{variant}__probabilities"].append(probabilities)
            if batch_index < retained_activation_batches:
                for name in activation_groups:
                    if name not in activations:
                        raise RuntimeError(
                            f"configured E06 activation group is unavailable: {name}"
                        )
                    retained_activations.setdefault(f"{variant}__{name}", []).append(
                        activations[name]
                    )
        reference, _, reference_activations = host["R"]
        reference_decision = np.argmax(reference, axis=-1)
        for variant, (candidate, probabilities, candidate_activations) in host.items():
            arrays = {
                "logits": candidate,
                **{name: candidate_activations[name] for name in activation_groups},
            }
            reference_arrays = {
                "logits": reference,
                **{name: reference_activations[name] for name in activation_groups},
            }
            for variable_group, values in arrays.items():
                group_statistics = array_sums[variant].setdefault(
                    variable_group, _new_array_statistics()
                )
                _update_array_statistics(
                    group_statistics,
                    reference_arrays[variable_group],
                    values,
                    curvature=_curvature(task_name, task, params),
                    projection_relative_tolerance=projection_relative_tolerance,
                )
            stats = task_sums[variant]
            stats.setdefault("nonfinite_probability_count", 0)
            stats["nonfinite_probability_count"] += int(
                probabilities.size - np.count_nonzero(np.isfinite(probabilities))
            )
            decision = np.argmax(candidate, axis=-1)
            changed = decision != reference_decision
            stats["disagreements"] += int(np.sum(changed))
            stats["decisions"] += int(decision.size)
            changed_by_example = np.reshape(changed, (len(indices), -1)).any(axis=1)
            stats["affected_examples"] += int(np.sum(changed_by_example))
            stats["examples"] += len(indices)
            if task_name == "isic16_poincare_unet":
                predicted = decision == 1
                expected = targets == 1
                axes = tuple(range(1, expected.ndim))
                true_positive = np.sum(predicted & expected, axis=axes)
                false_positive = np.sum(predicted & ~expected, axis=axes)
                false_negative = np.sum(~predicted & expected, axis=axes)
                denominator = 2 * true_positive + false_positive + false_negative
                dice = np.divide(
                    2 * true_positive,
                    denominator,
                    out=np.ones_like(denominator, dtype=np.float64),
                    where=denominator > 0,
                )
                stats["metric_sum"] += float(np.sum(dice))
            else:
                stats["metric_sum"] += int(np.sum(decision == targets))
    metrics: dict[str, dict[str, Any]] = {}
    for variant in variants:
        task_stats = task_sums[variant]
        metrics[variant] = {
            "task": {
                "decision_disagreement_fraction": int(task_stats["disagreements"])
                / int(task_stats["decisions"]),
                "affected_examples": int(task_stats["affected_examples"]),
                "affected_example_fraction": int(task_stats["affected_examples"])
                / int(task_stats["examples"]),
                "task_metric": float(task_stats["metric_sum"])
                / int(task_stats["examples"]),
                "examples": int(task_stats["examples"]),
                "nonfinite_probability_count": int(
                    task_stats["nonfinite_probability_count"]
                ),
            },
            "arrays": {
                name: _finalize_array_statistics(statistics)
                for name, statistics in array_sums[variant].items()
            },
        }
    raw_outputs = {
        "sample_ids": np.asarray(sample_ids),
        "targets": np.concatenate(target_batches),
        **{name: np.concatenate(batches) for name, batches in output_batches.items()},
    }
    retained = {
        name: np.concatenate(batches) for name, batches in retained_activations.items()
    }
    retained["sample_ids"] = np.asarray(
        sample_ids[: min(retained_activation_batches * batch_size, len(sample_ids))]
    )
    return metrics, raw_outputs, retained


def _compatibility_rows(
    protocol: dict[str, Any],
    *,
    config_digest: str,
    task_name: str,
    checkpoint_name: str,
    split: str,
    seed: int,
    metrics: dict[str, dict[str, Any]],
    projection_relative_tolerance: float,
    source_checkpoint: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    source_fields = {} if source_checkpoint is None else source_checkpoint
    for variant, variant_metrics in metrics.items():
        for variable_group, array_metrics in variant_metrics["arrays"].items():
            status = (
                "ok"
                if int(array_metrics["comparison_nonfinite_count"]) == 0
                else "nonfinite"
            )
            rows.append(
                {
                    **base_result_row(
                        protocol,
                        experiment_id="E06",
                        variant=variant,
                        configuration_hash=config_digest,
                        dtype="float32",
                        seed_or_session=seed,
                        status=status,
                    ),
                    "task": task_name,
                    "checkpoint": checkpoint_name,
                    "split": split,
                    "record_type": (
                        "output" if variable_group == "logits" else "activation"
                    ),
                    "variable_group": variable_group,
                    "bn_state": "current_batch_no_running_state",
                    "projection_detection_relative_tolerance": (
                        projection_relative_tolerance
                    ),
                    **source_fields,
                    **array_metrics,
                    **(variant_metrics["task"] if variable_group == "logits" else {}),
                }
            )
    return rows


def run_trained_checkpoint_evaluation(
    protocol_path: str | Path,
    checkpoint_config_path: str | Path,
    training_config_path: str | Path,
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Apply R/F/T to one retained final R/T checkpoint for E06."""

    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(protocol_path)
    checkpoint_config = load_yaml(checkpoint_config_path)
    training_config = load_yaml(training_config_path)
    require_keys(
        checkpoint_config,
        (
            "experiment_id",
            "variants",
            "retained_activation_batches",
            "projection_detection_relative_tolerance",
            "post_training_splits",
            "tasks",
        ),
        name="checkpoint config",
    )
    require_keys(
        training_config,
        ("experiment_id", "task", "dataset", "variants", "seeds"),
        name="training config",
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError("E06 retained-checkpoint evaluation requires CUDA")
    source_path = Path(checkpoint_path).expanduser().resolve()
    source = load_checkpoint(source_path)
    task_name = str(training_config["task"])
    task = protocol["tasks"][task_name]
    task_entry = next(
        item for item in checkpoint_config["tasks"] if item["task"] == task_name
    )
    if task_entry["dataset"] != training_config["dataset"]:
        raise RuntimeError("training and checkpoint configs select different datasets")
    source_variant = str(source["variant"])
    seed = int(source["seed"])
    update = int(source["update"])
    expected_training_hash = config_hash(protocol, training_config)
    source_identity = (
        source.get("config_hash"),
        source.get("protocol_version"),
        source.get("task"),
        source_variant,
        seed,
        update,
    )
    expected_identity = (
        expected_training_hash,
        protocol["protocol_version"],
        task_name,
        source_variant,
        seed,
        int(task["max_updates"]),
    )
    if source_identity != expected_identity:
        raise RuntimeError("retained checkpoint does not match the frozen E06 campaign")
    if source_variant not in {"R", "T"}:
        raise ValueError("E06 retained checkpoint must originate from R or T")
    if (
        source_variant not in training_config["variants"]
        or seed not in training_config["seeds"]
    ):
        raise ValueError("retained checkpoint is outside the frozen training matrix")
    if source.get("code_dirty") is not False or not source.get("code_commit"):
        raise RuntimeError("retained checkpoint lacks clean-code provenance")

    config_digest = config_hash(protocol, checkpoint_config, task_entry, training_config)
    run_identity = base_result_row(
        protocol,
        experiment_id="E06",
        variant="shared",
        configuration_hash=config_digest,
        dtype="float32",
        seed_or_session=seed,
    )
    if run_identity["code_dirty"]:
        raise RuntimeError("E06 retained-checkpoint evaluation requires a clean tree")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    splits = tuple(str(value) for value in checkpoint_config["post_training_splits"])
    if not splits or any(split not in {"test", "test_deduplicated"} for split in splits):
        raise ValueError("post_training_splits contains an unsupported partition")
    expected_files = [summary_path, destination / "e06_compatibility.jsonl"]
    for split in splits:
        expected_files.extend(
            (
                destination / f"e06_{split}_outputs.npz",
                destination / f"e06_{split}_selected_activations.npz",
            )
        )
    existing = [str(path) for path in expected_files if path.exists()]
    if existing:
        raise FileExistsError(
            "retained-checkpoint evaluation requires audit before rerun: "
            + ", ".join(existing)
        )

    activation_groups = tuple(task_entry["activation_groups"])
    retained_activation_batches = int(checkpoint_config["retained_activation_batches"])
    projection_relative_tolerance = float(
        checkpoint_config["projection_detection_relative_tolerance"]
    )
    if retained_activation_batches <= 0:
        raise ValueError("retained_activation_batches must be positive")
    if not 0.0 <= projection_relative_tolerance < 1.0:
        raise ValueError("projection_detection_relative_tolerance must be in [0, 1)")
    checkpoint_name = (
        f"{training_config['experiment_id']}:{source_variant}:seed_{seed}:"
        f"final_update_{update}"
    )
    source_sha256 = _file_sha256(source_path)
    source_fields = {
        "source_checkpoint_experiment": training_config["experiment_id"],
        "source_checkpoint_variant": source_variant,
        "source_checkpoint_seed": seed,
        "source_checkpoint_update": update,
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_code_commit": source.get("code_commit"),
        "source_checkpoint_code_dirty": source.get("code_dirty"),
    }
    all_metrics = {}
    compatibility_rows = []
    for split in splits:
        dataset = materialize_dataset(
            load_dataset(training_config["dataset"], data_root, manifest_dir, split)
        )
        metrics, raw_outputs, retained_activations = _compatibility_metrics(
            task_name,
            task,
            source["params"],
            dataset,
            list(checkpoint_config["variants"]),
            activation_groups=activation_groups,
            retained_activation_batches=retained_activation_batches,
            projection_relative_tolerance=projection_relative_tolerance,
        )
        _save_npz(destination / f"e06_{split}_outputs.npz", **raw_outputs)
        _save_npz(
            destination / f"e06_{split}_selected_activations.npz",
            **retained_activations,
        )
        compatibility_rows.extend(
            _compatibility_rows(
                protocol,
                config_digest=config_digest,
                task_name=task_name,
                checkpoint_name=checkpoint_name,
                split=split,
                seed=seed,
                metrics=metrics,
                projection_relative_tolerance=projection_relative_tolerance,
                source_checkpoint=source_fields,
            )
        )
        all_metrics[split] = metrics
    append_jsonl(destination / "e06_compatibility.jsonl", compatibility_rows)
    write_json(
        summary_path,
        {
            **run_identity,
            **source_fields,
            "task": task_name,
            "checkpoint": checkpoint_name,
            "config_hash": config_digest,
            "splits": list(splits),
            "activation_groups": list(activation_groups),
            "retained_activation_batches": retained_activation_batches,
            "projection_detection_relative_tolerance": (projection_relative_tolerance),
            "compatibility": all_metrics,
        },
    )
