"""Publication-scale matched training workers for E09 and E10."""

from __future__ import annotations

import math
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from poincare_exec.artifacts import append_jsonl, base_result_row, write_json
from poincare_exec.artifacts.provenance import timestamp_utc
from poincare_exec.config.loader import config_hash, load_yaml, require_keys
from poincare_exec.data.datasets import DatasetBundle, load_dataset
from poincare_exec.training.batches import (
    load_training_batch,
    materialize_dataset,
    ordered_batches,
)
from poincare_exec.training.checkpoints import load_checkpoint, save_checkpoint
from poincare_exec.training.diagnostics import finite_float, tree_nonfinite_count
from poincare_exec.training.optimizers import AdamState, init_adam
from poincare_exec.training.tasks import (
    forward_task,
    initialize_task,
    make_train_step,
    task_loss,
)

Array = jax.Array


def _synchronize(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _compile_train_steps(
    train_step,
    params: Any,
    state: AdamState,
    dataset: DatasetBundle,
    batch_size: int,
) -> tuple[dict[int, Any], dict[int, float]]:
    shapes = sorted({batch_size, len(dataset) % batch_size or batch_size})
    compiled = {}
    compilation_seconds = {}
    image_shape = dataset.images.shape[1:]
    target_shape = dataset.labels.shape[1:]
    for count in shapes:
        inputs = jnp.zeros((count, *image_shape), dtype=jnp.float32)
        targets = jnp.zeros((count, *target_shape), dtype=jnp.int32)
        started = time.perf_counter()
        compiled[count] = train_step.lower(params, state, inputs, targets).compile()
        compilation_seconds[count] = time.perf_counter() - started
    return compiled, compilation_seconds


def _segmentation_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=-1).astype(np.uint8)
    predicted = predictions == 1
    expected = targets == 1
    axes = tuple(range(1, expected.ndim))
    true_positive = np.sum(predicted & expected, axis=axes)
    false_positive = np.sum(predicted & ~expected, axis=axes)
    false_negative = np.sum(~predicted & expected, axis=axes)
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    dice = np.where(dice_denominator > 0, 2 * true_positive / dice_denominator, 1.0)
    iou = np.where(iou_denominator > 0, true_positive / iou_denominator, 1.0)
    return {"predictions": predictions, "dice": dice, "iou": iou}


def _make_evaluation_step(
    task_name: str,
    task: dict[str, Any],
    variant: str,
) -> Any:
    """Create one reusable compiled evaluation boundary per training worker."""

    @jax.jit
    def evaluation_step(selected_params, inputs, targets):
        logits = forward_task(task_name, task, selected_params, inputs, variant)
        loss = task_loss(task_name, task, logits, targets)
        return loss, logits

    return evaluation_step


def _evaluate(
    task_name: str,
    task: dict[str, Any],
    params: Any,
    dataset: DatasetBundle,
    evaluation_step: Any,
    *,
    retain_predictions: bool,
) -> tuple[dict[str, float | int | None], np.ndarray | None]:
    batch_size = int(task["evaluation_batch_size"])

    weighted_loss = 0.0
    finite_losses = True
    output_batches = []
    target_batches = []
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        images, targets = dataset.batch(indices)
        loss, logits = evaluation_step(params, images, targets)
        _synchronize((loss, logits))
        loss_value = finite_float(loss)
        if loss_value is None:
            finite_losses = False
        else:
            weighted_loss += loss_value * len(indices)
        output_batches.append(np.asarray(logits))
        target_batches.append(targets)
    outputs = np.concatenate(output_batches)
    targets = np.concatenate(target_batches)
    output_nonfinite_count = int(outputs.size - np.count_nonzero(np.isfinite(outputs)))
    outputs_are_finite = output_nonfinite_count == 0
    metrics: dict[str, float | int | None] = {
        "examples": len(dataset),
        "loss": weighted_loss / len(dataset) if finite_losses else None,
        "output_nonfinite_count": output_nonfinite_count,
    }
    if task_name == "isic16_poincare_unet":
        segmentation = _segmentation_metrics(outputs, targets)
        metrics["mean_foreground_dice"] = (
            float(np.mean(segmentation["dice"])) if outputs_are_finite else None
        )
        metrics["mean_foreground_iou"] = (
            float(np.mean(segmentation["iou"])) if outputs_are_finite else None
        )
        predictions = segmentation["predictions"]
    else:
        predictions = np.argmax(outputs, axis=-1).astype(np.int16)
        metrics["top1_accuracy"] = (
            float(np.mean(predictions == targets)) if outputs_are_finite else None
        )
    return metrics, predictions if retain_predictions else None


def _save_predictions(
    path: Path, sample_ids: tuple[str, ...], predictions: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary_name,
            sample_id=np.asarray(sample_ids),
            prediction=predictions,
        )
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _checkpoint_payload(
    *,
    params: Any,
    optimizer_state: AdamState,
    config_digest: str,
    task_name: str,
    variant: str,
    seed: int,
    epoch: int,
    update: int,
    elapsed_seconds: float,
    compute_seconds: float,
    best_metric: float,
    best_update: int,
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "poincare-exec-checkpoint-v1",
        "config_hash": config_digest,
        "protocol_version": run_identity["protocol_version"],
        "code_commit": run_identity["code_commit"],
        "code_dirty": run_identity["code_dirty"],
        "task": task_name,
        "variant": variant,
        "seed": seed,
        "epoch": epoch,
        "update": update,
        "elapsed_seconds": elapsed_seconds,
        "compute_seconds": compute_seconds,
        "best_metric": best_metric,
        "best_update": best_update,
        "params": params,
        "optimizer_state": optimizer_state,
    }


def _run_training_worker(
    protocol_path: str | Path,
    config_path: str | Path,
    *,
    variant: str,
    seed: int,
    data_root: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
) -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    require_keys(
        config,
        ("experiment_id", "task", "dataset", "variants", "seeds"),
        name="training config",
    )
    if variant not in config["variants"] or seed not in config["seeds"]:
        raise ValueError("variant/seed is outside the frozen training campaign")
    if jax.default_backend() != "gpu":
        raise RuntimeError("publication training requires a CUDA backend")
    task_name = config["task"]
    task = protocol["tasks"][task_name]
    config_digest = config_hash(protocol, config)
    base = base_result_row(
        protocol,
        experiment_id=config["experiment_id"],
        variant=variant,
        configuration_hash=config_digest,
        dtype="float32",
        seed_or_session=seed,
    )
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        raise FileExistsError(f"completed training run already exists: {run_dir}")

    run_started = time.perf_counter()
    train_data = materialize_dataset(
        load_dataset(config["dataset"], data_root, manifest_dir, "train")
    )
    development_data = materialize_dataset(
        load_dataset(config["dataset"], data_root, manifest_dir, "development")
    )
    batch_size = int(task["batch_size"])
    updates_per_epoch = math.ceil(len(train_data) / batch_size)
    expected_updates = updates_per_epoch * int(task["epochs"])
    if expected_updates != int(task["max_updates"]):
        raise RuntimeError(
            "frozen max_updates disagrees with the manifest and batch policy"
        )

    resume_path = run_dir / "resume.pkl"
    if resume_path.exists():
        resumed = load_checkpoint(resume_path)
        identity = (
            resumed["config_hash"],
            resumed.get("protocol_version"),
            resumed.get("code_commit"),
            resumed.get("code_dirty"),
            resumed["task"],
            resumed["variant"],
            int(resumed["seed"]),
        )
        expected_identity = (
            config_digest,
            base["protocol_version"],
            base["code_commit"],
            base["code_dirty"],
            task_name,
            variant,
            seed,
        )
        if identity != expected_identity:
            raise RuntimeError("resume checkpoint identity does not match this run")
        params = resumed["params"]
        optimizer_state = resumed["optimizer_state"]
        first_epoch = int(resumed["epoch"]) + 1
        update = int(resumed["update"])
        elapsed_offset = float(resumed["elapsed_seconds"])
        compute_seconds = float(resumed["compute_seconds"])
        best_metric = float(resumed["best_metric"])
        best_update = int(resumed["best_update"])
    else:
        params = initialize_task(task_name, seed, task)
        optimizer_state = init_adam(params)
        first_epoch = 1
        update = 0
        elapsed_offset = 0.0
        compute_seconds = 0.0
        best_metric = -math.inf
        best_update = -1

    def elapsed() -> float:
        return elapsed_offset + time.perf_counter() - run_started

    checkpoint_steps = {
        0: "initial",
        math.ceil(expected_updates * 0.10): "10_percent",
        math.ceil(expected_updates * 0.50): "50_percent",
        expected_updates: "final",
    }
    if update == 0:
        save_checkpoint(
            run_dir / "checkpoints" / "initial.pkl",
            _checkpoint_payload(
                params=params,
                optimizer_state=optimizer_state,
                config_digest=config_digest,
                task_name=task_name,
                variant=variant,
                seed=seed,
                epoch=0,
                update=0,
                elapsed_seconds=elapsed(),
                compute_seconds=compute_seconds,
                best_metric=best_metric,
                best_update=best_update,
                run_identity=base,
            ),
        )

    train_step = make_train_step(task_name, task, variant)
    compiled_steps, compilation_seconds = _compile_train_steps(
        train_step, params, optimizer_state, train_data, batch_size
    )
    evaluation_step = _make_evaluation_step(task_name, task, variant)

    def finish_nonfinite(*, epoch: int, reason: str, details: dict[str, Any]) -> None:
        failure_checkpoint = f"nonfinite_update_{update}.pkl"
        save_checkpoint(
            run_dir / "checkpoints" / failure_checkpoint,
            _checkpoint_payload(
                params=params,
                optimizer_state=optimizer_state,
                config_digest=config_digest,
                task_name=task_name,
                variant=variant,
                seed=seed,
                epoch=epoch,
                update=update,
                elapsed_seconds=elapsed(),
                compute_seconds=compute_seconds,
                best_metric=best_metric,
                best_update=best_update,
                run_identity=base,
            ),
        )
        write_json(
            run_dir / "summary.json",
            {
                **base,
                "status": "nonfinite",
                "task": task_name,
                "epochs": int(task["epochs"]),
                "stopped_epoch": epoch,
                "updates": update,
                "expected_updates": expected_updates,
                "batch_size": batch_size,
                "remainder_handling": task["remainder_handling"],
                "compilation_seconds_by_batch_size": compilation_seconds,
                "compute_seconds": compute_seconds,
                "elapsed_seconds": elapsed(),
                "best_development_metric": finite_float(best_metric),
                "best_update": best_update if best_update >= 0 else None,
                "official_test": None,
                "deduplicated_test": None,
                "validation_best": None,
                "failure": {
                    "reason": reason,
                    "checkpoint": f"checkpoints/{failure_checkpoint}",
                    **details,
                },
                "completed_at_utc": timestamp_utc(),
            },
        )

    resume_payload = _checkpoint_payload(
        params=params,
        optimizer_state=optimizer_state,
        config_digest=config_digest,
        task_name=task_name,
        variant=variant,
        seed=seed,
        epoch=first_epoch - 1,
        update=update,
        elapsed_seconds=elapsed(),
        compute_seconds=compute_seconds,
        best_metric=best_metric,
        best_update=best_update,
        run_identity=base,
    )
    for epoch in range(first_epoch, int(task["epochs"]) + 1):
        epoch_rows = []
        for indices in ordered_batches(len(train_data), batch_size, seed, epoch):
            images, targets = load_training_batch(
                train_data,
                indices,
                task_name=task_name,
                seed=seed,
                epoch=epoch,
            )
            step_started = time.perf_counter()
            params, optimizer_state, loss, diagnostics = compiled_steps[len(indices)](
                params, optimizer_state, images, targets
            )
            _synchronize((params, optimizer_state, loss, diagnostics))
            step_seconds = time.perf_counter() - step_started
            compute_seconds += step_seconds
            update += 1
            curvature = finite_float(
                jax.nn.softplus(params["curvature_raw"])
                if task_name == "isic16_poincare_unet"
                else float(task["curvature"]["initial_magnitude"])
            )
            loss_value = finite_float(loss)
            gradient_norm = finite_float(diagnostics.gradient_norm)
            parameter_update_norm = finite_float(diagnostics.update_norm)
            parameter_nonfinite = tree_nonfinite_count(params)
            optimizer_nonfinite = tree_nonfinite_count(optimizer_state)
            status = (
                "ok"
                if loss_value is not None
                and gradient_norm is not None
                and parameter_update_norm is not None
                and curvature is not None
                and parameter_nonfinite == 0
                and optimizer_nonfinite == 0
                else "nonfinite"
            )
            epoch_rows.append(
                {
                    **base,
                    "status": status,
                    "task": task_name,
                    "split": "train",
                    "epoch": epoch,
                    "update": update,
                    "examples": len(indices),
                    "loss": loss_value,
                    "gradient_norm": gradient_norm,
                    "parameter_update_norm": parameter_update_norm,
                    "curvature": curvature,
                    "parameter_nonfinite_count": parameter_nonfinite,
                    "optimizer_nonfinite_count": optimizer_nonfinite,
                    "step_seconds": step_seconds,
                    "compute_seconds": compute_seconds,
                    "elapsed_seconds": elapsed(),
                }
            )
            if status == "nonfinite":
                append_jsonl(run_dir / "training_metrics.jsonl", epoch_rows)
                finish_nonfinite(
                    epoch=epoch,
                    reason="nonfinite_training_update",
                    details={
                        "loss": loss_value,
                        "gradient_norm": gradient_norm,
                        "parameter_update_norm": parameter_update_norm,
                        "curvature": curvature,
                        "parameter_nonfinite_count": parameter_nonfinite,
                        "optimizer_nonfinite_count": optimizer_nonfinite,
                    },
                )
                return
            if update in checkpoint_steps and update != expected_updates:
                name = checkpoint_steps[update]
                save_checkpoint(
                    run_dir / "checkpoints" / f"{name}.pkl",
                    _checkpoint_payload(
                        params=params,
                        optimizer_state=optimizer_state,
                        config_digest=config_digest,
                        task_name=task_name,
                        variant=variant,
                        seed=seed,
                        epoch=epoch,
                        update=update,
                        elapsed_seconds=elapsed(),
                        compute_seconds=compute_seconds,
                        best_metric=best_metric,
                        best_update=best_update,
                        run_identity=base,
                    ),
                )
        append_jsonl(run_dir / "training_metrics.jsonl", epoch_rows)
        development_metrics, _ = _evaluate(
            task_name,
            task,
            params,
            development_data,
            evaluation_step,
            retain_predictions=False,
        )
        primary_name = (
            "mean_foreground_dice"
            if task_name == "isic16_poincare_unet"
            else "top1_accuracy"
        )
        primary_value = finite_float(development_metrics[primary_name])
        development_loss = finite_float(development_metrics["loss"])
        development_curvature = finite_float(
            jax.nn.softplus(params["curvature_raw"])
            if task_name == "isic16_poincare_unet"
            else float(task["curvature"]["initial_magnitude"])
        )
        development_status = (
            "ok"
            if primary_value is not None
            and development_loss is not None
            and int(development_metrics["output_nonfinite_count"]) == 0
            and development_curvature is not None
            else "nonfinite"
        )
        append_jsonl(
            run_dir / "training_metrics.jsonl",
            [
                {
                    **base,
                    "status": development_status,
                    "task": task_name,
                    "split": "development",
                    "epoch": epoch,
                    "update": update,
                    "examples": int(development_metrics["examples"]),
                    "loss": development_loss,
                    "gradient_norm": None,
                    "parameter_update_norm": None,
                    "curvature": development_curvature,
                    "step_seconds": None,
                    "compute_seconds": compute_seconds,
                    "elapsed_seconds": elapsed(),
                    **{
                        key: value
                        for key, value in development_metrics.items()
                        if key not in {"examples", "loss"}
                    },
                }
            ],
        )
        if development_status == "nonfinite":
            finish_nonfinite(
                epoch=epoch,
                reason="nonfinite_development_evaluation",
                details={
                    "loss": development_loss,
                    primary_name: primary_value,
                    "output_nonfinite_count": int(
                        development_metrics["output_nonfinite_count"]
                    ),
                    "curvature": development_curvature,
                },
            )
            return
        assert primary_value is not None
        if primary_value > best_metric:
            best_metric = primary_value
            best_update = update
            save_checkpoint(
                run_dir / "checkpoints" / "validation_best.pkl",
                _checkpoint_payload(
                    params=params,
                    optimizer_state=optimizer_state,
                    config_digest=config_digest,
                    task_name=task_name,
                    variant=variant,
                    seed=seed,
                    epoch=epoch,
                    update=update,
                    elapsed_seconds=elapsed(),
                    compute_seconds=compute_seconds,
                    best_metric=best_metric,
                    best_update=best_update,
                    run_identity=base,
                ),
            )
        resume_payload = _checkpoint_payload(
            params=params,
            optimizer_state=optimizer_state,
            config_digest=config_digest,
            task_name=task_name,
            variant=variant,
            seed=seed,
            epoch=epoch,
            update=update,
            elapsed_seconds=elapsed(),
            compute_seconds=compute_seconds,
            best_metric=best_metric,
            best_update=best_update,
            run_identity=base,
        )
        save_checkpoint(resume_path, resume_payload)

    if update != expected_updates:
        raise RuntimeError(
            f"training ended at update {update}, expected {expected_updates}"
        )
    save_checkpoint(run_dir / "checkpoints" / "final.pkl", resume_payload)
    official_test = materialize_dataset(
        load_dataset(config["dataset"], data_root, manifest_dir, "test")
    )
    official_metrics, official_predictions = _evaluate(
        task_name,
        task,
        params,
        official_test,
        evaluation_step,
        retain_predictions=True,
    )
    deduplicated_test = materialize_dataset(
        load_dataset(config["dataset"], data_root, manifest_dir, "test_deduplicated")
    )
    deduplicated_metrics, deduplicated_predictions = _evaluate(
        task_name,
        task,
        params,
        deduplicated_test,
        evaluation_step,
        retain_predictions=True,
    )
    validation_best = load_checkpoint(run_dir / "checkpoints" / "validation_best.pkl")
    if int(validation_best["update"]) != best_update:
        raise RuntimeError(
            "validation-best checkpoint update disagrees with the training record"
        )
    validation_best_params = validation_best["params"]
    validation_best_official_metrics, validation_best_official_predictions = _evaluate(
        task_name,
        task,
        validation_best_params,
        official_test,
        evaluation_step,
        retain_predictions=True,
    )
    (
        validation_best_deduplicated_metrics,
        validation_best_deduplicated_predictions,
    ) = _evaluate(
        task_name,
        task,
        validation_best_params,
        deduplicated_test,
        evaluation_step,
        retain_predictions=True,
    )
    _save_predictions(
        run_dir / "predictions_official_test.npz",
        official_test.sample_ids,
        official_predictions,
    )
    _save_predictions(
        run_dir / "predictions_deduplicated_test.npz",
        deduplicated_test.sample_ids,
        deduplicated_predictions,
    )
    _save_predictions(
        run_dir / "predictions_validation_best_official_test.npz",
        official_test.sample_ids,
        validation_best_official_predictions,
    )
    _save_predictions(
        run_dir / "predictions_validation_best_deduplicated_test.npz",
        deduplicated_test.sample_ids,
        validation_best_deduplicated_predictions,
    )
    endpoint_metrics = (
        official_metrics,
        deduplicated_metrics,
        validation_best_official_metrics,
        validation_best_deduplicated_metrics,
    )
    final_status = (
        "ok"
        if all(
            metrics["loss"] is not None and int(metrics["output_nonfinite_count"]) == 0
            for metrics in endpoint_metrics
        )
        else "nonfinite"
    )
    write_json(
        run_dir / "summary.json",
        {
            **base,
            "status": final_status,
            "task": task_name,
            "epochs": int(task["epochs"]),
            "updates": update,
            "expected_updates": expected_updates,
            "batch_size": batch_size,
            "remainder_handling": task["remainder_handling"],
            "compilation_seconds_by_batch_size": compilation_seconds,
            "compute_seconds": compute_seconds,
            "elapsed_seconds": elapsed(),
            "best_development_metric": best_metric,
            "best_update": best_update,
            "official_test": official_metrics,
            "deduplicated_test": deduplicated_metrics,
            "validation_best": {
                "update": best_update,
                "official_test": validation_best_official_metrics,
                "deduplicated_test": validation_best_deduplicated_metrics,
            },
            "completed_at_utc": timestamp_utc(),
        },
    )


def run_training_worker(
    protocol_path: str | Path,
    config_path: str | Path,
    *,
    variant: str,
    seed: int,
    data_root: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Run or resume one frozen variant/seed, preserving any terminal failure."""

    output = Path(output_dir).expanduser().resolve()
    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    failure_base = base_result_row(
        protocol,
        experiment_id=config["experiment_id"],
        variant=variant,
        configuration_hash=config_hash(protocol, config),
        dtype="float32",
        seed_or_session=seed,
    )
    try:
        _run_training_worker(
            protocol_path,
            config_path,
            variant=variant,
            seed=seed,
            data_root=data_root,
            manifest_dir=manifest_dir,
            output_dir=output,
        )
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        failure_name = timestamp_utc().replace(":", "-") + ".failure.json"
        write_json(
            output / failure_name,
            {
                **failure_base,
                "status": "failed",
                "task": config["task"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
