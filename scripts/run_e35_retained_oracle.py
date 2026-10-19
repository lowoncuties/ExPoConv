#!/usr/bin/env python3
"""E35: FP64-oracle accuracy of the E07-J2 retained-state ResNet-20 FP32 gradients.

E07-J2 compared the FP32 R, F, and T raw loss gradients with one another at the
retained seed-11 E10 final-update states and its update-0 batch. This script
evaluates the same gradients and the E33 fixed-margin FP64 oracle so that each path's
error can be measured. One invocation runs one stage in its own process:

- ``--stage float32``: R, F, and T in FP32 with 64-bit types disabled, as in E07-J2.
- ``--stage float64``: the oracle ``R_float64_fixed_float32_margin``, i.e. R in FP64
  on the FP32 state and batch cast to FP64 with every projection margin fixed to
  the FP32 value.

Each stage writes one NPZ of gradient tensors per source state and path plus a JSON
record. scripts/export_e35_retained_oracle.py compares them.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from run_oracle_accuracy import ORACLE, fixed_projection_margin

from poincare_exec.artifacts.provenance import git_provenance, timestamp_utc
from poincare_exec.config.loader import config_hash, load_yaml
from poincare_exec.data.datasets import load_dataset
from poincare_exec.experiments.benchmark import (
    _gpu_snapshot,
    _require_idle_gpu,
    _selected_physical_gpu,
)
from poincare_exec.experiments.numerics import _cast_tree
from poincare_exec.experiments.retained_diagnostics import (
    _make_loss_gradient,
    _path_string,
)
from poincare_exec.training.batches import (
    load_training_batch,
    materialize_dataset,
    ordered_batches,
)
from poincare_exec.training.checkpoints import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocols/protocol_1_20.yaml"
CONFIG = ROOT / "configs/checkpoints/e35_retained_oracle.yaml"
SOURCE_REPOSITORY = Path("/data2/joc0027/poincare-execution-experiments")
E07_DIRECTORY = "artifacts/checkpoint_experiments/journal_diagnostics_v2/e10"
STAGES = ("float32", "float64")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_design() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the protocol snapshot and the E35 config and check their bound inputs."""

    protocol = load_yaml(PROTOCOL)
    config = load_yaml(CONFIG)
    if protocol["revision_benchmarks"]["E35"]["config"] != str(CONFIG.relative_to(ROOT)):
        raise ValueError("protocol does not register this E35 config")
    for field in ("reference_config", "reference_gradient_table", "checkpoint_inventory"):
        bound = config[field]
        if file_sha256(ROOT / bound["path"]) != bound["sha256"]:
            raise ValueError(f"{bound['path']} differs from the registered SHA-256")
    margins = protocol["numerics"]["float32_contract_oracle_projection_margin"]
    if margins != protocol["projection"]["float32_relative_margin"]:
        raise ValueError("oracle margin differs from the FP32 projection margin")
    if config["oracle"]["name"] != ORACLE:
        raise ValueError("E35 oracle differs from the E33 oracle")
    return protocol, config


def inventory_rows(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return the registered final checkpoint row for each source variant."""

    with (ROOT / config["checkpoint_inventory"]["path"]).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for variant in config["source_variants"]:
        matches = [
            row
            for row in rows
            if row["experiment_id"] == config["training_experiment_id"]
            and row["task"] == config["task"]
            and row["variant"] == variant
            and int(row["seed"]) == int(config["source_seed"])
            and row["role"] == config["source_role"]
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one inventory row for source {variant}")
        selected[variant] = matches[0]
    return selected


def rebuild_batch(
    config: dict[str, Any], task: dict[str, Any], source_repository: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the E07-J2 update-0 batch through the frozen data path."""

    training_config = load_yaml(ROOT / config["training_config"])
    training = materialize_dataset(
        load_dataset(
            training_config["dataset"],
            source_repository / "data/raw",
            ROOT / "splits",
            "train",
        )
    )
    seed = int(config["source_seed"])
    epoch = int(config["continuation_epoch"])
    indices = ordered_batches(len(training), int(task["batch_size"]), seed, epoch)[
        int(config["batch_position"])
    ]
    images, targets = load_training_batch(
        training, indices, task_name=config["task"], seed=seed, epoch=epoch
    )
    return images, targets, np.asarray(indices)


def check_e07_source(
    e07_directory: Path,
    variant: str,
    checkpoint_sha256: str,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
) -> tuple[dict[str, Any], Any]:
    """Require the E07-J2 record of this source to match the checkpoint and batch."""

    run_directory = e07_directory / variant / f"seed_{seed}"
    with (run_directory / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    outputs_path = run_directory / "first_batch_outputs.npz"
    if summary["source_checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"E07-J2 source {variant} used a different checkpoint")
    if summary["artifact_sha256"]["first_batch_outputs.npz"] != file_sha256(outputs_path):
        raise ValueError(f"E07-J2 outputs of source {variant} changed after the run")
    outputs = np.load(outputs_path)
    images, targets, indices = batch
    equal = {
        name: bool(
            outputs[name].dtype == value.dtype and np.array_equal(outputs[name], value)
        )
        for name, value in (
            ("inputs", images),
            ("targets", targets),
            ("batch_indices", indices),
        )
    }
    if not all(equal.values()):
        raise ValueError(f"rebuilt batch differs from E07-J2 source {variant}: {equal}")
    record = {
        "e07_summary_sha256": file_sha256(run_directory / "summary.json"),
        "e07_first_batch_outputs_sha256": file_sha256(outputs_path),
        "e07_code_commit": summary["code_commit"],
        "e07_device_id": summary["device_id"],
        "batch_equal_to_e07": equal,
    }
    return record, outputs


def gradient_arrays(gradients: Any) -> tuple[list[str], list[np.ndarray]]:
    leaves = jax.tree_util.tree_flatten_with_path(gradients)[0]
    paths = [_path_string(path) for path, _ in leaves]
    arrays = [np.asarray(jax.device_get(leaf)) for _, leaf in leaves]
    return paths, arrays


def save_gradients(
    path: Path, paths: list[str], arrays: list[np.ndarray], loss: Any, logits: Any
) -> None:
    np.savez(
        path,
        paths=np.asarray(paths),
        loss=np.asarray(jax.device_get(loss)),
        logits=np.asarray(jax.device_get(logits)),
        **{f"gradient_{index:03d}": array for index, array in enumerate(arrays)},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--source-repository", type=Path, default=SOURCE_REPOSITORY)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    provenance = git_provenance()  # before any output is written
    started = timestamp_utc()
    if provenance["dirty"]:
        raise RuntimeError("E35 requires a clean code tree")
    stage = arguments.stage
    if stage == "float32" and arguments.device != "gpu":
        raise RuntimeError("E35 FP32 paths must run on the E07-J2 GPU")
    protocol, config = load_design()
    task = protocol["tasks"][config["task"]]
    output_dir = arguments.output_dir.expanduser().resolve()
    record_path = output_dir / f"e35_{stage}_record.json"
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite {record_path}")
    # Check idleness before this process initializes the GPU.
    snapshot_before = _require_idle_gpu(config) if arguments.device == "gpu" else None
    jax.config.update("jax_enable_x64", stage == "float64")
    jax.config.update("jax_default_matmul_precision", "highest")
    device = jax.devices(arguments.device)[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    source_repository = arguments.source_repository.expanduser().resolve()
    e07_directory = source_repository / E07_DIRECTORY
    batch = rebuild_batch(config, task, source_repository)
    images, targets, indices = batch
    dtype = jnp.float32 if stage == "float32" else jnp.float64
    variants = config["execution_variants"] if stage == "float32" else ["R"]
    margin = float(protocol["numerics"]["float32_contract_oracle_projection_margin"])
    functions = {
        variant: _make_loss_gradient(config["task"], task, variant) for variant in variants
    }

    sources = []
    with (
        jax.default_device(device),
        fixed_projection_margin(margin) if stage == "float64" else contextlib.nullcontext(),
    ):
        for source_variant, inventory in inventory_rows(config).items():
            checkpoint_path = source_repository / inventory["path"]
            checkpoint_sha256 = file_sha256(checkpoint_path)
            if checkpoint_sha256 != inventory["sha256"]:
                raise ValueError(f"checkpoint {checkpoint_path} differs from inventory")
            checkpoint = load_checkpoint(checkpoint_path)
            identity = (
                checkpoint["task"],
                checkpoint["variant"],
                int(checkpoint["seed"]),
                int(checkpoint["update"]),
            )
            expected = (
                config["task"],
                source_variant,
                int(config["source_seed"]),
                int(config["source_update"]),
            )
            if identity != expected:
                raise ValueError(f"unexpected checkpoint identity {identity}")
            e07_record, e07_outputs = check_e07_source(
                e07_directory,
                source_variant,
                checkpoint_sha256,
                batch,
                int(config["source_seed"]),
            )
            params = _cast_tree(checkpoint["params"], dtype)
            inputs = jnp.asarray(images, dtype=dtype)
            results = {}
            for variant in variants:
                begin = time.perf_counter()
                (loss, logits), gradients = functions[variant](params, inputs, targets)
                jax.block_until_ready((loss, logits, gradients))
                seconds = time.perf_counter() - begin
                paths, arrays = gradient_arrays(gradients)
                dtypes = sorted({str(array.dtype) for array in arrays})
                if dtypes != [jnp.dtype(dtype).name] or logits.dtype != dtype:
                    raise RuntimeError(f"{variant} did not run in {stage}: {dtypes}")
                if not all(np.isfinite(array).all() for array in arrays):
                    raise RuntimeError(f"{source_variant}/{variant} gradient is nonfinite")
                name = f"e35_{stage}_{source_variant}_{variant}.npz"
                save_gradients(output_dir / name, paths, arrays, loss, logits)
                results[variant] = {
                    "file": name,
                    "sha256": file_sha256(output_dir / name),
                    "loss": float(loss),
                    "tensor_count": len(arrays),
                    "element_count": int(sum(array.size for array in arrays)),
                    "gradient_dtype": dtypes[0],
                    "seconds_including_first_compilation": seconds,
                    "logits_bitwise_equal_to_e07": (
                        bool(
                            np.asarray(logits).tobytes()
                            == e07_outputs[f"{variant}__logits"].tobytes()
                        )
                        if stage == "float32"
                        else None
                    ),
                }
            sources.append(
                {
                    "source_variant": source_variant,
                    "checkpoint_path": inventory["path"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "source_update": int(checkpoint["update"]),
                    **e07_record,
                    "results": results,
                }
            )

    record = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "stage": stage,
        "evaluated_paths": variants if stage == "float32" else [ORACLE],
        "code_commit": provenance["commit"],
        "code_dirty": provenance["dirty"],
        "protocol_version": protocol["protocol_version"],
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_hash": config_hash(protocol, config),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "matmul_precision": "highest",
        "projection_margin": margin if stage == "float64" else None,
        "device": str(device),
        "device_platform": device.platform,
        "device_kind": device.device_kind,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "gpu_snapshot_before": snapshot_before,
        "gpu_snapshot_after": (
            _gpu_snapshot(_selected_physical_gpu()) if device.platform == "gpu" else None
        ),
        "batch": {
            "continuation_epoch": int(config["continuation_epoch"]),
            "batch_position": int(config["batch_position"]),
            "examples": len(indices),
            "indices_sha256": hashlib.sha256(
                np.asarray(indices, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "sources": sources,
        "started_utc": started,
        "finished_utc": timestamp_utc(),
    }
    with record_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"stage": stage, "device": str(device), "sources": len(sources)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
