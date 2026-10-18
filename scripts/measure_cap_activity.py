#!/usr/bin/env python3
"""E33 cap activity: how often the source-radius cap is active in trained models.

For every Poincare convolution, the normalized patch radius is
rho = tanh(sqrt(c) * ||P||), where P is the beta-scaled tangent patch. When
rho >= m_c = 1 - 0.004 (FP32), the clipped HNN++ score depends on P only through
the cosine D / (R * g) and the layer acts as a cosine-similarity convolution.

Each retained final-update checkpoint is evaluated in its own training variant on
the official test split with the registered evaluation batch size and manifest
order (normalization uses current-batch statistics, so batching matters). The
radius is evaluated in FP32 on each layer's actual input with F's expression:
R^2 = beta^2 * conv(sum_c t^2, ones).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import multiprocessing
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from run_oracle_accuracy import pin_cores

from poincare_exec.artifacts.provenance import git_provenance
from poincare_exec.config.constants import CONV_DIMENSION_NUMBERS
from poincare_exec.config.loader import load_yaml
from poincare_exec.core.hnnpp import beta_ratio, spatial_pair
from poincare_exec.data.datasets import load_dataset
from poincare_exec.models import common, resnet, unet
from poincare_exec.training.batches import materialize_dataset
from poincare_exec.training.checkpoints import load_checkpoint
from poincare_exec.training.tasks import forward_task

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path("/data2/joc0027/poincare-execution-experiments/artifacts/training")
DATA_ROOT = Path("/data2/joc0027/poincare-execution-experiments/data/raw")
EXPERIMENTS = {
    "E09": "configs/training/e09_isic16_unet.yaml",
    "E10": "configs/training/e10_cifar100_resnet20.yaml",
}


def cap_counts(params, tangent, curvature, cap, *, stride=1, padding="SAME", dilation=1):
    """Return (active positions, positions, sum of rho, max rho) for one convolution."""

    kernel_height, kernel_width, channels, _ = params.weight.shape
    ratio = beta_ratio(
        kernel_height * kernel_width * channels, channels, dtype=tangent.dtype
    )
    energy = lax.conv_general_dilated(
        jnp.sum(jnp.square(tangent), axis=-1, keepdims=True),
        jnp.ones((kernel_height, kernel_width, 1, 1), dtype=tangent.dtype),
        window_strides=spatial_pair(stride, name="stride"),
        padding=padding.upper(),
        rhs_dilation=spatial_pair(dilation, name="dilation"),
        dimension_numbers=CONV_DIMENSION_NUMBERS,
    )
    rho = jnp.tanh(jnp.sqrt(curvature * jnp.square(ratio) * energy))
    active = jnp.square(rho) >= jnp.square(cap)
    return jnp.sum(active), active.size, jnp.sum(rho), jnp.max(rho)


@contextlib.contextmanager
def recorded_convolutions(names: dict[int, str], records: list, cap) -> Iterator[None]:
    """Record cap statistics at every model-level Poincare convolution call."""

    modules = (common, unet, resnet)
    original = common.conv2d

    def recorded(params, tangent, curvature, **options):
        if options["variant"] != "E":
            selected = {
                key: options[key]
                for key in ("stride", "padding", "dilation")
                if key in options
            }
            records.append(
                (
                    names[id(params.weight)],
                    params.weight.shape[0],
                    cap_counts(params, tangent, curvature, cap, **selected),
                )
            )
        return original(params, tangent, curvature, **options)

    for module in modules:
        module.conv2d = recorded
    try:
        yield
    finally:
        for module in modules:
            module.conv2d = original


def measure(job: tuple[int, str, str, int]) -> list[dict[str, Any]]:
    """Measure one checkpoint over the full official test split."""

    block, experiment, variant, seed = job
    pin_cores(block)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(ROOT / "protocol.yaml")
    config = load_yaml(ROOT / EXPERIMENTS[experiment])
    task_name = config["task"]
    task = protocol["tasks"][task_name]
    path = (
        TRAINING_ROOT / experiment.lower() / variant / f"seed_{seed}/checkpoints/final.pkl"
    )
    checkpoint = load_checkpoint(path)
    if (checkpoint["variant"], checkpoint["seed"], checkpoint["task"]) != (
        variant,
        seed,
        task_name,
    ) or checkpoint["update"] != task["max_updates"]:
        raise ValueError(f"unexpected checkpoint identity: {path}")
    margin = float(protocol["projection"]["float32_relative_margin"])
    cap = np.float32(1.0) - np.float32(margin)

    layers: list[tuple[str, int]] = []

    def forward(params, images):
        names = {
            id(leaf): jax.tree_util.keystr(path[:-1])
            for path, leaf in jax.tree_util.tree_leaves_with_path(params)
        }
        records: list = []
        with recorded_convolutions(names, records, cap):
            forward_task(task_name, task, params, images, variant)
        layers[:] = [(layer, kernel) for layer, kernel, _ in records]
        return [counts for _, _, counts in records]

    dataset = materialize_dataset(
        load_dataset(config["dataset"], DATA_ROOT, ROOT / "splits", "test")
    )
    compiled = jax.jit(forward)
    totals: dict[str, list] = {}
    batch_size = int(task["evaluation_batch_size"])
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        images, _ = dataset.batch(indices)
        counts = compiled(checkpoint["params"], jnp.asarray(images))
        for (layer, kernel), (active, positions, rho_sum, rho_max) in zip(
            layers, counts, strict=True
        ):
            total = totals.setdefault(layer, [kernel, 0, 0, 0.0, 0.0])
            total[1] += int(active)
            total[2] += int(positions)
            total[3] += float(rho_sum)
            total[4] = max(total[4], float(rho_max))
    curvature = (
        float(jax.nn.softplus(checkpoint["params"]["curvature_raw"]))
        if task_name == "isic16_poincare_unet"
        else float(task["curvature"]["initial_magnitude"])
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return [
        {
            "experiment_id": experiment,
            "task": task_name,
            "variant": variant,
            "seed": seed,
            "checkpoint": str(path.relative_to(TRAINING_ROOT)),
            "checkpoint_sha256": digest,
            "update": int(checkpoint["update"]),
            "curvature": curvature,
            "test_examples": len(dataset),
            "layer_order": order,
            "layer": layer,
            "kernel_size": kernel,
            "positions": positions,
            "cap_active_positions": active,
            "cap_active_fraction": active / positions,
            "mean_normalized_radius": rho_sum / positions,
            "max_normalized_radius": rho_max,
        }
        for order, (layer, (kernel, active, positions, rho_sum, rho_max)) in enumerate(
            totals.items()
        )
    ]


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per layer over seeds and over all layers per experiment/variant."""

    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["experiment_id"], row["variant"], row["layer_order"], row["layer"])
        groups[key].append(row)
        groups[(row["experiment_id"], row["variant"], 10_000, "all_layers")].append(row)
    output = []
    for key in sorted(groups):
        selected = groups[key]
        by_seed: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for row in selected:
            by_seed[row["seed"]][0] += row["cap_active_positions"]
            by_seed[row["seed"]][1] += row["positions"]
        fractions = [active / positions for active, positions in by_seed.values()]
        output.append(
            {
                "experiment_id": key[0],
                "variant": key[1],
                "layer": key[3],
                "kernel_size": selected[0]["kernel_size"] if key[3] != "all_layers" else "",
                "seeds": len(by_seed),
                "pooled_cap_active_fraction": (
                    sum(a for a, _ in by_seed.values())
                    / sum(p for _, p in by_seed.values())
                ),
                "min_seed_fraction": min(fractions),
                "max_seed_fraction": max(fractions),
                "max_normalized_radius": max(
                    row["max_normalized_radius"] for row in selected
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--variants", nargs="+", default=["R", "F"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/reviewer_requested"
    )
    arguments = parser.parse_args()
    provenance = git_provenance()  # before any output is written
    protocol = load_yaml(ROOT / "protocol.yaml")
    cells = [
        (experiment, variant, seed)
        for experiment in arguments.experiments
        for variant in arguments.variants
        for seed in load_yaml(ROOT / EXPERIMENTS[experiment])["seeds"]
    ]
    jobs = [(block, *cell) for block, cell in enumerate(cells)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        arguments.jobs, mp_context=context, max_tasks_per_child=1
    ) as pool:
        rows = [row for result in pool.map(measure, jobs) for row in result]
    summary = summary_rows(rows)
    write_csv(arguments.output_dir / "e33_cap_activity_by_checkpoint.csv", rows)
    write_csv(arguments.output_dir / "e33_cap_activity_summary.csv", summary)
    metadata = {
        "experiment_id": "E33",
        "analysis_status": "post_hoc_reviewer_requested_trained_model_diagnostic",
        "code_commit": provenance["commit"],
        "code_dirty": provenance["dirty"],
        "protocol_version": protocol["protocol_version"],
        "device": str(jax.devices()[0]),
        "checkpoints": "final-update checkpoints (checkpoints/final.pkl)",
        "split": "official test, manifest order, registered evaluation batch size",
        "cap_definition": "rho^2 >= m_c^2 with rho = tanh(sqrt(c R^2)) in FP32, "
        "m_c = 1 - float32_relative_margin",
        "output_rows": {
            "e33_cap_activity_by_checkpoint.csv": len(rows),
            "e33_cap_activity_summary.csv": len(summary),
        },
        "all_layer_fractions": {
            f"{row['experiment_id']}_{row['variant']}": row["pooled_cap_active_fraction"]
            for row in summary
            if row["layer"] == "all_layers"
        },
    }
    with (arguments.output_dir / "e33_cap_activity_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metadata["all_layer_fractions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
