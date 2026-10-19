#!/usr/bin/env python3
"""Plot reviewer-requested learned-curvature trajectories from retained rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/reviewer_requested/isic16_curvature_trajectories.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories: dict[tuple[str, str, int], list[tuple[int, float]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["experiment_id"], row["variant"], int(row["seed"]))
            trajectories[key].append((int(row["epoch"]), float(row["curvature"])))

    experiments = ("E09", "E20")
    colors = {"R": "#3567a9", "F": "#d9792b"}
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharex=True, sharey=True)
    for axis, experiment_id in zip(axes, experiments, strict=True):
        for variant in ("R", "F"):
            series = [
                sorted(values)
                for (experiment, path, _), values in trajectories.items()
                if experiment == experiment_id and path == variant
            ]
            for values in series:
                axis.plot(
                    [epoch for epoch, _ in values],
                    [curvature for _, curvature in values],
                    color=colors[variant],
                    alpha=0.20,
                    linewidth=0.75,
                )
            epochs = [epoch for epoch, _ in series[0]]
            mean_curve = [
                fmean(values[index][1] for values in series) for index in range(len(epochs))
            ]
            axis.plot(
                epochs,
                mean_curve,
                color=colors[variant],
                linewidth=2.0,
                label=f"{variant} mean",
            )
        pairs = len(
            {seed for experiment, _, seed in trajectories if experiment == experiment_id}
        )
        axis.set_title(f"{experiment_id}: {pairs} matched seeds")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Learned curvature magnitude")
    figure.tight_layout(pad=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
