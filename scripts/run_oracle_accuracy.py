#!/usr/bin/env python3
"""E33: FP32 accuracy of R, F, and T against a fixed-margin FP64 oracle.

E02/E26 compare F and T with R evaluated in the same precision, which measures
agreement, not accuracy. Here every FP32 structured realization is also cast to
FP64 and evaluated by R with the FP32 projection margin (protocol key
``numerics.float32_contract_oracle_projection_margin``). This is the repository's
``R_float64_fixed_float32_margin`` oracle: the same clipped function as the FP32
paths, evaluated with FP64 rounding. R, F, and T in FP32, and F and T in FP64 at the
same margin, are compared with it tensor by tensor.

Curvature can be overridden without changing anything else: the convolution stack
receives curvature as an argument, and the small U-Net's only curvature-dependent
initial value is its unconstrained curvature parameter. Keys, inputs, weights, and
cotangents are those of the registered design at every curvature.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing
import os
import statistics
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.analysis.reviewer_requested import elementwise_relative_error_summary
from poincare_exec.artifacts.provenance import git_provenance
from poincare_exec.config.loader import config_hash, load_yaml
from poincare_exec.core import geometry
from poincare_exec.experiments.numerical_structured import (
    _case,
    _elementwise_settings,
    _evaluate,
    _gradient_groups,
)
from poincare_exec.experiments.numerics import _cast_tree, _finite_metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol.yaml"
E02_CONFIG = ROOT / "configs/numerics/e02_repair_structured.yaml"
E26_CONFIG = ROOT / "configs/numerics/e26_elementwise_agreement.yaml"
E26_TENSORS = ROOT / "results/reviewer_requested/e26_elementwise_tensor_statistics.csv"
ORACLE = "R_float64_fixed_float32_margin"
CANDIDATES = (
    ("R", "float32"),
    ("F", "float32"),
    ("T", "float32"),
    ("F", "float64"),
    ("T", "float64"),
)
SHARED_DESIGN_KEYS = (
    "cases",
    "realizations",
    "curvature",
    "input_scale",
    "normalization_epsilon",
)
QUANTILE_COLUMNS = ("p50", "p90", "p95", "p99", "p99_9")
CORES_PER_JOB = 8


@contextlib.contextmanager
def fixed_projection_margin(margin: float) -> Iterator[None]:
    """Use one projection margin for every dtype whose margin is not explicit.

    All default margins resolve through ``geometry.projection_margin``, so this
    applies the FP32 clipping contract to every projection inside FP64 models.
    """

    original = geometry.projection_margin
    geometry.projection_margin = lambda dtype: jnp.asarray(margin, dtype=jnp.dtype(dtype))
    try:
        yield
    finally:
        geometry.projection_margin = original


def pin_cores(block: int) -> None:
    """Pin this process to one block of ``CORES_PER_JOB`` cores before JAX starts.

    XLA sizes its CPU thread pool from the affinity mask, so every job runs with the
    same pool size (and the same reduction partitioning) whatever ``--jobs`` is.
    Eager dispatch with one pool per host core is also far slower.
    """

    available = sorted(os.sched_getaffinity(0))
    blocks = max(len(available) // CORES_PER_JOB, 1)
    start = (block % blocks) * CORES_PER_JOB
    os.sched_setaffinity(0, available[start : start + CORES_PER_JOB])


def _configure_jax() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


def load_design() -> tuple[dict[str, Any], dict[str, Any], float, tuple[float, ...]]:
    """Return the protocol, the E26 design, and its elementwise settings."""

    protocol = load_yaml(PROTOCOL)
    e02 = load_yaml(E02_CONFIG)
    e26 = load_yaml(E26_CONFIG)
    for key in SHARED_DESIGN_KEYS:
        if e02[key] != e26[key]:
            raise ValueError(f"E02 and E26 structured designs differ in {key}")
    numerics = protocol["numerics"]
    if (
        numerics["float32_contract_oracle_projection_margin"]
        != (protocol["projection"]["float32_relative_margin"])
    ):
        raise ValueError("oracle margin differs from the FP32 projection margin")
    floor, quantiles = _elementwise_settings(e26, protocol)
    return protocol, e26, floor, quantiles


def realization(
    spec: dict[str, Any],
    case_index: int,
    index: int,
    curvature: float,
    *,
    seed: int,
    input_scale: float,
    epsilon: float,
):
    """Rebuild one FP32 structured realization exactly as the E02/E26 runner does."""

    selected = dict(spec)
    if spec["kind"] == "small_unet":
        selected["initial_curvature"] = curvature
    key = jax.random.fold_in(jax.random.PRNGKey(seed), case_index * 10_000 + index)
    case_key, cotangent_key, _ = jax.random.split(key, 3)
    case = _case(selected, case_key, jnp.float32, input_scale, epsilon)
    external_curvature = jnp.asarray(curvature, dtype=jnp.float32)
    output = case.function(case.parameters, case.inputs, external_curvature, "R")
    cotangent = jax.random.normal(cotangent_key, output.shape, dtype=jnp.float32)
    return case, external_curvature, cotangent


def evaluate_paths(case, curvature, cotangent, margin: float) -> dict[tuple[str, str], Any]:
    """Return named output/gradient groups for FP32 R/F/T and FP64 fixed-margin R/F/T."""

    def groups(result):
        return _gradient_groups(
            result, include_external_curvature=case.uses_external_curvature
        )

    arguments32 = (case.parameters, case.inputs, curvature, cotangent)
    arguments64 = tuple(_cast_tree(value, jnp.float64) for value in arguments32)
    evaluated = {
        (variant, "float32"): groups(_evaluate(case.function, variant, *arguments32))
        for variant in ("R", "F", "T")
    }
    with fixed_projection_margin(margin):
        for variant in ("R", "F", "T"):
            evaluated[(variant, "float64")] = groups(
                _evaluate(case.function, variant, *arguments64)
            )
    return evaluated


def tensor_rows(
    evaluated: dict[tuple[str, str], Any],
    *,
    norm_floor: float,
    quantiles: Sequence[float],
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare every candidate tensor with the oracle tensor of the same name."""

    oracle = evaluated[("R", "float64")]
    rows = []
    for variant, dtype in CANDIDATES:
        for group, reference in oracle.items():
            candidate = evaluated[(variant, dtype)][group]
            metrics = _finite_metrics(reference, candidate, norm_floor)
            elementwise = elementwise_relative_error_summary(
                reference, candidate, norm_floor, quantiles=quantiles
            )
            prefix = "elementwise_relative_error_"
            rows.append(
                {
                    **identity,
                    "variant": variant,
                    "dtype": dtype,
                    "scope": "output" if group == "output" else "gradient",
                    "variable_group": group,
                    "finite": metrics["finite"],
                    "elements": elementwise[prefix + "elements"],
                    "oracle_l2_norm": metrics["reference_l2_norm"],
                    "relative_l2_error": metrics["relative_l2_error"],
                    "max_absolute_error": metrics["max_absolute_error"],
                    "denominator_floored_elements": elementwise[
                        prefix + "denominator_floored_elements"
                    ],
                    **{name: elementwise[prefix + name] for name in QUANTILE_COLUMNS},
                    "maximum": elementwise[prefix + "max"],
                }
            )
    return rows


def run_job(job: tuple[int, float, int, int]) -> list[dict[str, Any]]:
    """Evaluate one (curvature, case, realization) cell in a fresh worker process."""

    block, curvature, case_index, index = job
    pin_cores(block)
    _configure_jax()
    protocol, design, floor, quantiles = load_design()
    spec = design["cases"][case_index]
    case, external_curvature, cotangent = realization(
        spec,
        case_index,
        index,
        curvature,
        seed=int(protocol["seeds"]["numerical"]),
        input_scale=float(design["input_scale"]),
        epsilon=float(design["normalization_epsilon"]),
    )
    margin = float(protocol["numerics"]["float32_contract_oracle_projection_margin"])
    evaluated = evaluate_paths(case, external_curvature, cotangent, margin)
    identity = {"curvature": curvature, "case_id": spec["id"], "realization": index}
    return tensor_rows(evaluated, norm_floor=floor, quantiles=quantiles, identity=identity)


def _case_scopes(row: dict[str, Any]) -> tuple[str, str]:
    return ("all_cases", row["case_id"])


def scope_summary(
    rows: Sequence[dict[str, Any]], tolerances: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Summarize errors per curvature, case scope, dtype, path, and tensor scope.

    Like E26, quantile columns are maxima of per-tensor quantiles; medians across
    tensors are added because a single realization can set every maximum. The
    registered mixed criterion (max abs <= atol + rtol * ||oracle||) is applied
    against the oracle with the candidate dtype's protocol tolerances.
    """

    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for case_scope in _case_scopes(row):
            key = (row["curvature"], case_scope, row["dtype"], row["variant"], row["scope"])
            cells[key].append(row)
    output = []
    for key in sorted(cells):
        selected = cells[key]
        relative = [float(row["relative_l2_error"]) for row in selected]
        output.append(
            {
                **dict(
                    zip(
                        ("curvature", "case_scope", "dtype", "variant", "scope"),
                        key,
                        strict=True,
                    )
                ),
                "reference": ORACLE,
                "tensor_count": len(selected),
                "mixed_criterion_failed_tensors": sum(
                    not _mixed_pass(row, tolerances[row["dtype"]]) for row in selected
                ),
                "median_tensor_relative_l2_error": statistics.median(relative),
                "largest_tensor_relative_l2_error": max(relative),
                "largest_max_absolute_error": max(
                    float(row["max_absolute_error"]) for row in selected
                ),
                "median_tensor_p50": statistics.median(
                    float(row["p50"]) for row in selected
                ),
                "median_tensor_p99": statistics.median(
                    float(row["p99"]) for row in selected
                ),
                **{
                    f"largest_tensor_{name}": max(float(row[name]) for row in selected)
                    for name in QUANTILE_COLUMNS
                },
                "largest_elementwise_relative_error": max(
                    float(row["maximum"]) for row in selected
                ),
                "pooled_denominator_floored_fraction": sum(
                    int(row["denominator_floored_elements"]) for row in selected
                )
                / sum(int(row["elements"]) for row in selected),
            }
        )
    return output


def _mixed_pass(row: dict[str, Any], tolerance: dict[str, float]) -> bool:
    threshold = tolerance["absolute"] + tolerance["relative"] * row["oracle_l2_norm"]
    return bool(row["finite"]) and row["max_absolute_error"] <= threshold


def paired_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each FP32 F/T tensor error with the FP32 R error of the same tensor."""

    reference = {
        (row["curvature"], row["case_id"], row["realization"], row["variable_group"]): row
        for row in rows
        if row["variant"] == "R" and row["dtype"] == "float32"
    }
    cells: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    for row in rows:
        if row["dtype"] != "float32" or row["variant"] == "R":
            continue
        pair = (
            row,
            reference[
                (
                    row["curvature"],
                    row["case_id"],
                    row["realization"],
                    row["variable_group"],
                )
            ],
        )
        for case_scope in _case_scopes(row):
            cells[(row["curvature"], case_scope, row["variant"], row["scope"])].append(pair)
    output = []
    for key in sorted(cells):
        pairs = cells[key]
        compared = [
            (candidate, base)
            for candidate, base in pairs
            if candidate["relative_l2_error"] > 0 or base["relative_l2_error"] > 0
        ]
        ratios = [
            candidate["relative_l2_error"] / base["relative_l2_error"]
            for candidate, base in compared
            if base["relative_l2_error"] > 0
        ]
        output.append(
            {
                **dict(
                    zip(("curvature", "case_scope", "variant", "scope"), key, strict=True)
                ),
                "baseline": "R_float32",
                "tensor_count": len(pairs),
                "both_exact_tensors": len(pairs) - len(compared),
                "compared_tensors": len(compared),
                "fraction_relative_l2_le_R": _fraction(
                    c["relative_l2_error"] <= b["relative_l2_error"] for c, b in compared
                ),
                "fraction_max_absolute_le_R": _fraction(
                    c["max_absolute_error"] <= b["max_absolute_error"] for c, b in compared
                ),
                "fraction_relative_l2_le_2x_R": _fraction(
                    c["relative_l2_error"] <= 2.0 * b["relative_l2_error"]
                    for c, b in compared
                ),
                "median_relative_l2_ratio_to_R": statistics.median(ratios)
                if ratios
                else None,
                "min_relative_l2_ratio_to_R": min(ratios) if ratios else None,
                "max_relative_l2_ratio_to_R": max(ratios) if ratios else None,
            }
        )
    return output


def _fraction(values: Iterator[bool]) -> float | None:
    selected = list(values)
    return sum(selected) / len(selected) if selected else None


def e26_output_norm_check(
    rows: Sequence[dict[str, Any]], curvature: float
) -> dict[str, Any]:
    """Check that the rebuilt c=0.1 realizations are E26's by output L2 norms."""

    recorded = {}
    with E26_TENSORS.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["dtype"], row["variant"], row["scope"]) == ("float32", "F", "output"):
                recorded[(row["case_id"], int(row["realization"]))] = float(
                    row["reference_l2_norm"]
                )
    differences = [
        abs(row["oracle_l2_norm"] - recorded[(row["case_id"], row["realization"])])
        / recorded[(row["case_id"], row["realization"])]
        for row in rows
        if row["curvature"] == curvature
        and row["scope"] == "output"
        and (row["variant"], row["dtype"]) == ("R", "float32")
    ]
    return {
        "compared_outputs": len(differences),
        "e26_outputs": len(recorded),
        "max_relative_difference_oracle_vs_e26_v100_R_fp32_output_norm": max(differences),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curvatures", type=float, nargs="+", default=[0.06, 0.1, 0.13])
    parser.add_argument(
        "--realizations",
        type=int,
        default=None,
        help="limit realizations per case (default: registered count)",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/reviewer_requested"
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    provenance = git_provenance()  # before any output is written
    protocol, design, floor, quantiles = load_design()
    count = arguments.realizations or int(design["realizations"])
    cells = [
        (curvature, case_index, index)
        for curvature in arguments.curvatures
        for case_index in range(len(design["cases"]))
        for index in range(count)
    ]
    jobs = [(block, *cell) for block, cell in enumerate(cells)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        arguments.jobs, mp_context=context, max_tasks_per_child=1
    ) as pool:
        rows = [row for result in pool.map(run_job, jobs) for row in result]
    pin_cores(0)
    _configure_jax()

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scopes = scope_summary(rows, protocol["numerics"]["tolerances"])
    pairs = paired_summary(rows)
    write_csv(output_dir / "e33_oracle_accuracy_tensor_statistics.csv", rows)
    write_csv(output_dir / "e33_oracle_accuracy_scope_summary.csv", scopes)
    write_csv(output_dir / "e33_oracle_accuracy_paired_summary.csv", pairs)
    registered = float(design["curvature"])
    summary = {
        "schema_version": "1.0",
        "experiment_id": "E33",
        "analysis_status": "post_hoc_reviewer_requested_accuracy_diagnostic",
        "scientific_behavior_change": False,
        "identity": {
            "code_commit": provenance["commit"],
            "code_dirty": provenance["dirty"],
            "protocol_version": protocol["protocol_version"],
            "design_config": str(E26_CONFIG.relative_to(ROOT)),
            "design_config_hash": config_hash(protocol, design),
            "jax_version": jax.__version__,
            "device": str(jax.devices()[0]),
            "seed": int(protocol["seeds"]["numerical"]),
        },
        "oracle": {
            "name": ORACLE,
            "definition": (
                "R evaluated in FP64 on the FP32 realization cast to FP64, with every "
                "projection margin fixed to the FP32 value"
            ),
            "projection_margin": protocol["numerics"][
                "float32_contract_oracle_projection_margin"
            ],
        },
        "candidates": [f"{variant}_{dtype}" for variant, dtype in CANDIDATES],
        "elementwise_definition": "abs(candidate-oracle)/max(abs(oracle),1e-12)",
        "denominator_floor": floor,
        "quantiles": list(quantiles),
        "relative_l2_definition": "||candidate-oracle||_2 / max(||oracle||_2, 1e-12)",
        "curvatures": arguments.curvatures,
        "registered_curvature": registered,
        "curvature_override": (
            "convolution_stack_8 receives curvature as an argument; small_unet sets "
            "initial_curvature, which only sets curvature_raw; keys, inputs, weights, "
            "and cotangents are unchanged"
        ),
        "realizations_per_case": count,
        "e26_realization_check": (
            e26_output_norm_check(rows, registered)
            if registered in arguments.curvatures and count == 10
            else None
        ),
        "output_rows": {
            "e33_oracle_accuracy_tensor_statistics.csv": len(rows),
            "e33_oracle_accuracy_scope_summary.csv": len(scopes),
            "e33_oracle_accuracy_paired_summary.csv": len(pairs),
        },
    }
    with (output_dir / "e33_oracle_accuracy_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary["output_rows"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
