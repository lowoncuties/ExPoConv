#!/usr/bin/env python3
"""E33 branch trace: rounding-sensitive branch decisions of one structured realization.

Every branch decision in the forward pass of R, F, and T is recorded with its distance
from the decision boundary in FP32 units in the last place (ulps):

- comparisons (source-radius caps, output projections, series cutoffs, the
  scaled-readout switch, norm guards): ulps of the larger operand;
- max/min gates, including every ReLU, whose derivative routes the cotangent to one
  operand: ulps of the larger operand, or for a gate against zero (ReLU) ulps of the
  tensor's largest magnitude;
- 2x2 max-pool windows, whose derivative routes the cotangent to the argmax: the gap
  between the two largest entries in ulps of the largest.

Records carry the parameter path of the issuing layer (or "after <layer>" for an
unscoped gate) and the source call chain. The FP64 fixed-margin trace of the same
realization gives the oracle's decision at the same element.

Two flip tests change one decision at a time: moving the FP32 cap by a few ulps, and
negating the pre-ReLU element closest to zero. Their gradient changes are compared with
the recorded V100 F-vs-R (E26) and T-vs-R (E02) differences for this realization.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import linecache
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import source_info_util
from run_oracle_accuracy import (
    ROOT,
    _configure_jax,
    fixed_projection_margin,
    load_design,
    pin_cores,
    realization,
)

from poincare_exec.artifacts.provenance import git_provenance
from poincare_exec.experiments.numerical_structured import _evaluate, _gradient_groups
from poincare_exec.experiments.numerics import _cast_tree
from poincare_exec.models import common, unet

COMPARISONS = {"gt", "ge", "lt", "le"}
GATES = {"max", "min"}
SOURCE_ROOT = ROOT / "src"
E26_TENSORS = ROOT / "results/reviewer_requested/e26_elementwise_tensor_statistics.csv"
E02_FAILURES = ROOT / "results/numerics/structured_agreement_failures.csv"


def _layer_names(params: Any) -> dict[int, str]:
    return {
        id(leaf): jax.tree_util.keystr(path[:-1])
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
    }


@contextlib.contextmanager
def layer_scopes(names: dict[int, str], negate: tuple[str, tuple] | None = None):
    """Name each model-level layer call by its parameter path.

    ``negate=(layer, element)`` additionally flips the sign of one output element of
    that normalization layer, which flips the ReLU gate that follows it.
    """

    targets = [
        (common, "conv2d", "weight"),
        (common, "midpoint_batch_norm", "bias"),
        (unet, "conv2d", "weight"),
        (unet, "transpose_conv2d", "weight"),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in targets]

    def scoped(function, field):
        def wrapper(params, *args, **kwargs):
            name = names.get(id(getattr(params, field)), "unnamed")
            with jax.named_scope(name):
                output = function(params, *args, **kwargs)
            if negate is not None and name == negate[0]:
                output = output.at[negate[1]].multiply(-1.0)
            return output

        return wrapper

    for (module, name, field), (_, _, function) in zip(targets, originals, strict=True):
        setattr(module, name, scoped(function, field))
    try:
        yield
    finally:
        for module, name, function in originals:
            setattr(module, name, function)


def _call_chain(eqn) -> str:
    frames = [
        frame
        for frame in source_info_util.user_frames(eqn.source_info)
        if Path(frame.file_name).is_relative_to(SOURCE_ROOT)
    ]
    return " < ".join(f"{Path(f.file_name).name}:{f.start_line}" for f in frames[:3])


def _max_pool_record(value: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    window = tuple(params["window_dimensions"])
    if window != (1, 2, 2, 1) or tuple(params["window_strides"]) != window:
        raise ValueError("only non-overlapping 2x2 max pooling is traced")
    batch, height, width, channels = value.shape
    blocks = value[:, : height // 2 * 2, : width // 2 * 2, :].reshape(
        batch, height // 2, 2, width // 2, 2, channels
    )
    windows = blocks.transpose(0, 1, 3, 5, 2, 4).reshape(*blocks.shape[:1], -1, 4)
    ordered = np.sort(windows, axis=-1)
    return {
        "lhs": ordered[..., -1],
        "rhs": ordered[..., -2],
        "decision": np.argmax(windows, axis=-1),
    }


def _interpret(jaxpr, consts, args, records, state, scope=""):
    """Evaluate a jaxpr eagerly and keep every branch decision."""

    env: dict[Any, Any] = {}

    def read(variable):
        return variable.val if isinstance(variable, jax.core.Literal) else env[variable]

    env.update(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, args, strict=True))
    for eqn in jaxpr.eqns:
        values = [read(variable) for variable in eqn.invars]
        name = "/".join(part for part in (scope, str(eqn.source_info.name_stack)) if part)
        paths = [part for part in name.split("/") if part.startswith("[")]
        if paths:
            state["last"] = paths[-1]
        inner = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
        if isinstance(inner, jax.core.ClosedJaxpr):
            outputs = _interpret(inner.jaxpr, inner.consts, values, records, state, name)
            env.update(zip(eqn.outvars, outputs, strict=True))
            continue
        outputs = eqn.primitive.bind(*values, **eqn.params)
        if not eqn.primitive.multiple_results:
            outputs = [outputs]
        env.update(zip(eqn.outvars, outputs, strict=True))
        operator = eqn.primitive.name
        arrays = [np.asarray(value) for value in values]
        if not arrays or not np.issubdtype(arrays[0].dtype, np.floating):
            continue
        if operator in COMPARISONS or operator in GATES:
            lhs, rhs = np.broadcast_arrays(*arrays)
            decision = np.asarray(outputs[0]) if operator in COMPARISONS else lhs > rhs
            record = {"lhs": lhs, "rhs": rhs, "decision": decision}
        elif operator == "reduce_window_max":
            record = _max_pool_record(arrays[0], eqn.params)
        else:
            continue
        record.update(
            layer=paths[-1] if paths else f"after {state['last']}",
            site=_call_chain(eqn),
            operator=operator,
        )
        records.append(record)
    return [read(variable) for variable in jaxpr.outvars]


def trace(case, curvature, variant: str) -> list[dict[str, Any]]:
    """Return keyed branch records for one forward evaluation."""

    def forward(params, inputs, selected_curvature):
        with layer_scopes(_layer_names(params)):
            return case.function(params, inputs, selected_curvature, variant)

    arguments = (case.parameters, case.inputs, curvature)
    closed = jax.make_jaxpr(forward)(*arguments)
    records: list[dict[str, Any]] = []
    leaves = jax.tree_util.tree_leaves(arguments)
    _interpret(closed.jaxpr, closed.consts, leaves, records, {"last": "input"})
    occurrences: Counter[tuple[str, str]] = Counter()
    for record in records:
        key = (record["layer"], record["site"])
        record["occurrence"] = occurrences[key]
        occurrences[key] += 1
    return records


def gap_ulps(record: dict[str, Any]) -> np.ndarray:
    """Distance from the decision boundary in FP32 ulps (see module docstring)."""

    lhs = record["lhs"].astype(np.float64)
    rhs = record["rhs"].astype(np.float64)
    if record["operator"] in GATES and (not rhs.any() or not lhs.any()):
        quantity = rhs if not lhs.any() else lhs
        scale = np.full_like(quantity, np.max(np.abs(quantity)))
    else:
        scale = np.maximum(np.abs(lhs), np.abs(rhs))
    return np.abs(lhs - rhs) / np.spacing(scale.astype(np.float32)).astype(np.float64)


def _code(site: str) -> str:
    if not site:
        return ""
    file_name, line = site.split(" < ")[0].split(":")
    matches = list((SOURCE_ROOT / "poincare_exec").rglob(file_name))
    return linecache.getline(str(matches[0]), int(line)).strip() if matches else ""


def margin_rows(variant: str, fp32: list, fp64: list) -> list[dict[str, Any]]:
    """Summarize each FP32 branch decision at its element closest to the boundary."""

    oracle = {(r["layer"], r["site"], r["occurrence"]): r for r in fp64}
    rows = []
    for record in fp32:
        tied = record["lhs"] == record["rhs"]
        gap = np.where(tied, np.inf, gap_ulps(record))
        index = int(np.argmin(gap)) if gap.size else 0
        partner = oracle.get((record["layer"], record["site"], record["occurrence"]))
        paired = partner is not None and partner["lhs"].shape == record["lhs"].shape
        # Dtype-dependent constant thresholds (series cutoffs) make FP64 decisions
        # incomparable; data-dependent operands and fixed-margin caps are compared.
        rhs = record["rhs"]
        constant = rhs.size > 0 and bool(np.all(rhs == rhs.flat[0]))
        threshold_changes = (
            paired
            and constant
            and not np.isclose(partner["rhs"].flat[0], rhs.flat[0], rtol=1.0e-5, atol=0.0)
        )
        row = {
            "variant": variant,
            "layer": record["layer"],
            "operator": record["operator"],
            "site": record["site"],
            "code": _code(record["site"]),
            "occurrence": record["occurrence"],
            "elements": int(gap.size),
            "exact_ties": int(np.count_nonzero(tied)),
            "min_gap_fp32_ulps": float(gap.flat[index]) if gap.size else None,
            "elements_within_4_ulps": int(np.count_nonzero(gap <= 4.0)),
            "elements_within_64_ulps": int(np.count_nonzero(gap <= 64.0)),
            "closest_element": json.dumps(
                [int(value) for value in np.unravel_index(index, gap.shape)]
            ),
            "closest_lhs_fp32": float(record["lhs"].flat[index]) if gap.size else None,
            "closest_rhs_fp32": float(record["rhs"].flat[index]) if gap.size else None,
            "closest_lhs_fp64": None,
            "closest_rhs_fp64": None,
            "fp64_threshold_differs": bool(threshold_changes),
            "decisions_differing_from_fp64": None,
        }
        if paired and gap.size:
            row["closest_lhs_fp64"] = float(partner["lhs"].flat[index])
            row["closest_rhs_fp64"] = float(partner["rhs"].flat[index])
            if not threshold_changes:
                row["decisions_differing_from_fp64"] = int(
                    np.count_nonzero(partner["decision"] != record["decision"])
                )
        rows.append(row)
    return rows


def flipped_decisions(baseline: list, changed: list) -> list[dict[str, Any]]:
    """Return every decision element that differs between two traces."""

    changes = []
    for before, after in zip(baseline, changed, strict=True):
        if (before["layer"], before["site"]) != (after["layer"], after["site"]):
            raise ValueError("trace structure changed under a flip")
        for index in np.argwhere(before["decision"] != after["decision"]):
            element = tuple(int(value) for value in index)
            changes.append(
                {
                    "layer": before["layer"],
                    "site": before["site"],
                    "element": list(element),
                    "lhs_before": float(before["lhs"][element]),
                    "rhs_before": float(before["rhs"][element]),
                }
            )
    return changes


def shifted_margin(margin: float, ulps: int) -> float:
    """Return a margin whose FP32 cap 1-margin is moved by ``ulps`` FP32 ulps."""

    cap = np.float32(1.0) - np.float32(margin)
    for _ in range(abs(ulps)):
        cap = np.nextafter(cap, np.float32(2.0 if ulps > 0 else 0.0))
    return float(np.float32(1.0) - cap)


def _relative_l2(candidate, reference) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = np.linalg.norm((candidate - reference).ravel())
    return float(difference / max(np.linalg.norm(reference.ravel()), 1.0e-12))


def _norms(groups: dict[str, Any]) -> dict[str, float]:
    return {
        group: float(np.linalg.norm(np.asarray(value, dtype=np.float64).ravel()))
        for group, value in groups.items()
    }


def recorded_v100(field: str) -> dict[str, dict[str, float]]:
    """A field of the V100 E26 F-vs-R and E02 T-vs-R rows for small_unet r2."""

    recorded: dict[str, dict[str, float]] = {"E26_F_vs_R": {}, "E02_T_vs_R": {}}
    for path, label, variant in (
        (E26_TENSORS, "E26_F_vs_R", "F"),
        (E02_FAILURES, "E02_T_vs_R", "T"),
    ):
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["case_id"], row["dtype"], row["realization"], row["variant"])
                if key == ("small_unet", "float32", "2", variant):
                    recorded[label][row["variable_group"]] = float(row[field])
    return recorded


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="small_unet")
    parser.add_argument("--realization", type=int, default=2)
    parser.add_argument("--curvature", type=float, default=0.1)
    parser.add_argument("--cap-shifts", type=int, nargs="+", default=[-2, -1, 1, 2])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/reviewer_requested"
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    provenance = git_provenance()  # before any output is written
    pin_cores(0)
    _configure_jax()
    protocol, design, _, _ = load_design()
    case_index = [spec["id"] for spec in design["cases"]].index(arguments.case_id)
    case, curvature, cotangent = realization(
        design["cases"][case_index],
        case_index,
        arguments.realization,
        arguments.curvature,
        seed=int(protocol["seeds"]["numerical"]),
        input_scale=float(design["input_scale"]),
        epsilon=float(design["normalization_epsilon"]),
    )
    margin = float(protocol["numerics"]["float32_contract_oracle_projection_margin"])
    case64 = dataclasses.replace(
        case,
        parameters=_cast_tree(case.parameters, jnp.float64),
        inputs=_cast_tree(case.inputs, jnp.float64),
    )

    rows = []
    baselines = {}
    for variant in ("R", "F", "T"):
        baselines[variant] = trace(case, curvature, variant)
        with fixed_projection_margin(margin):
            oracle = trace(case64, curvature.astype(jnp.float64), variant)
        rows.extend(margin_rows(variant, baselines[variant], oracle))
    rows.sort(key=lambda row: (row["min_gap_fp32_ulps"] is None, row["min_gap_fp32_ulps"]))

    def gradients(variant: str, selected_margin: float, negate=None):
        def function(params, inputs, selected_curvature, selected_variant):
            with layer_scopes(_layer_names(params), negate):
                return case.function(params, inputs, selected_curvature, selected_variant)

        with fixed_projection_margin(selected_margin):
            result = _evaluate(
                function, variant, case.parameters, case.inputs, curvature, cotangent
            )
        return _gradient_groups(result, include_external_curvature=False)

    def flip(variant, label, reference, selected_margin, negate=None):
        def forward(params, inputs, selected_curvature, selected_variant):
            with layer_scopes(_layer_names(params), negate):
                return case.function(params, inputs, selected_curvature, selected_variant)

        with fixed_projection_margin(selected_margin):
            changed = trace(dataclasses.replace(case, function=forward), curvature, variant)
        moved = gradients(variant, selected_margin, negate)
        return {
            "variant": variant,
            "flip": label,
            "changed_decisions": flipped_decisions(baselines[variant], changed),
            "relative_l2_change_by_tensor": {
                group: _relative_l2(moved[group], reference[group]) for group in reference
            },
            "l2_norm_before_by_tensor": _norms(reference),
            "l2_norm_after_by_tensor": _norms(moved),
        }

    gate = next(
        row
        for row in rows
        if row["variant"] == "F"
        and row["operator"] == "max"
        and row["layer"].startswith("after ")
        and "relu" in row["code"]
    )
    gate_target = (
        gate["layer"].removeprefix("after "),
        tuple(json.loads(gate["closest_element"])),
    )
    flips = []
    for variant in ("R", "F", "T"):
        reference = gradients(variant, margin)
        for shift in arguments.cap_shifts:
            flips.append(
                flip(
                    variant,
                    f"cap_{shift:+d}_ulps",
                    reference,
                    shifted_margin(margin, shift),
                )
            )
        flips.append(
            flip(variant, "negate_closest_pre_relu_element", reference, margin, gate_target)
        )

    stem = f"e33_branch_trace_{arguments.case_id}_r{arguments.realization}"
    write_csv(arguments.output_dir / f"{stem}.csv", rows)
    summary = {
        "experiment_id": "E33",
        "analysis_status": "post_hoc_branch_diagnostic",
        "case_id": arguments.case_id,
        "realization": arguments.realization,
        "curvature": arguments.curvature,
        "margin": margin,
        "device": str(jax.devices()[0]),
        "gate_flip_target": {"layer": gate_target[0], "element": list(gate_target[1])},
        "recorded_v100_relative_l2": recorded_v100("relative_l2_error"),
        "recorded_v100_R_reference_l2_norm": recorded_v100("reference_l2_norm"),
        "code_commit": provenance["commit"],
        "code_dirty": provenance["dirty"],
        "flip_tests": flips,
    }
    with (arguments.output_dir / f"{stem}_flip_test.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for row in rows[:12]:
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "variant",
                        "layer",
                        "operator",
                        "code",
                        "min_gap_fp32_ulps",
                        "closest_lhs_fp32",
                        "closest_lhs_fp64",
                        "decisions_differing_from_fp64",
                    )
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
