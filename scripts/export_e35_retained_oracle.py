#!/usr/bin/env python3
"""Validate the E35 run records and export FP64-oracle gradient accuracy tables.

Reads the two stage records and gradient NPZ files written by
scripts/run_e35_retained_oracle.py and writes compact tables to
results/reviewer_requested/e35_retained_oracle_*:

- ``tensor_errors.csv``: relative L2 and maximum absolute error of every FP32 R/F/T
  parameter-gradient tensor against the FP64 oracle, per source state;
- ``path_summary.csv``: per source state and path, the median and maximum over
  tensors, the whole-gradient relative L2, and tensor counts closer to or farther
  from the oracle than R;
- ``f_vs_r.csv``: the whole-gradient F-versus-R relative L2 recomputed from the same
  FP32 run next to the E07-J2 update-0 value;
- ``summary.json``: provenance, devices, definitions, and validation results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from poincare_exec.config.loader import config_hash, load_yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocols/protocol_1_20.yaml"
CONFIG = ROOT / "configs/checkpoints/e35_retained_oracle.yaml"
ORACLE = "R_float64_fixed_float32_margin"
NORM_FLOOR = 1.0e-12
PREFIX = "e35_retained_oracle"
STAGE_DTYPES = {"float32": "float32", "float64": "float64"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registration(
    protocol_path: Path = PROTOCOL, config_path: Path = CONFIG
) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    return protocol, config, config_hash(protocol, config)


def inventory_sha256(config: dict[str, Any], root: Path = ROOT) -> dict[str, str]:
    with (root / config["checkpoint_inventory"]["path"]).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["variant"]: row["sha256"]
        for row in rows
        if row["experiment_id"] == config["training_experiment_id"]
        and row["task"] == config["task"]
        and int(row["seed"]) == int(config["source_seed"])
        and row["role"] == config["source_role"]
    }


def load_gradients(path: Path) -> tuple[list[str], list[np.ndarray]]:
    with np.load(path) as stored:
        paths = [str(value) for value in stored["paths"]]
        arrays = [stored[f"gradient_{index:03d}"] for index in range(len(paths))]
    return paths, arrays


def validate_record(
    record: dict[str, Any],
    *,
    stage: str,
    artifact_dir: Path,
    protocol: dict[str, Any],
    config: dict[str, Any],
    config_digest: str,
    checkpoints: dict[str, str],
) -> None:
    """Reject a record that is not a clean, complete run of the registered design."""

    expected_paths = list(config["execution_variants"]) if stage == "float32" else [ORACLE]
    expected = {
        "experiment_id": config["experiment_id"],
        "stage": stage,
        "code_dirty": False,
        "protocol_version": protocol["protocol_version"],
        "config_hash": config_digest,
        "evaluated_paths": expected_paths,
        "jax_enable_x64": stage == "float64",
    }
    mismatches = {
        key: record.get(key) for key, value in expected.items() if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{stage} record does not match the registration: {mismatches}")
    if stage == "float32" and record["device_platform"] != "gpu":
        raise ValueError("FP32 paths did not run on a GPU")
    sources = [source["source_variant"] for source in record["sources"]]
    if sources != list(config["source_variants"]):
        raise ValueError(f"{stage} record has sources {sources}")
    for source in record["sources"]:
        variant = source["source_variant"]
        if source["checkpoint_sha256"] != checkpoints[variant]:
            raise ValueError(f"{stage} source {variant} used an unregistered checkpoint")
        if not all(source["batch_equal_to_e07"].values()):
            raise ValueError(f"{stage} source {variant} batch differs from E07-J2")
        names = list(config["execution_variants"]) if stage == "float32" else ["R"]
        if sorted(source["results"]) != sorted(names):
            raise ValueError(f"{stage} source {variant} lacks paths")
        for result in source["results"].values():
            if sha256(artifact_dir / result["file"]) != result["sha256"]:
                raise ValueError(f"{result['file']} changed after the run")
            if result["gradient_dtype"] != STAGE_DTYPES[stage]:
                raise ValueError(f"{result['file']} has dtype {result['gradient_dtype']}")


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    difference = np.asarray(candidate, dtype=np.float64) - reference64
    if not (np.isfinite(reference64).all() and np.isfinite(difference).all()):
        raise ValueError("nonfinite gradient tensor")
    reference_norm = float(np.linalg.norm(reference64.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    return {
        "reference_l2_norm": reference_norm,
        "difference_l2_norm": difference_norm,
        "relative_l2_error": difference_norm / max(reference_norm, NORM_FLOOR),
        "max_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
    }


def whole_relative(rows: Sequence[dict[str, Any]], norm: str, difference: str) -> float:
    reference = math.sqrt(sum(float(row[norm]) ** 2 for row in rows))
    error = math.sqrt(sum(float(row[difference]) ** 2 for row in rows))
    return error / max(reference, NORM_FLOOR)


def e07_f_vs_r(path: Path, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return the E07-J2 update-0 whole-gradient F-versus-R values per source."""

    with path.open(encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["source_experiment_id"] == config["training_experiment_id"]
            and int(row["source_seed"]) == int(config["source_seed"])
            and row["run_variant"] == "F"
            and row["comparison_type"] == "loss_gradient"
            and int(row["update"]) == 0
        ]
    values = {
        row["source_variant"]: {
            "relative_l2_error": float(row["relative_l2_error"]),
            "reference_l2_norm": float(row["reference_l2_norm"]),
            "parameter_paths": int(row["parameter_paths"]),
        }
        for row in rows
    }
    if sorted(values) != sorted(config["source_variants"]) or len(rows) != len(values):
        raise ValueError("E07-J2 gradient table lacks a unique F row per source")
    return values


def build_tables(
    fp32: dict[str, dict[str, tuple[list[str], list[np.ndarray]]]],
    oracle: dict[str, tuple[list[str], list[np.ndarray]]],
    e07: dict[str, dict[str, float]],
    *,
    sources: Sequence[str],
    variants: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tensor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    for source in sources:
        oracle_paths, oracle_arrays = oracle[source]
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for variant in variants:
            paths, arrays = fp32[source][variant]
            if paths != oracle_paths:
                raise ValueError(f"{source}/{variant} tensor paths differ from the oracle")
            rows = []
            for name, candidate, reference in zip(
                paths, arrays, oracle_arrays, strict=True
            ):
                if candidate.shape != reference.shape:
                    raise ValueError(f"{source}/{variant}/{name} shape differs")
                rows.append(
                    {
                        "source_variant": source,
                        "path": variant,
                        "dtype": str(candidate.dtype),
                        "reference": ORACLE,
                        "parameter_path": name,
                        "elements": int(candidate.size),
                        **compare(candidate, reference),
                    }
                )
            by_variant[variant] = rows
            tensor_rows.extend(rows)
        baseline = [row["relative_l2_error"] for row in by_variant["R"]]
        for variant in variants:
            rows = by_variant[variant]
            errors = [row["relative_l2_error"] for row in rows]
            largest = max(rows, key=lambda row: row["relative_l2_error"])
            paired = list(zip(errors, baseline, strict=True))
            summary_rows.append(
                {
                    "source_variant": source,
                    "path": variant,
                    "dtype": rows[0]["dtype"],
                    "reference": ORACLE,
                    "tensor_count": len(rows),
                    "elements": sum(row["elements"] for row in rows),
                    "median_tensor_relative_l2_error": statistics.median(errors),
                    "max_tensor_relative_l2_error": largest["relative_l2_error"],
                    "max_tensor_parameter_path": largest["parameter_path"],
                    "whole_gradient_relative_l2_error": whole_relative(
                        rows, "reference_l2_norm", "difference_l2_norm"
                    ),
                    "tensors_closer_than_R": (
                        None if variant == "R" else sum(e < r for e, r in paired)
                    ),
                    "tensors_farther_than_R": (
                        None if variant == "R" else sum(e > r for e, r in paired)
                    ),
                }
            )
        r_paths, r_arrays = fp32[source]["R"]
        f_paths, f_arrays = fp32[source]["F"]
        if r_paths != f_paths:
            raise ValueError(f"{source} F and R tensor paths differ")
        pairs = [compare(f, r) for f, r in zip(f_arrays, r_arrays, strict=True)]
        recomputed = whole_relative(pairs, "reference_l2_norm", "difference_l2_norm")
        reference_norm = math.sqrt(sum(row["reference_l2_norm"] ** 2 for row in pairs))
        agreement_rows.append(
            {
                "source_variant": source,
                "comparison": "F_vs_R",
                "dtype": str(r_arrays[0].dtype),
                "tensor_count": len(pairs),
                "recomputed_relative_l2": recomputed,
                "e07_j2_relative_l2": e07[source]["relative_l2_error"],
                "recomputed_minus_e07_j2": recomputed - e07[source]["relative_l2_error"],
                "recomputed_R_l2_norm": reference_norm,
                "e07_j2_R_l2_norm": e07[source]["reference_l2_norm"],
            }
        )
    return tensor_rows, summary_rows, agreement_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _gpu_uuid(record: dict[str, Any]) -> str | None:
    snapshot = record.get("gpu_snapshot_before") or {}
    return snapshot.get("uuid")


def _device(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": record["device"],
        "platform": record["device_platform"],
        "device_kind": record["device_kind"],
        "cuda_visible_devices": record["cuda_visible_devices"],
        "gpu_uuid": _gpu_uuid(record),
        "jax_version": record["jax_version"],
        "jaxlib_version": record["jaxlib_version"],
    }


def export(
    artifact_dir: Path,
    output_dir: Path,
    *,
    protocol_path: Path = PROTOCOL,
    config_path: Path = CONFIG,
    root: Path = ROOT,
) -> dict[str, Any]:
    protocol, config, digest = load_registration(protocol_path, config_path)
    checkpoints = inventory_sha256(config, root)
    records = {}
    for stage in STAGE_DTYPES:
        with (artifact_dir / f"e35_{stage}_record.json").open(encoding="utf-8") as stream:
            records[stage] = json.load(stream)
        validate_record(
            records[stage],
            stage=stage,
            artifact_dir=artifact_dir,
            protocol=protocol,
            config=config,
            config_digest=digest,
            checkpoints=checkpoints,
        )
    if records["float32"]["code_commit"] != records["float64"]["code_commit"]:
        raise ValueError("FP32 and FP64 stages ran at different commits")
    if records["float32"]["batch"] != records["float64"]["batch"]:
        raise ValueError("FP32 and FP64 stages used different batches")
    sources = list(config["source_variants"])
    variants = list(config["execution_variants"])
    fp32 = {
        source["source_variant"]: {
            variant: load_gradients(artifact_dir / result["file"])
            for variant, result in source["results"].items()
        }
        for source in records["float32"]["sources"]
    }
    oracle = {
        source["source_variant"]: load_gradients(
            artifact_dir / source["results"]["R"]["file"]
        )
        for source in records["float64"]["sources"]
    }
    e07 = e07_f_vs_r(root / config["reference_gradient_table"]["path"], config)
    tensor_rows, summary_rows, agreement_rows = build_tables(
        fp32, oracle, e07, sources=sources, variants=variants
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        f"{PREFIX}_tensor_errors.csv": tensor_rows,
        f"{PREFIX}_path_summary.csv": summary_rows,
        f"{PREFIX}_f_vs_r.csv": agreement_rows,
    }
    for name, rows in files.items():
        write_csv(output_dir / name, rows)
    fp32_sources = {
        source["source_variant"]: source for source in records["float32"]["sources"]
    }
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "analysis_status": "post_hoc_reviewer_requested_accuracy_diagnostic",
        "scientific_behavior_change": False,
        "identity": {
            "code_commit": records["float32"]["code_commit"],
            "code_dirty": False,
            "protocol_version": protocol["protocol_version"],
            "protocol_path": records["float32"]["protocol_path"],
            "config_path": records["float32"]["config_path"],
            "config_hash": digest,
        },
        "reference_experiment": {
            "experiment_id": config["reference_experiment"],
            "code_commit": fp32_sources[sources[0]]["e07_code_commit"],
            "device_id": fp32_sources[sources[0]]["e07_device_id"],
            "gradient_table": config["reference_gradient_table"]["path"],
        },
        "devices": {stage: _device(record) for stage, record in records.items()},
        "oracle": {
            "name": ORACLE,
            "definition": config["oracle"]["definition"].strip(),
            "projection_margin": records["float64"]["projection_margin"],
        },
        "sources": {
            source: {
                "checkpoint_path": fp32_sources[source]["checkpoint_path"],
                "checkpoint_sha256": fp32_sources[source]["checkpoint_sha256"],
                "fp32_logits_bitwise_equal_to_e07_j2": {
                    variant: result["logits_bitwise_equal_to_e07"]
                    for variant, result in fp32_sources[source]["results"].items()
                },
            }
            for source in sources
        },
        "batch": records["float32"]["batch"],
        "batch_equal_to_e07_j2": True,
        "definitions": {
            "tensor_relative_l2": "||g - g_oracle||_2 / max(||g_oracle||_2, 1e-12)",
            "whole_gradient_relative_l2": (
                "sqrt(sum ||g - g_oracle||^2) / sqrt(sum ||g_oracle||^2) over all tensors"
            ),
            "f_vs_r": "sqrt(sum ||g_F - g_R||^2) / sqrt(sum ||g_R||^2), FP32, same run",
            "tensors_closer_than_R": "tensors whose relative L2 error is below R's",
        },
        "whole_gradient_relative_l2_error": {
            source: {
                row["path"]: row["whole_gradient_relative_l2_error"]
                for row in summary_rows
                if row["source_variant"] == source
            }
            for source in sources
        },
        "output_rows": {name: len(rows) for name, rows in files.items()},
    }
    with (output_dir / f"{PREFIX}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/reviewer_requested"
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    summary = export(arguments.artifact_dir.resolve(), arguments.output_dir.resolve())
    print(json.dumps(summary["whole_gradient_relative_l2_error"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
