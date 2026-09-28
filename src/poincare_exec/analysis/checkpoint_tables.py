"""Publication tables and completeness audits for checkpoint evidence."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def checkpoint_rows(artifact_dir: Path, filename: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((artifact_dir / "checkpoint_experiments").glob(f"**/{filename}")):
        if "rejected" in path.relative_to(artifact_dir).parts:
            continue
        for row in _jsonl(path):
            # Retained-state diagnostics share this tree but use a different schema.
            if "seed" not in row:
                continue
            row["_evidence_path"] = str(path)
            row["_summary_path"] = str(path.with_name("summary.json"))
            rows.append(row)
    return rows


def checkpoint_attempts(artifact_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        (artifact_dir / "checkpoint_experiments").glob("**/campaign_attempts.jsonl")
    ):
        if "rejected" in path.relative_to(artifact_dir).parts:
            continue
        for row in _jsonl(path):
            row["_evidence_path"] = str(path)
            rows.append(row)
    return rows


def checkpoint_compatibility_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten registered E06 output and activation evidence for Table 2."""

    return [
        {
            "experiment_id": row["experiment_id"],
            "protocol_version": row["protocol_version"],
            "config_hash": row["config_hash"],
            "code_commit": row["code_commit"],
            "code_dirty": row["code_dirty"],
            "device_id": row["device_id"],
            "task": row["task"],
            "checkpoint": row["checkpoint"],
            "source_checkpoint_experiment": row.get("source_checkpoint_experiment"),
            "source_checkpoint_variant": row.get("source_checkpoint_variant"),
            "source_checkpoint_seed": row.get("source_checkpoint_seed"),
            "source_checkpoint_update": row.get("source_checkpoint_update"),
            "source_checkpoint_sha256": row.get("source_checkpoint_sha256"),
            "source_checkpoint_code_commit": row.get(
                "source_checkpoint_code_commit"
            ),
            "source_checkpoint_code_dirty": row.get("source_checkpoint_code_dirty"),
            "split": row["split"],
            "variant": row["variant"],
            "record_type": row["record_type"],
            "variable_group": row["variable_group"],
            "status": row["status"],
            "max_absolute_error": row["max_absolute_error"],
            "relative_l2_error": row["relative_l2_error"],
            "nonfinite_count": row["nonfinite_count"],
            "reference_nonfinite_count": row["reference_nonfinite_count"],
            "comparison_nonfinite_count": row["comparison_nonfinite_count"],
            "projection_saturation_count": row["projection_saturation_count"],
            "projection_vector_count": row["projection_vector_count"],
            "maximum_normalized_radius": row["maximum_normalized_radius"],
            "decision_disagreement_fraction": row.get("decision_disagreement_fraction"),
            "affected_examples": row.get("affected_examples"),
            "affected_example_fraction": row.get("affected_example_fraction"),
            "task_metric": row.get("task_metric"),
            "examples": row.get("examples"),
            "evidence_path": row.get("_evidence_path"),
            "summary_path": row.get("_summary_path"),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                item["task"],
                item["checkpoint"],
                item["variant"],
                item["record_type"],
                item["variable_group"],
            ),
        )
    ]


def trained_checkpoint_campaign_audit(
    rows: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    protocol: dict[str, Any],
    checkpoint_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate complete trained-checkpoint E06 matrices before aggregation."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_experiment = row.get("source_checkpoint_experiment")
        if source_experiment is not None:
            grouped[(str(source_experiment), str(row["task"]))].append(row)

    task_entries = {
        str(entry["task"]): entry for entry in checkpoint_config["tasks"]
    }
    expected_seeds = {int(value) for value in protocol["seeds"]["training"]}
    source_variants = {str(value) for value in checkpoint_config["update_variants"]}
    comparison_variants = {str(value) for value in checkpoint_config["variants"]}
    splits = {str(value) for value in checkpoint_config["post_training_splits"]}
    output = []
    for (source_experiment, task), selected in sorted(grouped.items()):
        expected_sources = {
            (variant, seed) for variant in source_variants for seed in expected_seeds
        }
        observed_sources = {
            (str(row["source_checkpoint_variant"]), int(row["source_checkpoint_seed"]))
            for row in selected
        }
        activation_groups = {
            str(value) for value in task_entries[task]["activation_groups"]
        }
        expected_output_keys = {
            (variant, seed, split, candidate)
            for variant, seed in expected_sources
            for split in splits
            for candidate in comparison_variants
        }
        expected_activation_keys = {
            (*key, variable_group)
            for key in expected_output_keys
            for variable_group in activation_groups
        }
        output_rows = [
            row
            for row in selected
            if row["record_type"] == "output" and row["variable_group"] == "logits"
        ]
        activation_rows = [
            row
            for row in selected
            if row["record_type"] == "activation"
            and row["variable_group"] in activation_groups
        ]

        def base_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
            return (
                str(row["source_checkpoint_variant"]),
                int(row["source_checkpoint_seed"]),
                str(row["split"]),
                str(row["variant"]),
            )

        observed_output_keys = [base_key(row) for row in output_rows]
        observed_activation_keys = [
            (*base_key(row), str(row["variable_group"])) for row in activation_rows
        ]
        selected_attempts = [
            row
            for row in attempts
            if row.get("training_experiment_id") == source_experiment
            and row.get("task") == task
        ]
        observed_attempts = [
            (str(row["source_variant"]), int(row["seed"]))
            for row in selected_attempts
        ]
        source_metadata: dict[tuple[str, int], set[tuple[Any, ...]]] = defaultdict(set)
        for row in selected:
            key = (
                str(row["source_checkpoint_variant"]),
                int(row["source_checkpoint_seed"]),
            )
            source_metadata[key].add(
                (
                    row["source_checkpoint_sha256"],
                    row["source_checkpoint_code_commit"],
                    row["source_checkpoint_code_dirty"],
                    int(row["source_checkpoint_update"]),
                )
            )
        source_metadata_consistent = all(
            len(values) == 1 for values in source_metadata.values()
        )
        expected_update = int(protocol["tasks"][task]["max_updates"])
        record_status_ok = all(row["status"] == "ok" for row in selected)
        clean_evaluation_code = all(not bool(row["code_dirty"]) for row in selected)
        clean_source_code = all(
            not bool(row["source_checkpoint_code_dirty"]) for row in selected
        )
        source_updates_match = all(
            int(row["source_checkpoint_update"]) == expected_update for row in selected
        )
        output_matrix_exact = (
            len(observed_output_keys) == len(expected_output_keys)
            and set(observed_output_keys) == expected_output_keys
        )
        activation_matrix_exact = (
            len(observed_activation_keys) == len(expected_activation_keys)
            and set(observed_activation_keys) == expected_activation_keys
        )
        attempt_matrix_exact = (
            len(observed_attempts) == len(expected_sources)
            and set(observed_attempts) == expected_sources
        )
        attempts_ok = all(
            row["status"] == "ok"
            and int(row["returncode"]) == 0
            and row["gpu_snapshot_before"]["status"] == "ok"
            and row["gpu_snapshot_after"]["status"] == "ok"
            and row["gpu_snapshot_before"]["uuid"]
            == row["gpu_snapshot_after"]["uuid"]
            for row in selected_attempts
        )
        idle_snapshots = all(
            not row["gpu_snapshot_before"]["compute_processes"]
            and not row["gpu_snapshot_after"]["compute_processes"]
            for row in selected_attempts
        )
        protocol_versions = {str(row["protocol_version"]) for row in selected}
        config_hashes = {str(row["config_hash"]) for row in selected}
        evaluation_commits = {str(row["code_commit"]) for row in selected}
        device_ids = {str(row["device_id"]) for row in selected}
        total_matrix_exact = len(selected) == (
            len(expected_output_keys) + len(expected_activation_keys)
        )
        identity_consistent = (
            protocol_versions == {str(protocol["protocol_version"])}
            and len(config_hashes) == 1
            and len(evaluation_commits) == 1
            and len(device_ids) == 1
        )
        complete = all(
            (
                observed_sources == expected_sources,
                total_matrix_exact,
                output_matrix_exact,
                activation_matrix_exact,
                attempt_matrix_exact,
                attempts_ok,
                idle_snapshots,
                record_status_ok,
                clean_evaluation_code,
                clean_source_code,
                source_updates_match,
                source_metadata_consistent,
                identity_consistent,
            )
        )
        checkpoint_hashes = {
            f"{variant}:{seed}": next(iter(source_metadata[(variant, seed)]))[0]
            for variant, seed in sorted(source_metadata)
            if len(source_metadata[(variant, seed)]) == 1
        }
        output.append(
            {
                "status": "complete" if complete else "incomplete",
                "source_checkpoint_experiment": source_experiment,
                "task": task,
                "expected_sources": len(expected_sources),
                "observed_sources": len(observed_sources),
                "expected_output_rows": len(expected_output_keys),
                "observed_output_rows": len(output_rows),
                "expected_activation_rows": len(expected_activation_keys),
                "observed_activation_rows": len(activation_rows),
                "expected_attempts": len(expected_sources),
                "observed_attempts": len(selected_attempts),
                "record_status_ok": record_status_ok,
                "total_matrix_exact": total_matrix_exact,
                "identity_consistent": identity_consistent,
                "clean_evaluation_code": clean_evaluation_code,
                "clean_source_code": clean_source_code,
                "source_updates_match": source_updates_match,
                "source_metadata_consistent": source_metadata_consistent,
                "output_matrix_exact": output_matrix_exact,
                "activation_matrix_exact": activation_matrix_exact,
                "attempt_matrix_exact": attempt_matrix_exact,
                "attempts_ok": attempts_ok,
                "idle_before_and_after": idle_snapshots,
                "protocol_versions": json.dumps(sorted(protocol_versions)),
                "config_hashes": json.dumps(sorted(config_hashes)),
                "evaluation_code_commits": json.dumps(sorted(evaluation_commits)),
                "source_checkpoint_code_commits": json.dumps(
                    sorted(
                        {str(row["source_checkpoint_code_commit"]) for row in selected}
                    )
                ),
                "device_ids": json.dumps(sorted(device_ids)),
                "physical_gpu_indices": json.dumps(
                    sorted(
                        {
                            int(row["gpu_snapshot_before"]["index"])
                            for row in selected_attempts
                        }
                    )
                ),
                "physical_gpu_uuids": json.dumps(
                    sorted(
                        {
                            str(row["gpu_snapshot_before"]["uuid"])
                            for row in selected_attempts
                        }
                    )
                ),
                "source_checkpoint_sha256": json.dumps(
                    checkpoint_hashes, sort_keys=True
                ),
                "evidence_paths": json.dumps(
                    sorted({str(row["_evidence_path"]) for row in selected})
                ),
                "attempt_paths": json.dumps(
                    sorted({str(row["_evidence_path"]) for row in selected_attempts})
                ),
            }
        )
    return output


def trained_checkpoint_task_agreement(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    """Report fixed-checkpoint task and logit agreement for F/T against R."""

    baseline = str(protocol["variants"]["training_roles"]["baseline"])
    candidates = {
        str(protocol["variants"]["training_roles"]["primary_method"]),
        str(protocol["variants"]["training_roles"]["ablation"]),
    }
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if (
            row.get("source_checkpoint_experiment") is not None
            and row["record_type"] == "output"
            and row["variable_group"] == "logits"
            and row["status"] == "ok"
        ):
            grouped[(row["task"], row["checkpoint"], row["split"])][
                row["variant"]
            ] = row

    output = []
    for (task, checkpoint, split), variants in sorted(grouped.items()):
        if baseline not in variants or not candidates.issubset(variants):
            continue
        reference = variants[baseline]
        for candidate in sorted(candidates):
            row = variants[candidate]
            task_difference = float(row["task_metric"]) - float(
                reference["task_metric"]
            )
            output.append(
                {
                    "source_checkpoint_experiment": row[
                        "source_checkpoint_experiment"
                    ],
                    "task": task,
                    "checkpoint": checkpoint,
                    "source_checkpoint_variant": row["source_checkpoint_variant"],
                    "source_checkpoint_seed": int(row["source_checkpoint_seed"]),
                    "source_checkpoint_update": int(row["source_checkpoint_update"]),
                    "source_checkpoint_sha256": row["source_checkpoint_sha256"],
                    "source_checkpoint_code_commit": row[
                        "source_checkpoint_code_commit"
                    ],
                    "source_checkpoint_code_dirty": row[
                        "source_checkpoint_code_dirty"
                    ],
                    "evaluation_code_commit": row["code_commit"],
                    "evaluation_code_dirty": row["code_dirty"],
                    "protocol_version": row["protocol_version"],
                    "config_hash": row["config_hash"],
                    "device_id": row["device_id"],
                    "split": split,
                    "task_metric_name": protocol["tasks"][task]["primary_metric"],
                    "reference_variant": baseline,
                    "candidate_variant": candidate,
                    "reference_task_metric": reference["task_metric"],
                    "candidate_task_metric": row["task_metric"],
                    "task_metric_difference": task_difference,
                    "absolute_task_metric_difference": abs(task_difference),
                    "logit_max_absolute_error": row["max_absolute_error"],
                    "logit_relative_l2_error": row["relative_l2_error"],
                    "decision_disagreement_fraction": row[
                        "decision_disagreement_fraction"
                    ],
                    "affected_examples": row["affected_examples"],
                    "affected_example_fraction": row["affected_example_fraction"],
                    "examples": row["examples"],
                    "logit_nonfinite_count": row["nonfinite_count"],
                    "reference_nonfinite_count": row["reference_nonfinite_count"],
                    "comparison_nonfinite_count": row[
                        "comparison_nonfinite_count"
                    ],
                    "nonfinite_probability_count": row[
                        "nonfinite_probability_count"
                    ],
                    "projection_saturation_count": row[
                        "projection_saturation_count"
                    ],
                    "projection_vector_count": row["projection_vector_count"],
                    "maximum_normalized_radius": row["maximum_normalized_radius"],
                    "evidence_path": row.get("_evidence_path"),
                    "summary_path": row.get("_summary_path"),
                }
            )
    return output


def trained_checkpoint_compatibility_aggregate(
    task_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate deterministic fixed-checkpoint agreement across retained sources."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[
            (
                row["source_checkpoint_experiment"],
                row["task"],
                row["split"],
                row["candidate_variant"],
            )
        ].append(row)
    activation_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("source_checkpoint_experiment") is None:
            continue
        if row["record_type"] != "activation" or row["status"] != "ok":
            continue
        activation_rows[(row["checkpoint"], row["split"], row["variant"])].append(
            row
        )

    output = []
    for key, selected in sorted(grouped.items()):
        source_experiment, task, split, candidate = key
        activations = [
            activation
            for row in selected
            for activation in activation_rows[
                (row["checkpoint"], row["split"], candidate)
            ]
        ]
        task_differences = [float(row["task_metric_difference"]) for row in selected]
        absolute_task_differences = [abs(value) for value in task_differences]
        output.append(
            {
                "source_checkpoint_experiment": source_experiment,
                "task": task,
                "split": split,
                "task_metric_name": selected[0]["task_metric_name"],
                "reference_variant": selected[0]["reference_variant"],
                "candidate_variant": candidate,
                "source_checkpoints": len(selected),
                "source_variants": json.dumps(
                    sorted({str(row["source_checkpoint_variant"]) for row in selected})
                ),
                "unique_training_seeds": len(
                    {int(row["source_checkpoint_seed"]) for row in selected}
                ),
                "mean_task_metric_difference": statistics.mean(task_differences),
                "median_task_metric_difference": statistics.median(task_differences),
                "maximum_absolute_task_metric_difference": max(
                    absolute_task_differences
                ),
                "mean_absolute_task_metric_difference": statistics.mean(
                    absolute_task_differences
                ),
                "maximum_logit_relative_l2_error": max(
                    float(row["logit_relative_l2_error"]) for row in selected
                ),
                "median_logit_relative_l2_error": statistics.median(
                    float(row["logit_relative_l2_error"]) for row in selected
                ),
                "maximum_logit_absolute_error": max(
                    float(row["logit_max_absolute_error"]) for row in selected
                ),
                "maximum_decision_disagreement_fraction": max(
                    float(row["decision_disagreement_fraction"]) for row in selected
                ),
                "maximum_affected_examples": max(
                    int(row["affected_examples"]) for row in selected
                ),
                "total_logit_nonfinite_count": sum(
                    int(row["logit_nonfinite_count"]) for row in selected
                ),
                "total_probability_nonfinite_count": sum(
                    int(row["nonfinite_probability_count"]) for row in selected
                ),
                "activation_rows": len(activations),
                "maximum_activation_relative_l2_error": max(
                    float(row["relative_l2_error"]) for row in activations
                ),
                "maximum_activation_absolute_error": max(
                    float(row["max_absolute_error"]) for row in activations
                ),
                "total_activation_nonfinite_count": sum(
                    int(row["nonfinite_count"]) for row in activations
                ),
                "maximum_activation_normalized_radius": max(
                    float(row["maximum_normalized_radius"]) for row in activations
                ),
                "protocol_versions": json.dumps(
                    sorted({str(row["protocol_version"]) for row in selected})
                ),
                "evaluation_code_commits": json.dumps(
                    sorted({str(row["evaluation_code_commit"]) for row in selected})
                ),
                "source_checkpoint_code_commits": json.dumps(
                    sorted(
                        {str(row["source_checkpoint_code_commit"]) for row in selected}
                    )
                ),
                "config_hashes": json.dumps(
                    sorted({str(row["config_hash"]) for row in selected})
                ),
            }
        )
    return output
