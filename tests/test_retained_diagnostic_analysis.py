import hashlib
import json
import math

import numpy as np

from poincare_exec.analysis.retained_diagnostics import (
    aggregate_comparison_rows,
    export_retained_diagnostic_tables,
)


def _row(path, reference_norm, difference_norm, maximum_error, exact=False):
    return {
        "source_experiment_id": "E09",
        "source_task": "isic16_poincare_unet",
        "source_variant": "F",
        "source_seed": 11,
        "run_variant": "F",
        "comparison_type": "loss_gradient",
        "update": 0,
        "parameter_path": path,
        "reference_l2_norm": reference_norm,
        "difference_l2_norm": difference_norm,
        "max_absolute_error": maximum_error,
        "comparison_nonfinite_count": 0,
        "element_count": 2,
        "exactly_equal": exact,
    }


def test_aggregate_comparison_rows_builds_global_l2_and_maximum_error():
    result = aggregate_comparison_rows(
        [
            _row("a", 3.0, 0.3, 0.2),
            _row("b", 4.0, 0.4, 0.1),
        ]
    )

    assert len(result) == 1
    assert result[0]["reference_l2_norm"] == 5.0
    assert math.isclose(result[0]["difference_l2_norm"], 0.5)
    assert math.isclose(result[0]["relative_l2_error"], 0.1)
    assert result[0]["max_absolute_error"] == 0.2
    assert result[0]["element_count"] == 4
    assert result[0]["exactly_equal"] is False


def test_export_requires_and_writes_one_complete_source(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "e99" / "R" / "seed_11"
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
experiment_id: E07-J1
design_document:
  path: docs/journal_diagnostic_protocol_v2.md
  sha256: c356b7c7325295b4d93aaaa090bc5f1c47398d975b2293a88bf34512bcc0e31f
execution_variants: [R]
continuation_updates: 1
observation_updates: [0, 1]
controls:
  unperturbed_repeat: true
  one_ulp_perturbation: true
tasks:
  - task: fake_task
    training_experiment_id: E99
    source_seed: 11
    source_variants: [R]
    source_config_hash: source-config
    source_protocol_version: "1.10"
""",
        encoding="utf-8",
    )
    run_variants = ["R", "R_repeat", "R_ulp"]
    design_sha256 = "c356b7c7325295b4d93aaaa090bc5f1c47398d975b2293a88bf34512bcc0e31f"
    identity = {
        "experiment_id": "E07-J1",
        "protocol_version": "1.10",
        "variant": "shared",
        "config_hash": "diagnostic-config",
        "code_commit": "a" * 40,
        "code_dirty": False,
        "reference_commit": "{}",
        "device_id": "host::gpu:0::test",
        "precision_policy": "{}",
        "seed_or_session": 11,
        "design_document_path": "docs/journal_diagnostic_protocol_v2.md",
        "source_experiment_id": "E99",
        "source_task": "fake_task",
        "source_variant": "R",
        "source_seed": 11,
        "source_update": 100,
        "source_checkpoint_sha256": "b" * 64,
        "source_summary_sha256": "d" * 64,
        "source_code_commit": "e" * 40,
        "source_config_hash": "source-config",
        "source_protocol_version": "1.10",
        "design_document_sha256": design_sha256,
    }
    comparison = {
        "status": "ok",
        "parameter_path": "weight",
        "reference_l2_norm": 1.0,
        "difference_l2_norm": 0.0,
        "max_absolute_error": 0.0,
        "comparison_nonfinite_count": 0,
        "element_count": 1,
        "exactly_equal": True,
    }
    initial = []
    first_update = []
    trajectory = []
    for run_variant in run_variants:
        initial.extend(
            [
                {
                    **identity,
                    "run_variant": run_variant,
                    "comparison_type": "initial_scalar",
                    "parameter_path": None,
                    "update": 0,
                    "status": "ok",
                    "loss_difference_from_R": 0.0,
                    "output_nonfinite_count": 0,
                    "gradient_nonfinite_count": 0,
                },
                {
                    **identity,
                    **comparison,
                    "run_variant": run_variant,
                    "comparison_type": "logits",
                    "parameter_path": None,
                    "update": 0,
                    "status": "ok",
                },
                {
                    **identity,
                    **comparison,
                    "run_variant": run_variant,
                    "comparison_type": "probabilities",
                    "parameter_path": None,
                    "update": 0,
                    "status": "ok",
                },
                {
                    **identity,
                    **comparison,
                    "run_variant": run_variant,
                    "comparison_type": "loss_gradient",
                    "update": 0,
                },
            ]
        )
        for comparison_type in (
            "parameter_update",
            "updated_parameter",
            "updated_first_moment",
            "updated_second_moment",
        ):
            first_update.append(
                {
                    **identity,
                    **comparison,
                    "run_variant": run_variant,
                    "comparison_type": comparison_type,
                    "update": 1,
                }
            )
        for update in (0, 1):
            trajectory.append(
                {
                    **identity,
                    "run_variant": run_variant,
                    "update": update,
                    "optimizer_step": 100 + update,
                    "status": "ok",
                    "loss": 1.0,
                    "parameter_l2_difference_from_R": (
                        0.0 if run_variant in {"R", "R_repeat"} else 1.0e-7
                    ),
                    "optimizer_first_moment_difference_from_R": 0.0,
                    "optimizer_second_moment_difference_from_R": 0.0,
                    "parameter_nonfinite_count": 0,
                    "optimizer_nonfinite_count": 0,
                }
            )

    artifacts = {
        "initial_comparisons.jsonl": initial,
        "first_update_comparisons.jsonl": first_update,
        "trajectory.jsonl": trajectory,
    }
    hashes = {}
    for name, rows in artifacts.items():
        path = run_dir / name
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    outputs = run_dir / "first_batch_outputs.npz"
    np.savez_compressed(
        outputs,
        inputs=np.ones((1, 2), dtype=np.float32),
        targets=np.asarray([0], dtype=np.int32),
        batch_indices=np.asarray([4], dtype=np.int64),
        **{
            f"{run_variant}__logits": np.ones((1, 3), dtype=np.float32)
            for run_variant in run_variants
        },
        **{
            f"{run_variant}__probabilities": np.full((1, 3), 1.0 / 3.0, dtype=np.float32)
            for run_variant in run_variants
        },
    )
    hashes[outputs.name] = hashlib.sha256(outputs.read_bytes()).hexdigest()
    summary = {
        **identity,
        "status": "ok",
        "task": "fake_task",
        "curvature_gradient_applicable": False,
        "unperturbed_repeat_bitwise_exact": True,
        "unperturbed_repeat_policy": "bitwise",
        "unperturbed_repeat_check_updates": [0, 1],
        "run_variants": run_variants,
        "observation_updates": [0, 1],
        "continuation_updates": 1,
        "continuation_index_schedule_sha256": "c" * 64,
        "continuation_batch_sizes": [1],
        "initial_comparison_rows": len(initial),
        "first_update_comparison_rows": len(first_update),
        "trajectory_rows": len(trajectory),
        "artifact_sha256": hashes,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = export_retained_diagnostic_tables(
        artifact_root, config_path, tmp_path / "tables"
    )

    assert result["status"] == "complete"
    assert result["sources"] == 1
    assert (tmp_path / "tables" / "diagnostic_source_audit.csv").is_file()
    assert (tmp_path / "tables" / "logits_agreement_summary.csv").is_file()
    assert (tmp_path / "tables" / "probability_agreement_summary.csv").is_file()
    assert (tmp_path / "tables" / "diagnostic_key_metrics.csv").is_file()
    assert (tmp_path / "figures" / "figure_retained_state_trajectory.pdf").is_file()
    assert (tmp_path / "figures" / "figure_retained_state_trajectory.png").is_file()
    assert result["key_metric_rows"] == len(run_variants)
