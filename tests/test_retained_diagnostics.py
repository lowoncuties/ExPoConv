import jax.numpy as jnp
import numpy as np
import pytest

from poincare_exec.experiments.retained_diagnostics import (
    _validate_source,
    array_comparison,
    arrays_bitwise_equal,
    perturb_parameters_one_ulp,
    tree_comparison_rows,
    trees_bitwise_equal,
)
from poincare_exec.training.optimizers import AdamState


def test_array_comparison_reports_absolute_relative_and_exact_fields():
    reference = np.asarray([3.0, 4.0], dtype=np.float32)
    candidate = np.asarray([3.0, 5.0], dtype=np.float32)

    result = array_comparison(reference, candidate)

    assert result["reference_l2_norm"] == 5.0
    assert result["difference_l2_norm"] == 1.0
    assert result["max_absolute_error"] == 1.0
    assert result["relative_l2_error"] == 0.2
    assert result["comparison_nonfinite_count"] == 0
    assert result["exactly_equal"] is False


def test_tree_comparison_rows_use_stable_parameter_paths():
    reference = {"block": {"weight": jnp.asarray([1.0, 2.0])}}
    candidate = {"block": {"weight": jnp.asarray([1.0, 2.0])}}

    rows = tree_comparison_rows(reference, candidate)

    assert [row["parameter_path"] for row in rows] == ["block/weight"]
    assert rows[0]["exactly_equal"] is True
    assert trees_bitwise_equal(reference, candidate) is True


def test_bitwise_equality_distinguishes_signed_zero_and_dtype():
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)

    assert np.array_equal(positive_zero, negative_zero)
    assert arrays_bitwise_equal(positive_zero, negative_zero) is False
    assert arrays_bitwise_equal(positive_zero, positive_zero.copy()) is True
    assert arrays_bitwise_equal(positive_zero, positive_zero.astype(np.float64)) is False
    assert trees_bitwise_equal({"x": positive_zero}, {"x": negative_zero}) is False


def test_one_ulp_control_changes_every_float_element_and_preserves_integers():
    params = {
        "float": jnp.asarray([0.0, 1.0, -1.0], dtype=jnp.float32),
        "integer": jnp.asarray([1, 2], dtype=jnp.int32),
    }

    perturbed, metadata = perturb_parameters_one_ulp(params)

    original_float = np.asarray(params["float"])
    perturbed_float = np.asarray(perturbed["float"])
    assert np.count_nonzero(original_float != perturbed_float) == 3
    assert np.array_equal(np.asarray(params["integer"]), perturbed["integer"])
    assert metadata["changed_element_count"] == 3
    assert metadata["perturbation_l2_norm"] > 0.0
    assert metadata["perturbation_max_absolute"] > 0.0


def test_array_comparison_marks_nonfinite_pairs_without_error_magnitudes():
    result = array_comparison(np.asarray([1.0, np.nan]), np.asarray([1.0, np.nan]))

    assert result["comparison_nonfinite_count"] == 1
    assert result["difference_l2_norm"] is None
    assert result["max_absolute_error"] is None
    assert result["relative_l2_error"] is None
    assert result["exactly_equal"] is False


def _source_fixture():
    params = {"weight": jnp.asarray([1.0], dtype=jnp.float32)}
    moments = {"weight": jnp.zeros(1, dtype=jnp.float32)}
    source = {
        "format": "poincare-exec-checkpoint-v1",
        "task": "fake_task",
        "variant": "F",
        "seed": 11,
        "update": 100,
        "epoch": 10,
        "config_hash": "source-config",
        "protocol_version": "1.10",
        "code_commit": "a" * 40,
        "code_dirty": False,
        "params": params,
        "optimizer_state": AdamState(
            step=jnp.asarray(100, dtype=jnp.int32),
            first_moment=moments,
            second_moment=moments,
        ),
    }
    summary = {
        "status": "ok",
        "experiment_id": "E99",
        "task": "fake_task",
        "variant": "F",
        "seed_or_session": 11,
        "updates": 100,
        "epochs": 10,
        "config_hash": "source-config",
        "protocol_version": "1.10",
        "code_commit": "a" * 40,
        "code_dirty": False,
    }
    task_entry = {
        "source_seed": 11,
        "source_variants": ["R", "F", "T"],
        "source_config_hash": "source-config",
        "source_protocol_version": "1.10",
    }
    training_config = {"experiment_id": "E99", "task": "fake_task"}
    task = {"max_updates": 100, "epochs": 10}
    return source, summary, task_entry, training_config, task


def test_source_validation_accepts_complete_matching_training_state():
    source, summary, task_entry, training_config, task = _source_fixture()

    _validate_source(
        source,
        summary,
        task_entry=task_entry,
        training_config=training_config,
        task=task,
    )


def test_source_validation_rejects_summary_or_optimizer_mismatch():
    source, summary, task_entry, training_config, task = _source_fixture()
    wrong_summary = {**summary, "code_commit": "b" * 40}

    with pytest.raises(RuntimeError, match="summary and checkpoint"):
        _validate_source(
            source,
            wrong_summary,
            task_entry=task_entry,
            training_config=training_config,
            task=task,
        )

    source["optimizer_state"] = source["optimizer_state"]._replace(
        step=jnp.asarray(99, dtype=jnp.int32)
    )
    with pytest.raises(RuntimeError, match="Adam step"):
        _validate_source(
            source,
            summary,
            task_entry=task_entry,
            training_config=training_config,
            task=task,
        )


def test_source_validation_rejects_wrong_experiment_or_checkpoint_format():
    source, summary, task_entry, training_config, task = _source_fixture()
    wrong_summary = {**summary, "experiment_id": "E98"}

    with pytest.raises(RuntimeError, match="summary and checkpoint"):
        _validate_source(
            source,
            wrong_summary,
            task_entry=task_entry,
            training_config=training_config,
            task=task,
        )

    source["format"] = "unknown"
    with pytest.raises(RuntimeError, match="format"):
        _validate_source(
            source,
            summary,
            task_entry=task_entry,
            training_config=training_config,
            task=task,
        )
