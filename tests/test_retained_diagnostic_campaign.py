import pytest

from poincare_exec.config.loader import config_hash
from poincare_exec.experiments.retained_diagnostic_campaign import (
    _validate_completed_summary,
)


def _fixture():
    protocol = {"protocol_version": "1.10"}
    diagnostic_config = {
        "experiment_id": "E07-J1",
        "design_document": {
            "path": "docs/journal_diagnostic_protocol_v2.md",
            "sha256": ("c356b7c7325295b4d93aaaa090bc5f1c47398d975b2293a88bf34512bcc0e31f"),
        },
        "controls": {
            "unperturbed_repeat": True,
            "one_ulp_perturbation": True,
        },
    }
    training_config = {"experiment_id": "E99", "task": "fake_task"}
    task_entry = {
        "source_seed": 11,
        "source_config_hash": "source-config",
        "source_protocol_version": "1.10",
    }
    code = {"commit": "a" * 40, "dirty": False}
    checkpoint_sha256 = "b" * 64
    source_summary_sha256 = "c" * 64
    summary = {
        "experiment_id": "E07-J1",
        "protocol_version": "1.10",
        "variant": "shared",
        "config_hash": config_hash(
            protocol, diagnostic_config, task_entry, training_config
        ),
        "code_commit": code["commit"],
        "code_dirty": False,
        "seed_or_session": 11,
        "task": "fake_task",
        "source_experiment_id": "E99",
        "source_variant": "F",
        "source_seed": 11,
        "source_config_hash": "source-config",
        "source_protocol_version": "1.10",
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_summary_sha256": source_summary_sha256,
        "status": "ok",
        "unperturbed_repeat_bitwise_exact": True,
        "unperturbed_repeat_policy": "bitwise",
        "design_document_path": "docs/journal_diagnostic_protocol_v2.md",
        "design_document_sha256": diagnostic_config["design_document"]["sha256"],
    }
    return (
        summary,
        protocol,
        diagnostic_config,
        training_config,
        task_entry,
        code,
        checkpoint_sha256,
        source_summary_sha256,
    )


def test_completed_summary_validation_binds_current_source_hashes():
    (
        summary,
        protocol,
        diagnostic_config,
        training_config,
        task_entry,
        code,
        checkpoint_sha256,
        source_summary_sha256,
    ) = _fixture()

    _validate_completed_summary(
        summary,
        protocol=protocol,
        diagnostic_config=diagnostic_config,
        training_config=training_config,
        task_entry=task_entry,
        source_variant="F",
        source_checkpoint_sha256=checkpoint_sha256,
        source_summary_sha256=source_summary_sha256,
        code=code,
    )

    with pytest.raises(RuntimeError, match="source_checkpoint_sha256"):
        _validate_completed_summary(
            summary,
            protocol=protocol,
            diagnostic_config=diagnostic_config,
            training_config=training_config,
            task_entry=task_entry,
            source_variant="F",
            source_checkpoint_sha256="d" * 64,
            source_summary_sha256=source_summary_sha256,
            code=code,
        )


def test_quantified_repeat_policy_accepts_finite_nonbitwise_repeat():
    (
        summary,
        protocol,
        diagnostic_config,
        training_config,
        task_entry,
        code,
        checkpoint_sha256,
        source_summary_sha256,
    ) = _fixture()
    diagnostic_config["controls"]["unperturbed_repeat_policy"] = "quantify"
    summary["config_hash"] = config_hash(
        protocol, diagnostic_config, task_entry, training_config
    )
    summary["unperturbed_repeat_policy"] = "quantify"
    summary["unperturbed_repeat_bitwise_exact"] = False

    _validate_completed_summary(
        summary,
        protocol=protocol,
        diagnostic_config=diagnostic_config,
        training_config=training_config,
        task_entry=task_entry,
        source_variant="F",
        source_checkpoint_sha256=checkpoint_sha256,
        source_summary_sha256=source_summary_sha256,
        code=code,
    )
