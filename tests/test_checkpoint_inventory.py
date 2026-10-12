from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from poincare_exec.analysis import checkpoint_inventory as inventory
from poincare_exec.config.loader import config_hash

HISTORICAL_SEEDS = [11, 29, 47, 71, 101]
EXTENSION_SEEDS = list(range(201, 216))
TASKS = {
    "isic16_poincare_unet": {"epochs": 10, "max_updates": 100},
    "cifar100_poincare_resnet20": {"epochs": 10, "max_updates": 100},
}
COHORTS = {
    "E09": ("isic16_poincare_unet", ["R", "F", "T"], HISTORICAL_SEEDS),
    "E10": ("cifar100_poincare_resnet20", ["R", "F", "T"], HISTORICAL_SEEDS),
    "E20": ("isic16_poincare_unet", ["R", "F"], EXTENSION_SEEDS),
    "E21": ("cifar100_poincare_resnet20", ["R", "F"], EXTENSION_SEEDS),
}
COMMITS = {
    "E09": "1" * 40,
    "E10": "2" * 40,
    "E20": "3" * 40,
    "E21": "4" * 40,
}
FROZEN_HISTORICAL_INVENTORY_SHA256 = (
    "bc112cfa4a3d725d3e763c792325def98eafe589fb2671684b9bb229009bc52b"
)


@dataclass(frozen=True)
class Campaign:
    repository: Path
    manifest: Path
    analysis_output: Path
    roots: dict[str, Path]
    existing_inventory: Path
    config_hashes: dict[str, str]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _milestone(role: str) -> tuple[int, int, float, int]:
    if role == "initial":
        return 0, 0, -math.inf, -1
    if role == "10_percent":
        return 10, 1, -math.inf, -1
    if role == "50_percent":
        return 50, 5, 0.4, 40
    if role == "validation_best":
        return 20, 2, 0.5, 20
    return 100, 10, 0.5, 20


def _payload(
    *,
    experiment_id: str,
    task: str,
    variant: str,
    seed: int,
    role: str,
    digest: str,
) -> dict[str, Any]:
    update, epoch, best_metric, best_update = _milestone(role)
    return {
        "format": "poincare-exec-checkpoint-v1",
        "config_hash": digest,
        "protocol_version": "1.10" if experiment_id in {"E09", "E10"} else "1.14",
        "code_commit": COMMITS[experiment_id],
        "code_dirty": False,
        "task": task,
        "variant": variant,
        "seed": seed,
        "epoch": epoch,
        "update": update,
        "best_metric": best_metric,
        "best_update": best_update,
        "params": {"weight": 1.0},
        "optimizer_state": {
            "step": update,
            "first_moment": {"weight": 0.0},
            "second_moment": {"weight": 0.0},
        },
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=inventory.INVENTORY_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _historical_inventory(roots: dict[str, Path], config_hashes: dict[str, str]) -> bytes:
    rows: list[dict[str, Any]] = []
    for experiment_id in ("E09", "E10"):
        task, variants, seeds = COHORTS[experiment_id]
        for variant in sorted(variants):
            for seed in seeds:
                run_dir = roots[experiment_id] / variant / f"seed_{seed}"
                for role, relative in inventory.ROLE_PATHS.items():
                    if role == "resume":
                        continue
                    checkpoint = run_dir / relative
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "task": task,
                            "variant": variant,
                            "seed": seed,
                            "role": role,
                            "path": (
                                Path("artifacts")
                                / "training"
                                / experiment_id.lower()
                                / variant
                                / f"seed_{seed}"
                                / relative
                            ).as_posix(),
                            "bytes": checkpoint.stat().st_size,
                            "sha256": _digest(checkpoint),
                            "code_commit": COMMITS[experiment_id],
                            "config_hash": config_hashes[experiment_id],
                            "protocol_version": "1.10",
                        }
                    )
    return _csv_bytes(rows)


def _make_campaign(root: Path) -> Campaign:
    repository = root / "repository"
    repository.mkdir()
    protocol = {"protocol_version": "1.14", "tasks": TASKS}
    protocol_path = repository / "protocol.yaml"
    _write_yaml(protocol_path, protocol)

    roots: dict[str, Path] = {}
    registrations: dict[str, Any] = {}
    config_hashes: dict[str, str] = {}
    for experiment_id, (task, variants, seeds) in COHORTS.items():
        config = {
            "experiment_id": experiment_id,
            "task": task,
            "dataset": "synthetic",
            "variants": variants,
            "seeds": seeds,
        }
        config_path = repository / "configs" / f"{experiment_id.lower()}.yaml"
        _write_yaml(config_path, config)
        digest = (
            config_hash(protocol, config)
            if experiment_id in {"E20", "E21"}
            else ("a" if experiment_id == "E09" else "b") * 64
        )
        config_hashes[experiment_id] = digest
        registrations[experiment_id] = {
            "task": task,
            "config_document": {
                "path": config_path.relative_to(repository).as_posix(),
                "sha256": _digest(config_path),
            },
            "run_config_hash": digest,
            "protocol_version": ("1.10" if experiment_id in {"E09", "E10"} else "1.14"),
            "variants": variants,
            "seeds": seeds,
            "training_health": {"total_train_rows_per_epoch": 10},
            "accepted_code_assignments": [
                {"commit": COMMITS[experiment_id], "variants": variants, "seeds": seeds}
            ],
        }
        cohort_root = root / "accepted" / experiment_id.lower()
        roots[experiment_id] = cohort_root
        for variant in variants:
            for seed in seeds:
                run_dir = cohort_root / variant / f"seed_{seed}"
                for role, relative in inventory.ROLE_PATHS.items():
                    _write_checkpoint(
                        run_dir / relative,
                        _payload(
                            experiment_id=experiment_id,
                            task=task,
                            variant=variant,
                            seed=seed,
                            role=role,
                            digest=digest,
                        ),
                    )
                (run_dir / "summary.json").write_text("{poisoned", encoding="utf-8")
                (run_dir / "predictions.npz").write_bytes(b"poisoned endpoint payload")

    manifest_value = {
        "analysis_id": "synthetic_e20_e21",
        "protocol_document": {
            "path": "protocol.yaml",
            "sha256": _digest(protocol_path),
        },
        "global_embargo": {
            "experiments": ["E20", "E21"],
            "required_successful_runs": 60,
            "required_complete_pairs": 30,
        },
        "cohorts": registrations,
        "required_outputs": {"gate_audit": "e20_e21_gate_audit.json"},
    }
    manifest = repository / "manifest.yaml"
    _write_yaml(manifest, manifest_value)
    analysis_output = root / "analysis"
    analysis_output.mkdir()
    gate = {
        "analysis_id": "synthetic_e20_e21",
        "status": "open",
        "opened_only_after_joint_validation": True,
        "required_successful_runs": 60,
        "required_complete_pairs": 30,
        "cohorts": [
            {
                "experiment_id": experiment_id,
                "status": "complete",
                "expected_runs": 30,
                "validated_runs": 30,
                "expected_pairs": 15,
                "validated_pairs": 15,
                "official_endpoint_embargo": "opened_after_global_validation",
            }
            for experiment_id in reversed(inventory.EXTENSION_IDS)
        ],
    }
    (analysis_output / "e20_e21_gate_audit.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )
    existing = root / "existing.csv"
    existing.write_bytes(_historical_inventory(roots, config_hashes))
    return Campaign(
        repository=repository,
        manifest=manifest,
        analysis_output=analysis_output,
        roots=roots,
        existing_inventory=existing,
        config_hashes=config_hashes,
    )


@pytest.fixture
def campaign(tmp_path: Path) -> Campaign:
    return _make_campaign(tmp_path)


def test_checked_in_historical_projection_is_byte_stable() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "results/checkpoints/checkpoint_inventory.csv"
    )
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == inventory.INVENTORY_FIELDS
        rows = list(reader)
    projection = [
        row
        for row in rows
        if row["experiment_id"] in {"E09", "E10"}
        and row["role"] in inventory.HISTORICAL_ROLES
    ]

    assert len(rows) in {150, 540}
    assert len(projection) == 150
    assert (
        hashlib.sha256(_csv_bytes(projection)).hexdigest()
        == FROZEN_HISTORICAL_INVENTORY_SHA256
    )


def _arguments(campaign: Campaign, output: Path, order: list[str]) -> list[str]:
    arguments = [
        "--manifest",
        str(campaign.manifest),
        "--repository-root",
        str(campaign.repository),
        "--analysis-output-dir",
        str(campaign.analysis_output),
        "--existing-inventory",
        str(campaign.existing_inventory),
        "--output",
        str(output),
    ]
    for experiment_id in order:
        arguments.extend(
            ["--cohort-root", f"{experiment_id}={campaign.roots[experiment_id]}"]
        )
    return arguments


def _checkpoint_stats(campaign: Campaign) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for root in campaign.roots.values()
        for path in root.rglob("*.pkl")
    }


def test_cli_order_is_deterministic_poisoned_endpoints_are_never_opened(
    campaign: Campaign, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _checkpoint_stats(campaign)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name in {"summary.json", "predictions.npz"}:
            raise AssertionError(f"endpoint artifact was opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    assert inventory.main(_arguments(campaign, first, ["E21", "E09", "E20", "E10"])) == 0
    second_arguments = _arguments(campaign, second, ["E10", "E20", "E09", "E21"])
    existing_index = second_arguments.index("--existing-inventory") + 1
    second_arguments[existing_index] = str(first)
    assert inventory.main(second_arguments) == 0

    assert first.read_bytes() == second.read_bytes()
    assert _checkpoint_stats(campaign) == before
    with first.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 540
    assert set(rows[0]) == set(inventory.INVENTORY_FIELDS)
    assert {(row["experiment_id"], row["variant"], row["seed"]) for row in rows} == {
        (experiment_id, variant, str(seed))
        for experiment_id, (_, variants, seeds) in COHORTS.items()
        for variant in variants
        for seed in seeds
    }
    assert {row["role"] for row in rows} == set(inventory.ROLE_PATHS)


def test_gate_is_required_before_any_cohort_root_resolution(campaign: Campaign) -> None:
    gate_path = campaign.analysis_output / "e20_e21_gate_audit.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["status"] = "closed"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    class TraversalTrap:
        def __fspath__(self) -> str:
            raise AssertionError("cohort root was resolved before gate acceptance")

    with pytest.raises(inventory.CheckpointInventoryError, match="invalid status"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            {experiment_id: TraversalTrap() for experiment_id in COHORTS},
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_checkpoint_role_matrix_rejects_missing_and_extra_files(
    campaign: Campaign, mutation: str
) -> None:
    checkpoint_dir = campaign.roots["E09"] / "F" / "seed_11" / "checkpoints"
    if mutation == "missing":
        (checkpoint_dir / "initial.pkl").unlink()
    else:
        (checkpoint_dir / "rejected_attempt.pkl").write_bytes(b"not accepted")

    with pytest.raises(inventory.CheckpointInventoryError, match="checkpoint roles"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


def _rewrite(path: Path, mutate: Any) -> None:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    mutate(payload)
    _write_checkpoint(path, payload)


def test_checkpoint_identity_is_registered(campaign: Campaign) -> None:
    path = campaign.roots["E09"] / "F" / "seed_11" / "checkpoints" / "initial.pkl"
    _rewrite(path, lambda payload: payload.__setitem__("task", "wrong_task"))

    with pytest.raises(inventory.CheckpointInventoryError, match="identity field task"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


@pytest.mark.parametrize("mutation", ["epoch", "optimizer_step"])
def test_checkpoint_milestones_and_optimizer_steps_are_validated(
    campaign: Campaign, mutation: str
) -> None:
    path = campaign.roots["E09"] / "F" / "seed_11" / "checkpoints" / "10_percent.pkl"

    def mutate(payload: dict[str, Any]) -> None:
        if mutation == "epoch":
            payload["epoch"] = 2
        else:
            payload["optimizer_state"]["step"] = 11

    _rewrite(path, mutate)

    expected = "milestone" if mutation == "epoch" else "optimizer step"
    with pytest.raises(inventory.CheckpointInventoryError, match=expected):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("seed", 201.0, "identity field seed"),
        ("update", 10.5, "milestone"),
        ("epoch", True, "milestone"),
        ("optimizer_step", math.inf, "optimizer step"),
        ("best_update", 20.5, "best-checkpoint metadata"),
    ],
)
def test_checkpoint_integer_fields_require_exact_scalars(
    campaign: Campaign,
    field: str,
    value: Any,
    expected: str,
) -> None:
    role = "validation_best" if field == "best_update" else "10_percent"
    path = campaign.roots["E20"] / "F" / "seed_201" / "checkpoints" / f"{role}.pkl"

    def mutate(payload: dict[str, Any]) -> None:
        if field == "optimizer_step":
            payload["optimizer_state"]["step"] = value
        else:
            payload[field] = value

    _rewrite(path, mutate)

    with pytest.raises(inventory.CheckpointInventoryError, match=expected):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


def test_terminal_checkpoints_share_validation_best_metadata(campaign: Campaign) -> None:
    path = campaign.roots["E20"] / "F" / "seed_201" / "checkpoints" / "validation_best.pkl"

    def mutate(payload: dict[str, Any]) -> None:
        payload["update"] = 30
        payload["epoch"] = 3
        payload["optimizer_state"]["step"] = 30
        payload["best_update"] = 30
        payload["best_metric"] = 0.6

    _rewrite(path, mutate)

    with pytest.raises(
        inventory.CheckpointInventoryError,
        match="best-checkpoint metadata disagrees with validation_best",
    ):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


def test_checkpoint_directory_symlink_cannot_escape_cohort_root(
    campaign: Campaign, tmp_path: Path
) -> None:
    run_dir = campaign.roots["E20"] / "F" / "seed_201"
    outside = tmp_path / "outside" / "checkpoints"
    outside.parent.mkdir()
    shutil.move(run_dir / "checkpoints", outside)
    (run_dir / "checkpoints").symlink_to(outside, target_is_directory=True)

    with pytest.raises(inventory.CheckpointInventoryError, match="non-symlink"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


def test_checkpoint_hash_and_validation_use_the_same_byte_snapshot(
    campaign: Campaign,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = campaign.roots["E20"] / "F" / "seed_201" / "checkpoints" / "10_percent.pkl"
    original_bytes = target.read_bytes()
    replacement = pickle.loads(original_bytes)
    replacement["params"]["weight"] = 999.0
    replacement_bytes = pickle.dumps(replacement, protocol=5)
    original_loads = pickle.loads
    swapped = False

    def swap_after_snapshot(payload: bytes) -> Any:
        nonlocal swapped
        value = original_loads(payload)
        if not swapped and payload == original_bytes:
            target.write_bytes(replacement_bytes)
            swapped = True
        return value

    monkeypatch.setattr(inventory.pickle, "loads", swap_after_snapshot)
    output = tmp_path / "inventory.csv"
    rows = inventory.export_checkpoint_inventory(
        campaign.manifest,
        campaign.roots,
        campaign.analysis_output,
        campaign.existing_inventory,
        output,
        repository_root=campaign.repository,
    )

    row = next(
        row
        for row in rows
        if row["experiment_id"] == "E20"
        and row["variant"] == "F"
        and row["seed"] == 201
        and row["role"] == "10_percent"
    )
    assert swapped
    assert target.read_bytes() == replacement_bytes
    assert row["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert row["sha256"] != hashlib.sha256(replacement_bytes).hexdigest()


def test_gate_audit_symlink_cannot_escape_analysis_root(
    campaign: Campaign, tmp_path: Path
) -> None:
    gate = campaign.analysis_output / "e20_e21_gate_audit.json"
    outside = tmp_path / "outside_gate.json"
    gate.replace(outside)
    gate.symlink_to(outside)

    with pytest.raises(inventory.CheckpointInventoryError, match="non-symlink"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


def test_historical_checkpoint_hashes_must_match_frozen_projection(
    campaign: Campaign,
) -> None:
    path = campaign.roots["E09"] / "F" / "seed_11" / "checkpoints" / "initial.pkl"
    _rewrite(path, lambda payload: payload.__setitem__("unused_tamper", True))

    with pytest.raises(inventory.CheckpointInventoryError, match="not byte-identical"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            campaign.repository / "output.csv",
            repository_root=campaign.repository,
        )


def test_output_replacement_is_atomic(
    campaign: Campaign, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "inventory.csv"
    output.write_bytes(b"previous complete inventory\n")

    def fail_replace(source: Path, destination: Path, **_: Any) -> None:
        raise OSError(f"simulated replace failure: {source} -> {destination}")

    monkeypatch.setattr(inventory.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            output,
            repository_root=campaign.repository,
        )

    assert output.read_bytes() == b"previous complete inventory\n"
    assert not list(tmp_path.glob(".inventory.csv.*.tmp"))


def test_output_symlink_is_refused_without_replacing_target(
    campaign: Campaign, tmp_path: Path
) -> None:
    target = tmp_path / "target.csv"
    target.write_bytes(b"keep me\n")
    output = tmp_path / "inventory.csv"
    output.symlink_to(target)

    with pytest.raises(inventory.CheckpointInventoryError, match="non-symlink"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            output,
            repository_root=campaign.repository,
        )

    assert output.is_symlink()
    assert target.read_bytes() == b"keep me\n"


def test_output_parent_symlink_is_refused(campaign: Campaign, tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(target, target_is_directory=True)

    with pytest.raises(inventory.CheckpointInventoryError, match="output parent"):
        inventory.export_checkpoint_inventory(
            campaign.manifest,
            campaign.roots,
            campaign.analysis_output,
            campaign.existing_inventory,
            linked_parent / "inventory.csv",
            repository_root=campaign.repository,
        )

    assert not (target / "inventory.csv").exists()
