import hashlib
from pathlib import Path

from poincare_exec.config.loader import load_yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_1_19 = REPOSITORY_ROOT / "configs/protocols/protocol_1_19.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_1_19_registers_e31_before_execution() -> None:
    protocol = load_yaml(PROTOCOL_1_19)
    amendment = protocol["protocol_amendments"][-1]
    registered = protocol["revision_benchmarks"]["E31"]
    config = load_yaml(REPOSITORY_ROOT / registered["config"])

    assert protocol["protocol_version"] == "1.19"
    assert [item["version"] for item in protocol["protocol_amendments"][-2:]] == [
        "1.17",
        "1.19",
    ]
    assert amendment["affected_experiments"] == ["E31"]
    assert "1.18 is reserved for the E29/E30 registration" in amendment["reason"]
    assert config["experiment_id"] == "E31"
    assert registered["variants"] == config["variants"] == ["R", "Rg", "Mg", "F"]
    assert registered["workloads"] == [workload["id"] for workload in config["workloads"]]
    for field in ("sessions", "warmup_calls", "timed_calls_per_session"):
        assert registered[field] == config[field]
    assert (config["sessions"], config["warmup_calls"]) == (5, 50)
    assert config["timed_calls_per_session"] == 200
    assert registered["allocator_fraction"] == 0.95
    assert registered["preallocate"] is False
    assert config["require_idle_gpu"] is True
    assert config["max_preexisting_memory_mib"] == 512
    assert config["max_preexisting_utilization_percent"] == 5
    assert registered["contrasts"] == [
        ["R", "Rg"],
        ["Rg", "F"],
        ["R", "F"],
        ["Rg", "Mg"],
        ["Mg", "F"],
    ]
    assert registered["compiled_memory_analysis"] == {
        "script": "scripts/capture_e31_compiled_memory.py",
        "workload": "unet_b8_update",
        "variants": ["R", "Rg", "F"],
    }
    assert registered["gpu_equivalence_check"] == {
        "script": "scripts/check_e31_gpu_equivalence.py",
        "pairs": [["R", "Rg"], ["M", "Mg"]],
    }


def test_e31_workloads_copy_the_e24_and_e25_shapes_verbatim() -> None:
    e31 = {
        workload["id"]: workload
        for workload in load_yaml(
            REPOSITORY_ROOT / "configs/benchmarks/e31_gather_reference.yaml"
        )["workloads"]
    }
    e24 = load_yaml(REPOSITORY_ROOT / "configs/benchmarks/e24_unet_mechanism_ablation.yaml")
    e25 = load_yaml(REPOSITORY_ROOT / "configs/benchmarks/e25_euclidean_context.yaml")
    expected = [*e24["workloads"]]
    expected.extend(
        workload
        for workload in e25["workloads"]
        if workload["id"] == "resnet20_b128_update"
    )

    assert list(e31.values()) == expected


def test_protocol_snapshots_preserve_the_run_protocols() -> None:
    assert _sha256(REPOSITORY_ROOT / "configs/protocols/protocol_1_17.yaml") == (
        "3dadcc5e27bc93f535506168d48ff9f3e8667e6f6f98c24c44cb56beba23eb1e"
    )
    assert _sha256(PROTOCOL_1_19) == (
        "e39d1f658fc9e963c517e06388adee754ba2f8621fc9bf8d9c7921a388620de5"
    )
    assert (
        load_yaml(REPOSITORY_ROOT / "configs/protocols/protocol_1_17.yaml")[
            "protocol_version"
        ]
        == "1.17"
    )


def test_e31_launcher_pins_the_e24_e25_allocator_policy() -> None:
    launcher = (REPOSITORY_ROOT / "scripts/run_e31.sh").read_text(encoding="utf-8")

    assert "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95" in launcher
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "--config configs/benchmarks/e31_gather_reference.yaml" in launcher
