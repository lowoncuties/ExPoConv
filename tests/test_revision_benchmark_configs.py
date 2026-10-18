import hashlib
import json
from pathlib import Path

from poincare_exec.config.loader import config_hash, load_yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_e27_is_a_separate_current_jax_batch_eight_sanity_check() -> None:
    config = load_yaml(
        REPOSITORY_ROOT / "configs/benchmarks/e27_current_jax_unet_update.yaml"
    )

    assert config["experiment_id"] == "E27"
    assert config["variants"] == ["R", "F"]
    assert config["sessions"] == 5
    assert config["warmup_calls"] == 50
    assert config["timed_calls_per_session"] == 200
    assert config["require_idle_gpu"] is True
    assert config["max_preexisting_memory_mib"] == 512
    assert config["max_preexisting_utilization_percent"] == 5
    assert len(config["workloads"]) == 1

    workload = config["workloads"][0]
    assert workload == {
        "id": "unet_update_b8_r256",
        "kind": "unet",
        "mode": "update",
        "shape": {"batch": 8, "height": 256, "width": 256},
        "curvature": 0.1,
        "dtype": "float32",
    }


def test_e28_replays_the_first_three_e26_fp32_realizations() -> None:
    config = load_yaml(
        REPOSITORY_ROOT / "configs/numerics/e28_elementwise_replay_stability.yaml"
    )

    assert config["experiment_id"] == "E28"
    assert config["precisions"] == ["float32"]
    assert config["realizations"] == 3
    assert config["finite_difference_relative_steps"] == []
    assert [case["id"] for case in config["cases"]] == [
        "convolution_stack_8",
        "small_unet",
    ]
    assert config["elementwise_relative_error"] == {
        "denominator_floor": 1.0e-12,
        "quantiles": [0.5, 0.9, 0.95, 0.99, 0.999],
        "quantile_method": "linear",
        "report_denominator_floored_fraction": True,
    }


def _registration(experiment_id: str) -> tuple[dict, dict]:
    protocol = load_yaml(REPOSITORY_ROOT / "protocol.yaml")
    registered = protocol["revision_benchmarks"][experiment_id]
    config = load_yaml(REPOSITORY_ROOT / registered["config"])
    return registered, config


def _assert_matches_registration(registered: dict, config: dict) -> None:
    assert registered["variants"] == config["variants"]
    assert registered["workloads"] == [workload["id"] for workload in config["workloads"]]
    for field in ("sessions", "warmup_calls", "timed_calls_per_session"):
        assert registered[field] == config[field]
    assert registered["allocator_fraction"] == "default"
    assert registered["preallocate"] is False
    assert config["require_idle_gpu"] is True
    assert config["max_preexisting_memory_mib"] == 512
    assert config["max_preexisting_utilization_percent"] == 5


def test_protocol_1_18_registers_e29_and_e30_before_execution() -> None:
    protocol = load_yaml(REPOSITORY_ROOT / "protocol.yaml")
    amendment = protocol["protocol_amendments"][-1]

    assert protocol["protocol_version"] == "1.18"
    assert amendment["version"] == "1.18"
    assert amendment["affected_experiments"] == ["E29", "E30"]
    for experiment_id in ("E29", "E30"):
        registered, config = _registration(experiment_id)
        assert config["experiment_id"] == experiment_id
        _assert_matches_registration(registered, config)


def test_e29_copies_the_e08_and_e13_workloads_verbatim() -> None:
    _, config = _registration("E29")
    e08 = load_yaml(REPOSITORY_ROOT / "configs/benchmarks/e08_unet_bounded.yaml")
    e13 = load_yaml(REPOSITORY_ROOT / "configs/benchmarks/e13_unet_scaling.yaml")

    assert config["variants"] == ["E"]
    assert config["workloads"] == e08["workloads"] + e13["workloads"]
    for field in ("sessions", "warmup_calls", "timed_calls_per_session"):
        assert config[field] == e13[field] == e08[field]


def test_e30_changes_only_the_initial_width_of_the_e13_pivot() -> None:
    _, config = _registration("E30")
    e13 = load_yaml(REPOSITORY_ROOT / "configs/benchmarks/e13_unet_scaling.yaml")
    pivot = next(item for item in e13["workloads"] if item["id"] == "unet_update_b8_r256")

    assert config["variants"] == e13["variants"] == ["R", "F"]
    assert [workload["initial_width"] for workload in config["workloads"]] == [16, 32]
    for workload in config["workloads"]:
        expected = {
            **pivot,
            "id": f"unet_update_b8_r256_iw{workload['initial_width']}",
            "sweep_axes": ["initial_width"],
            "initial_width": workload["initial_width"],
        }
        assert workload == expected


def test_only_e30_benchmark_configs_set_initial_width() -> None:
    configured = {
        path.name: [
            workload["initial_width"]
            for workload in load_yaml(path)["workloads"]
            if "initial_width" in workload
        ]
        for path in sorted((REPOSITORY_ROOT / "configs" / "benchmarks").glob("*.yaml"))
    }

    assert {name: widths for name, widths in configured.items() if widths} == {
        "e30_v100_unet_width.yaml": [16, 32]
    }


def test_protocol_1_17_snapshot_binds_the_committed_e24_e25_e27_hashes() -> None:
    snapshot = REPOSITORY_ROOT / "configs/protocols/protocol_1_17.yaml"
    audit = json.loads(
        (REPOSITORY_ROOT / "results/reviewer_requested/e24_e25_e27_audit.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = load_yaml(snapshot)

    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == audit["protocol"]["sha256"]
    assert protocol["protocol_version"] == "1.17"
    for cohort in audit["cohorts"]:
        registered = protocol["revision_benchmarks"][cohort["experiment_id"]]
        config = load_yaml(REPOSITORY_ROOT / registered["config"])
        assert {
            workload["id"]: config_hash(protocol, config, workload)
            for workload in config["workloads"]
        } == cohort["expected_config_hashes"]
