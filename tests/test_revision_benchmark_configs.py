from pathlib import Path

from poincare_exec.config.loader import load_yaml

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
        "id": "unet_b8_update_current_jax",
        "kind": "unet",
        "mode": "update",
        "shape": {"batch": 8, "height": 256, "width": 256},
        "curvature": 0.1,
        "dtype": "float32",
    }
