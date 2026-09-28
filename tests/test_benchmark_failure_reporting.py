from __future__ import annotations

from poincare_exec.experiments.benchmark import (
    _classify_worker_failure,
    _logical_gpu_device_identity,
    _read_worker_phase,
    _write_worker_phase,
)


def test_worker_phase_marker_is_overwritten_at_phase_boundaries(tmp_path) -> None:
    path = tmp_path / "phase.json"

    _write_worker_phase(path, "compilation")
    _write_worker_phase(path, "first_update")

    assert _read_worker_phase(path) == "first_update"


def test_worker_phase_reader_retains_unknown_for_missing_or_invalid_marker(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")

    assert _read_worker_phase(missing) == "unknown"
    assert _read_worker_phase(invalid) == "unknown"


def test_failure_classifier_requires_an_explicit_oom_diagnostic() -> None:
    assert (
        _classify_worker_failure(
            stdout="", stderr="RESOURCE_EXHAUSTED: failed to allocate buffer", returncode=1
        )
        == "oom"
    )
    assert (
        _classify_worker_failure(stdout="", stderr="", returncode=-9)
        == "terminated_by_signal"
    )
    assert (
        _classify_worker_failure(stdout="", stderr="shape mismatch", returncode=1)
        == "failed"
    )
    assert (
        _classify_worker_failure(
            stdout="", stderr="RESOURCE_EXHAUSTED: queue limit", returncode=1
        )
        == "failed"
    )


def test_failure_device_identity_matches_single_visible_gpu_worker(monkeypatch) -> None:
    monkeypatch.setattr("poincare_exec.experiments.benchmark.platform.node", lambda: "host")

    assert (
        _logical_gpu_device_identity({"status": "ok", "name": "Tesla V100-PCIE-32GB"})
        == "host::gpu:0::Tesla V100-PCIE-32GB"
    )
