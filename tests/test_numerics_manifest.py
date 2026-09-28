from __future__ import annotations

from poincare_exec.config.loader import load_yaml
from poincare_exec.experiments.numerics import _assert_manifest, generate_case_manifest


def test_manifest_is_deterministic_and_pairwise_complete() -> None:
    protocol = load_yaml("protocol.yaml")
    first = generate_case_manifest(protocol)
    second = generate_case_manifest(protocol)
    assert first == second
    assert len(first["cases"]) == 100
    _assert_manifest(first, protocol)
