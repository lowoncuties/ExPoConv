from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.verify_provenance_bundle import verify_bundle


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    repository = tmp_path / "source"
    repository.mkdir()
    _run(["git", "init", "--quiet"], cwd=repository)
    _run(["git", "config", "user.name", "Test Reproducer"], cwd=repository)
    _run(["git", "config", "user.email", "test@example.invalid"], cwd=repository)
    _run(["git", "checkout", "-b", "accepted-provenance"], cwd=repository)

    commits = []
    for index in range(2):
        (repository / "evidence.txt").write_text(f"revision {index}\n", encoding="utf-8")
        _run(["git", "add", "evidence.txt"], cwd=repository)
        _run(["git", "commit", "--quiet", "-m", f"revision {index}"], cwd=repository)
        commits.append(_run(["git", "rev-parse", "HEAD"], cwd=repository))

    bundle = tmp_path / "accepted-provenance.bundle"
    _run(
        [
            "git",
            "bundle",
            "create",
            str(bundle),
            "refs/heads/accepted-provenance",
        ],
        cwd=repository,
    )
    manifest = {
        "artifact_name": bundle.name,
        "size_bytes": bundle.stat().st_size,
        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "bundle_ref": "refs/heads/accepted-provenance",
        "bundle_tip": commits[-1],
        "release_history_commits": [],
        "accepted_commits": [
            {"family": f"fixture-{index}", "commit": commit}
            for index, commit in enumerate(commits)
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, bundle, manifest


def test_verifies_declared_bundle_and_every_commit(tmp_path: Path) -> None:
    manifest_path, bundle, manifest = _fixture(tmp_path)

    result = verify_bundle(manifest_path, bundle)

    assert result["status"] == "ok"
    assert result["bundle_tip"] == manifest["bundle_tip"]
    assert result["verified_commits"] == 2


def test_rejects_bundle_with_wrong_digest(tmp_path: Path) -> None:
    manifest_path, bundle, manifest = _fixture(tmp_path)
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_bundle(manifest_path, bundle)


def test_tracked_manifest_covers_every_results_readme_revision() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (repository / "provenance/v100_accepted_history_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    results_readme = (repository / "results/README.md").read_text(encoding="utf-8")
    provenance_table = results_readme.split(
        "## Provenance of each result family", maxsplit=1
    )[1].split("The E08 forward rows", maxsplit=1)[0]
    documented_prefixes = set(re.findall(r"`([0-9a-f]{8})`", provenance_table))
    bundled_prefixes = {
        row["commit"][:8] for row in manifest["accepted_commits"]
    }
    release_prefixes = {
        row["commit"][:8] for row in manifest["release_history_commits"]
    }

    assert bundled_prefixes.isdisjoint(release_prefixes)
    assert documented_prefixes == bundled_prefixes | release_prefixes
    for row in manifest["release_history_commits"]:
        commit = row["commit"]
        _run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repository)
        _run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repository)
