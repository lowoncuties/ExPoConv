from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import scripts.build_release_package as builder_module
from scripts.build_release_package import build_release_package
from scripts.verify_release_package import (
    CURRENT_E19_STATUS,
    MANIFEST_NAME,
    OUTER_CHECKSUM_NAME,
    verify_release_package,
)


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evidence_checksums(repository: Path, tree: str) -> None:
    root = repository / tree
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(root).as_posix()
            rows.append(f"{_sha256(path)}  ./{relative}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")


def _commit(repository: Path, message: str) -> str:
    _run(["git", "add", "."], cwd=repository)
    _run(["git", "commit", "--quiet", "-m", message], cwd=repository)
    return _run(["git", "rev-parse", "HEAD"], cwd=repository)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "source"
    repository.mkdir()
    _run(["git", "init", "--quiet"], cwd=repository)
    _run(["git", "config", "user.name", "Release Test"], cwd=repository)
    _run(["git", "config", "user.email", "release@example.invalid"], cwd=repository)
    _run(["git", "checkout", "-b", "release-test"], cwd=repository)

    (repository / ".gitignore").write_text(
        "/artifacts/\n/data/\n*.ckpt\n", encoding="utf-8"
    )
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    (repository / "run.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repository / "run.sh").chmod(0o755)
    (repository / "results/tables").mkdir(parents=True)
    (repository / "results/README.md").write_text("evidence\n", encoding="utf-8")
    (repository / "results/tables/metric.csv").write_text(
        "metric,value\naccuracy,1\n", encoding="utf-8"
    )
    (repository / "splits").mkdir()
    (repository / "splits/train.csv").write_text("sample,split\na,train\n")
    _write_evidence_checksums(repository, "results")
    _write_evidence_checksums(repository, "splits")
    _commit(repository, "fixture release")
    return repository


def _rewrite_outer_checksums(package: Path) -> None:
    names = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != OUTER_CHECKSUM_NAME
    }
    (package / OUTER_CHECKSUM_NAME).write_text(
        "".join(
            f"{_sha256(package / name)}  ./{name}\n" for name in sorted(names)
        ),
        encoding="utf-8",
    )


def _refresh_artifact_fixity(package: Path, artifact_key: str) -> dict[str, object]:
    manifest_path = package / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"][artifact_key]
    artifact_path = package / artifact["path"]
    artifact["size_bytes"] = artifact_path.stat().st_size
    artifact["sha256"] = _sha256(artifact_path)
    sidecar_key = {
        "reviewer_source_zip": "reviewer_sidecar",
        "release_git_bundle": "repository_bundle_sidecar",
    }[artifact_key]
    sidecar = package / manifest["fixity"][sidecar_key]
    sidecar.write_text(
        f"{artifact['sha256']}  {artifact_path.name}\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_outer_checksums(package)
    return manifest


def _legacy_bundle(repository: Path, tmp_path: Path) -> Path:
    legacy_repository = tmp_path / "legacy-source"
    legacy_repository.mkdir()
    _run(["git", "init", "--quiet"], cwd=legacy_repository)
    _run(["git", "config", "user.name", "Legacy Test"], cwd=legacy_repository)
    _run(
        ["git", "config", "user.email", "legacy@example.invalid"],
        cwd=legacy_repository,
    )
    _run(
        ["git", "checkout", "-b", "v100-accepted-provenance"],
        cwd=legacy_repository,
    )
    (legacy_repository / "legacy.txt").write_text("accepted evidence\n")
    legacy_commit = _commit(legacy_repository, "legacy evidence")
    bundle = tmp_path / "v100-accepted-provenance.bundle"
    _run(
        [
            "git",
            "bundle",
            "create",
            str(bundle),
            "refs/heads/v100-accepted-provenance",
        ],
        cwd=legacy_repository,
    )
    manifest = {
        "artifact_name": bundle.name,
        "size_bytes": bundle.stat().st_size,
        "sha256": _sha256(bundle),
        "bundle_ref": "refs/heads/v100-accepted-provenance",
        "bundle_tip": legacy_commit,
        "release_history_commits": [],
        "accepted_commits": [{"family": "fixture", "commit": legacy_commit}],
    }
    manifest_path = repository / "provenance/v100_accepted_history_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _commit(repository, "legacy manifest")
    return bundle


def test_builds_and_verifies_exact_head_package(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "artifacts").mkdir()
    (repository / "artifacts/raw.jsonl").write_text("ignored\n")
    (repository / "data").mkdir()
    (repository / "data/sample.bin").write_bytes(b"dataset")
    (repository / "model.ckpt").write_bytes(b"checkpoint")
    release_commit = _run(["git", "rev-parse", "HEAD"], cwd=repository)

    first = tmp_path / "release-one"
    second = tmp_path / "release-two"
    result = build_release_package(repository, first)
    build_release_package(repository, second)

    assert result["release_commit"] == release_commit
    verification = verify_release_package(first)
    assert verification["status"] == "ok"
    assert verification["artifacts_verified"] == 2

    short_commit = release_commit[:12]
    reviewer_name = f"reviewer/ExPoConv-reviewer-{short_commit}.zip"
    assert _sha256(first / reviewer_name) == _sha256(second / reviewer_name)
    manifest = json.loads((first / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert set(manifest) == {
        "artifacts",
        "created_utc",
        "fixity",
        "package_id",
        "release",
        "schema_version",
        "scope_limits",
        "visibility",
    }
    assert set(manifest["artifacts"]) == {
        "reviewer_source_zip",
        "release_git_bundle",
    }
    assert manifest["release"]["e19_status"] == CURRENT_E19_STATUS
    with zipfile.ZipFile(first / reviewer_name) as archive:
        names = set(archive.namelist())
    assert f"ExPoConv-reviewer-{short_commit}/README.md" in names
    assert all("/.git/" not in name for name in names)
    assert all("/artifacts/" not in name for name in names)
    assert all("/data/" not in name for name in names)
    assert all(not name.endswith(".ckpt") for name in names)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "coverage mismatch"),
        ("unlisted", "coverage mismatch"),
        ("duplicate", "duplicate checksum path"),
        ("mismatch", "SHA-256 mismatch"),
    ],
)
def test_rejects_inexact_evidence_checksum_manifest(
    tmp_path: Path, case: str, match: str
) -> None:
    repository = _repository(tmp_path)
    checksum_path = repository / "results/SHA256SUMS"
    if case == "missing":
        replacement = ""
    elif case == "unlisted":
        replacement = f"{'0' * 64}  ./ghost.csv\n"
    elif case == "duplicate":
        replacement = checksum_path.read_text(encoding="utf-8")
        replacement += replacement.splitlines(keepends=True)[0]
    else:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
        lines[0] = f"{'0' * 64}{lines[0][64:]}"
        replacement = "\n".join(lines) + "\n"
    checksum_path.write_text(replacement, encoding="utf-8")
    _commit(repository, "invalid results checksums")

    with pytest.raises(ValueError, match=match):
        build_release_package(repository, tmp_path / "release")


def test_rejects_dirty_worktree_and_non_head_ref(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first_commit = _run(["git", "rev-parse", "HEAD"], cwd=repository)
    (repository / "README.md").write_text("second revision\n", encoding="utf-8")
    _commit(repository, "second revision")

    with pytest.raises(ValueError, match="must be HEAD or the current branch"):
        build_release_package(repository, tmp_path / "old", ref=first_commit)

    _run(["git", "checkout", "--quiet", "--detach"], cwd=repository)
    with pytest.raises(ValueError, match="attached to a local branch"):
        build_release_package(repository, tmp_path / "detached")
    _run(["git", "checkout", "--quiet", "release-test"], cwd=repository)

    (repository / "untracked.txt").write_text("not released\n")
    with pytest.raises(ValueError, match="worktree is dirty"):
        build_release_package(repository, tmp_path / "dirty")


def test_rejects_tracked_raw_checkpoint(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "model.ckpt").write_bytes(b"raw checkpoint")
    _run(["git", "add", "-f", "model.ckpt"], cwd=repository)
    _run(["git", "commit", "--quiet", "-m", "bad checkpoint"], cwd=repository)

    with pytest.raises(ValueError, match="raw checkpoint"):
        build_release_package(repository, tmp_path / "release")


def test_verifier_detects_reviewer_zip_content_not_in_commit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    package = tmp_path / "release"
    build_release_package(repository, package)
    manifest_path = package / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reviewer = manifest["artifacts"]["reviewer_source_zip"]
    zip_path = package / reviewer["path"]
    with zipfile.ZipFile(zip_path, "a") as archive:
        info = zipfile.ZipInfo(f"{reviewer['archive_root']}extra.txt")
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, "tampered\n")
    _refresh_artifact_fixity(package, "reviewer_source_zip")

    with pytest.raises(ValueError, match="tracked-file coverage mismatch"):
        verify_release_package(package)


def test_legacy_bundle_requires_explicit_verified_inclusion(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _legacy_bundle(repository, tmp_path)
    without_legacy = tmp_path / "release-without-legacy"
    with_legacy = tmp_path / "release-with-legacy"

    build_release_package(repository, without_legacy)
    manifest = json.loads(
        (without_legacy / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert set(manifest["artifacts"]) == {
        "reviewer_source_zip",
        "release_git_bundle",
    }

    build_release_package(repository, with_legacy, legacy_v100_bundle=bundle)
    manifest = json.loads((with_legacy / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {
        "reviewer_source_zip",
        "release_git_bundle",
        "legacy_v100_provenance_bundle",
    }
    assert verify_release_package(with_legacy)["artifacts_verified"] == 3


def test_rejects_unverified_legacy_bundle_before_creating_output(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    bundle = _legacy_bundle(repository, tmp_path)
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    output = tmp_path / "release"

    with pytest.raises(ValueError, match="bundle size mismatch"):
        build_release_package(repository, output, legacy_v100_bundle=bundle)

    assert not output.exists()


def test_rejects_deleted_raw_checkpoint_from_complete_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    checkpoint = repository / "deleted.ckpt"
    checkpoint.write_bytes(b"historical raw checkpoint")
    _run(["git", "add", "-f", checkpoint.name], cwd=repository)
    _run(["git", "commit", "--quiet", "-m", "historical checkpoint"], cwd=repository)
    _run(["git", "rm", "--quiet", checkpoint.name], cwd=repository)
    _run(["git", "commit", "--quiet", "-m", "remove checkpoint"], cwd=repository)

    with pytest.raises(ValueError, match="release history contains forbidden entries"):
        build_release_package(repository, tmp_path / "rejected")

    monkeypatch.setattr(builder_module, "_verify_release_history", lambda *_: 0)
    monkeypatch.setattr(builder_module, "verify_release_package", lambda *_: {})
    package = tmp_path / "adversarial-package"
    build_release_package(repository, package)
    with pytest.raises(ValueError, match="release history contains forbidden entries"):
        verify_release_package(package)


def test_rejects_nonregular_tracked_mode(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "readme-link").symlink_to("README.md")
    _commit(repository, "tracked symlink")

    with pytest.raises(ValueError, match="mode=120000"):
        build_release_package(repository, tmp_path / "release")


def test_verifier_rechecks_internal_evidence_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    (repository / "results/SHA256SUMS").write_text("", encoding="utf-8")
    _commit(repository, "invalid evidence manifest")
    monkeypatch.setattr(builder_module, "_verify_evidence_tree", lambda *_: None)
    monkeypatch.setattr(builder_module, "verify_release_package", lambda *_: {})
    package = tmp_path / "adversarial-package"
    build_release_package(repository, package)

    with pytest.raises(ValueError, match="results/SHA256SUMS coverage mismatch"):
        verify_release_package(package)


def test_verifier_rejects_reviewer_zip_mode_change(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    package = tmp_path / "release"
    build_release_package(repository, package)
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    reviewer = manifest["artifacts"]["reviewer_source_zip"]
    zip_path = package / reviewer["path"]
    rewritten = zip_path.with_suffix(".rewritten")
    target = f"{reviewer['archive_root']}run.sh"
    with zipfile.ZipFile(zip_path) as source, zipfile.ZipFile(rewritten, "w") as output:
        output.comment = source.comment
        for info in source.infolist():
            contents = source.read(info)
            rewritten_info = zipfile.ZipInfo(info.filename, info.date_time)
            rewritten_info.compress_type = info.compress_type
            rewritten_info.comment = info.comment
            rewritten_info.extra = info.extra
            rewritten_info.create_system = 3
            if info.is_dir():
                rewritten_info.external_attr = 0o40755 << 16
            elif info.filename == target:
                rewritten_info.external_attr = 0o100644 << 16
            else:
                rewritten_info.external_attr = 0o100644 << 16
            output.writestr(rewritten_info, contents)
    rewritten.replace(zip_path)
    _refresh_artifact_fixity(package, "reviewer_source_zip")

    with pytest.raises(ValueError, match="reviewer ZIP mode mismatch"):
        verify_release_package(package)


def test_release_tag_is_normalized_bundled_and_bound_to_head(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _run(["git", "tag", "-a", "release-v1", "-m", "release"], cwd=repository)
    package = tmp_path / "tagged-release"

    build_release_package(repository, package, release_tag="release-v1")

    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["release"]["release_tag"] == "refs/tags/release-v1"
    assert "refs/tags/release-v1" in manifest["artifacts"]["release_git_bundle"][
        "bundle_refs"
    ]
    assert verify_release_package(package)["status"] == "ok"

    (repository / "README.md").write_text("later commit\n", encoding="utf-8")
    _commit(repository, "later commit")
    with pytest.raises(ValueError, match="release tag .* is not HEAD"):
        build_release_package(
            repository,
            tmp_path / "stale-tag",
            release_tag="refs/tags/release-v1",
        )


def test_zip_and_bundle_are_deterministic_across_environment_and_git_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repository)
    short_commit = commit[:12]
    first = tmp_path / "first"
    second = tmp_path / "second"

    _run(["git", "config", "core.compression", "1"], cwd=repository)
    _run(["git", "config", "pack.threads", "8"], cwd=repository)
    monkeypatch.setenv("TZ", "Pacific/Auckland")
    build_release_package(repository, first)

    _run(["git", "config", "core.compression", "9"], cwd=repository)
    _run(["git", "config", "pack.threads", "1"], cwd=repository)
    monkeypatch.setenv("TZ", "America/New_York")
    build_release_package(repository, second)

    reviewer = f"reviewer/ExPoConv-reviewer-{short_commit}.zip"
    bundle = f"repository/ExPoConv-git-{short_commit}.bundle"
    assert _sha256(first / reviewer) == _sha256(second / reviewer)
    assert _sha256(first / bundle) == _sha256(second / bundle)


def test_rejects_undeclared_legacy_inventory(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _legacy_bundle(repository, tmp_path)
    package = tmp_path / "release"
    build_release_package(repository, package, legacy_v100_bundle=bundle)
    legacy_root = package / "legacy-v100-provenance"
    extra = legacy_root / "undeclared.txt"
    extra.write_text("not declared by the release schema\n", encoding="utf-8")
    internal_fixity = legacy_root / OUTER_CHECKSUM_NAME
    with internal_fixity.open("a", encoding="utf-8") as stream:
        stream.write(f"{_sha256(extra)}  {extra.name}\n")
    _rewrite_outer_checksums(package)

    with pytest.raises(ValueError, match="undeclared files"):
        verify_release_package(package)


def test_legacy_release_history_commit_must_exist_in_current_bundle(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    bundle = _legacy_bundle(repository, tmp_path)
    manifest_path = repository / "provenance/v100_accepted_history_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_history_commits"] = [
        {"family": "missing fixture", "commit": "f" * 40}
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _commit(repository, "declare missing release-history commit")

    with pytest.raises(ValueError, match="absent from current bundle"):
        build_release_package(
            repository,
            tmp_path / "release",
            legacy_v100_bundle=bundle,
        )
