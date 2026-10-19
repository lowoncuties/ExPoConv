#!/usr/bin/env python3
"""Build a fail-closed ExPoConv release package from the exact clean HEAD.

The package contains a deterministic tracked-file reviewer ZIP, a cloneable
current-release Git bundle, a machine-readable artifact manifest, and outer
SHA-256 checksums.  A separately supplied legacy V100 bundle is included only
when ``--include-legacy-v100-bundle`` is given and its existing provenance
manifest verifier succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from .verify_provenance_bundle import verify_bundle as verify_legacy_bundle
    from .verify_release_package import (
        CURRENT_E19_STATUS,
        MANIFEST_NAME,
        OUTER_CHECKSUM_NAME,
        SCHEMA_VERSION,
        _validate_tracked_paths,
        _verify_release_history,
        verify_release_package,
    )
else:
    from verify_provenance_bundle import verify_bundle as verify_legacy_bundle
    from verify_release_package import (
        CURRENT_E19_STATUS,
        MANIFEST_NAME,
        OUTER_CHECKSUM_NAME,
        SCHEMA_VERSION,
        _validate_tracked_paths,
        _verify_release_history,
        verify_release_package,
    )

_REGULAR_FILE_MODES = {"100644", "100755"}


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def _run_bytes(command: Sequence[str], *, cwd: Path | None = None) -> bytes:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _deterministic_git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C", "TZ": "UTC0"})
    return environment


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_root(repository: Path) -> Path:
    repository = repository.resolve()
    if not repository.is_dir():
        raise ValueError(f"repository directory does not exist: {repository}")
    try:
        root = Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=repository)).resolve()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"not a Git worktree: {repository}") from error
    return root


def _require_clean_worktree(repository: Path) -> None:
    status = _run_bytes(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository
    )
    if status:
        summary = status.decode("utf-8", errors="replace").splitlines()[:5]
        raise ValueError(f"Git worktree is dirty: {summary}")


def _resolve_commit(repository: Path, ref: str) -> str:
    if not ref or ref.startswith("-") or any(character.isspace() for character in ref):
        raise ValueError(f"invalid Git ref: {ref!r}")
    try:
        return _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repository)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"cannot resolve Git ref: {ref!r}") from error


def _release_identity(repository: Path, ref: str) -> tuple[str, str]:
    head = _resolve_commit(repository, "HEAD")
    try:
        release_ref = _run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repository)
    except subprocess.CalledProcessError as error:
        raise ValueError("release HEAD must be attached to a local branch") from error
    if not release_ref.startswith("refs/heads/"):
        raise ValueError(f"release HEAD is not a local branch: {release_ref!r}")
    allowed_refs = {"HEAD", release_ref, release_ref.removeprefix("refs/heads/")}
    if ref not in allowed_refs:
        raise ValueError(
            f"release ref must be HEAD or the current branch {release_ref!r}: {ref!r}"
        )
    requested = _resolve_commit(repository, ref)
    if requested != head:
        raise ValueError(f"release ref {ref!r} is not HEAD: {requested} != {head}")
    return head, release_ref


def _release_tag_identity(
    repository: Path, release_tag: str | None, release_commit: str
) -> tuple[str | None, str | None]:
    if release_tag is None:
        return None, None
    if not release_tag or any(character.isspace() for character in release_tag):
        raise ValueError(f"invalid release tag: {release_tag!r}")
    if release_tag.startswith("refs/") and not release_tag.startswith("refs/tags/"):
        raise ValueError(f"release tag must name refs/tags: {release_tag!r}")
    full_ref = (
        release_tag if release_tag.startswith("refs/tags/") else f"refs/tags/{release_tag}"
    )
    try:
        _run(["git", "check-ref-format", full_ref], cwd=repository)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"invalid release tag: {release_tag!r}") from error
    tag_commit = _resolve_commit(repository, full_ref)
    if tag_commit != release_commit:
        raise ValueError(
            f"release tag {full_ref!r} is not HEAD: {tag_commit} != {release_commit}"
        )
    tag_object = _run(["git", "rev-parse", "--verify", full_ref], cwd=repository)
    return full_ref, tag_object


def _tree_entries(
    repository: Path, commit: str, tree: str | None = None
) -> dict[str, tuple[str, str, str]]:
    command = ["git", "ls-tree", "-r", "-z", commit]
    if tree is not None:
        command.extend(["--", tree])
    output = _run_bytes(command, cwd=repository)
    entries: dict[str, tuple[str, str, str]] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        header, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise ValueError("invalid git ls-tree record")
        fields = header.decode("ascii").split()
        if len(fields) != 3:
            raise ValueError("invalid git ls-tree header")
        mode, object_type, object_id = fields
        path = raw_path.decode("utf-8")
        entries[path] = (mode, object_type, object_id)
    return entries


def _normalized_evidence_path(raw_path: str, *, manifest: str, line: int) -> str:
    path = raw_path.removeprefix("./")
    normalized = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or "\x00" in path
        or "\n" in path
        or "\r" in path
        or normalized.is_absolute()
        or path != normalized.as_posix()
        or ".." in normalized.parts
        or any(part in {"", "."} for part in normalized.parts)
    ):
        raise ValueError(
            f"invalid evidence path in {manifest} at line {line}: {raw_path!r}"
        )
    return path


def _parse_evidence_checksums(contents: bytes, *, manifest: str) -> dict[str, str]:
    try:
        lines = contents.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{manifest} is not UTF-8") from error
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise ValueError(f"invalid checksum line in {manifest} at line {line_number}")
        digest = line[:64]
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA-256 in {manifest} at line {line_number}")
        path = _normalized_evidence_path(line[66:], manifest=manifest, line=line_number)
        if path in checksums:
            raise ValueError(f"duplicate checksum path in {manifest}: {path!r}")
        checksums[path] = digest
    return checksums


def _verify_evidence_tree(repository: Path, commit: str, tree: str) -> None:
    manifest_path = f"{tree}/SHA256SUMS"
    entries = _tree_entries(repository, commit, tree)
    manifest_entry = entries.get(manifest_path)
    if manifest_entry is None or manifest_entry[0] not in _REGULAR_FILE_MODES:
        raise ValueError(f"missing regular tracked checksum manifest: {manifest_path}")
    manifest_bytes = _run_bytes(
        ["git", "cat-file", "blob", manifest_entry[2]], cwd=repository
    )
    checksums = _parse_evidence_checksums(manifest_bytes, manifest=manifest_path)

    expected: dict[str, str] = {}
    prefix = f"{tree}/"
    for path, (mode, object_type, object_id) in entries.items():
        if path == manifest_path or mode not in _REGULAR_FILE_MODES:
            continue
        if object_type != "blob":
            raise ValueError(f"non-blob regular entry under {tree}: {path}")
        expected[path.removeprefix(prefix)] = object_id

    missing = sorted(set(expected) - set(checksums))
    unlisted = sorted(set(checksums) - set(expected))
    if missing or unlisted:
        raise ValueError(
            f"{manifest_path} coverage mismatch: missing={missing}, unlisted={unlisted}"
        )
    for relative_path, object_id in expected.items():
        contents = _run_bytes(["git", "cat-file", "blob", object_id], cwd=repository)
        actual = _sha256_bytes(contents)
        declared = checksums[relative_path]
        if actual != declared:
            raise ValueError(
                f"{manifest_path} SHA-256 mismatch for {relative_path!r}: "
                f"{actual} != {declared}"
            )


def _artifact(path: Path, *, package_root: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(package_root).as_posix(),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checksums(
    path: Path,
    *,
    root: Path,
    names: set[str],
    dot_prefix: bool = False,
) -> None:
    prefix = "./" if dot_prefix else ""
    lines = [f"{_sha256_file(root / name)}  {prefix}{name}\n" for name in sorted(names)]
    path.write_text("".join(lines), encoding="utf-8")


def _write_sidecar(path: Path, target: Path) -> None:
    path.write_text(f"{_sha256_file(target)}  {target.name}\n", encoding="utf-8")


def _created_utc() -> str:
    created = datetime.now(UTC).replace(microsecond=0)
    return created.isoformat().replace("+00:00", "Z")


def _recheck_source(
    repository: Path,
    *,
    release_commit: str,
    release_ref: str,
    release_tag: str | None,
    release_tag_object: str | None,
) -> None:
    observed_commit, observed_ref = _release_identity(repository, "HEAD")
    if observed_commit != release_commit or observed_ref != release_ref:
        raise ValueError("repository HEAD changed while building the release package")
    observed_tag, observed_tag_object = _release_tag_identity(
        repository, release_tag, release_commit
    )
    if (observed_tag, observed_tag_object) != (release_tag, release_tag_object):
        raise ValueError("release tag changed while building the release package")
    _require_clean_worktree(repository)


def build_release_package(
    repository: Path,
    output_directory: Path,
    *,
    ref: str = "HEAD",
    release_tag: str | None = None,
    legacy_v100_bundle: Path | None = None,
) -> dict[str, Any]:
    """Build and self-verify a package from a clean, attached exact HEAD."""

    repository = _repository_root(Path(repository))
    _require_clean_worktree(repository)
    release_commit, release_ref = _release_identity(repository, ref)
    release_tag, release_tag_object = _release_tag_identity(
        repository, release_tag, release_commit
    )
    _verify_release_history(repository, release_ref)

    all_entries = _tree_entries(repository, release_commit)
    unsupported = sorted(
        path for path, (_, object_type, _) in all_entries.items() if object_type != "blob"
    )
    if unsupported:
        raise ValueError(
            f"release commit contains unsupported tracked entries: {unsupported}"
        )
    _validate_tracked_paths(set(all_entries))
    for evidence_tree in ("results", "splits"):
        _verify_evidence_tree(repository, release_commit, evidence_tree)
    release_tree = _run(["git", "rev-parse", f"{release_commit}^{{tree}}"], cwd=repository)
    tracked_file_count = len(all_entries)

    verified_legacy_path: Path | None = None
    legacy_manifest: dict[str, Any] | None = None
    legacy_manifest_path = repository / "provenance/v100_accepted_history_manifest.json"
    if legacy_v100_bundle is not None:
        verified_legacy_path = Path(legacy_v100_bundle).resolve()
        if not verified_legacy_path.is_file() or verified_legacy_path.is_symlink():
            raise ValueError(
                f"legacy V100 bundle must be a regular file: {verified_legacy_path}"
            )
        if legacy_manifest_path.is_symlink() or not legacy_manifest_path.is_file():
            raise ValueError("repository lacks a regular legacy V100 manifest")
        verify_legacy_bundle(legacy_manifest_path, verified_legacy_path)
        loaded_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, dict):
            raise ValueError("legacy V100 manifest must be a JSON object")
        legacy_manifest = loaded_manifest

    output_directory = Path(output_directory).resolve()
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError(f"output directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_directory.name}.staging-", dir=output_directory.parent
    ) as temporary:
        package_root = Path(temporary)
        short_commit = release_commit[:12]
        reviewer_directory = package_root / "reviewer"
        reviewer_directory.mkdir()
        reviewer_zip = reviewer_directory / f"ExPoConv-reviewer-{short_commit}.zip"
        archive_prefix = f"ExPoConv-reviewer-{short_commit}/"
        _run(
            [
                "git",
                "-c",
                "core.compression=9",
                "archive",
                "--format=zip",
                "-9",
                f"--prefix={archive_prefix}",
                f"--output={reviewer_zip}",
                release_commit,
            ],
            cwd=repository,
            env=_deterministic_git_environment(),
        )

        bundle_directory = package_root / "repository"
        bundle_directory.mkdir()
        release_bundle = bundle_directory / f"ExPoConv-git-{short_commit}.bundle"
        bundle_revisions = ["HEAD", release_ref]
        if release_tag is not None:
            bundle_revisions.append(release_tag)
        _run(
            [
                "git",
                "-c",
                "core.compression=9",
                "-c",
                "pack.compression=9",
                "-c",
                "pack.threads=1",
                "bundle",
                "create",
                str(release_bundle),
                *bundle_revisions,
            ],
            cwd=repository,
            env=_deterministic_git_environment(),
        )

        reviewer_sidecar = reviewer_zip.with_name(f"{reviewer_zip.name}.sha256")
        bundle_sidecar = release_bundle.with_name(f"{release_bundle.name}.sha256")
        _write_sidecar(reviewer_sidecar, reviewer_zip)
        _write_sidecar(bundle_sidecar, release_bundle)

        branch_name = release_ref.removeprefix("refs/heads/")
        artifacts: dict[str, dict[str, Any]] = {
            "reviewer_source_zip": {
                **_artifact(
                    reviewer_zip,
                    package_root=package_root,
                    role="Tracked-file reviewer snapshot at the release commit",
                ),
                "archive_root": archive_prefix,
                "git_commit": release_commit,
                "tracked_file_count": tracked_file_count,
            },
            "release_git_bundle": {
                **_artifact(
                    release_bundle,
                    package_root=package_root,
                    role=(
                        f"Complete ExPoConv {branch_name} history for cloning "
                        "and E19 Git identity"
                    ),
                ),
                "bundle_refs": {
                    "HEAD": release_commit,
                    release_ref: release_commit,
                },
                "complete_history": True,
                "git_commit": release_commit,
            },
        }
        if release_tag is not None and release_tag_object is not None:
            artifacts["release_git_bundle"]["bundle_refs"][release_tag] = release_tag_object
        if verified_legacy_path is not None and legacy_manifest is not None:
            accepted_commit_count = len(legacy_manifest["accepted_commits"])
            accepted_count_label = (
                "eleven" if accepted_commit_count == 11 else str(accepted_commit_count)
            )
            legacy_directory = package_root / "legacy-v100-provenance"
            legacy_directory.mkdir()
            legacy_copy = legacy_directory / verified_legacy_path.name
            shutil.copyfile(verified_legacy_path, legacy_copy)
            internal_manifest = legacy_directory / legacy_manifest_path.name
            shutil.copyfile(legacy_manifest_path, internal_manifest)
            internal_fixity = legacy_directory / OUTER_CHECKSUM_NAME
            _write_checksums(
                internal_fixity,
                root=legacy_directory,
                names={legacy_copy.name, internal_manifest.name},
            )
            artifacts["legacy_v100_provenance_bundle"] = {
                **_artifact(
                    legacy_copy,
                    package_root=package_root,
                    role=(
                        f"Exact history for {accepted_count_label} legacy accepted "
                        "V100 result revisions"
                    ),
                ),
                "accepted_commit_count": accepted_commit_count,
                "bundle_ref": legacy_manifest["bundle_ref"],
                "bundle_tip": legacy_manifest["bundle_tip"],
                "internal_fixity": internal_fixity.relative_to(package_root).as_posix(),
                "internal_manifest": internal_manifest.relative_to(package_root).as_posix(),
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "package_id": f"ExPoConv-release-{short_commit}",
            "created_utc": _created_utc(),
            "visibility": (
                "Confidential editor/reviewer transport unless the author "
                "explicitly approves public disclosure"
            ),
            "release": {
                "commit": release_commit,
                "e19_status": CURRENT_E19_STATUS,
                "external_archive_identifier": None,
                "ref": release_ref,
                "release_tag": release_tag,
                "repository_name": "ExPoConv",
                "tracked_file_count": tracked_file_count,
                "tree": release_tree,
            },
            "artifacts": artifacts,
            "fixity": {
                "package_checksums": OUTER_CHECKSUM_NAME,
                "repository_bundle_sidecar": bundle_sidecar.relative_to(
                    package_root
                ).as_posix(),
                "reviewer_sidecar": reviewer_sidecar.relative_to(package_root).as_posix(),
            },
            "scope_limits": [
                "The reviewer ZIP contains only files tracked at the release commit.",
                (
                    "The reviewer ZIP alone is insufficient for commit-bound reruns "
                    "because it has no Git object database."
                ),
                (
                    f"The ExPoConv Git bundle carries the complete {branch_name} "
                    "history needed for identity capture in an optional E19 rerun."
                ),
                (
                    "The legacy V100 bundle is a separate confidential artifact "
                    "and is not required for an optional E19 rerun."
                ),
                (
                    "Creating these artifacts does not constitute an independent-"
                    "human reproduction."
                ),
            ],
        }
        _write_json(package_root / MANIFEST_NAME, manifest)
        checksum_names = {
            path.relative_to(package_root).as_posix()
            for path in package_root.rglob("*")
            if path.is_file()
        }
        _write_checksums(
            package_root / OUTER_CHECKSUM_NAME,
            root=package_root,
            names=checksum_names,
            dot_prefix=True,
        )

        _recheck_source(
            repository,
            release_commit=release_commit,
            release_ref=release_ref,
            release_tag=release_tag,
            release_tag_object=release_tag_object,
        )
        verify_release_package(package_root)
        package_root.rename(output_directory)

    return {
        "status": "ok",
        "package": str(output_directory),
        "release_commit": release_commit,
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and self-verify an ExPoConv release package from a clean, "
            "attached exact HEAD."
        )
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Git worktree to package (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to create; it must not already exist",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help=("release ref, limited to HEAD or the checked-out branch (default: HEAD)"),
    )
    parser.add_argument(
        "--release-tag",
        help=(
            "optional tag name (short or refs/tags/...) that must already resolve "
            "to HEAD and will be included in the release bundle"
        ),
    )
    parser.add_argument(
        "--include-legacy-v100-bundle",
        type=Path,
        metavar="PATH",
        help=(
            "explicitly include this separately supplied legacy bundle after "
            "provenance/v100_accepted_history_manifest.json verifies it"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = build_release_package(
        arguments.repository,
        arguments.output_dir,
        ref=arguments.ref,
        release_tag=arguments.release_tag,
        legacy_v100_bundle=arguments.include_legacy_v100_bundle,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
