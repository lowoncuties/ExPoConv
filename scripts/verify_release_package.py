#!/usr/bin/env python3
"""Verify an ExPoConv release package without trusting an extracted ZIP.

The verifier checks the release manifest, every fixity layer, the exact refs and
clone commit in the current-release Git bundle, and every file in the reviewer
ZIP against the corresponding tracked blob.  An included legacy V100 bundle is
also passed through the existing dedicated provenance verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "RELEASE_ARTIFACT_MANIFEST.json"
OUTER_CHECKSUM_NAME = "SHA256SUMS"
SCHEMA_VERSION = "1.0"
CURRENT_E19_STATUS = "technical_preflight_complete_independent_optional"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CHECKSUM_LINE_RE = re.compile(r"([0-9a-f]{64}) ([ *])(.+)")
_ARTIFACT_KEYS = {"reviewer_source_zip", "release_git_bundle"}
_OPTIONAL_ARTIFACT_KEY = "legacy_v100_provenance_bundle"
_REGULAR_FILE_MODES = {"100644", "100755"}
_SUPPORTED_E19_STATUSES = {"open", CURRENT_E19_STATUS}
_LEGACY_ALLOWED_FILES = {
    "README.md",
    "SCOPED_HYPERJAX_OVERLAP_AUDIT.md",
    "v100-accepted-provenance.bundle",
    "v100_accepted_history_manifest.json",
}
_FORBIDDEN_ROOTS = {
    "artifacts",
    "checkpoint",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "raw_checkpoints",
}
_RAW_CHECKPOINT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
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
        if path in entries:
            raise ValueError(f"duplicate tracked path: {path!r}")
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
    prefix = f"{tree}/"
    expected = {
        path.removeprefix(prefix): object_id
        for path, (mode, object_type, object_id) in entries.items()
        if path != manifest_path and mode in _REGULAR_FILE_MODES and object_type == "blob"
    }
    missing = sorted(set(expected) - set(checksums))
    unlisted = sorted(set(checksums) - set(expected))
    if missing or unlisted:
        raise ValueError(
            f"{manifest_path} coverage mismatch: missing={missing}, unlisted={unlisted}"
        )
    for relative_path, object_id in expected.items():
        contents = _run_bytes(["git", "cat-file", "blob", object_id], cwd=repository)
        actual = hashlib.sha256(contents).hexdigest()
        declared = checksums[relative_path]
        if actual != declared:
            raise ValueError(
                f"{manifest_path} SHA-256 mismatch for {relative_path!r}: "
                f"{actual} != {declared}"
            )


def _verify_release_history(repository: Path, release_ref: str) -> int:
    commits = _run(["git", "rev-list", release_ref], cwd=repository).splitlines()
    if not commits:
        raise ValueError(f"release ref has no reachable commits: {release_ref}")
    violations: list[str] = []
    for commit in commits:
        for path, (mode, object_type, _) in _tree_entries(repository, commit).items():
            if mode not in _REGULAR_FILE_MODES or object_type != "blob":
                violations.append(f"{commit} {path!r}: mode={mode}, type={object_type}")
                continue
            try:
                _validate_tracked_paths({path})
            except ValueError as error:
                violations.append(f"{commit} {path!r}: {error}")
    if violations:
        detail = "; ".join(violations[:20])
        remaining = len(violations) - min(len(violations), 20)
        suffix = f"; plus {remaining} more" if remaining else ""
        raise ValueError(f"release history contains forbidden entries: {detail}{suffix}")
    return len(commits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if "\\" in value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{field} is not a portable relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ValueError(f"{field} is not a normalized relative path: {value!r}")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field} is not a normalized relative path: {value!r}")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read release manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("release manifest must be a JSON object")

    required = {
        "schema_version",
        "package_id",
        "created_utc",
        "visibility",
        "release",
        "artifacts",
        "fixity",
        "scope_limits",
    }
    if set(value) != required:
        missing = sorted(required.difference(value))
        unexpected = sorted(set(value).difference(required))
        raise ValueError(
            "release manifest fields differ from schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported release manifest version: {value['schema_version']!r}"
        )
    if not isinstance(value["created_utc"], str):
        raise ValueError("created_utc must be an ISO-8601 UTC string")
    try:
        created = datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_utc must be an ISO-8601 UTC string") from error
    if not value["created_utc"].endswith("Z") or created.utcoffset() is None:
        raise ValueError("created_utc must be an ISO-8601 UTC string")
    if not isinstance(value["visibility"], str) or not value["visibility"]:
        raise ValueError("visibility must be a non-empty string")
    if not isinstance(value["scope_limits"], list) or not value["scope_limits"]:
        raise ValueError("scope_limits must be a non-empty list")
    if not all(isinstance(row, str) and row for row in value["scope_limits"]):
        raise ValueError("every scope limit must be a non-empty string")
    return value


def _validate_release(manifest: dict[str, Any]) -> dict[str, Any]:
    release = manifest["release"]
    if not isinstance(release, dict):
        raise ValueError("release must be an object")
    required = {
        "commit",
        "e19_status",
        "external_archive_identifier",
        "ref",
        "release_tag",
        "repository_name",
        "tracked_file_count",
        "tree",
    }
    if set(release) != required:
        raise ValueError("release fields differ from schema")
    commit = release["commit"]
    tree = release["tree"]
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("release.commit must be a full lowercase SHA-1")
    if not isinstance(tree, str) or _COMMIT_RE.fullmatch(tree) is None:
        raise ValueError("release.tree must be a full lowercase SHA-1")
    release_ref = release["ref"]
    if (
        not isinstance(release_ref, str)
        or not release_ref.startswith("refs/heads/")
        or any(character.isspace() for character in release_ref)
    ):
        raise ValueError("release.ref must be a full refs/heads name")
    count = release["tracked_file_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("release.tracked_file_count must be a positive integer")
    if release["repository_name"] != "ExPoConv":
        raise ValueError("release.repository_name must be 'ExPoConv'")
    if release["e19_status"] not in _SUPPORTED_E19_STATUSES:
        raise ValueError("release.e19_status must be a supported reproduction status")
    release_tag = release["release_tag"]
    if release_tag is not None and (
        not isinstance(release_tag, str)
        or not release_tag.startswith("refs/tags/")
        or any(character.isspace() for character in release_tag)
    ):
        raise ValueError("release.release_tag must be null or a full refs/tags name")
    if release["external_archive_identifier"] is not None:
        raise ValueError(
            "release.external_archive_identifier must be null for this package schema"
        )
    expected_id = f"ExPoConv-release-{commit[:12]}"
    if manifest["package_id"] != expected_id:
        raise ValueError(
            f"package_id mismatch: {manifest['package_id']!r} != {expected_id!r}"
        )
    return release


def _validate_size_and_digest(artifact: dict[str, Any], *, name: str) -> None:
    size = artifact["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"invalid size for artifact {name!r}")
    digest = artifact["sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"invalid SHA-256 for artifact {name!r}")
    if not isinstance(artifact["role"], str) or not artifact["role"]:
        raise ValueError(f"invalid role for artifact {name!r}")


def _validate_artifacts(
    manifest: dict[str, Any], release: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    keys = set(artifacts)
    if not _ARTIFACT_KEYS.issubset(keys) or not keys.issubset(
        _ARTIFACT_KEYS | {_OPTIONAL_ARTIFACT_KEY}
    ):
        raise ValueError(f"invalid artifact key set: {sorted(keys)}")
    if not all(isinstance(row, dict) for row in artifacts.values()):
        raise ValueError("every artifact must be an object")

    commit = release["commit"]
    short_commit = commit[:12]
    reviewer = artifacts["reviewer_source_zip"]
    reviewer_fields = {
        "archive_root",
        "git_commit",
        "path",
        "role",
        "sha256",
        "size_bytes",
        "tracked_file_count",
    }
    if set(reviewer) != reviewer_fields:
        raise ValueError("reviewer_source_zip fields differ from schema")
    _validate_size_and_digest(reviewer, name="reviewer_source_zip")
    reviewer_path = _normalized_relative_path(
        reviewer["path"], field="reviewer_source_zip.path"
    )
    expected_reviewer_path = f"reviewer/ExPoConv-reviewer-{short_commit}.zip"
    if reviewer_path != expected_reviewer_path:
        raise ValueError("reviewer_source_zip path does not match the release commit")
    if reviewer["archive_root"] != f"ExPoConv-reviewer-{short_commit}/":
        raise ValueError("reviewer_source_zip archive_root mismatch")
    if reviewer["git_commit"] != commit:
        raise ValueError("reviewer_source_zip git_commit mismatch")
    if reviewer["tracked_file_count"] != release["tracked_file_count"]:
        raise ValueError("reviewer_source_zip tracked_file_count mismatch")

    bundle = artifacts["release_git_bundle"]
    bundle_fields = {
        "bundle_refs",
        "complete_history",
        "git_commit",
        "path",
        "role",
        "sha256",
        "size_bytes",
    }
    if set(bundle) != bundle_fields:
        raise ValueError("release_git_bundle fields differ from schema")
    _validate_size_and_digest(bundle, name="release_git_bundle")
    bundle_path = _normalized_relative_path(bundle["path"], field="release_git_bundle.path")
    expected_bundle_path = f"repository/ExPoConv-git-{short_commit}.bundle"
    if bundle_path != expected_bundle_path:
        raise ValueError("release_git_bundle path does not match the release commit")
    bundle_refs = bundle["bundle_refs"]
    expected_ref_names = {"HEAD", release["ref"]}
    if release["release_tag"] is not None:
        expected_ref_names.add(release["release_tag"])
    if (
        not isinstance(bundle_refs, dict)
        or set(bundle_refs) != expected_ref_names
        or bundle_refs.get("HEAD") != commit
        or bundle_refs.get(release["ref"]) != commit
        or any(
            not isinstance(object_id, str) or _COMMIT_RE.fullmatch(object_id) is None
            for object_id in bundle_refs.values()
        )
    ):
        raise ValueError("release_git_bundle bundle_refs mismatch")
    if bundle["git_commit"] != commit or bundle["complete_history"] is not True:
        raise ValueError("release_git_bundle release identity mismatch")

    legacy = artifacts.get(_OPTIONAL_ARTIFACT_KEY)
    if legacy is not None:
        legacy_fields = {
            "accepted_commit_count",
            "bundle_ref",
            "bundle_tip",
            "internal_fixity",
            "internal_manifest",
            "path",
            "role",
            "sha256",
            "size_bytes",
        }
        if set(legacy) != legacy_fields:
            raise ValueError("legacy_v100_provenance_bundle fields differ from schema")
        _validate_size_and_digest(legacy, name=_OPTIONAL_ARTIFACT_KEY)
        if legacy["path"] != ("legacy-v100-provenance/v100-accepted-provenance.bundle"):
            raise ValueError("legacy V100 bundle path mismatch")
        if legacy["internal_fixity"] != "legacy-v100-provenance/SHA256SUMS":
            raise ValueError("legacy V100 internal_fixity path mismatch")
        if legacy["internal_manifest"] != (
            "legacy-v100-provenance/v100_accepted_history_manifest.json"
        ):
            raise ValueError("legacy V100 internal_manifest path mismatch")
        if (
            not isinstance(legacy["bundle_tip"], str)
            or _COMMIT_RE.fullmatch(legacy["bundle_tip"]) is None
        ):
            raise ValueError("legacy V100 bundle_tip must be a full SHA-1")
        if not isinstance(legacy["bundle_ref"], str) or not legacy["bundle_ref"].startswith(
            "refs/heads/"
        ):
            raise ValueError("legacy V100 bundle_ref must be a branch ref")
        count = legacy["accepted_commit_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("legacy accepted_commit_count must be positive")
    return artifacts


def _validate_fixity(manifest: dict[str, Any], release: dict[str, Any]) -> dict[str, str]:
    fixity = manifest["fixity"]
    required = {
        "package_checksums",
        "repository_bundle_sidecar",
        "reviewer_sidecar",
    }
    if not isinstance(fixity, dict) or set(fixity) != required:
        raise ValueError("fixity fields differ from schema")
    short_commit = release["commit"][:12]
    expected = {
        "package_checksums": OUTER_CHECKSUM_NAME,
        "repository_bundle_sidecar": (
            f"repository/ExPoConv-git-{short_commit}.bundle.sha256"
        ),
        "reviewer_sidecar": f"reviewer/ExPoConv-reviewer-{short_commit}.zip.sha256",
    }
    if fixity != expected:
        raise ValueError("fixity paths do not match the release commit")
    return fixity


def _parse_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read checksum file {path}: {error}") from error
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid checksum line {line_number} in {path.name}")
        digest, _, raw_name = match.groups()
        name = raw_name.removeprefix("./")
        name = _normalized_relative_path(name, field=f"{path.name} line {line_number}")
        if name in checksums:
            raise ValueError(f"duplicate checksum path in {path.name}: {name!r}")
        checksums[name] = digest
    return checksums


def _package_files(package_directory: Path) -> set[str]:
    files: set[str] = set()
    for path in package_directory.rglob("*"):
        relative = path.relative_to(package_directory).as_posix()
        if path.is_symlink():
            raise ValueError(f"release package contains a symbolic link: {relative}")
        if path.is_file():
            files.add(relative)
        elif not path.is_dir():
            raise ValueError(f"release package contains a non-regular entry: {relative}")
    return files


def _verify_checksum_map(root: Path, checksums: dict[str, str]) -> None:
    for name, expected_digest in checksums.items():
        actual_digest = _sha256(root / name)
        if actual_digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {name!r}: {actual_digest} != {expected_digest}"
            )


def _verify_legacy_internal_fixity(
    package_directory: Path, legacy: dict[str, Any] | None
) -> set[str]:
    if legacy is None:
        return set()
    fixity_path = package_directory / legacy["internal_fixity"]
    checksums = _parse_checksums(fixity_path)
    legacy_root = fixity_path.parent
    actual = {
        path.relative_to(legacy_root).as_posix()
        for path in legacy_root.rglob("*")
        if path.is_file() and path != fixity_path
    }
    undeclared_inventory = sorted(actual - _LEGACY_ALLOWED_FILES)
    if undeclared_inventory:
        raise ValueError(
            f"legacy provenance inventory contains undeclared files: {undeclared_inventory}"
        )
    if set(checksums) != actual:
        missing = sorted(actual - set(checksums))
        unexpected = sorted(set(checksums) - actual)
        raise ValueError(
            f"legacy internal checksum coverage mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    _verify_checksum_map(legacy_root, checksums)
    required = {
        Path(legacy["path"]).name,
        Path(legacy["internal_manifest"]).name,
    }
    if not required.issubset(checksums):
        raise ValueError("legacy internal checksums omit the bundle or manifest")
    return {
        f"{legacy_root.relative_to(package_directory).as_posix()}/{name}"
        for name in checksums
    } | {legacy["internal_fixity"]}


def _verify_sidecar(
    package_directory: Path, *, sidecar_name: str, target_name: str
) -> None:
    sidecar_path = package_directory / sidecar_name
    checksums = _parse_checksums(sidecar_path)
    expected_relative = Path(target_name).name
    if checksums != {expected_relative: _sha256(package_directory / target_name)}:
        raise ValueError(f"sidecar checksum mismatch: {sidecar_name}")


def _verify_outer_fixity(
    package_directory: Path,
    artifacts: dict[str, dict[str, Any]],
    fixity: dict[str, str],
    legacy_files: set[str],
) -> None:
    expected_files = {
        MANIFEST_NAME,
        OUTER_CHECKSUM_NAME,
        fixity["repository_bundle_sidecar"],
        fixity["reviewer_sidecar"],
        artifacts["reviewer_source_zip"]["path"],
        artifacts["release_git_bundle"]["path"],
    } | legacy_files
    actual_files = _package_files(package_directory)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            f"release package file coverage mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    checksums = _parse_checksums(package_directory / OUTER_CHECKSUM_NAME)
    if set(checksums) != actual_files - {OUTER_CHECKSUM_NAME}:
        missing = sorted((actual_files - {OUTER_CHECKSUM_NAME}) - set(checksums))
        unexpected = sorted(set(checksums) - actual_files)
        raise ValueError(
            f"outer checksum coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    _verify_checksum_map(package_directory, checksums)
    for name, artifact in artifacts.items():
        path = package_directory / artifact["path"]
        if path.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {name}")
        if checksums[artifact["path"]] != artifact["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {name}")
    _verify_sidecar(
        package_directory,
        sidecar_name=fixity["reviewer_sidecar"],
        target_name=artifacts["reviewer_source_zip"]["path"],
    )
    _verify_sidecar(
        package_directory,
        sidecar_name=fixity["repository_bundle_sidecar"],
        target_name=artifacts["release_git_bundle"]["path"],
    )


def _bundle_heads(bundle_path: Path) -> list[tuple[str, str]]:
    output = _run(["git", "bundle", "list-heads", str(bundle_path)])
    heads: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid bundle head line: {line!r}")
        heads.append((parts[0], parts[1]))
    return heads


def _clone_release_bundle(
    bundle_path: Path,
    *,
    release: dict[str, Any],
    artifact: dict[str, Any],
    destination: Path,
) -> None:
    expected_heads = {(commit, ref) for ref, commit in artifact["bundle_refs"].items()}
    heads = _bundle_heads(bundle_path)
    if len(heads) != len(expected_heads) or set(heads) != expected_heads:
        raise ValueError(f"unexpected release bundle heads: {heads!r}")
    _run(["git", "clone", "--quiet", str(bundle_path), str(destination)])
    observed_commit = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    if observed_commit != release["commit"]:
        raise ValueError(
            f"cloned release commit mismatch: {observed_commit} != {release['commit']}"
        )
    observed_tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=destination)
    if observed_tree != release["tree"]:
        raise ValueError(
            f"cloned release tree mismatch: {observed_tree} != {release['tree']}"
        )
    release_tag = release["release_tag"]
    if release_tag is not None:
        observed_tag_object = _run(
            ["git", "rev-parse", "--verify", release_tag], cwd=destination
        )
        if observed_tag_object != artifact["bundle_refs"][release_tag]:
            raise ValueError("cloned release tag object mismatch")
        observed_tag_commit = _run(
            ["git", "rev-parse", "--verify", f"{release_tag}^{{commit}}"],
            cwd=destination,
        )
        if observed_tag_commit != release["commit"]:
            raise ValueError("cloned release tag does not resolve to the release commit")


def _tracked_blobs(repository: Path, commit: str) -> dict[str, tuple[str, bytes]]:
    blobs: dict[str, tuple[str, bytes]] = {}
    for path, (mode, object_type, object_id) in _tree_entries(repository, commit).items():
        if object_type != "blob" or mode not in _REGULAR_FILE_MODES:
            raise ValueError(
                f"release commit contains unsupported tracked entry {path!r} "
                f"with mode/type {mode}/{object_type}"
            )
        blobs[path] = (
            mode,
            _run_bytes(["git", "cat-file", "blob", object_id], cwd=repository),
        )
    return blobs


def _validate_tracked_paths(paths: set[str]) -> None:
    for value in paths:
        path = PurePosixPath(value)
        lowered_parts = tuple(part.lower() for part in path.parts)
        if ".git" in lowered_parts:
            raise ValueError(f"release commit contains forbidden .git path: {value}")
        if lowered_parts and lowered_parts[0] in _FORBIDDEN_ROOTS:
            raise ValueError(f"release commit contains forbidden data path: {value}")
        lowered = value.lower()
        if lowered.endswith(".bundle") or lowered.endswith(
            (".tar", ".tar.gz", ".tgz", ".zip")
        ):
            raise ValueError(f"release commit contains a packaged artifact: {value}")
        if path.suffix.lower() in _RAW_CHECKPOINT_SUFFIXES:
            raise ValueError(f"release commit contains a raw checkpoint: {value}")


def _expected_directories(paths: set[str], prefix: str) -> set[str]:
    directories = {prefix}
    for value in paths:
        parent = PurePosixPath(value).parent
        while parent != PurePosixPath("."):
            directories.add(f"{prefix}{parent.as_posix()}/")
            parent = parent.parent
    return directories


def _zip_file_mode(info: zipfile.ZipInfo) -> str:
    raw_mode = (info.external_attr >> 16) & 0o177777
    if raw_mode == 0 and info.create_system == 0:
        return "100644"
    if raw_mode in {0o644, 0o100644}:
        return "100644"
    if raw_mode in {0o755, 0o100755}:
        return "100755"
    raise ValueError(
        f"reviewer ZIP entry has unsupported Unix mode {oct(raw_mode)}: {info.filename!r}"
    )


def _verify_reviewer_zip(
    zip_path: Path,
    *,
    repository: Path,
    release: dict[str, Any],
    artifact: dict[str, Any],
) -> int:
    blobs = _tracked_blobs(repository, release["commit"])
    _validate_tracked_paths(set(blobs))
    observed_files: dict[str, tuple[str, bytes]] = {}
    observed_directories: set[str] = set()
    observed_names: set[str] = set()
    prefix = artifact["archive_root"]
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.filename in observed_names:
                    raise ValueError(f"duplicate reviewer ZIP entry: {info.filename!r}")
                observed_names.add(info.filename)
                if not info.filename.startswith(prefix):
                    raise ValueError(
                        f"reviewer ZIP entry is outside {prefix!r}: {info.filename!r}"
                    )
                if info.is_dir():
                    observed_directories.add(info.filename)
                    continue
                relative = info.filename.removeprefix(prefix)
                _normalized_relative_path(relative, field="reviewer ZIP entry")
                if info.flag_bits & 0x1:
                    raise ValueError(f"encrypted reviewer ZIP entry: {info.filename!r}")
                observed_files[relative] = (_zip_file_mode(info), archive.read(info))
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError(f"cannot read reviewer ZIP: {error}") from error

    if set(observed_files) != set(blobs):
        missing = sorted(set(blobs) - set(observed_files))
        unexpected = sorted(set(observed_files) - set(blobs))
        raise ValueError(
            f"reviewer ZIP tracked-file coverage mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    expected_directories = _expected_directories(set(blobs), prefix)
    if observed_directories != expected_directories:
        missing = sorted(expected_directories - observed_directories)
        unexpected = sorted(observed_directories - expected_directories)
        raise ValueError(
            f"reviewer ZIP directory coverage mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for path, (expected_mode, expected_bytes) in blobs.items():
        observed_mode, observed_bytes = observed_files[path]
        if observed_mode != expected_mode:
            raise ValueError(
                f"reviewer ZIP mode mismatch for {path!r}: "
                f"{observed_mode} != {expected_mode}"
            )
        if observed_bytes != expected_bytes:
            raise ValueError(f"reviewer ZIP content mismatch for {path!r}")
    if len(blobs) != artifact["tracked_file_count"]:
        raise ValueError("reviewer ZIP tracked_file_count does not match the commit")
    return len(blobs)


def _verify_legacy_bundle(
    checkout: Path,
    package_directory: Path,
    artifact: dict[str, Any],
) -> None:
    try:
        if __package__:
            from .verify_provenance_bundle import resolve_recorded_commit, verify_bundle
        else:
            from verify_provenance_bundle import resolve_recorded_commit, verify_bundle
    except ImportError as error:
        raise ValueError("legacy provenance verifier is unavailable") from error
    internal_manifest = package_directory / artifact["internal_manifest"]
    tracked_manifest = checkout / "provenance/v100_accepted_history_manifest.json"
    if internal_manifest.is_symlink() or not internal_manifest.is_file():
        raise ValueError("release package lacks a regular legacy V100 manifest")
    if tracked_manifest.is_symlink() or not tracked_manifest.is_file():
        raise ValueError("release commit lacks a regular legacy V100 manifest")
    if internal_manifest.read_bytes() != tracked_manifest.read_bytes():
        raise ValueError("legacy V100 manifest differs from the release commit")
    legacy_manifest = json.loads(internal_manifest.read_text(encoding="utf-8"))
    release_history = legacy_manifest.get("release_history_commits")
    if not isinstance(release_history, list):
        raise ValueError("legacy release_history_commits must be a list")
    release_commits: list[str] = []
    for index, row in enumerate(release_history):
        if not isinstance(row, dict) or set(row) != {"family", "commit"}:
            raise ValueError(f"invalid legacy release-history row {index}")
        commit = row["commit"]
        if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
            raise ValueError(f"invalid legacy release-history commit {index}")
        release_commits.append(commit)
    if len(release_commits) != len(set(release_commits)):
        raise ValueError("legacy release_history_commits contains duplicates")
    for commit in release_commits:
        try:
            resolve_recorded_commit(checkout, commit)
        except (ValueError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"legacy release-history commit is absent from current bundle: {commit}"
            ) from error
    result = verify_bundle(internal_manifest, package_directory / artifact["path"])
    expected = (
        artifact["bundle_ref"],
        artifact["bundle_tip"],
        artifact["accepted_commit_count"],
    )
    observed = (
        result["bundle_ref"],
        result["bundle_tip"],
        result["verified_commits"],
    )
    if observed != expected:
        raise ValueError(
            f"legacy V100 artifact metadata mismatch: {observed!r} != {expected!r}"
        )


def verify_release_package(package_directory: Path) -> dict[str, Any]:
    """Verify one release-package directory and return a machine-readable result."""

    package_directory = Path(package_directory)
    if package_directory.is_symlink():
        raise ValueError("package directory must not be a symbolic link")
    package_directory = package_directory.resolve()
    if not package_directory.is_dir():
        raise ValueError(f"package directory does not exist: {package_directory}")
    manifest_path = package_directory / MANIFEST_NAME
    checksum_path = package_directory / OUTER_CHECKSUM_NAME
    if manifest_path.is_symlink() or checksum_path.is_symlink():
        raise ValueError("release manifest and outer checksum file must be regular files")
    manifest = _load_manifest(manifest_path)
    _package_files(package_directory)
    release = _validate_release(manifest)
    artifacts = _validate_artifacts(manifest, release)
    fixity = _validate_fixity(manifest, release)
    legacy = artifacts.get(_OPTIONAL_ARTIFACT_KEY)
    legacy_files = _verify_legacy_internal_fixity(package_directory, legacy)
    _verify_outer_fixity(
        package_directory,
        artifacts,
        fixity,
        legacy_files,
    )

    with tempfile.TemporaryDirectory(prefix="expoconv-release-verify-") as temporary:
        checkout = Path(temporary) / "repository"
        _clone_release_bundle(
            package_directory / artifacts["release_git_bundle"]["path"],
            release=release,
            artifact=artifacts["release_git_bundle"],
            destination=checkout,
        )
        history_commits = _verify_release_history(checkout, release["ref"])
        for evidence_tree in ("results", "splits"):
            _verify_evidence_tree(checkout, release["commit"], evidence_tree)
        tracked_files = _verify_reviewer_zip(
            package_directory / artifacts["reviewer_source_zip"]["path"],
            repository=checkout,
            release=release,
            artifact=artifacts["reviewer_source_zip"],
        )
        if legacy is not None:
            _verify_legacy_bundle(checkout, package_directory, legacy)

    return {
        "status": "ok",
        "package": str(package_directory),
        "release_commit": release["commit"],
        "artifacts_verified": len(artifacts),
        "history_commits_verified": history_commits,
        "tracked_files_verified": tracked_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an ExPoConv release package, its fixity files, reviewer ZIP, "
            "and cloneable current-release Git bundle."
        )
    )
    parser.add_argument(
        "package_directory",
        type=Path,
        help="directory containing RELEASE_ARTIFACT_MANIFEST.json",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = verify_release_package(arguments.package_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
