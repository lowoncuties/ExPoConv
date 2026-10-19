#!/usr/bin/env python3
"""Verify the archived exact-history bundle for accepted results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COMMIT_MAP_PATH = Path("provenance/commit_map.csv")
_COMMIT_MAP_FIELDS = ["old_commit", "new_commit", "tree"]


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_full_sha1(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def load_commit_map(repository: Path) -> dict[str, tuple[str, str]]:
    """Read ``recorded commit -> (published commit, tree)``; no map file is empty."""

    path = Path(repository) / COMMIT_MAP_PATH
    if not path.is_file():
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    with path.open(encoding="ascii", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _COMMIT_MAP_FIELDS:
            raise ValueError(f"{COMMIT_MAP_PATH} header must be old_commit,new_commit,tree")
        for row in reader:
            values = [row[field] for field in _COMMIT_MAP_FIELDS]
            if not all(_is_full_sha1(value) for value in values) or values[0] in mapping:
                raise ValueError(f"invalid {COMMIT_MAP_PATH} row: {values}")
            mapping[values[0]] = (values[1], values[2])
    return mapping


def resolve_recorded_commit(
    repository: Path,
    commit: str,
    tip: str = "HEAD",
    commit_map: Mapping[str, tuple[str, str]] | None = None,
) -> str:
    """Return the commit reachable from ``tip`` that holds a recorded commit's tree.

    A recorded commit that is an ancestor of ``tip`` resolves to itself. Otherwise
    the commit map must name an ancestor of ``tip`` whose tree is the mapped tree.
    """

    def is_ancestor(candidate: str) -> bool:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", candidate, tip],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        return result.returncode == 0

    if not _is_full_sha1(commit):
        raise ValueError(f"recorded commit is not a full lowercase SHA-1: {commit!r}")
    if is_ancestor(commit):
        return commit
    if commit_map is None:
        commit_map = load_commit_map(repository)
    if commit not in commit_map:
        raise ValueError(f"recorded commit is not reachable from {tip}: {commit}")
    published, tree = commit_map[commit]
    if not is_ancestor(published) or (
        _run(["git", "rev-parse", f"{published}^{{tree}}"], cwd=repository) != tree
    ):
        raise ValueError(
            f"mapped commit {published} for {commit} is not reachable with tree {tree}"
        )
    return published


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "artifact_name",
        "size_bytes",
        "sha256",
        "bundle_ref",
        "bundle_tip",
        "accepted_commits",
        "release_history_commits",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"manifest is missing required fields: {sorted(missing)}")
    commits = manifest["accepted_commits"]
    if not isinstance(commits, list) or not commits:
        raise ValueError("accepted_commits must be a non-empty list")
    return manifest


def verify_bundle(manifest_path: Path, bundle_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    bundle_path = bundle_path.resolve()
    manifest = _load_manifest(manifest_path)

    if bundle_path.name != manifest["artifact_name"]:
        raise ValueError(
            f"bundle filename mismatch: {bundle_path.name!r} != "
            f"{manifest['artifact_name']!r}"
        )
    actual_size = bundle_path.stat().st_size
    if actual_size != int(manifest["size_bytes"]):
        raise ValueError(f"bundle size mismatch: {actual_size} != {manifest['size_bytes']}")
    actual_sha256 = _sha256(bundle_path)
    if actual_sha256 != manifest["sha256"]:
        raise ValueError(
            f"bundle SHA-256 mismatch: {actual_sha256} != {manifest['sha256']}"
        )

    heads_output = _run(["git", "bundle", "list-heads", str(bundle_path)])
    heads = [line.split(maxsplit=1) for line in heads_output.splitlines() if line]
    expected_head = [manifest["bundle_tip"], manifest["bundle_ref"]]
    if heads != [expected_head]:
        raise ValueError(f"unexpected bundle heads: {heads!r}")

    branch = str(manifest["bundle_ref"]).removeprefix("refs/heads/")
    if not branch or branch == manifest["bundle_ref"]:
        raise ValueError("bundle_ref must name one refs/heads branch")

    commits = [str(row["commit"]) for row in manifest["accepted_commits"]]
    if len(commits) != len(set(commits)):
        raise ValueError("accepted commit list contains duplicates")
    for commit in commits:
        invalid_character = any(character not in "0123456789abcdef" for character in commit)
        if len(commit) != 40 or invalid_character:
            raise ValueError(f"accepted commit is not a full lowercase SHA-1: {commit!r}")

    with tempfile.TemporaryDirectory(prefix="expoconv-provenance-") as temporary:
        checkout = Path(temporary) / "repository"
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--branch",
                branch,
                str(bundle_path),
                str(checkout),
            ]
        )
        git_directory = checkout / ".git"
        observed_tip = _run(["git", f"--git-dir={git_directory}", "rev-parse", "HEAD"])
        if observed_tip != manifest["bundle_tip"]:
            raise ValueError(
                f"cloned bundle tip mismatch: {observed_tip} != {manifest['bundle_tip']}"
            )
        for commit in commits:
            _run(
                [
                    "git",
                    f"--git-dir={git_directory}",
                    "cat-file",
                    "-e",
                    f"{commit}^{{commit}}",
                ]
            )
            _run(
                [
                    "git",
                    f"--git-dir={git_directory}",
                    "merge-base",
                    "--is-ancestor",
                    commit,
                    str(manifest["bundle_tip"]),
                ]
            )

    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "bundle": str(bundle_path),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "bundle_ref": manifest["bundle_ref"],
        "bundle_tip": manifest["bundle_tip"],
        "verified_commits": len(commits),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the archived exact-history bundle for accepted results."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = verify_bundle(arguments.manifest, arguments.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
