# Accepted-result development history

The release snapshot is intentionally a compact, standalone repository. Some
accepted V100 result families were measured at earlier revisions of the
standalone development repository. Their exact commits are preserved in the
separate `v100-accepted-provenance.bundle` evidence artifact described by
[`v100_accepted_history_manifest.json`](v100_accepted_history_manifest.json).

The bundle is prepared as a separate evidence artifact rather than committed
inside this source tree. This keeps the public checkout simple while allowing
an editor, reviewer, or archivist to inspect the eleven legacy result-producing
trees. E13 revision `75f10dd9` is already reachable in ExPoConv's own history.
The bundle is not required to run the release snapshot or the bounded E19
reproduction.

A release package therefore carries two different Git bundles. Clone
`repository/ExPoConv-git-<release-commit>.bundle` for the release snapshot and
E19; clone `v100-accepted-provenance.bundle` only for the legacy result history
described here. The reviewer ZIP is a tracked-file snapshot without `.git`.
It supports inspection and checksum verification, but provenance-bound
commands must run in a clone of the appropriate bundle (or released
repository), not in the extracted ZIP.

## Building the final release package

Only build a package after the author has approved and committed the intended
tree and created the immutable release tag. The builder fails if the checkout
is dirty or detached, if the tag does not resolve to `HEAD`, if committed
result or split checksums are incomplete, or if the complete branch history
contains a packaged artifact, raw checkpoint, dataset root, symlink, submodule,
or other non-regular tracked entry.

```bash
release_tag=v0.1.0
release_commit=$(git rev-parse HEAD)
release_short=$(git rev-parse --short=12 "$release_commit")

./.venv/bin/python scripts/build_release_package.py \
  --repository . \
  --release-tag "$release_tag" \
  --output-dir "/path/to/ExPoConv-release-$release_short" \
  --include-legacy-v100-bundle \
  /path/to/v100-accepted-provenance.bundle

./.venv/bin/python scripts/verify_release_package.py \
  "/path/to/ExPoConv-release-$release_short"
```

Omit `--include-legacy-v100-bundle` for a package that does not carry the
confidential legacy history. The builder uses the tracked commit rather than
the working-directory files, creates a deterministic reviewer ZIP, bundles
only `HEAD`, the current branch, and the declared release tag, writes package
and sidecar checksums, and runs the independent verifier before publishing the
output directory. The verifier clones the bundle and independently checks the
reachable history, tag and commit identities, evidence checksum manifests,
reviewer-ZIP paths, contents and executable modes, package inventory, and all
fixity layers. Creating and verifying this package does not turn the completed
same-machine E19 technical preflight into an independent-human reproduction.

The accepted E20/E21 A6000 training cohorts were produced from the clean
revision `fef30ecb5063fb8823368997bde10d16654b8b5e` under protocol 1.14.
Their registered configuration hashes are
`b226335a56314f0947c81f193621802ffa0c4aea90acc1f2d1117212de59bed8`
for E20 and
`6b7973dad0db1b2578f88987c7527a586f5fac68eda1bae3592ff96387d883b6`
for E21. The joint gate and endpoint tables were exported from clean revision
`072c9ddd23571dc3137d44ec1a9c51e4df7ce198`; the post-gate checkpoint
inventory was exported with the hardened implementation at clean revision
`8a233b94c33297ee566af8ff073e85586a8e953d`. All three commits are reachable
in this repository's history. The gate audit, exporter source hashes, output
hashes, and the retained pre-update infrastructure failure are recorded under
`results/e20_e21_training_replication/`. These training cohorts remain
separate from the earlier V100 cohorts; the repository contains no pooled or
causal accelerator comparison.

The archival `configs/numerics/e02_structured.yaml` is the byte-exact broad
E02 configuration introduced at standalone development commit
`a486aab6fb0fd296fb42273032e4741094e017db` (file SHA-256
`e1b6b624be66d1a7352b4b74925ccb4876f22714ce94cd1540e658ac83ebf60b`).
Historical E02 rows bind that file together with their contemporaneous
protocol; it is not the bounded repaired E02 configuration supporting the
manuscript claims.

The confidential evidence archive includes the checksummed
`SCOPED_HYPERJAX_OVERLAP_AUDIT.md` record. It freezes the bundle ref and tip,
the comparator's sorted-ref snapshot and distinct tip IDs, the exact commands,
the result counts, and the audit's limitations. The scoped audit found no
HyperJAX-named paths or direct Python imports and no identical complete Git
objects shared with the audited committed HyperJAX history. It is not a
source-origin or non-exposure attestation and is not enforced by the bundle
verifier. The record deliberately excludes the comparator's repository path
and remote URL.

The historical commits do contain textual HyperJAX references, author identity,
host metadata, and absolute private-workspace paths. Keep the bundle and audit
record in a confidential editor/reviewer deposit unless the author explicitly
approves those metadata for public release.

After obtaining the bundle and placing it beside the manifest, verify it with:

```bash
./.venv/bin/python scripts/verify_provenance_bundle.py \
  --manifest provenance/v100_accepted_history_manifest.json \
  --bundle /path/to/v100-accepted-provenance.bundle
```

The verifier checks the byte length and SHA-256, requires exactly the declared
bundle ref and tip, clones the bundle into a temporary directory, and confirms
that every full legacy commit in the manifest is present and is an ancestor of
the declared tip. To create a persistent worktree for inspection or a legacy
rerun, clone the verified bundle rather than copying files from it:

```bash
git clone --branch v100-accepted-provenance \
  /path/to/v100-accepted-provenance.bundle ExPoConv-v100-history
cd ExPoConv-v100-history
git checkout --detach 04aca39f7c99b9881995889ca4f50ff42d2c0425
```

That detached revision is the E07-J2 evaluator and retains protocol 1.10. The
bundle supplies Git history, not the datasets or archived E09/E10 checkpoints
and source summaries needed to rerun the diagnostic. The external archive
identifier remains an author release action until the evidence deposit is
frozen.
