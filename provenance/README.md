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
python scripts/verify_provenance_bundle.py \
  --manifest provenance/v100_accepted_history_manifest.json \
  --bundle /path/to/v100-accepted-provenance.bundle
```

The verifier checks the byte length and SHA-256, requires exactly the declared
bundle ref and tip, clones the bundle into a temporary directory, and confirms
that every full legacy commit in the manifest is present and is an ancestor of
the declared tip. The external archive identifier remains an author release
action until the evidence deposit is frozen.
