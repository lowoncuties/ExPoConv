# E13 R/F U-Net scaling analysis

This is the CPU-only analysis path used for the completed, bounded E13 scaling
campaign. The exporter does not launch benchmarks. The accepted raw sources
account for all 70 planned fresh-process path/session groups: all 35 F groups
and 30 of 35 R groups completed, while all five R groups at batch 8 and
512 x 512 ended in recorded first-update OOMs. The corresponding five F groups
completed.

## Registered analysis grid

The grid in `configs/analysis/e13_unet_scaling.yaml` contains seven distinct
complete-update workloads:

- batch sizes 1, 2, 4, 8, and 16 at 256 x 256;
- resolutions 128, 256, and 512 at batch 8.

The batch-8, 256 x 256 workload is shared by both sweeps and must be measured
once. Only R and F are included. This differs from the older R/T wording for
E13 in the original experiment plan; protocol amendment 1.11 records the R/F
scope prospectively, before GPU execution. The analysis manifest pins both
the byte-exact protocol 1.11 snapshot at
`configs/protocols/protocol_1_11.yaml` and
`configs/benchmarks/e13_unet_scaling.yaml` by SHA-256. The exporter recomputes
every expected per-workload configuration hash from those two pinned documents.
At the recorded E13 code revision, the same protocol bytes were at top-level
`protocol.yaml`; `results/e13_scaling/summary.json` retains that historical
path, while the versioned snapshot permits validation after later amendments.

## Accepted raw formats and identities

`scripts/export_e13_scaling.py` accepts any combination of CSV, JSON
(`{"rows": [...]}` or a row list), and JSONL files. Pass both the benchmark
table and its failure sidecar. Records use the existing benchmark schema and
must contain at least:

- `experiment_id`, `protocol_version`, `code_commit`, `code_dirty`,
  `config_hash`, `device_id`, and `precision_policy`;
- `workload_id`, `variant`, `seed_or_session`, `status`, `shape`, and
  `operation_boundary`;
- for completed groups, `phase`, `observation`, `latency_ms`, and the measured
  peak-memory fields.

The exporter requires one clean code revision and one physical GPU UUID across
the campaign. It normalizes the two supported FP32 precision-policy encodings
and retains all literal device and precision provenance strings. Each row's
protocol version and configuration hash must match the pinned documents.
Shapes must match the analysis grid. Every planned
workload/session/path appears in `session_outcomes.csv`, including missing,
partial, failed, nonfinite, and OOM cases.

Failure rows should record `failure_kind: oom` and one of the following
`failure_phase` values when applicable: `initialization`, `compilation`,
`first_update`, `warmup_updates`, `measured_updates`, or `post_measurement`.
A generic worker failure is retained as `failed`/`unknown`; the exporter does
not guess OOM from free-form stderr. Legacy `failure_stage` is accepted as an
input alias but the compact table always uses `failure_phase`.

## Export

```bash
PYTHONPATH=src ./.venv/bin/python scripts/export_e13_scaling.py \
  --manifest configs/analysis/e13_unet_scaling.yaml \
  --raw artifacts/e13_unet_scaling.csv \
  --raw artifacts/e13_unet_scaling.csv.failures.jsonl \
  --output-dir artifacts/analysis/e13_unet_scaling
```

The exporter refuses to overwrite an existing output. It writes:

- `session_outcomes.csv`: one row for every planned session and path;
- `workload_summary.csv`: descriptive latency, peak-allocation, and outcome
  counts by workload and path;
- `paired_ratio_summary.csv`: paired R/F latency and peak-allocation ratios
  using only sessions in which both paths succeeded;
- `unet_scaling.pdf` and `.png`: a four-panel batch/resolution latency/memory
  figure; and
- `summary.json`: source hashes, validated identities, output hashes, and
  outcome counts.

The figure draws a point only when all five sessions for that path/workload
completed. OOM and other incomplete outcomes remain visible as markers at the
top of the affected panel. This prevents a capacity failure from disappearing
from the visual summary.

The accepted compact outputs are copied byte-for-byte to
`results/e13_scaling/`. The 24-MiB per-call timing table and JSONL failure
sidecar remain in the untracked artifact archive; their paths, sizes, row
counts, and SHA-256 hashes are retained in
`results/e13_scaling/summary.json`.
