# Reproducibility details

This document holds the detailed reproduction protocol, measurement
definitions, and provenance rules that are summarized in the top-level
`README.md`. Results that a command regenerates are compared with the tracked
files in `results/`; new runs write to `artifacts/` and `out/` (both ignored)
and never overwrite `results/`.

## 1. Measured platforms

| Platform | Experiments | Runtime |
|---|---|---|
| NVIDIA Tesla V100-PCIE-32GB, driver 520.61.05 | E02, E03/E05, E08, E09/E10, E11, E13, E19, E22, E23, E26, E28, E29, E30, E35 | Python 3.11.16, JAX/JAXLIB 0.4.25 (`jaxlib 0.4.25+cuda11.cudnn86`), CUDA 11.8.89, cuDNN 8.6.0.163 |
| NVIDIA RTX A6000, driver 560.35.03 | E20/E21, E24, E25, E31 | the same JAX/JAXLIB 0.4.25 with pinned CUDA-11.8 and cuDNN-8.6 user-space packages |
| NVIDIA RTX A6000, driver 560.35.03 | E27 only | JAX/JAXLIB 0.11.2 with the CUDA-12 plugin, reported separately |
| NVIDIA RTX A6000, driver 560.35.03 | E32 only | Python 3.10.21, PyTorch 2.4.0 (CUDA 12.1), HypLL 0.1.1; see `pytorch_crosscheck/README.md` |
| CPU | E33 | the locked CPU environment (JAX 0.4.25) |

All timed tensors are FP32 with the highest matrix-multiplication precision,
TF32 disabled, and JAX preallocation disabled. The E22, E24, E25, E27, E28,
and E31 runner scripts additionally set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`;
E29 and E30 keep the default fraction of E08 and E13. The
machine-readable V100 record is `results/environment.json`; the E20/E21 A6000
runtime is in `results/e20_e21_training_replication/e20_e21_gate_audit.json`;
the E24/E25, E27, E29/E30, and E31 runtime captures are named and hashed in
their audits in `results/reviewer_requested/`. A driver that advertises
CUDA 12 compatibility does not make the pinned CUDA-11.8 environment a CUDA-12
software stack. A CUDA GPU is needed only to rerun experiments; the CPU
environment is enough for the tests, table inspection, and figure regeneration.

## 2. Installation

```bash
./scripts/bootstrap.sh          # CPU environment in ./.venv from uv.lock
./.venv/bin/pytest              # CPU tests
```

`bootstrap.sh` checks `uv.lock` with `uv lock --check --offline` before
syncing. If `python3.11` is not on `PATH`, select an interpreter explicitly;
`POINCARE_EXEC_ENV_DIR` places the environment outside the checkout without
changing the locked dependencies:

```bash
POINCARE_EXEC_PYTHON=/absolute/path/to/python3.11 \
POINCARE_EXEC_ENV_DIR=/absolute/path/to/expoconv-venv \
./scripts/bootstrap.sh
```

Measured GPU platform (V100, also used for the A6000 E20/E21/E24/E25 runs):

```bash
./scripts/bootstrap_cuda11_v100.sh   # CUDA-11.8 jaxlib wheel and pinned NVIDIA libraries
CUDA_VISIBLE_DEVICES=0 ./scripts/with_cuda_11_8.sh ./scripts/check_cuda.sh
```

`scripts/with_cuda_11_8.sh` puts the pip-installed CUDA 11.8 libraries on the
library path and disables preallocation. It appends the system CUDA 11.8
library directory, `/usr/local/cuda-11.8/targets/x86_64-linux/lib` unless
`EXPOCONV_CUDA_TOOLKIT_LIB` names another one. Every runner requires exactly one
physical index in `CUDA_VISIBLE_DEVICES` and runs its jobs sequentially on
that device.

E27 used a separate Python environment with JAX/JAXLIB 0.11.2 and the CUDA-12
plugin; `scripts/run_e27.sh` expects it at `.venv-jax0112-cuda12/` (ignored by
Git) or at `POINCARE_EXEC_ENV_DIR`. That environment is not lock-pinned in this
repository; its full package capture (`e27_environment.json`) is part of the
external raw evidence and is hashed in
`results/reviewer_requested/e24_e25_e27_audit.json`.

Use a Git clone for provenance-recording reruns: every result row records the
commit and whether the worktree was clean. A source archive or the reviewer
ZIP (which omits `.git`) is sufficient for source inspection, CPU tests,
checksum verification, and figure regeneration. The Python wheel is code-only
and is not a reproduction package.

## 3. Data

`./scripts/fetch_core_datasets.sh` downloads ISIC 2016 Part 1 and CIFAR-100
into `data/raw/` (ignored) and verifies their SHA-256 hashes. No image bytes
are redistributed. The tracked manifests in `splits/` are the authoritative
partitions (ISIC16 810/90/379, CIFAR-100 45,000/5,000/10,000). They record
membership, order, per-example input and target hashes, and byte-identical
duplicate groups. Verify and optionally reconstruct them:

```bash
(cd splits && sha256sum --check SHA256SUMS)
candidate=$(mktemp -d)
./.venv/bin/poincare-exec make-splits --data-root data/raw --output-dir "$candidate"
cmp splits/isic16.csv "$candidate/isic16.csv"
cmp splits/cifar100.csv "$candidate/cifar100.csv"
```

## 4. Running the experiments

All commands run from the repository root of a clean Git clone.

```bash
export CUDA_VISIBLE_DEVICES=0
W=./scripts/with_cuda_11_8.sh

# E02: numerical agreement
$W ./scripts/run_e02_repair.sh artifacts/numerical_repair/e02_repair
$W ./.venv/bin/poincare-exec structured-numerics --protocol protocol.yaml \
    --config configs/numerics/e02_repair_structured.yaml \
    --output artifacts/numerical_repair/e02_repair_structured.jsonl

# E03/E05, E08, E11, E13: operator, complete U-Net, profile, and scaling
$W ./scripts/run_corrected_operator_anchors.sh
$W ./scripts/run_e08.sh
$W ./scripts/run_e11.sh
$W ./scripts/run_e13.sh

# E22/E23: allocator sensitivity and batch-one workspace diagnostic (V100)
$W ./scripts/run_e22.sh
$W ./scripts/run_e23.sh default
$W ./scripts/run_e23.sh autotune0

# E26/E28: elementwise replay and three fresh-process replays (V100)
$W ./.venv/bin/poincare-exec structured-numerics --protocol protocol.yaml \
    --config configs/numerics/e26_elementwise_agreement.yaml \
    --output artifacts/e26_elementwise_agreement.jsonl
$W ./scripts/run_e28.sh

# E29/E30: Euclidean reference and U-Net width check (V100, default allocator)
$W ./scripts/run_e29.sh
$W ./scripts/run_e30.sh

# E09/E10: five matched seeds per path (R/F/T run sequentially)
$W ./scripts/run_training_campaign.sh configs/training/e09_isic16_unet.yaml
$W ./scripts/run_training_campaign.sh configs/training/e10_cifar100_resnet20.yaml

# E20/E21: 15 matched seeds per path (R/F); official-test endpoints are read only
# after all 60 runs pass the checks registered in docs/e20_e21_analysis_contract.md
$W ./scripts/run_training_campaign.sh configs/training/e20_isic16_unet_rf_replication.yaml
$W ./scripts/run_training_campaign.sh configs/training/e21_cifar100_resnet20_rf_replication.yaml

# E06: checkpoint replay of fresh E09 checkpoints
$W ./scripts/run_trained_checkpoint_campaign.sh configs/training/e09_isic16_unet.yaml
```

The A6000 controls E24/E25 use the pinned environment; E27 uses its separate
environment. The E24/E25/E27 exporter expects a runtime check
(`*_runtime_gate.json`) and an environment capture beside each raw CSV:

```bash
export CUDA_VISIBLE_DEVICES=1
$W ./.venv/bin/python scripts/check_revision_runtime.py --expected-jax 0.4.25 \
    --expected-jaxlib 0.4.25 --expected-device-substring "RTX A6000" \
    --output artifacts/e24_e25_runtime_gate.json
$W ./.venv/bin/poincare-exec capture-environment --output artifacts/e24_e25_environment.json
$W ./scripts/run_e24.sh
$W ./scripts/run_e25.sh
$W ./scripts/run_e31.sh       # E31 needs e31_runtime_gate.json and e31_environment.json too
$W ./.venv/bin/python -m scripts.check_e31_gpu_equivalence --output artifacts/e31_gpu_equivalence.json
$W ./.venv/bin/python -m scripts.capture_e31_compiled_memory --output artifacts/e31_compiled_memory.json

E27=.venv-jax0112-cuda12/bin/python
$E27 scripts/check_revision_runtime.py --expected-jax 0.11.2 \
    --expected-jaxlib 0.11.2 --expected-device-substring "RTX A6000" \
    --output artifacts/e27_runtime_gate.json
PYTHONPATH=src $E27 -m poincare_exec.cli capture-environment \
    --output artifacts/e27_environment.json
./scripts/run_e27.sh
```

E32 runs in its own PyTorch environment; its steps are in
`pytorch_crosscheck/README.md`. E33 runs on CPU and writes its tables directly:

```bash
./.venv/bin/python scripts/run_oracle_accuracy.py --output-dir out/e33
./.venv/bin/python scripts/trace_branch_margins.py --output-dir out/e33
EXPOCONV_TRAINING_ROOT=/path/to/training EXPOCONV_DATA_ROOT=data/raw \
    ./.venv/bin/python scripts/measure_cap_activity.py --output-dir out/e33
```

`measure_cap_activity.py` needs the archived final E09/E10 checkpoints
(`<experiment>/<variant>/seed_<seed>/checkpoints/final.pkl`) and the datasets.

`scripts/run_training.sh CONFIG VARIANT SEED` runs a single training worker.
E09/E10 contain 30 publication-scale runs and E20/E21 60 more; they take hours,
and nothing is launched by installation or tests.

Settings that must not change between a recorded and a new run: the tracked
split manifests, the registered seeds (split seed `1729`), preprocessing, loss
reductions, optimizer, batching, checkpoint selection, metric aggregation,
precision policy, synchronization, and timing scope.

Measurement definitions:

- Steady-state timing uses five fresh-process paired sessions with 50 warmup
  and 200 synchronized calls; compilation and host transfer are outside the
  measured boundary. Each session contributes its median latency.
- An operator or block forward-plus-gradient call differentiates the mean
  squared output with respect to parameters, inputs, and curvature and does
  not include an optimizer update.
- Training `compute_seconds` sums synchronized optimizer updates only.
  `elapsed_seconds` starts before data materialization and includes
  initialization, compilation, training, development evaluation,
  checkpointing, final evaluations, and prediction writing.
- Peak memory is the process-lifetime allocator peak, including
  initialization, compilation, and warmup; it is not isolated activation
  memory.
- Normalization uses current-batch statistics without running state, so
  evaluation always uses manifest order and the declared batch size.
- A nonfinite value, out-of-memory outcome, or divergence is a result and is
  never replaced.

E13 ran on a 32-GiB V100 with the captured default JAX allocator limit of
25,567,444,992 bytes (23.81 GiB); its batch-8, 512 x 512 reference OOM is
specific to that allocator configuration. E22 repeats that point and two
others with the allocator fraction raised to 0.95; requested allocations and
BufferAssignment totals of failed groups are compiler diagnostics, not process
peaks or minimum-memory requirements.

The E09/E10 commands above create outputs under the current protocol. They
cannot feed the historical E07-J2 configuration, which accepts only
protocol-1.10 seed-11 sources with the two configuration hashes frozen in
`configs/checkpoints/e07_journal_diagnostics_v2.yaml`. An exact E07-J2 rerun
requires the legacy V100 Git bundle at evaluator revision `04aca39f` plus the
archived E09/E10 checkpoints and source summaries (Section 7).

### Same-machine rerun check (E19)

E19 repeated the bounded E02 suite and the batch-8 U-Net update timing from a
clean clone of release-candidate commit `93fb31a1` on the original V100
(`results/e19_preflight/`). It is a technical check by the author, not an
independent reproduction. To repeat it from a clean clone:

```bash
r=artifacts/e19; c=$(git rev-parse HEAD)
./.venv/bin/poincare-exec capture-environment --output "$r/environment.json"
$W ./scripts/run_e02_repair.sh "$r/numerics"
POINCARE_EXEC_E08_CONFIG=configs/benchmarks/e08_unet_update_bounded.yaml \
POINCARE_EXEC_E08_OUTPUT="$r/unet_b8_update.csv" $W ./scripts/run_e08.sh
./.venv/bin/python scripts/export_existing_timing_sessions.py \
    --input "$r/unet_b8_update.csv" --output "$r/unet_b8_update_session_medians.csv" \
    --environment "$r/environment.json" --expected-code-commit "$c"
./.venv/bin/python scripts/export_corrected_anchor_tables.py \
    --raw "$r/unet_b8_update.csv" --session-medians "$r/unet_b8_update_session_medians.csv" \
    --config configs/benchmarks/e08_unet_update_bounded.yaml \
    --environment "$r/environment.json" --output-dir "$r/unet_b8_update_analysis"
```

The recorded check passed all 424 bounded rows (40/40 required same-precision
groups) and gave a paired R/F update speedup of 1.745 [1.557, 1.956].

## 5. Re-exporting compact evidence

Exporters validate raw artifacts against the protocol version, configuration
hash, and code commit of the accepted runs, and fail otherwise. A new run
therefore needs its own declared identity; regenerated numbers are comparable
with `results/`, but byte identity is not expected after paths, timestamps, or
code revisions change. The raw inputs are archived outside Git.

```bash
./.venv/bin/python scripts/export_e02_repair_tables.py --help
./.venv/bin/python scripts/export_numerical_claim_rows.py --help
./.venv/bin/python scripts/export_corrected_anchor_tables.py --help
./.venv/bin/python scripts/export_e08_unet_tables.py --help
./.venv/bin/python scripts/export_e11_profile_tables.py --help
./.venv/bin/python scripts/export_e13_scaling.py \
    --manifest configs/analysis/e13_unet_scaling.yaml \
    --raw artifacts/e13_unet_scaling.csv \
    --raw artifacts/e13_unet_scaling.csv.failures.jsonl \
    --output-dir artifacts/analysis/e13_unet_scaling
./.venv/bin/python -m scripts.export_e22_allocator_fraction --help
./.venv/bin/python -m scripts.export_e23_autotune_memory --help
./.venv/bin/python scripts/export_e24_e25_e27_tables.py --help
./.venv/bin/python scripts/export_e26_elementwise_tables.py --help
./.venv/bin/python -m scripts.export_e28_replay_stability --help
./.venv/bin/python -m scripts.export_e29_e30_tables --help
./.venv/bin/python -m scripts.export_e31_gather_reference --help
./.venv/bin/python scripts/export_reviewer_requested_analysis.py --help
./.venv/bin/python -m poincare_exec.analysis.build_all \
    --artifact-dir artifacts --table-dir out/tables --figure-dir out/figures
./.venv/bin/python scripts/export_retained_diagnostic_tables.py --output-dir out/diagnostics
./.venv/bin/python scripts/export_checkpoint_inventory.py --help
```

The E22, E23, E28, E29/E30, and E31 exporters and the E25 and E29 figure
scripts import shared helpers from `scripts/`; run them with
`python -m scripts.<name>` from the repository root. The E13 analysis grid and allocator reference are
described in `docs/e13_scaling_analysis.md`.

E20/E21 export. Set `E09_ROOT`, `E10_ROOT`, `E20_ROOT`, and `E21_ROOT` to the
cohort directories (each containing `R/seed_<seed>/` and `F/seed_<seed>/`),
`ACCEPTED_RUNTIME_ROOT` to the registered protocol-1.14 runtime evidence,
`REJECTED_ATTEMPT_ROOT` to the registered pre-update failure evidence, and
`E20_POST_WORKER_ROOT`/`E21_POST_WORKER_ROOT` to the two infrastructure
evidence archives. The exporter runs only from a clean committed checkout,
writes to a new directory, and validates every registered run before it opens
any endpoint value:

```bash
./.venv/bin/python scripts/export_e20_e21_training.py \
    --manifest configs/analysis/e20_e21_training_replication_v3.yaml \
    --cohort-root E09="$E09_ROOT" --cohort-root E10="$E10_ROOT" \
    --cohort-root E20="$E20_ROOT" --cohort-root E21="$E21_ROOT" \
    --runtime-evidence e20_e21_fef30ec_environment_before="$ACCEPTED_RUNTIME_ROOT" \
    --supplemental-attempt-evidence e20_p112_jax053_preupdate_compile_failure="$REJECTED_ATTEMPT_ROOT" \
    --post-worker-evidence e20_r1338995025_post_worker_ollama="$E20_POST_WORKER_ROOT" \
    --post-worker-evidence e21_r1619140427_post_worker_ollama="$E21_POST_WORKER_ROOT" \
    --post-worker-evidence e21_f646228129_post_worker_ollama="$E21_POST_WORKER_ROOT" \
    --output-dir artifacts/analysis/e20_e21_training_replication
```

The frozen E20/E21 analysis manifests pin `protocol.yaml` at version 1.14
(SHA-256 `5f0824340f67b3c0...`). Version 0.2.0 amends `protocol.yaml` to 1.17
and adds the byte-identical snapshot `configs/protocols/protocol_1_14.yaml`.
When the top-level file no longer matches such a versioned pin, the validators
use `configs/protocols/protocol_<version>.yaml` if, and only if, its SHA-256
equals the pin. This exporter and `scripts/export_checkpoint_inventory.py`
therefore still validate the recorded E20/E21 hashes against the exact
protocol-1.14 bytes.

## 6. Rendering figures from tracked compact evidence

These CPU-only commands read tracked compact tables and reproduce the tracked
figures byte for byte in the locked environment:

```bash
./.venv/bin/python scripts/plot_training_all_endpoints.py --output-stem out/figures/training_all_endpoints
./.venv/bin/python scripts/plot_e20_e21_quality_strata.py --output-stem out/figures/quality_strata
./.venv/bin/python -m scripts.plot_e25_cost_context \
    --validated-dir results/reviewer_requested --output-stem out/figures/e25_cost_context
./.venv/bin/python -m scripts.plot_scaling_with_euclidean --output-stem out/figures/e29_unet_update_scaling
./.venv/bin/python scripts/plot_curvature_trajectories.py --output out/figures/curvature.pdf
```

Without `--output-stem` the first two scripts rewrite their tracked figures in
place.

## 7. Results and provenance

`results/README.md` maps result families to files and records the code
revision, protocol version, and identity record of each family.
`results/claims.csv` maps each manuscript claim to its supporting files.
`results/checkpoints/checkpoint_inventory.csv` lists the 540 checkpoint
artifacts of the 90 accepted training runs (paths, sizes, SHA-256); the
checkpoints themselves are archived outside Git. Verify the compact evidence:

```bash
(cd results && sha256sum --check SHA256SUMS)
```

The E13, E19, E20/E21, and E22--E28 revisions are reachable in this
repository's own history. Some commit messages after v0.1.0 were edited before
publication, which changed later commit identifiers but not their file trees;
`provenance/commit_map.csv` maps each recorded commit to the published commit
with the same tree. Eleven earlier V100 result-producing commits
(E02, E03/E05, E06, E07-J2, E08, E09, E10, E11) predate this repository. They
are kept in a separate Git bundle, `v100-accepted-provenance.bundle`, whose
size, SHA-256, branch, tip, and commits are listed in
`provenance/v100_accepted_history_manifest.json`. The bundle is distributed
separately from the source tree and is not needed to run the code. Verify it
and open the E07-J2 evaluator revision with:

```bash
./.venv/bin/python scripts/verify_provenance_bundle.py \
    --manifest provenance/v100_accepted_history_manifest.json \
    --bundle /path/to/v100-accepted-provenance.bundle
git clone --branch v100-accepted-provenance \
    /path/to/v100-accepted-provenance.bundle ExPoConv-v100-history
(cd ExPoConv-v100-history && git checkout --detach 04aca39f7c99b9881995889ca4f50ff42d2c0425)
```

The E20/E21 cohorts were produced at clean revision `fef30ecb` under protocol
1.14 (configuration hashes `b226335a...` for E20 and `6b7973da...` for E21);
their tables were exported at `072c9ddd` and the checkpoint inventory at
`8a233b94`. Compact evidence keeps the original machine-readable provenance,
including host names, device identifiers, and absolute paths of the measuring
host; these fields are evidence metadata, not runtime requirements, and are not
rewritten.

Release packages are built from a clean, tagged commit and checked with the
independent verifier, which clones the included Git bundle and re-checks every
tracked file, checksum manifest, and fixity record:

```bash
./.venv/bin/python scripts/build_release_package.py --repository . \
    --release-tag v0.2.0 --output-dir /path/to/ExPoConv-release
./.venv/bin/python scripts/verify_release_package.py /path/to/ExPoConv-release
```

The reviewer ZIP in a package is a tracked-file snapshot without `.git`; use
the package's Git bundle (or a clone of the repository) for provenance-recording
reruns.

## 8. Protocol and configuration hashes

The controlled choices that differ from the inspected upstream code (published
conformal factor, midpoint normalization, explicit Dice--focal loss, fixed
learning rate, test-label isolation) are listed in
`docs/reference_resolution.md`; the source-level comparison of R with the
pinned upstream layers is in `docs/upstream_execution_audit.md`.

`protocol.yaml` is the registered protocol, amended only through its dated
amendment log (1.15--1.19 register E22--E31). Each result row records a
configuration hash: the SHA-256 of the canonical JSON of the parsed protocol
mapping together with the experiment configuration and, for benchmarks, the
workload (`src/poincare_exec/config/loader.py`, `config_hash`). The binding is
to protocol content, not to its raw bytes or the protocol version string
alone, so every amendment changes the hash of new runs while recorded hashes
stay as they were.

Accepted results are verified against frozen values rather than against the
current top-level protocol:

- The E02 and E03/E05 exporters read identities from the raw rows and do not
  recompute hashes; the E08 and E11 exporters compare recorded protocol
  versions and hashes with their manifests in `configs/analysis/`.
- E13 recomputes its hashes from the byte-exact snapshot
  `configs/protocols/protocol_1_11.yaml`.
- E07-J2 accepts only the two source hashes frozen in
  `configs/checkpoints/e07_journal_diagnostics_v2.yaml`.
- E06 recomputes the source training hash from the protocol given by
  `--protocol`; replaying the accepted protocol-1.10 E09 checkpoints needs the
  protocol-1.10 bytes from the legacy bundle (unchanged since v0.1.0).
- The E22, E23, E26, and E28 exporters hold their registered commit, protocol
  version, and configuration hashes as constants.
- E20/E21 and the checkpoint inventory pin the protocol-1.14 bytes, which
  are kept as `configs/protocols/protocol_1_14.yaml`; see Section 5.
- The E24/E25/E27, E29/E30, and E31 exporters recompute their hashes from the
  byte-identical snapshots `configs/protocols/protocol_1_17.yaml`,
  `protocol_1_18.yaml`, and `protocol_1_19.yaml` of the run commits.
- E32 and E33 were not registered in `protocol.yaml`; their summaries record
  the measurement commits.

Versions 1.18 (E29/E30) and 1.19 (E31) were registered on parallel branches
from 1.17. `protocol_1_19.yaml` is therefore the E31 run protocol without the
1.18 amendment, whereas the top-level `protocol.yaml` (version 1.19) records
both amendments. Every recorded hash of E20--E31 recomputes from the protocol
bytes retained in this repository (the snapshots above;
`git show 30142ce:protocol.yaml` for 1.15). Protocol versions up to 1.10 are
available only in the legacy bundle.

Historical details that are kept verbatim:

- `configs/numerics/e02_structured.yaml` is the byte-exact protocol-1.3 broad
  E02 matrix (SHA-256 `e1b6b624...`). The paper uses the bounded repaired
  configurations `configs/numerics/e02_repair*.yaml`.
- The hashed E20/E21 configuration descriptions say "secondary CUDA-12
  platform"; protocol 1.14 and the captured runtime are authoritative that the
  accepted training used the pinned CUDA-11.8 stack.
- `docs/journal_diagnostic_protocol_v2.md` is bound by SHA-256 to the E07-J2
  configuration and refers to the superseded first attempt that it amends.
- The top-level `confirmatory_status` field is protocol metadata;
  `results/README.md` and `results/claims.csv` describe the completed
  evidence.

## 9. Revision analyses (E22--E35 and exports)

The compact outputs are in `results/reviewer_requested/`; run and export
commands are in `docs/release_notes_v0.2.0.md`. Definitions:

- **E22.** The E13 workloads at batch 8 (256 x 256 and 512 x 512) and a batch-16,
  512 x 512 point are rerun with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
  (nominal capacity 30.4 GiB of 32 GiB) and preallocation disabled. Requested
  allocations and BufferAssignment totals of failed groups are parsed from the
  worker diagnostics; they are not process peaks or minimum-memory
  requirements. Paired contrasts use only sessions in which both paths
  completed.
- **E23.** The batch-1 U-Net forward is measured with `XLA_FLAGS` unset and with
  `--xla_gpu_autotune_level=0`. Latency with autotuning disabled is a
  diagnostic, not a performance result.
- **E24, E25, E27.** Same session design as E08. Throughput is
  `1000 * batch_size / session_median_latency_ms` (samples/s) for each
  accepted session. Contrasts are geometric means of paired session ratios
  with log-t 95% intervals; failed groups would be retained and excluded from
  ratios (none occurred). E is a cost reference with no quality evaluation;
  E27 is not pooled with the JAX 0.4.25 results.
- **E26.** For every output and gradient tensor of the structured cases, the
  elementwise relative error is `abs(candidate - reference) /
  max(abs(reference), 1e-12)`, summarized by linear p50, p90, p95, p99, p99.9,
  maximum, and the fraction of floored denominators. Scope summaries report
  the maximum of the per-tensor quantiles. E26 does not replace the E02
  acceptance criterion.
- **E28.** The first three FP32 realizations of both structured cases (408
  tensors) are replayed in three fresh processes and evaluated with the mixed
  criterion `max_abs_error <= 2e-5 + 2e-5 * reference_l2_norm`. Every replay is
  reported; none is selected.
- **E29.** The Euclidean path E is timed alone on the three E08 and seven E13
  U-Net workloads with the E08/E13 allocator settings. Its ratios to the
  published E08/E13 R/F medians are unpaired and descriptive. E29 and E30 ran
  while the V100 host recorded driver soft lockups. A rule fixed before any
  affected latency was inspected flags every group whose execution window
  overlapped a recorded stall interval (widened by 180 s); flagged groups are
  excluded from the primary medians and paired contrasts, and all-session
  values are reported beside them as a sensitivity. The rule file and the
  stall intervals are named and hashed in `e29_audit.json` and
  `e30_audit.json`.
- **E30.** Paired R/F complete updates at batch 8 and 256 x 256 with U-Net
  initial widths 16 and 32 (E13 uses 8), under the default allocator cap.
- **E31.** Rg and Mg build R's patch by zero padding and concatenating the
  kernel-offset strided slices, as the official HNN++ code does, instead of
  R's one-hot patch convolution; all later operations are shared with R or M.
  R, Rg, Mg, and F share randomized paired sessions. Contrasts are geometric
  means of paired session ratios with log-t 95% intervals: the five registered
  contrasts and a post-hoc R/Mg contrast. A GPU check compares the R and Rg
  patch tensors (bitwise equal in all 53 comparisons) and one compiled call of
  R with Rg and of M with Mg.
- **E32.** The factorized PyTorch layer computes the patch energy with an
  all-ones convolution and the directional products with one convolution, then
  applies HypLL's own score and output expressions. Agreement is measured on
  S1--S3 and a stride-2 case in FP64 and FP32 (FP32 also against an FP64
  oracle); latency and memory use five paired fresh-process sessions with 50
  warmup and 200 timed calls (`pytorch_crosscheck/README.md`).
- **E33.** The oracle is R in FP64 on the FP32 realization with the FP32
  projection margin; R, F, and T in FP32 are compared with it at curvatures
  0.06, 0.1, and 0.13. The branch trace records the margin of every branch
  decision in small-U-Net realization 2 and tests single flips. Cap activity
  is the fraction of patch evaluations at the source-radius cap on the official
  test batches of the final E09/E10 checkpoints.
- **E35.** At the three retained seed-11 E10 states and the E07-J2 batch, R, F,
  and T gradients in FP32 are compared with R in FP64 with the FP32 margin, on
  the E07-J2 V100. Run `./scripts/run_e35.sh` and export with
  `scripts/export_e35_retained_oracle.py`.
- **Learned curvature.** End-of-epoch development records of all E09 and E20
  R/F seeds (2,000 rows), final paired differences, and paired 95% t
  intervals, exported from the retained training logs.
- **Post-hoc TOST.** Paired two one-sided t tests on the native 0--1 scale with
  margins of 0.01 Dice and 0.005 accuracy at alpha 0.05, for each cohort,
  endpoint, and test partition separately. The margins were not prespecified.
- **Training minutes.** Per-seed compute minutes (synchronized updates) and
  elapsed worker minutes for R and F in E09, E10, E20, and E21.

## 10. Scope of the evidence

| Status | Experiments |
|---|---|
| Complete | E09, E10, E13, E20, E21, E22--E33, E35; E19 as a same-machine technical check |
| Bounded or partial | E01 (bounded operator), E02 (424-row repair suite and structured follow-up), E03 (S1--S3 and Conv--BN--ReLU), E05 (one dense stack), E06 (ten R/T checkpoints), E07 (as E07-J2), E08 (U-Net forward and update), E11 (one update trace) |
| Excluded or unrun | E00 (pilot that could not be audited), E04, E12 (broader CUDA-12 study), E14 (dense training selected in protocol 1.14 but not run), E15--E18 |

E25 and E27 are later post-hoc checks and do not relabel E15 or E12. The
manuscript makes no dense-training or architecture-general claim.
