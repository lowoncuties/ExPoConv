# ExPoConv

Code and compact evidence for the manuscript

Jochymek L. **Efficient Poincaré Convolutions through Energy–Projection
Factorization.** (2026, manuscript in preparation)

The repository is a paper companion, not a general library. It contains the
standalone JAX implementation, the frozen experiment protocol and
configurations, the exact data-split manifests, the experiment runners and
table exporters, the compact result tables and figures behind every reported
number, and CPU tests. It has no dependency on any hyperbolic-learning
library; every operator is implemented from the published equations.

## 1. Overview

A literal HNN++ Poincaré convolution expands every receptive-field patch,
maps it to the ball, and evaluates a hyperplane score. The score depends on
the tangent-space patch only through its squared norm and its inner products
with the filters, so a standard convolution and a scalar spatial reduction
compute the same clipped score without ball-valued patches. Three execution
paths of the same layer are compared. They share parameters, clipping,
normalization, curvature, data order, and optimizer:

| Path | Execution | Role |
|---|---|---|
| **R** | materializes and ball-maps the beta-scaled patch for spatial layers; ball-maps the input to dense layers; evaluates the clipped score and reference output round trip | reference |
| **F** | directional convolution plus scalar patch energy for spatial layers; analogous matrix products plus input energy for dense layers; same clipped score and output round trip | primary method |
| **T** | F plus direct evaluation of the projected origin-tangent output | ablation only |

Spatial patch factorization is the primary contribution. The packaged F/T
model paths also apply the analogous energy--projection score evaluation to
HNN++ dense layers, including the ResNet classifier and the fully connected
part of U-Net transpose decoding; `docs/derivation.md` maps both cases to code.

Experiments reported in the manuscript, with the identifiers used in
`configs/` and `results/`:

| ID | Experiment |
|---|---|
| E02 | bounded numerical validation (derivative repair suite, structured Conv-stack and small-U-Net agreement in FP32/FP64) |
| E03, E05 | paired operator and block timing: three convolution shapes, Conv-BN-ReLU, depth-four dense stack |
| E08 | complete Poincaré U-Net forward and optimizer-update latency and process-lifetime peak allocation |
| E09 | ISIC16 Poincaré U-Net, five matched seeds, fixed 5,100-update budget |
| E10 | CIFAR-100 Poincaré ResNet-20, five matched seeds, fixed 35,200-update budget |
| E11 | descriptive device-kernel profile of the U-Net update |
| E13 | complete-update scaling across U-Net batch size and input resolution, including retained OOM outcomes |
| E19 | same-machine clean-clone technical preflight of the bounded numerical gate and batch-eight U-Net update timing |
| E20 | prospective ISIC16 Poincaré U-Net replication, 15 new matched R/F seeds on one RTX A6000 |
| E21 | prospective CIFAR-100 Poincaré ResNet-20 replication, the same 15 new matched R/F seeds and platform |

All 60 E20/E21 runs passed the frozen health and provenance gate in
[`docs/e20_e21_analysis_contract.md`](docs/e20_e21_analysis_contract.md) and
its pre-unblinding
[`v3 amendment`](docs/e20_e21_analysis_contract_amendment_v3.md), which
preserves and supersedes the v2 infrastructure registration. The official-test
endpoints were opened only after that joint validation. E20/E21 are reported
as separate A6000 cohorts and are not pooled with E09/E10; the cohort/platform
change does not support a causal hardware comparison.

Two completed diagnostics are included as supporting evidence. E06 remains
outside the concise manuscript. E07-J2 is summarized there only by one bounded
limitations sentence; its full design, tables, controls, and trajectory remain
repository-only:

| ID | Supporting diagnostic |
|---|---|
| E06 | replay of the ten final R/T U-Net checkpoints through R/F/T at fixed weights; omitted from the manuscript |
| E07-J2 | gradient, update, and 100-update drift comparison at six seed-11 checkpoints, with repeated-R and one-ULP controls; summarized in one limitations sentence |

The original E09/E10 training cohorts and the timing, memory, and profile
experiments use the V100 platform in Section 3. E20/E21 add accepted RTX A6000
training cohorts using the same pinned JAX/CUDA-11.8 user-space runtime. They
are not CUDA-12 software-stack runs. The separate E12 CUDA-12 timing study,
reduced precision, and a dense-only training task remain unrun and unreported
(Section 8).

The original experiment-plan document is an archival design record rather
than a live status page. The bounded E02, E03/E05, E06, E07-J2, E08--E11
evidence listed above is complete. E13 completed its registered R/F scaling
grid, including five retained reference-path OOM outcomes at batch 8 and
512 × 512 resolution. E20/E21 completed their registered 60-run matrix and
passed the joint gate before endpoint export. E12 and E14--E18 are unrun
strengthening or conditional extensions. An E19 same-machine clean-clone
technical preflight completed successfully on 15 October 2026; optional
independent post-release reproduction remains unclaimed. The bounded handoff is documented in
[`docs/e19_independent_reproduction.md`](docs/e19_independent_reproduction.md),
and the completed technical record is in
[`results/e19_preflight/`](results/e19_preflight/). The
dated [`release-scope disposition`](docs/release_scope_disposition.md) maps
every E00--E21 item to completed, bounded, unrun, or open status without
rewriting the frozen protocol.

## 2. Repository structure

```text
configs/                 frozen experiment, analysis, and numerical-case definitions
docs/                    derivation and equation-to-code map, upstream reference resolution,
                         hashed E07-J2 design document
results/                 compact tables and figures behind the manuscript (results/README.md)
scripts/                 environment setup, dataset download, experiment runners, exporters
splits/                  exact ISIC16 and CIFAR-100 manifests with membership, order, hashes,
                         and duplicate annotations
src/poincare_exec/       implementation
  core/                  ball geometry, stable scalar primitives, HNN++ R/F/T kernels,
                         normalization, pooling, residual and transpose operators
  models/                controlled Poincaré U-Net and ResNet-20
  training/              fixed-budget training worker, losses, Adam, checkpoints
  experiments/           numerical, timing, profile, checkpoint-replay and diagnostic runners
  analysis/              table and figure builders
tests/                   CPU tests
protocol.yaml            frozen preregistered protocol (Section 8)
```

## 3. Requirements

Python 3.11 and [uv](https://docs.astral.sh/uv/). The environment is pinned
in `uv.lock`: JAX 0.4.25, NumPy 1.26.4, SciPy 1.11.4, Pillow 11.3.0,
matplotlib 3.8.4.

The original E09/E10 cohorts and all operator, complete-model timing, memory,
scaling, and profile measurements were taken on one NVIDIA Tesla V100-PCIE-32GB
(driver 520.61.05) with CUDA 11.8.89 and cuDNN 8.6.0.163 through
`jaxlib 0.4.25+cuda11.cudnn86`, Python 3.11.16, FP32 storage, the highest
matrix-multiplication precision, TF32 disabled, and JAX preallocation
disabled. The machine-readable V100 record is `results/environment.json`.
E20/E21 were trained on one NVIDIA RTX A6000 (driver 560.35.03) with the same
JAX/JAXLIB 0.4.25 CUDA-11.8 and cuDNN-8.6 user-space packages; their captured
runtime is retained in
`results/e20_e21_training_replication/e20_e21_gate_audit.json`. A driver
advertising CUDA 12 compatibility does not make that environment a CUDA-12
software stack. A CUDA GPU is needed only to rerun experiments. The CPU
environment is enough for the tests, table inspection, and figure
regeneration.

## 4. Installation

```bash
./scripts/bootstrap.sh          # CPU environment in ./.venv from uv.lock
./.venv/bin/pytest              # CPU tests (geometry, kernels, models, exporters)
```

If `python3.11` is not on `PATH`, select an existing Python 3.11 interpreter
explicitly. `POINCARE_EXEC_ENV_DIR` may likewise place the environment outside
the checkout without changing the locked dependencies:

```bash
POINCARE_EXEC_PYTHON=/absolute/path/to/python3.11 \
POINCARE_EXEC_ENV_DIR=/absolute/path/to/expoconv-venv \
./scripts/bootstrap.sh
```

Use a Git clone or a clone made from the release Git bundle for
provenance-recording experiment reruns. A release source archive or source
distribution is sufficient for source inspection, CPU tests, checksum
verification, and deterministic table and figure regeneration. The Python
wheel is intentionally code-only: it does not bundle the frozen protocols,
configurations, split manifests, compact results, or scripts, and therefore is
not a reproduction package. The documented lock-based source bootstrap above
is the supported standalone workflow.

For the measured GPU platform:

```bash
./scripts/bootstrap_cuda11_v100.sh   # CUDA-11.8 jaxlib wheel and pinned NVIDIA libraries
CUDA_VISIBLE_DEVICES=0 ./scripts/with_cuda_11_8.sh ./scripts/check_cuda.sh
```

`scripts/with_cuda_11_8.sh` puts the pip-installed CUDA 11.8 libraries on the library
path and disables preallocation. It exists because the login shell of the
measured host exposed a different default CUDA toolkit. Every runner requires
exactly one physical index in `CUDA_VISIBLE_DEVICES` and runs its jobs
sequentially on that device. `0` above is an example index.

## 5. Data

```bash
./scripts/fetch_core_datasets.sh     # writes to data/raw/ (ignored), SHA-256 verified
```

| Dataset | Source | Used by | Terms |
|---|---|---|---|
| ISIC 2016 challenge, Part 1 | four archives from the [ISIC challenge object store](https://isic-archive.s3.amazonaws.com/challenges/2016/) ([Gutman et al., 2016](https://arxiv.org/abs/1605.01397)) | E09, E20, E06, E07-J2 | [ISIC terms of use](https://challenge.isic-archive.com/terms-of-use/) |
| CIFAR-100 | Python archive from the [CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009) | E10, E21, E07-J2 | citation requested by the source page |

No image bytes are redistributed, and the MIT license covers only the code
and documentation in this repository.

The tracked manifests in `splits/` are the authoritative partitions: ISIC16
810 train / 90 development / 379 official test; CIFAR-100 45,000 / 5,000 /
10,000. They record membership, order, per-example input and target hashes,
and byte-identical duplicate groups. The CIFAR-100 manifest is 11 MB and is
tracked on purpose, because its ordering and duplicate annotations are part
of the evaluated design. Verify and, optionally, reconstruct them:

```bash
(cd splits && sha256sum --check SHA256SUMS)
candidate=$(mktemp -d)
./.venv/bin/poincare-exec make-splits --data-root data/raw --output-dir "$candidate"
cmp splits/isic16.csv "$candidate/isic16.csv"
cmp splits/cifar100.csv "$candidate/cifar100.csv"
```

## 6. Reproducing the experiments

All commands run from the repository root on the GPU environment of
Section 4. New outputs go to `artifacts/` (ignored) and never overwrite
`results/`. E09/E10 contain 30 publication-scale runs in total; E20/E21 add
60 registered R/F runs. They take hours and nothing is launched by
installation or tests.

Run provenance-recording commands from a real Git worktree. The tracked-file
reviewer ZIP omits `.git` and is suitable for source and compact-evidence
inspection, not commit-bound reruns. For a release-snapshot rerun, clone the
release package's `repository/ExPoConv-git-<release-commit>.bundle` (or the
released repository) rather than extracting the ZIP; the separate legacy V100
bundle has a different role, described below. The fail-closed final package
builder and verifier are documented in
[`provenance/README.md`](provenance/README.md#building-the-final-release-package).

```bash
export CUDA_VISIBLE_DEVICES=0
W=./scripts/with_cuda_11_8.sh

# E02: numerical validation
$W ./scripts/run_e02_repair.sh artifacts/numerical_repair/e02_repair
$W ./.venv/bin/poincare-exec structured-numerics --protocol protocol.yaml \
    --config configs/numerics/e02_repair_structured.yaml \
    --output artifacts/numerical_repair/e02_repair_structured.jsonl

# E03/E05, E08, E11: timing, complete U-Net timing, profile
$W ./scripts/run_corrected_operator_anchors.sh
$W ./scripts/run_e08.sh
$W ./scripts/run_e11.sh

# E13: registered R/F U-Net update scaling campaign
$W ./scripts/run_e13.sh

# E09, E10: five matched seeds per path (R/F/T run sequentially)
$W ./scripts/run_training_campaign.sh configs/training/e09_isic16_unet.yaml
$W ./scripts/run_training_campaign.sh configs/training/e10_cifar100_resnet20.yaml

# E20, E21: 15 new matched seeds per path (R/F run sequentially)
# Do not inspect official-test endpoints until both campaigns pass the joint gate.
$W ./scripts/run_training_campaign.sh configs/training/e20_isic16_unet_rf_replication.yaml
$W ./scripts/run_training_campaign.sh configs/training/e21_cifar100_resnet20_rf_replication.yaml

# E06: checkpoint replay of fresh E09 checkpoints
$W ./scripts/run_trained_checkpoint_campaign.sh configs/training/e09_isic16_unet.yaml
```

The E09/E10 commands above create protocol-1.14 outputs. They cannot feed the
historical E07-J2 configuration, which accepts only protocol-1.10 seed-11
sources with the two frozen configuration hashes recorded in
`configs/checkpoints/e07_journal_diagnostics_v2.yaml`. This rejection is an
intentional provenance check. An exact E07-J2 rerun requires a clone of the
legacy V100 Git bundle at evaluator revision `04aca39f`, plus the separately
archived accepted E09/E10 checkpoints and source summaries; see
[`provenance/README.md`](provenance/README.md). Fresh outputs require a newly
declared diagnostic rather than relabelling E07-J2.

`scripts/run_training.sh CONFIG VARIANT SEED` runs a single training worker.
Settings that must not change between the recorded and a new run: the
tracked manifests, registered seeds in the selected configuration (split seed
`1729`),
preprocessing, loss reductions, optimizer, batching, checkpoint selection,
metric aggregation, precision policy, synchronization, and timing scope.
Steady-state benchmark timing uses five fresh-process paired sessions with 50
warm-up and 200 synchronized calls; compilation and host transfer are outside
that measured boundary. An operator or block forward-plus-gradient call
differentiates the mean squared output with respect to parameters, inputs, and
curvature and does not include an optimizer update. Training `compute_seconds`
sums synchronized optimizer updates only. `elapsed_seconds` starts before
train/development data materialization and includes initialization,
compilation, training, development evaluation, checkpointing, final
evaluations, and prediction writing; the outer launcher and GPU guard are
outside that clock. Peak memory is the process-lifetime allocator peak, not
isolated activation memory.
Normalization uses current-batch statistics without running state, so
evaluation always uses manifest order and the declared batch size. A nonfinite
value, out-of-memory, or divergence is a result and is never replaced.

For E13, the accelerator has 32 GiB of physical memory, while the accepted
environment record captures a 25,567,444,992-byte (23.81-GiB) JAX allocator
limit. The scaling figure uses that captured allocator ceiling rather than a
nominal 32-GiB line; its R out-of-memory outcome is therefore specific to the
recorded allocator configuration and does not imply exhaustion of all physical
device memory.

### Re-exporting compact evidence from raw outputs

For the E20/E21 exporter, set `E09_ROOT`, `E10_ROOT`, `E20_ROOT`, and
`E21_ROOT` to the corresponding cohort directories (each containing the
variant/seed directories). Set `ACCEPTED_RUNTIME_ROOT` to the registered
protocol-1.14 evidence root and `REJECTED_ATTEMPT_ROOT` to the registered
pre-update failure evidence root. Set `E20_POST_WORKER_ROOT` and
`E21_POST_WORKER_ROOT` to the two immutable infrastructure evidence archives.
Run it only from a clean committed checkout and give it a new, nonexistent
output directory; it validates the joint gate before opening any
endpoint-bearing summary and otherwise fails closed.

```bash
./.venv/bin/python scripts/export_e02_repair_tables.py --help
./.venv/bin/python scripts/export_corrected_anchor_tables.py --help
./.venv/bin/python scripts/export_e08_unet_tables.py --help
./.venv/bin/python scripts/export_e11_profile_tables.py --help
./.venv/bin/python scripts/export_e13_scaling.py \
    --manifest configs/analysis/e13_unet_scaling.yaml \
    --raw artifacts/e13_unet_scaling.csv \
    --raw artifacts/e13_unet_scaling.csv.failures.jsonl \
    --output-dir artifacts/analysis/e13_unet_scaling
# Run only after all 60 E20/E21 runs pass the joint endpoint gate.
# Each cohort root contains R/seed_<seed>/ and F/seed_<seed>/ directories.
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
./.venv/bin/python -m poincare_exec.analysis.build_all \
    --artifact-dir artifacts --table-dir out/tables --figure-dir out/figures
./.venv/bin/python scripts/export_retained_diagnostic_tables.py --output-dir out/diagnostics
```

The exporters validate the raw artifacts against the protocol version,
configuration hash, and code identity declared in `configs/analysis/`, so a
new run must declare its own identities. Generated numbers are comparable
with `results/`; byte identity is not expected after paths, timestamps, or
code revisions change.

### Rendering figures from tracked compact evidence

These CPU-only commands render already exported compact evidence and do not
open or re-export raw run artifacts:

```bash
# Legacy five-seed runtime/quality rendering; retained for evidence history.
./.venv/bin/python scripts/plot_training_efficiency_quality.py
# Original V100 R/F/T final and validation-selected endpoints used in the appendix.
./.venv/bin/python scripts/plot_training_all_endpoints.py
# Current manuscript rendering of the four separate E09/E10/E20/E21 strata.
./.venv/bin/python scripts/plot_e20_e21_quality_strata.py
```

The first output is no longer identified by a manuscript figure number. The
second validates the exact 60-row accepted E09/E10 endpoint cohort and renders
the appendix figure directly from `results/training/training_runs.csv`. The
third is a presentation-only derivative of the unchanged compact
`e09_e10_e20_e21_platform_strata.csv`; the raw exporter's archival figure
remains byte-preserved.

The E19 same-machine technical preflight is complete: an author-authorized
Codex run cloned the verified commit-bound bundle, retained all outcomes, and
reproduced the bounded numerical gate and batch-eight R/F update advantage on
the original V100 platform. Its compact record is in
`results/e19_preflight/`; raw evidence remains checksummed outside Git. An
independent person may still follow the E19 runbook after public release, but
that optional validation is not claimed here and is not a submission gate.

## 7. Results and provenance

`results/README.md` maps every manuscript table and figure to its file and
lists the code revision, protocol version, and configuration hash of each
result family. `results/claims.csv` maps each claim of the manuscript to its
supporting files. `results/checkpoints/checkpoint_inventory.csv` lists the
540 checkpoint artifacts from the 90 accepted E09/E10/E20/E21 training runs
(six roles per run, with paths, sizes, and SHA-256) that are archived outside
Git. Verify the compact evidence with:

```bash
(cd results && sha256sum --check SHA256SUMS)
```

The recorded code revisions belong to the development history from which this
repository was extracted; the source tree here is the release snapshot of that
code and is not itself one of those revisions. Eleven legacy V100
result-producing commits are retained in a separate, hashed Git bundle
described by
[`provenance/v100_accepted_history_manifest.json`](provenance/v100_accepted_history_manifest.json).
The E13 revision and the E20/E21 training, exporter, and inventory revisions
are reachable in this repository's own history. The legacy bundle is prepared
for a separate evidence deposit rather than embedded in the public checkout;
[`provenance/README.md`](provenance/README.md) gives the verification command
and disclosure boundary. It is distinct from the current-release Git bundle
that supplies `.git` history when the tracked-file reviewer ZIP is used for
transport.

The compact evidence retains the original machine-readable provenance,
including local source paths, the measurement host name, physical-device
identifiers, and failed-run diagnostics. These fields are evidence metadata,
not runtime requirements. They must not be silently rewritten: a later public
release may either retain the exact captures or publish explicitly labelled
sanitized derivatives alongside hashes for the preserved originals.

## 8. Frozen protocol

`protocol.yaml` is the preregistered protocol, frozen before each result and
amended only through its dated amendment log. Its bytes are hashed into the
configuration hash of every recorded result, and `configs/checkpoints/e07_journal_diagnostics_v2.yaml`
validates source checkpoints against those hashes, so historical versions are
not silently relabelled. Byte-exact historical protocols needed to replay an
accepted analysis are retained under `configs/protocols/`; E13 uses the
protocol 1.11 snapshot there rather than the later-amended top-level file. E12
CUDA-12 timing, a Fashion-MNIST dense task, and the original local plan
documents remain planned without a result. In particular, protocol 1.14
retains the originally selected E14 dense-training extension (15 R/F/T runs):
E14 was not executed. After the convolutional campaigns, the manuscript scope
was narrowed to spatial HNN++ convolution factorization. This is a post-results
scope disposition, not a completed or prospectively deselected experiment;
the bounded E05 dense-stack timing remains descriptive and supports no
dense-training or architecture-general claim. The complete post-results
status map is retained in
[`docs/release_scope_disposition.md`](docs/release_scope_disposition.md). The hashed
E20/E21 configuration descriptions still say "secondary CUDA-12 platform";
protocol 1.14 and the captured environment record are authoritative that the
accepted training runtime is the pinned CUDA-11.8 stack on A6000. Likewise,
`docs/journal_diagnostic_protocol_v2.md`
is bound by SHA-256 to the E07-J2 configuration and is kept verbatim; it
refers to the earlier, superseded first attempt that it amends.
The top-level `confirmatory_status` field is likewise historical protocol
metadata; `results/README.md` and `results/claims.csv` describe the completed
evidence currently included in this snapshot.

The top-level protocol also references
`configs/numerics/e02_structured.yaml`. That file is the byte-exact historical
protocol-1.3 broad E02 matrix (SHA-256
`e1b6b624be66d1a7352b4b74925ccb4876f22714ce94cd1540e658ac83ebf60b`,
introduced at development commit
`a486aab6fb0fd296fb42273032e4741094e017db`). It is retained to make the
archival protocol self-contained. The paper-facing numerical claims use the
bounded repaired configurations under `configs/numerics/e02_repair*.yaml`, not
this broad historical matrix. Reconstructing the historical E02 result hash
also requires the matching protocol bytes from its result-producing commit;
the current protocol must not be substituted and relabelled.

## 9. Citation

See `CITATION.cff`. This source snapshot is version `0.1.0`. A public
repository URL or archive DOI will be added only after the author approves
public disclosure; the initial provenance-complete reviewer package is
confidential.

## 10. License

MIT for the code and documentation in this repository (`LICENSE`). Dataset
rights remain with their providers (Section 5).
