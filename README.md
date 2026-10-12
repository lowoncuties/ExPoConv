# ExPoConv

Code and compact evidence for the manuscript

Jochymek L. **Efficient Poincaré Convolutions through Sufficient-Statistics
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
| **R** | materializes the beta-scaled patch, maps it to the ball, evaluates the score, reconstructs and projects the output point | reference |
| **F** | directional products by native convolution, patch energy by a box sum of squared inputs, same clipped score and output round trip | primary method |
| **T** | F plus direct evaluation of the projected origin-tangent output | ablation only |

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

Two prospective R/F replications are registered and running, but are not part
of the compact results in this repository snapshot:

| ID | Registered extension |
|---|---|
| E20 | ISIC16 Poincaré U-Net, 15 new matched R/F seeds on one RTX A6000 |
| E21 | CIFAR-100 Poincaré ResNet-20, the same 15 new matched R/F seeds and platform |

Their official-test endpoints remain jointly blinded until all 60 runs pass
the frozen health and provenance gate in
[`docs/e20_e21_analysis_contract.md`](docs/e20_e21_analysis_contract.md) and
its pre-unblinding
[`v3 amendment`](docs/e20_e21_analysis_contract_amendment_v3.md), which
preserves and supersedes the v2 infrastructure registration.
E20/E21 are reported as separate A6000 cohorts and are never silently pooled
with E09/E10.

Two completed diagnostics are included as supporting evidence but are not
reported in the manuscript:

| ID | Supporting diagnostic |
|---|---|
| E06 | replay of the ten final R/T U-Net checkpoints through R/F/T at fixed weights |
| E07-J2 | gradient, update, and 100-update drift comparison at six seed-11 checkpoints, with repeated-R and one-ULP controls |

All results currently included in `results/` come from the V100 platform in
Section 3. E20/E21 add an RTX A6000 hardware cohort using the same pinned
JAX/CUDA-11.8 user-space runtime; they are not CUDA-12 software-stack runs and
have no accepted result until their joint gate passes. The separate E12
CUDA-12 timing study, reduced precision, and a dense-only training task remain
unrun and unreported (Section 8).

The original experiment-plan document is an archival design record rather
than a live status page. The bounded E02, E03/E05, E06, E07-J2, E08--E11
evidence listed above is complete. E13 completed its registered R/F scaling
grid, including five retained reference-path OOM outcomes at batch 8 and
512 × 512 resolution. E20/E21 are in-progress prospective replications;
their presence in the protocol is not a result. E12 and E14--E18 are unrun
strengthening or conditional extensions, and E19
independent reproduction remains a release gate rather than a completed
result. The bounded handoff is documented in
[`docs/e19_independent_reproduction.md`](docs/e19_independent_reproduction.md),
with a separate signed-record template for the independent reproducer.

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

Every reported measurement was taken on one NVIDIA Tesla V100-PCIE-32GB
(driver 520.61.05) with CUDA 11.8.89 and cuDNN 8.6.0.163 through
`jaxlib 0.4.25+cuda11.cudnn86`, Python 3.11.16, FP32 storage, the highest
matrix-multiplication precision, TF32 disabled, and JAX preallocation
disabled. The machine-readable record is `results/environment.json`. A CUDA
GPU is needed only to rerun experiments. The CPU environment is enough for
the tests, table inspection, and figure regeneration. Other hardware is
untested in the accepted results snapshot. The in-progress E20/E21 campaigns
run on one NVIDIA RTX A6000 with the same JAX/JAXLIB 0.4.25 CUDA-11.8 and
cuDNN-8.6 user-space packages. A driver advertising CUDA 12 compatibility
does not make that environment a CUDA-12 software stack.

## 4. Installation

```bash
./scripts/bootstrap.sh          # CPU environment in ./.venv from uv.lock
./.venv/bin/pytest              # CPU tests (geometry, kernels, models, exporters)
```

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

# E06, E07-J2: checkpoint replay and retained-state diagnostic (need the E09/E10 checkpoints)
$W ./scripts/run_trained_checkpoint_campaign.sh configs/training/e09_isic16_unet.yaml
$W ./scripts/run_journal_retained_diagnostics_v2.sh
```

`scripts/run_training.sh CONFIG VARIANT SEED` runs a single training worker.
Settings that must not change between the recorded and a new run: the
tracked manifests, registered seeds in the selected configuration (split seed
`1729`),
preprocessing, loss reductions, optimizer, batching, checkpoint selection,
metric aggregation, precision policy, synchronization, and timing scope.
Timing uses five fresh-process paired sessions with 50 warm-up and 200
synchronized calls; compilation and host transfer are outside the measured
boundary. Peak memory is the process-lifetime allocator peak, not isolated
activation memory. Normalization uses current-batch statistics without
running state, so evaluation always uses manifest order and the declared
batch size. A nonfinite value, out-of-memory, or divergence is a result and
is never replaced.

Rebuilding the compact tables from the raw outputs of a new run:

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
./.venv/bin/python scripts/plot_training_efficiency_quality.py   # Figure 2 from results/training
```

The exporters validate the raw artifacts against the protocol version,
configuration hash, and code identity declared in `configs/analysis/`, so a
new run must declare its own identities. Generated numbers are comparable
with `results/`; byte identity is not expected after paths, timestamps, or
code revisions change.

E19 is not an additional author-run benchmark. After an immutable release is
frozen, a person who did not implement the method must follow the E19 runbook,
retain unfavorable or failed outcomes, sign the reproduction record, and
archive it outside this working repository. An author- or agent-operated clean
clone is only a technical preflight and does not close that gate.

## 7. Results and provenance

`results/README.md` maps every manuscript table and figure to its file and
lists the code revision, protocol version, and configuration hash of each
result family. `results/claims.csv` maps each claim of the manuscript to its
supporting files. `results/checkpoints/checkpoint_inventory.csv` lists the
150 training checkpoints (roles, sizes, SHA-256) that are archived outside
Git. Verify the compact evidence with:

```bash
(cd results && sha256sum --check SHA256SUMS)
```

The recorded code revisions belong to the development history from which this
repository was extracted; the source tree here is the release snapshot of that
code and is not itself one of those revisions.

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
documents remain planned without a result. The hashed
E20/E21 configuration descriptions still say "secondary CUDA-12 platform";
protocol 1.14 and the captured environment record are authoritative that the
accepted training runtime is the pinned CUDA-11.8 stack on A6000. Likewise,
`docs/journal_diagnostic_protocol_v2.md`
is bound by SHA-256 to the E07-J2 configuration and is kept verbatim; it
refers to the earlier, superseded first attempt that it amends.
The top-level `confirmatory_status` field is likewise historical protocol
metadata; `results/README.md` and `results/claims.csv` describe the completed
evidence currently included in this snapshot.

## 9. Citation

See `CITATION.cff`. The public repository identifier and archived release
version will be added when the release is frozen.

## 10. License

MIT for the code and documentation in this repository (`LICENSE`). Dataset
rights remain with their providers (Section 5).
