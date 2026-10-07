# E20/E21 pre-unblinding analysis contract

Status: frozen before inspection of any E20/E21 official-test endpoint value

Frozen: 2026-10-07

Machine-readable manifest: `configs/analysis/e20_e21_training_replication.yaml`

## Purpose and scope

E20 and E21 are prospective, independent-platform R/F replications of the
completed E09 and E10 training studies. This document fixes their analysis
before unblinding. It does not change a training run, seed, endpoint, data
split, implementation, or the frozen protocol. In particular, it does not
retroactively define a noninferiority margin.

The four cohorts are:

| Cohort | Task | Platform | Registered paths | Registered seed pairs | Role |
|---|---|---|---|---:|---|
| E09 | ISIC16 Poincare U-Net | V100 | R/F/T | 5 | completed original study |
| E10 | CIFAR-100 Poincare ResNet-20 | V100 | R/F/T | 5 | completed original study |
| E20 | ISIC16 Poincare U-Net | A6000 | R/F | 15 | prospective replication |
| E21 | CIFAR-100 Poincare ResNet-20 | A6000 | R/F | 15 | prospective replication |

The exact configuration hashes, clean source commits, seeds, runtime
identities, and physical devices are fixed in the machine-readable manifest.
The analysis unit is one matched training seed, not an image, batch, epoch, or
optimizer update. E20 and E21 use the same numeric seed identifiers, but the
two tasks are not paired with each other and must not be combined into a
cross-task effect.

## Global endpoint embargo

All E20 and E21 official-test values remain blinded until the joint extension
gate passes. The embargo covers individual-run values, aggregate values,
predictions, plots, tables, and values from both the fixed-final and
validation-selected checkpoints on both official-test partitions. Reading
development histories and non-endpoint health fields is permitted only to
verify progress, finiteness, update counts, checkpoints, and provenance.

Runtime identity is a precondition to opening any E20/E21 `summary.json`.
The exporter must first validate the accepted campaign's registered
non-endpoint environment record through the repeatable command-line option
`--runtime-evidence EVIDENCE_ID=PATH`. The registered `PATH` is the
accepted protocol-1.14 evidence root, not a training-cohort subdirectory. The
environment record's SHA-256 and all declared JSON fields must match the
machine-readable manifest. Campaign-attempt physical-GPU snapshots then bind
the JAX-visible device to the registered physical index and UUID. Missing,
extra, hash-mismatched, or field-mismatched runtime evidence keeps the embargo
closed; the exporter must fail before reading a summary.

The one registered environment capture is shared by E20 and E21 because both
campaigns use the same accepted root, source commit, Python environment, host,
and physical GPU. It records Python 3.11.16, JAX 0.4.25,
`jaxlib 0.4.25+cuda11.cudnn86`, the pinned CUDA-11.8/cuDNN-8.6 packages,
enabled JAX 64-bit support, highest matmul precision, disabled TF32 override,
disabled JAX preallocation, and the A6000 JAX device. JAX 64-bit support is
enabled for the validation oracle; the registered training tensors remain
FP32. The host's system `nvcc` and driver-advertised CUDA
versions are captured context and are not substituted for the user-space
runtime packages.

The gate requires all 60 registered extension runs: 15 seeds times two paths
times two tasks. Every run must have status `ok`, the exact registered update
budget, finite training and development records, the required checkpoints,
the registered protocol/config/source identity, and the registered physical
GPU identity. A partial cohort does not unlock the other task. No analysis
command that reads official-test values may run before this gate passes.

Before the gate passes, the exporter must not open, parse, hash by reading
content, or scan keys or values from any E20/E21 `summary.json`, including for
apparently non-endpoint health fields. It may verify registered summary paths
exist without reading their contents. Pre-gate completion, status, finiteness,
update-budget, checkpoint, source, and device checks must come from registered
campaign-attempt logs, training/development metrics, checkpoint metadata, and
the accepted runtime record. Summary contents are opened only after all those
checks establish the joint gate.

The pre-gate training/development health shape is also fixed. For E09 and E20,
each epoch has 101 full training rows of eight examples and one final row of
two examples (102 updates), the development evaluation contains 90 examples,
and the end of one-indexed epoch \(e\) is update \(102e\). For E10 and E21,
each epoch has 351 full training rows of 128 examples and one final row of 72
examples (352 updates), the development evaluation contains 5,000 examples,
and the boundary is update \(352e\). These are health and identity checks, not
endpoint observations, and must be verified without reading a summary.

## Primary task-quality estimands

For task-specific metric \(Y\), registered seed \(s\), and path
\(v\in\{R,F\}\), let \(Y_{s,v}\) be the fixed-final official-test endpoint.
The paired difference is

\[
 d_s = Y_{s,F}-Y_{s,R}.
\]

The E20 metric is mean per-image foreground Dice on the official ISIC16 test
partition. The E21 metric is top-1 accuracy on the official CIFAR-100 test
partition. The primary point estimate in each extension cohort is

\[
 \widehat\Delta = \frac{1}{15}\sum_{s=1}^{15}d_s.
\]

Report the following without winner selection:

- the R and F arithmetic means and sample standard deviations across seeds;
- every registered seed's R endpoint, F endpoint, and paired difference;
- \(\widehat\Delta\), the paired-difference sample standard deviation and
  standard error; and
- the unadjusted two-sided 95% Student-\(t\) interval
  \(\widehat\Delta\pm t_{0.975,14}s_d/\sqrt{15}\).

Machine-readable tables retain the metrics on their native zero-to-one scale.
Manuscript tables may multiply Dice or accuracy and their differences by 100,
provided they label the results as points. There is no formal noninferiority
or equivalence test and no registered margin. An interval that includes zero
does not establish equality; an interval wholly on one side of zero does not
turn factorization into an accuracy contribution.

E20 and E21 are separate task-level estimands. Their intervals are
unadjusted, estimation-oriented intervals rather than a joint accept/reject
procedure.

## Secondary task-quality analyses

Apply the same paired F-minus-R estimator separately to:

1. the validation-selected checkpoint on the official test partition;
2. the fixed-final checkpoint on the official test partition after excluding
   byte-identical official-training inputs;
3. the validation-selected checkpoint on that duplicate-excluded partition;
   and
4. for E20 only, mean per-image foreground IoU at the same two checkpoints and
   two partitions.

Because both models use current-batch normalization, each duplicate-excluded
endpoint is a fresh manifest-order evaluation with its new batch composition;
it is not obtained by deleting stored per-image scores. Test loss and best
development score may be tabulated descriptively but are not additional
primary estimands.

For both fixed-final and validation-selected checkpoints, the expected ISIC16
example counts are 379 on `official_test` and 376 on the duplicate-excluded
partition. The expected CIFAR-100 counts are 10,000 and 9,990, respectively.
These counts apply to both the original and replication cohort for a task. A
missing or different count is an endpoint-identity failure, not a smaller
complete-case analysis.

For each registered paired quality analysis, export the median paired
difference, the exact two-sided sign-test value after omitting exact ties, and
a 10,000-replicate seed-pair percentile bootstrap interval as sensitivity
summaries. The Student-\(t\) interval remains the declared interval. Bootstrap
streams must derive from base seed 20260907 and an identity containing the
analysis ID, experiment ID, task, endpoint, partition, metric, comparison,
and path identities. The 2.5th and 97.5th percentiles use NumPy's `linear`
quantile method. All intervals and sensitivity tests are unadjusted.

## Training-efficiency estimands

For each registered seed and time scope \(T\), form

\[
 r_s = T_{s,R}/T_{s,F}.
\]

Values above one favor F. Synchronized optimizer-update compute summed across
the fixed training budget (`compute_seconds`) is the primary training-time
scope. Complete worker elapsed time (`elapsed_seconds`) is secondary and
includes compilation, evaluation, checkpointing, and other worker overhead.
For each scope, report

\[
 \widehat G=\exp\left(\frac{1}{15}\sum_s\log r_s\right)
\]

with the unadjusted two-sided 95% Student-\(t\) interval computed on the 15 log
ratios and exponentiated (14 degrees of freedom). Also export the median and
all seed-level ratios. Do not pool absolute seconds across devices.

## Relationship to E09/E10 and cross-platform reporting

E20 and E21 are the primary replication analyses and must be reported
separately from E09 and E10. The original five-seed estimates are not
recomputed with the new seeds, and the original T ablation is not extended.
The paper should show two labeled strata for each task:

- original V100 cohort: E09 or E10, five matched R/F pairs for the direct
  comparison, with T retained only in the original ablation table; and
- prospective A6000 cohort: E20 or E21, 15 matched R/F pairs.

A forest table or plot may display both stratum-specific F-minus-R estimates
and intervals. It must not contain a pooled diamond, grand pooled estimate,
pooled hypothesis test, or random-effects estimate. Two platform/cohort strata
do not support a stable random-effects analysis, and an inverse-variance
fixed-effect estimate would impose an unregistered common-effect assumption.
Platform is also completely confounded with seed cohort, so a difference
between strata is not a causal hardware effect. The same stratum-only rule
applies to timing ratios.

## Failures, retries, and missing pairs

Seeds are never replaced.

- A pre-update infrastructure or compiler failure may be retried only for the
  same seed and path. The retry is admissible only if the failed attempt's
  immutable evidence is archived and its files, SHA-256 values, and complete
  identity are explicitly registered as supplemental attempt evidence before
  endpoint unblinding. Every attempt is retained. An accepted retry must match
  the frozen cohort's protocol, configuration, clean source, runtime, and
  physical device identities. An unregistered failed attempt keeps the joint
  gate closed even if a later retry succeeds.
- An external interruption after updates may resume only from its immutable,
  contemporaneous, provenance-matched checkpoint. Both that checkpoint and
  the interruption evidence must be archived and explicitly registered,
  including their SHA-256 values and identities, before endpoint unblinding.
  Otherwise the joint gate stays closed. The interrupted attempt and resume
  evidence remain in the audit trail. The current campaign has no registered
  post-update interruption or resume.
- A nonfinite update, nonfinite state, or other scientific/training failure is
  terminal for that registered run. It is reported as a failure and is not
  discarded, restarted to seek a favorable endpoint, imputed, or replaced.
- If any registered extension run remains missing, unresolved, nonfinite, or
  provenance-incompatible, the joint endpoint embargo remains closed. Report
  the campaign as incomplete and do not silently produce a complete-case
  primary estimate.

The retained protocol-1.12/JAX-0.5.3 E20 compiler failure occurred before the
first update and belongs to the infrastructure-failure audit trail. It is not
a task-quality observation. Protocol 1.14 freezes the accepted restart. The
failure is stored outside the accepted protocol-1.14 artifact root, so the
analysis manifest registers it as supplemental attempt evidence rather than
allowing an exporter to discover it from an arbitrary path.

Its registered environment record identifies Python 3.11.16, JAX and jaxlib
0.5.3, `jax-cuda12-pjrt` and `jax-cuda12-plugin` 0.5.3, and the exact captured
CUDA-12 package versions. JAX 64-bit support was enabled, preallocation was
disabled, and the visible device was the registered A6000. These fields and
the environment file's SHA-256 are required audit evidence for classifying the
failure; they do not describe the accepted protocol-1.14 training runtime.

The exporter receives each supplemental evidence root through the repeatable
command-line option
`--supplemental-attempt-evidence EVIDENCE_ID=PATH`. For the registered
protocol-1.12 failure, `PATH` is the root containing the relative failure
manifest, initial checkpoint, environment record, and campaign-attempt log
listed in the YAML manifest. The exporter must verify every registered file's
SHA-256 before reading it, then verify the declared experiment, path, seed,
protocol, configuration, clean source commit, status, return code, error type,
and pre-update failure stage. Unregistered evidence IDs, missing files, hash
mismatches, or identity mismatches are fatal. The resulting audit row is
labeled as a rejected pre-update infrastructure attempt and cannot satisfy an
accepted E20 pair.

## Required machine-readable outputs

After the joint gate passes, the exporter must create provenance-complete,
cohort-keyed outputs for:

- the registered run audit, including failures and all accepted identities;
- per-run fixed-final and validation-selected endpoints;
- per-cohort variant summaries;
- paired task-quality and E20 IoU summaries;
- paired update-compute and elapsed-time summaries; and
- the four stratum rows used by the cross-platform forest display.

Every derived row must retain `experiment_id`; task name alone is not a cohort
key. The exporter must reject mixed protocol/config/source/device identities,
unexpected seeds or paths, duplicate run identities, incomplete R/F pairs,
and any attempt to evaluate E20/E21 before the 60-run gate passes. Bootstrap
stream identities must also include `experiment_id` so the replication cannot
silently reuse an original-cohort stream.

The run audit is the union of accepted-root campaign evidence and the exactly
registered supplemental evidence roots. It is not a recursive search over
nearby directories. Supplemental attempts remain visible even when a later
accepted attempt exists for the same experiment, path, and seed.

The gate audit must also inventory the accepted runtime evidence ID, resolved
root, relative file, verified SHA-256, and matched software/device fields. This
runtime row is provenance evidence, not a training run and not one of the 60
accepted-run gate members.

## CUDA-12 description erratum

The one-line descriptions in the hashed E20 and E21 training configuration
files say "secondary CUDA-12 platform." Those descriptions predate protocol
1.14 and are stale metadata. They are intentionally not edited while their
hashes identify in-flight runs.

The accepted E20/E21 training runtime is JAX/JAXLIB 0.4.25 with pinned
CUDA-11.8 and cuDNN-8.6 user-space libraries on one RTX A6000. The A6000 host
driver advertises CUDA 12.6 compatibility, but that does not make these runs a
CUDA-12 software-stack replication. The separate, still-unrun E12 experiment
is the registered CUDA-12 portability timing study. Protocol 1.14, captured
environment records, source provenance, and physical-GPU snapshots are
authoritative for E20/E21.

## Immutability

This contract and its YAML manifest must be committed before endpoint
unblinding. After that point they are immutable. A correction requires a new
versioned contract and manifest that preserve these files and explain the
reason and timing; it must not overwrite this analysis identity.
