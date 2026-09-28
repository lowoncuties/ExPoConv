# Bounded retained-checkpoint diagnostic protocol, amendment 1

Status: frozen on 24 September 2026 before any amended execution. This document
supersedes the repeated-R acceptance rule in
`docs/journal_diagnostic_protocol.md`; every source, checkpoint, batch,
optimizer, observation update and scientific comparison remains unchanged.

## Reason for the amendment

The original E07-J1 campaign completed the three E09 U-Net sources and then
stopped, as designed, at the first E10 ResNet source. The repeated R forward
logits and probabilities were bitwise identical, but the combined bitwise gate
over loss and the raw gradient tree failed at update 0. The attempt record and
partial output archive are retained under the original experiment identity.
Because the original protocol required all six sources, no E07-J1 cross-source
aggregate is promoted or interpreted.

The failure shows that bitwise equality is too strong for repeated GPU
backpropagation in this measured setting. E07-J2 therefore measures the
repeated-R discrepancy as the same-path numerical floor. It does not treat a
non-bitwise repeat as an implementation error when both repeats are finite and
complete. This amendment was made because the original control failed, not in
response to an R/F/T comparison.

## Question and frozen sources

At identical retained parameters, optimizer state, data and batch ordering, how
large are the R/F/T differences in forward loss, raw loss gradients and one
complete Adam update, and how do those differences evolve over a short
continuation relative to a repeated-R numerical floor and a one-ULP parameter
perturbation?

- Tasks: completed E09 ISIC16 Poincare U-Net and completed E10 CIFAR-100
  Poincare ResNet-20.
- Source seed: 11, selected before either diagnostic campaign.
- Source variants: final R, F and T checkpoints for each task, yielding six
  source states.
- Each source must match its accepted training experiment, task, seed, final
  update, configuration hash, protocol version, clean commit and state
  structure.
- No official-test example is used for an update or checkpoint selection.

## Identical continuation and comparisons

Use the first 100 deterministic post-budget training batches: epoch 51 for E09
and epoch 101 for E10. Use the frozen example ordering and example-indexed
augmentation keys. For R, F, T, repeated R and one-ULP R, record:

1. logits, probabilities, scalar task loss and nonfinite counts;
2. the raw loss-gradient tree before clipping, L2 regularization or Adam;
3. per-parameter-path and grouped maximum-absolute and relative-L2 gradient
   differences from R, including learned U-Net curvature;
4. the regularized update vector, updated parameters and both Adam moments
   after one complete optimizer update; and
5. loss, gradient and update norms, parameter and moment distances from R,
   curvature and nonfinite counts at updates 0, 1, 10, 50 and 100.

## Repeated-R and one-ULP controls

The repeated R path uses the identical source state, optimizer state, execution
variant and batches as R. Record bitwise exactness and every ordinary comparison
metric. Require both R paths to be finite and structurally complete, but do not
require bitwise equality. Interpret F/T discrepancies together with the
repeated-R magnitude; never subtract the repeated-R error or call it a formal
noise distribution.

For the one-ULP control, traverse floating-point parameter leaves and elements
in stable path/index order and alternate `nextafter` toward positive and
negative infinity. Do not perturb optimizer state. This is a declared
sensitivity control, not a model of hardware nondeterminism.

## Acceptance and reporting

- Run all six amended sources sequentially from one clean code commit on
  authorized `argexpr3` physical GPU 1 under CUDA 11.8. Do not use AICLUSTER or
  GPU 0.
- Preserve the stopped E07-J1 campaign and all E07-J2 failed or partial
  attempts. Never replace the seed or source because of a numerical outcome.
- Require all six E07-J2 sources, all five execution/control paths, all declared
  observations and all per-path rows before an aggregate is built.
- Report individual source states and descriptive extrema. The source states
  are not independent task-quality replicates.
- Fixed-state agreement can rule out a large immediate discrepancy under the
  tested conditions. Short-horizon drift cannot establish task-quality
  equivalence or identify the cause of a terminal training difference.

## Expected artifacts

- immutable amended configuration and this hashed design document;
- attempt and summary records for all six sources;
- raw output, per-path gradient, complete-update and trajectory artifacts;
- a source/checkpoint completeness audit, key-metric and descriptive tables;
- a deterministic parameter-drift figure; and
- hashes and code/config identities in the evidence manifest.
