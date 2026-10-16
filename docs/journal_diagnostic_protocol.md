# Bounded retained-checkpoint diagnostic protocol

Status: frozen before execution on 20 September 2026. Do not execute until the
registered E10 campaign has 15 accepted terminal endpoints. This is a bounded
E06/E07 extension, not a new task-quality campaign.

## Question

At identical retained parameters, optimizer state, data and batch ordering, how
large are the R/F/T differences in forward loss, loss gradients and optimizer
updates, and how do those differences evolve over a short continuation? The
diagnostic is intended to constrain explanations of the observed training
quality differences. It cannot establish task noninferiority or their cause.

## Frozen sources

- Tasks: completed E09 ISIC16 Poincare U-Net and completed E10 CIFAR-100
  Poincare ResNet-20 only.
- Source seed: 11, the smallest registered training seed, selected without
  reference to endpoint rank.
- Source variants: final R, F and T checkpoints for each task. This yields six
  source states and includes one F-trained checkpoint per architecture.
- Each source contributes its saved parameters and Adam state. The diagnostic
  must reject a source whose experiment, task, seed, update, configuration hash,
  protocol version, clean commit or state structure does not match the accepted
  training record.
- No official-test example is used for an update or for checkpoint selection.

## Identical continuation data

For each task, construct the deterministic first 100 training batches after the
registered budget: epoch 51 for E09 and epoch 101 for E10. Use the source seed,
the existing `ordered_batches` rule, and the existing example-indexed
augmentation-key rule. R/F/T and the sensitivity control consume the same batch
at every update. The sequence is fixed before any diagnostic result is read.

## Comparisons at each source state

Apply R, F and T to the same source state and first batch. Record:

1. logits, probabilities and scalar task loss;
2. the raw loss-gradient tree before clipping, L2 regularization or Adam;
3. per-parameter-path and grouped maximum-absolute and relative-L2 gradient
   differences from R;
4. the curvature gradient explicitly for the learned U-Net `curvature_raw`
   leaf; record `not_applicable` for the fixed-curvature ResNet rather than zero;
5. the regularized update vector, updated parameters and both Adam moments after
   one complete optimizer update; and
6. nonfinite counts for outputs, gradients, parameters and optimizer state.

Continue separate R/F/T copies from the identical source state for 100 updates.
At updates 0, 1, 10, 50 and 100, record loss, gradient norm, update norm,
parameter distance from the R continuation, Adam-moment distances, curvature,
and all nonfinite counts. Current-batch normalization remains unchanged and has
no running state.

## Floating-point sensitivity control

For each source, run a fourth continuation through R after a deterministic
one-ULP perturbation of the starting floating-point parameter leaves. Traverse
leaves and elements in stable path/index order and alternate `nextafter` toward
positive and negative infinity; do not perturb optimizer state. Record the
initial perturbation norm and the same trajectory fields. This is a sensitivity
control for amplification of a declared minimal parameter perturbation, not an
estimate of nondeterministic hardware noise and not evidence that R/F rounding
has the same distribution.

Run an unperturbed duplicate R continuation first. Its outputs and trajectory
must be bitwise identical to the primary R continuation under the frozen
deterministic configuration; otherwise stop and report nondeterminism before
interpreting the one-ULP control.

## Acceptance and reporting

- Run sequentially on the authorized `argexpr3` physical GPU 1 under the same
  CUDA 11.8 environment; do not use AICLUSTER or GPU 0.
- Execute from a clean, committed standalone-repository version. Store the
  protocol/config hash, code commit, source-checkpoint SHA-256, device identity,
  precision policy and attempt status with every source.
- Preserve failed and partial attempts. Never replace the frozen seed or source
  variant because of a numerical outcome.
- Require all six sources, all R/F/T comparisons, both R controls, all declared
  observation updates and all raw per-path rows before building an aggregate.
- Report individual source-state results and descriptive maxima. Do not turn
  the six source states into independent task-quality replicates or attach a
  noninferiority claim.
- Manuscript language may say only what the fixed-state evidence establishes.
  In particular, short-trajectory drift can demonstrate sensitivity or rule out
  a large immediate implementation discrepancy, but cannot identify the cause
  of a 50- or 100-epoch endpoint difference.

## Expected artifacts

- immutable diagnostic configuration;
- one attempt record and summary per source checkpoint;
- raw output, per-path gradient, complete-update and trajectory JSONL/NPZ files;
- a source/checkpoint completeness audit;
- per-source and cross-source descriptive CSV tables; and
- hashes and code/config identities added to the evidence manifest.
