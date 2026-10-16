# Post-results release-scope disposition

Status: updated on 15 October 2026 after the same-machine E19 technical
preflight and before an immutable public or reviewer release.

This document reports which experiments from the archival
`Neural_Networks_Experiment_Plan.md` were actually completed. It is not a
preregistration or a protocol amendment, does not change any accepted result,
and must not be used to relabel a historical protocol or configuration hash.
The byte-exact protocol snapshots and their dated amendments remain the
authoritative prospective records for each accepted result.

The current manuscript makes a bounded claim about energy--projection
factorization of the spatial HNN++ Poincare convolution. Author-run data for that
claim are complete. The broader original experiment plan is not complete.

## Status definitions

- **Complete**: the registered experiment or its explicitly amended version
  completed and its accepted evidence is retained.
- **Bounded/partial**: accepted evidence exists, but it covers less than the
  breadth proposed in the archival plan.
- **Unrun**: no accepted result exists. A configuration file alone is not a
  completed experiment.
- **Optional external validation**: an independent person may add stronger
  post-release evidence, but no current manuscript or release claim depends on
  it.

## Experiment disposition

| ID | Disposition | Release interpretation and evidence |
|---|---|---|
| E00 | Not completed; excluded | The supplied pilot could not be made a sufficiently auditable accepted result. No manuscript claim depends on it. Source and implementation decisions are recorded in `docs/reference_resolution.md`; exclusion is explicit in claim C10 of `results/claims.csv`. |
| E01 | Complete for the bounded operator | The proof, assumptions, numerical limits, complexity, and equation-to-code map are in `docs/derivation.md` and the implementation under `src/poincare_exec/core/`. |
| E02 | Modified/bounded complete | The accepted evidence is the 424-row repair suite and the structured convolution-stack/small-U-Net follow-up, not the full broad matrix proposed originally. See `configs/numerics/e02_repair*.yaml` and `results/numerics/`. |
| E03 | Bounded complete | Five paired V100 sessions cover S1--S3 and Conv--BN--ReLU for R/F/T. The original upstream-O and S4/S5 breadth was not measured. See `results/operators/`. |
| E04 | Unrun | No accepted post-repair one-factor operator sweep over spatial size, batch, channels, and kernel exists. E13 is a separate complete-model sweep and is not relabelled as E04. |
| E05 | Partial | One width-256, depth-four dense stack was timed. The broader dense correctness, memory, width, and depth grid was not run. The manuscript treats this result as descriptive and makes no dense-training claim. |
| E06 | Registered supporting diagnostic complete; original breadth partial | Ten final E09 R/T-trained U-Net checkpoints were replayed through R/F/T. F-trained sources and ResNet checkpoints were outside the registered replay. See `results/checkpoints/`. |
| E07 | Modified complete as E07-J2 | Six seed-11 source states from both tasks were evaluated through R/F/T, repeated-R, and one-ULP controls for one Adam update and a 100-update continuation. See `docs/journal_diagnostic_protocol_v2.md` and `results/diagnostics/`. |
| E08 | Bounded complete | Five-session U-Net batch-1/batch-8 forward and batch-8 Adam-update timing and allocation evidence is retained. A complete ResNet/MLP matrix and standalone U-Net backward boundary were deferred. See `results/unet_timing/`. |
| E09 | Complete under amendments 1.9--1.10 | Five matched seeds for each of R/F/T completed 5,100 updates on ISIC16. See `results/training/`. |
| E10 | Complete under amendments 1.9--1.10 | Five matched seeds for each of R/F/T completed 35,200 updates on CIFAR-100. See `results/training/`. |
| E11 | Bounded/partial | One descriptive 20-call batch-8 U-Net update trace is retained; the broader operator and ResNet profile matrix was not run. See `results/profile/`. |
| E12 | Unrun | No accepted CUDA-12 software-stack timing result exists. E20/E21 used CUDA-11.8 user-space libraries on an A6000 whose driver advertised CUDA 12.6 compatibility; they are not E12. |
| E13 | Modified/bounded complete under protocol 1.11 | The registered R/F U-Net grid retained all 70 outcomes. All 35 F sessions and 30 R sessions completed; five R sessions at batch 8 and 512 x 512 recorded first-update allocator OOMs. See `results/e13_scaling/`. |
| E14 | Selected but unrun | Protocol 1.14 retains `include_dense_training_extension: true` and 15 selected R/F/T runs after amendment 1.9. E14 was not executed. Narrowing the manuscript to spatial convolution is a post-results disposition, not prospective deselection. No dense-training or architecture-general claim is made. |
| E15 | Unrun | No Euclidean-overhead baseline result or claim is included. |
| E16 | Unrun | No reduced-precision campaign or claim is included. |
| E17 | Unrun | No frozen corruption or input-shift checkpoint study or claim is included. |
| E18 | Unrun | No additional application, restoration, or GAN campaign or claim is included. |
| E19 | Same-machine technical preflight complete; optional independent validation open | The author-authorized Codex run cloned commit `93fb31a11a008ba1a378434071cb418b504092d9`, passed the 424-row bounded numerical gate, retained all 15 timing groups, and reproduced the R/F update advantage at 1.745 [1.557, 1.956]. See `results/e19_preflight/`. It is not labelled independent, and no current publication claim depends on a later external signature. |
| E20 | Complete prospective addition | Fifteen matched R/F seed pairs completed the registered ISIC16 U-Net budget on one A6000 with the pinned CUDA-11.8 user-space stack. All 30 runs passed the joint gate. See `results/e20_e21_training_replication/`. |
| E21 | Complete prospective addition | Fifteen matched R/F seed pairs completed the registered CIFAR-100 ResNet-20 budget on the same registered platform. All 30 runs passed the joint gate. See `results/e20_e21_training_replication/`. |

## Release consequence

No additional author-run training is required for the claims currently made in
the manuscript. E04, broader E05/E06/E08/E11 coverage, E12, and E14--E18 are
possible strengthening studies, not hidden completed evidence. They must stay
unclaimed unless separately preregistered, executed, and released. The E19
technical preflight is complete; independent repetition is optional. Freezing
an immutable, citable repository/evidence release remains a release operation,
not a missing experiment.
