# Upstream reference resolution

Inspected on 6 September 2026 at the upstream revisions frozen in `protocol.yaml`.
"O" is the unmodified upstream PyTorch implementation, inspected to define the
controlled R/F/T testbeds; no O measurement is reported by the paper.

| Topic | Paper/plan setting | Inspected code behavior | Controlled R/F/T decision | O treatment |
|---|---|---|---|---|
| HNN++ conformal factor | `lambda_x = 2 / (1-c||x||^2)` | HyPLL uses the quotient. The Poincaré ResNet standard-autograd `linreg.py` path contains `2 * (1-c||x||^2)`. | Use the published quotient in all controlled variants. | Keep the affected upstream program unmodified and label the discrepancy. |
| HNN++ output | Equations (6)–(7), followed by source projection where present | The pinned Poincaré ResNet reconstructs the point as `y / (sqrt(c) * (1 + sqrt(1 + ||y||²)))`, which is smooth at an all-zero score. | R/F use that rationalized expression in the ordinary regime and a stable equivalent for large scores. T evaluates the same tangent result directly. All paths retain the selected projection margin. | Run upstream as supplied. |
| Zero orientation columns and zero scores | Source initialization can create zero columns when `in_features <= out_features`; source score code floors their norm at `1e-15`. | The archived controlled reconstruction normalized a zero sinh vector and selected a zero autodiff Jacobian, although the complete published readout has continuous derivative `I/2` at zero. | Preserve the orientation norm floor and its zero-norm convention, but use the smooth published readout consistently in R/F/T. Retain explicit zero/near-zero E02 cases and the archived v1.3/v1.4 evidence. | Unmodified. |
| Projection derivative | Radial clipping is inactive in a neighborhood of the origin. | The archived helper divided the cap by a substituted denominator of one at zero, selecting a curvature-dependent origin slope when the cap was below one. | Use an explicitly shielded inside/outside radial scale. The origin derivative is identity; a point exactly at the cap belongs to the unclipped identity branch. | Unmodified. |
| High-precision oracle | FP64 should separate arithmetic error from a changed function. | The archived E02 oracle changed both dtype and the dtype-selected projection margin from `0.004` to `1e-5`. | The primary FP64 oracle for an FP32 case keeps margin `0.004`. Dtype-default FP64 remains a separately labeled context comparison and is not used to adjudicate the FP32 clipping contract. | The archived O replay retains its original labels and is not silently reinterpreted. |
| U-Net normalization | Paper-informed midpoint normalization | `HBatchNorm2d` is instantiated with HyPLL's default `use_midpoint=False`, selecting an iterative Fréchet mean. | Use midpoint BN, symmetrically for R/F/T, because direct statistic reuse and the paper protocol target that named operator. | Report upstream Fréchet behavior separately; no checkpoint-equivalence claim across this change. |
| U-Net loss | Dice–Focal | `train.py` first selects Dice, then its `elif 'focal'` branch is skipped, leaving BCE/CE as the second term for `dice+focal`. | Use an explicit Dice + focal implementation with alpha 0.25, gamma 2, and smoothing recorded in the protocol. | Preserve and label the upstream Dice + CE branch behavior in E08 backward/update measurements. |
| U-Net weight decay | `1e-4` in the paper-informed plan | Function default is `1e-8`, and CLI does not expose it. | Use `1e-4` for every controlled variant. | Preserve upstream default. |
| U-Net optimizer/schedule | Riemannian Adam at fixed declared endpoint | The source applies `RiemannianAdam` and validation-driven `ReduceLROnPlateau`; all parameters represented in this study are the source's Euclidean coordinates. | Adam in those coordinates, coupled L2 decay, global-norm clipping at 1.0, fixed lr `1e-3`, no scheduler. This removes validation-dependent branch divergence from the matched R/F/T comparison. | Preserve upstream optimizer and scheduler. |
| U-Net epochs | 50 | README example uses 30. | Fixed-final-update endpoint at 50 epochs. | Record the invoked command literally. |
| U-Net decoder | Transposed convolution | Default CLI has `bilinear=False`; custom HNN++ FC plus Poincaré fold is used. | Retain a source-equation transpose path at explicit tangent boundaries. Ordinary-convolution factorization is not claimed for it. | Unmodified. |
| U-Net evaluation | Official train/test partitions; test only once after training | Upstream training evaluates the test loader every five epochs. | Never expose test labels to scheduling or model selection. | Do not use upstream repeated-test results as confirmatory quality evidence. |
| Poincaré ResNet size | ResNet-20 | Example script names `8-16-32-resnet-20`; parser yields three blocks per group. | Channels `[8,16,32]`, blocks `[3,3,3]`. | Run exact command and commit separately. |
| Poincaré ResNet recipe | Adam, lr `1e-3`, wd `1e-4`, batch 128, 100 epochs | Example script explicitly sets these values for an Euclidean ResNet; CLI defaults differ and direct test evaluation is used each epoch. | Freeze the explicit values, add a training-only stratified development split, and reserve test for final evaluation. | Label upstream recipe/result separately. |
| CIFAR preprocessing | Source recipe | `ToTensor`, channelwise `(x-0.5)/0.5`, then zero-padded random crop and horizontal flip. | Use the same operation order and indexed per-example augmentation RNG; describe scaling as `[-1,1]`. | Unmodified. |
| ISIC resizing | 256² RGB and source padding | Source pads the bottom/right of the original array to a square and then interpolates to 256². | Freeze bottom/right minimum-value image padding, zero mask padding, Pillow 11.3 bicubic image resizing and nearest mask resizing for R/F/T. | Report the exact upstream PyTorch interpolation separately. |
| Remainder batches | Exact fixed update budget | Source loaders retain the final variable-size batch. | Retain all examples with variable final-batch shape; freeze manifest order for evaluation. Exact updates are 5,100 (ISIC) and 35,200 (CIFAR-100). | Unmodified. |
| Linked official examples | Official partitions plus independence audit | The downloaded ISIC release contains byte-identical images both within training and across official training/test partitions. | Group exact duplicates within the training/development split. Keep the official test as primary comparability endpoint and add a clearly labeled test subset excluding exact training duplicates. Do not claim lesion/patient independence without metadata. | Unmodified official partition. |
| Batch-normalization evaluation | Current-batch source semantics | Neither inspected implementation maintains usable running statistics for the named hyperbolic BN path. | Preserve current-batch statistics and freeze evaluation batch composition. B=1 timing is not presented as the same prediction function. | Unmodified. |
| Curvature | Task-specific | U-Net can learn one or multiple curvatures; ResNet factory can learn per-operation curvature but examples do not establish a single paper recipe. | One trainable U-Net magnitude initialized at 0.1; fixed 0.1 for ResNet/MLP. Any learned-ResNet extension is exploratory. | Record exact source invocation. |

The controlled comparison is therefore an equation-preserving study of a
fully specified architecture, not an assertion that every upstream software
default reproduced its paper. Any later decision that changes this table must
increment the protocol version before affected confirmatory runs.

## Measured platforms

The original E09/E10 training cohorts and all operator, complete-model timing,
memory, profile, and scaling evidence use one Tesla V100-PCIE-32GB with driver
520.61.05 under the pinned CUDA 11.8 build; see `results/environment.json`.
The prospective E20/E21 training replications use one NVIDIA RTX A6000 with
driver 560.35.03 and the same pinned JAX/JAXLIB 0.4.25 CUDA-11.8 and cuDNN-8.6
user-space packages; see
`results/e20_e21_training_replication/e20_e21_gate_audit.json`. The driver on
that host advertised CUDA 12.6 compatibility, but the measured user-space
runtime remained CUDA 11.8. E09/E10 and E20/E21 are retained as separate
cohort/platform strata, no grand pool was performed, and the design does not
identify a causal accelerator effect or establish CUDA-12 portability.
