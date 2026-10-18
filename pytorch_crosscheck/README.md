# PyTorch cross-check of the factorized Poincaré convolution

This folder checks the factorization of the paper outside JAX. It compares the official
HypLL implementation of the HNN++ Poincaré convolution (`hypll.nn.HConvolution2d`, HypLL
0.1.1, the version pinned by Hyperbolic U-Net) with a factorized PyTorch module that has the
same parameters. The JAX code, `protocol.yaml` and `configs/` are not used or changed.

| File | Content |
| --- | --- |
| `factorized_hconv.py` | `FactorizedHConvolution2d` and `factorize_hconvs(model)`, which swaps every `HConvolution2d` in a model |
| `check_equivalence.py` | Output and gradient agreement with `HConvolution2d` in FP64 and FP32 |
| `benchmark.py` | Paired fresh-process latency and memory sessions, and their summary |

## 1. Design

`HConvolution2d.forward` calls `PoincareBall.unfold` and `PoincareBall.fully_connected`.
For each input pixel it computes `t = logmap0(x)`, scales it by
`a = B(K C_in / 2, 1/2) / B(C_in / 2, 1/2)`, unfolds the full patch `P`, maps it to the ball with
`expmap0(P)`, and evaluates the HNN++ hyperplane scores and the output reconstruction.
`expmap0` is radial (`expmap0(P) = kappa(||P||) P`). The score therefore depends on `P` only
through two statistics (Eqs. 2–5 of the paper):

- `||P||^2 = a^2 * box(sum_c t_c^2)`, where `box` is an all-ones `conv2d` with the layer's
  stride and padding;
- `z_j^T P = a * conv2d(t, Z)`, where `Z` is the `(K C_in, C_out)` weight matrix reshaped to
  `(C_out, C_in, k, k)`. The rows follow `torch.nn.functional.unfold` order (channel, ky, kx).

The module computes these two statistics and evaluates HypLL's own formulas without building
the `B x K C_in x H_out W_out` ball-valued patch:

1. Ball radius of the patch: HypLL's `expmap0` applied to the one-element vector `(||P||,)`.
   This reuses HypLL's norm floor (`1e-15`) and its projection margin (`4e-3` in FP32, `1e-5`
   in FP64, `diffgeom.project`). Then `z_j^T expmap0(P) = (||expmap0(P)|| / ||P||) z_j^T P`.
2. Score: the expression of `poincare_hyperplane_dists` (`linalg.py`). The conformal factor
   `2 / (1 - c ||x||^2)` is unclamped, `||z_j||` has the `1e-15` floor, and the bias-free branch is
   kept separate, as in HypLL.
3. Output: the expression of `poincare_fully_connected`, followed by `PoincareBall.project`.

HypLL's score function takes the ball point itself, not its statistics. Steps 2 and 3 therefore
restate HypLL's expressions with the same operation order, and call HypLL's `logmap0`,
`expmap0`, `project` and `beta_func` directly. A floor of `1e-30` on `||P||^2` equals HypLL's
`1e-15` norm floor and keeps the square root differentiable for an all-zero patch.

`factorize_hconvs` replaces each `HConvolution2d` in place. The new module shares the original
`weights` and `bias` parameters and the manifold, so the model's parameters, curvature and
initialization are unchanged. Like `HConvolution2d`, the module has no dilation argument.

## 2. Environment

The measured environment is on an RTX A6000 (driver 560.35.03). It was created with `uv`:

```bash
uv python install 3.10.21
uv venv --python 3.10.21 ENV
uv pip install --python ENV/bin/python torch==2.4.0 hypll==0.1.1 numpy==1.24.4
```

PyPI `torch==2.4.0` is the CUDA 12.1 build. It is also the version pinned by Hyperbolic U-Net's
`environment.yml`. The complete-model benchmark imports the unmodified upstream
[Hyperbolic U-Net](https://github.com/swastishreya/Hyperbolic-U-Net) sources at commit
`bec504e2f4d1` (`--unet-src`).

## 3. Steps

1. Check agreement. This writes one row per case, curvature, dtype and compared tensor:

   ```bash
   CUDA_VISIBLE_DEVICES=1 python check_equivalence.py --output equivalence.csv
   ```

   The cases are S1 (16x16, 256→256), S2 (256x256, 32→32), S3 (256x256, 96→32) and S2 with
   stride 2. All use batch 8, a 3x3 kernel and padding 1, with curvature `c ∈ {0.1, 1.0}`. Weights
   are `N(0, 1/(9 C_in))` and biases `N(0, 0.0625/c)`. Inputs are `expmap0` of Gaussian tangents
   with `sqrt(c)||t|| ≈ 1` per pixel. Three small stress cases probe HypLL's clamps: all-zero
   patches, inputs beyond HypLL's projection radius (source-radius cap active), and large weights
   (output projection active). The compared tensors are the output and the gradients with
   respect to the input, weights, bias and curvature for one shared Gaussian cotangent.
   Random values are drawn once in FP64 and cast to each dtype. Errors cover the entries that
   are finite in both implementations, and a separate column counts entries where only one of
   them is finite. Each FP32 result is also compared with an FP64 oracle: HypLL evaluated in FP64
   on the same FP32-rounded values with HypLL's FP32 projection margin. This separates FP32
   rounding from differences between the two formulations.
2. Run the benchmark sessions. The driver refuses to launch a session unless
   `nvidia-smi --query-compute-apps` shows no process on the selected GPU:

   ```bash
   nice -n 19 python benchmark.py run --gpu 1 --run-dir RUN --unet-src Hyperbolic-U-Net
   ```

3. Summarize:

   ```bash
   python benchmark.py summarize RUN/sessions.jsonl --prefix OUT/e32_pytorch
   ```

## 4. Benchmark protocol

- **Workloads:** S1–S3 forward (`torch.no_grad`) and forward+backward. The backward pass is
  `torch.autograd.grad` of the mean squared output with respect to the input, weights, bias and
  curvature, as in the paper's JAX benchmark. The complete model is the upstream `HUNet` with
  depth 4, widths 8–128, transpose-convolution decoding, two classes, batch 8 and 256x256 inputs.
  It is timed as a no-grad forward pass and as a training update. The update is the step of
  upstream `train.py` without AMP: Dice plus focal loss (upstream `losses`, multiclass,
  α = 0.25, γ = 2), backward, gradient clipping at norm 1, and `RiemannianAdam` (lr 1e-3, weight
  decay 1e-4). Curvature is learned and initialized at `c = 0.1` (`softplus(theta) = 0.1`).
  The factorized variant swaps all 19 `HConvolution2d` layers. The upstream `HConvTranspose2d`
  and `HBatchNorm2d` (iterative Fréchet mean) stay unchanged in both variants.
- **Inputs:** operator inputs are `expmap0(0.05 * N(0, 1))` with HypLL's default initialization
  (normal weights, zero bias) and `c = 0.1`. U-Net inputs are uniform images and random binary
  masks in channels-last layout, as in upstream `train.py`.
- **Sessions:** there are 5 sessions per workload. Within each session the variant order is
  shuffled with `random.Random(314159)`. Each variant runs in a fresh process, with 50 warmup
  and 200 timed calls, each followed by `torch.cuda.synchronize()`. A session contributes its
  median latency. Both variants of a session use seed `314159 + session`.
- **Speedup:** the geometric mean of the per-session ratios HypLL / factorized, with a log-t
  95% interval (df = 4).
- **Memory:** the steady-state peak is `torch.cuda.max_memory_allocated()` over the timed calls,
  after `reset_peak_memory_stats()` following warmup. The process peak is the maximum allocated
  over the whole process, including setup and warmup.
- **Numerics:** FP32, with `torch.backends.cuda.matmul.allow_tf32 = False` and
  `torch.backends.cudnn.allow_tf32 = False`. `cudnn.benchmark` stays at its default (`False`).
