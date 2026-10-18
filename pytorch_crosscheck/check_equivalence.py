"""Compare FactorizedHConvolution2d with HypLL's HConvolution2d: outputs and gradients.

For every case, curvature and dtype, both layers share one manifold and one set of random
parameters, receive the same ball-valued input and the same Gaussian output cotangent, and
we compare the output and the gradients with respect to input, weights, bias and curvature.

    python check_equivalence.py --output equivalence.csv     # first visible CUDA device
"""

import argparse
import csv
import math

import torch
from factorized_hconv import factorize_hconvs
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.poincare_ball.math.diffgeom import expmap0, logmap0
from hypll.nn import HConvolution2d
from hypll.tensors import ManifoldTensor
from hypll.utils.math import beta_func
from torch import nn
from torch.nn.functional import unfold

FIELDS = (
    "case",
    "batch",
    "height",
    "width",
    "in_channels",
    "out_channels",
    "stride",
    "regime",
)
CASES = [
    ("S1", 8, 16, 16, 256, 256, 1, "smooth"),
    ("S2", 8, 256, 256, 32, 32, 1, "smooth"),
    ("S3", 8, 256, 256, 96, 32, 1, "smooth"),
    ("S2_stride2", 8, 256, 256, 32, 32, 2, "smooth"),
    # Small probes of HypLL's clamping conventions.
    ("stress_zero_patches", 2, 16, 16, 8, 8, 1, "zero_block"),
    ("stress_source_cap", 2, 16, 16, 8, 8, 1, "beyond_projection_radius"),
    ("stress_output_cap", 2, 16, 16, 8, 8, 1, "large_weights"),
]
CURVATURES = (0.1, 1.0)
DTYPES = {"float64": torch.float64, "float32": torch.float32}
# HypLL projection margins (diffgeom.project); paper's tensor-level tolerance (Eq. 6).
MARGIN = {torch.float64: 1e-5, torch.float32: 4e-3}
TOLERANCE = {torch.float64: 1e-10, torch.float32: 2e-5}


def build(case, c, dtype, device, seed):
    _, batch, height, width, cin, cout, stride, regime = case
    torch.manual_seed(seed)
    manifold = PoincareBall(Curvature(1.0, requires_grad=True))
    # HypLL's Curvature stores a float32 softplus pre-image; set it in the test dtype.
    manifold.c.value.data = torch.tensor(math.log(math.expm1(c)), dtype=dtype)
    # HConvolution2d allocates its parameters in the default dtype (bias=True by default).
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        reference = HConvolution2d(cin, cout, 3, manifold, stride=stride, padding=1)
    finally:
        torch.set_default_dtype(default)
    reference.to(device)
    weight_scale = 20.0 if regime == "large_weights" else 1.0
    with torch.no_grad():
        reference.weights.tensor.normal_(0.0, weight_scale / math.sqrt(9 * cin))
        reference.bias.normal_(0.0, 0.25 / math.sqrt(c))
    factorized = factorize_hconvs(nn.ModuleList([reference]))[0]  # shares parameters

    tangent = torch.randn(batch, cin, height, width, dtype=dtype, device=device)
    tangent /= math.sqrt(c * cin)  # sqrt(c) * ||t|| is about 1 per pixel
    c_value = manifold.c().detach()
    if regime == "beyond_projection_radius":
        # Valid ball points with sqrt(c)||x|| = 1 - 1e-6, outside HypLL's projection radius.
        x = tangent / tangent.norm(dim=1, keepdim=True) * (1 - 1e-6) / math.sqrt(c)
    else:
        if regime == "zero_block":
            tangent[:, :, 4:10, 4:10] = 0.0  # gives 4 x 4 all-zero 3 x 3 patches per image
        x = expmap0(tangent, c_value, dim=1)
    return reference, factorized, manifold, x.detach()


def run(layer, manifold, x, cotangent):
    x = x.clone().requires_grad_(True)
    y = layer(ManifoldTensor(data=x, manifold=manifold, man_dim=1)).tensor
    leaves = [x, layer.weights.tensor, layer.bias, manifold.c.value]
    grads = torch.autograd.grad(y, leaves, grad_outputs=cotangent)
    names = ["output", "grad_input", "grad_weights", "grad_bias", "grad_curvature"]
    return dict(zip(names, [y.detach(), *grads], strict=True))


def cap_fractions(case, manifold, x, y_ref):
    """Fraction of patches at HypLL's source-radius cap and of outputs at the output cap."""
    cin, stride = case[4], case[6]
    c = manifold.c().detach()
    margin = MARGIN[x.dtype]
    max_norm = (1 - margin) / (c + 1e-15) ** 0.5
    a = beta_func(9 * cin / 2, 0.5) / beta_func(cin / 2, 0.5)
    patch_norm = a * unfold(logmap0(x, c, dim=1), 3, padding=1, stride=stride).norm(dim=1)
    source = (torch.tanh(c.sqrt() * patch_norm) / c.sqrt() > max_norm).double().mean()
    output = (y_ref.norm(dim=1) >= max_norm * (1 - 1e-6)).double().mean()
    return source.item(), output.item()


def errors(ref, cand, dtype):
    diff = cand.double() - ref.double()
    ref_l2 = ref.double().norm().item()
    max_abs = diff.abs().max().item()
    return {
        "numel": ref.numel(),
        "ref_l2": ref_l2,
        "ref_max_abs": ref.double().abs().max().item(),
        "max_abs_error": max_abs,
        "rel_l2_error": diff.norm().item() / ref_l2 if ref_l2 > 0 else math.nan,
        "paper_criterion_pass": max_abs <= TOLERANCE[dtype] * (1 + ref_l2),
        "nonfinite_ref": (~torch.isfinite(ref)).sum().item(),
        "nonfinite_factorized": (~torch.isfinite(cand)).sum().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rows = []
    for case_index, case in enumerate(CASES):
        for c in CURVATURES:
            for dtype_name, dtype in DTYPES.items():
                seed = args.seed + 100 * case_index + int(10 * c)
                reference, factorized, manifold, x = build(case, c, dtype, device, seed)
                with torch.no_grad():
                    shape = reference(ManifoldTensor(x, manifold=manifold, man_dim=1)).shape
                cotangent = torch.randn(shape, dtype=dtype, device=device)
                ref = run(reference, manifold, x, cotangent)
                fac = run(factorized, manifold, x, cotangent)
                source_cap, output_cap = cap_fractions(case, manifold, x, ref["output"])
                setting = dict(zip(FIELDS, case, strict=True))
                setting |= {"dtype": dtype_name, "curvature": c}
                caps = {
                    "source_cap_fraction": source_cap,
                    "output_cap_fraction": output_cap,
                }
                for tensor in ref:
                    error = errors(ref[tensor], fac[tensor], dtype)
                    rows.append(setting | {"tensor": tensor} | error | caps)
                    print(
                        f"{case[0]:20s} {dtype_name} c={c:<4} {tensor:15s} "
                        f"max_abs={error['max_abs_error']:.3e} "
                        f"rel_l2={error['rel_l2_error']:.3e}",
                        flush=True,
                    )
                del reference, factorized, ref, fac
                torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
