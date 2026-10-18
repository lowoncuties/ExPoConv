"""Compare FactorizedHConvolution2d with HypLL's HConvolution2d: outputs and gradients.

For every case and curvature, the random parameters, input tangents and output cotangent are
drawn once in FP64 and cast to each tested dtype. Both layers share one manifold and one set
of parameters, and we compare the output and the gradients with respect to input, weights,
bias and curvature. FP32 results are also compared with an FP64 evaluation of HypLL on the
same FP32-rounded values, with HypLL's FP32 projection margin (the FP64 oracle).

    python check_equivalence.py --output equivalence.csv     # first visible CUDA device
"""

import argparse
import contextlib
import csv
import math

import hypll.manifolds.poincare_ball.manifold as ball_module
import torch
from factorized_hconv import factorize_hconvs
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.poincare_ball.math import diffgeom
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


def draw(case, c, device, seed):
    """FP64 parameters, input tangents and output cotangent of one case."""
    _, batch, height, width, cin, cout, stride, regime = case
    generator = torch.Generator(device=device).manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64, device=device)

    weight_scale = 20.0 if regime == "large_weights" else 1.0
    out_size = ((height - 1) // stride + 1, (width - 1) // stride + 1)  # 3x3, padding 1
    tangent = normal(batch, cin, height, width) / math.sqrt(c * cin)  # sqrt(c)||t|| ~ 1
    if regime == "zero_block":
        tangent[:, :, 4:10, 4:10] = 0.0  # gives 4 x 4 all-zero 3 x 3 patches per image
    return {
        "theta": torch.tensor(math.log(math.expm1(c)), dtype=torch.float64),  # softplus^-1
        "weights": normal(9 * cin, cout) * weight_scale / math.sqrt(9 * cin),
        "bias": normal(cout) * 0.25 / math.sqrt(c),
        "tangent": tangent,
        "cotangent": normal(batch, cout, *out_size),
    }


def make_layer(case, values, dtype, device):
    """HypLL HConvolution2d in ``dtype`` with the given parameter values."""
    manifold = PoincareBall(Curvature(1.0, requires_grad=True))
    manifold.c.value.data = values["theta"].to(dtype)  # HypLL stores float32 by default
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # HConvolution2d allocates parameters in the default
    try:
        layer = HConvolution2d(case[4], case[5], 3, manifold, stride=case[6], padding=1)
    finally:
        torch.set_default_dtype(default)
    layer.to(device)
    with torch.no_grad():
        layer.weights.tensor.copy_(values["weights"])
        layer.bias.copy_(values["bias"])
    return layer, manifold


def ball_input(case, tangent, c):
    if case[7] == "beyond_projection_radius":
        # Valid ball points with sqrt(c)||x|| = 1 - 1e-6, outside HypLL's projection radius.
        return tangent / tangent.norm(dim=1, keepdim=True) * (1 - 1e-6) / c.sqrt()
    return diffgeom.expmap0(tangent, c, dim=1)


def run(layer, manifold, x, cotangent):
    x = x.clone().requires_grad_(True)
    y = layer(ManifoldTensor(data=x, manifold=manifold, man_dim=1)).tensor
    leaves = [x, layer.weights.tensor, layer.bias, manifold.c.value]
    grads = torch.autograd.grad(y, leaves, grad_outputs=cotangent)
    names = ["output", "grad_input", "grad_weights", "grad_bias", "grad_curvature"]
    return dict(zip(names, [y.detach(), *grads], strict=True))


@contextlib.contextmanager
def float32_margin():
    """Make HypLL's project use its FP32 margin (4e-3) for FP64 tensors."""
    original = diffgeom.project

    def project(x, c, dim=-1, eps=-1.0):
        return original(x, c, dim=dim, eps=MARGIN[torch.float32] if eps < 0 else eps)

    diffgeom.project = ball_module.project = project
    try:
        yield
    finally:
        diffgeom.project = ball_module.project = original


def cap_fractions(case, manifold, x, y_ref):
    """Fraction of patches at HypLL's source-radius cap and of outputs at the output cap."""
    cin, stride = case[4], case[6]
    c = manifold.c().detach()
    max_norm = (1 - MARGIN[x.dtype]) / (c + 1e-15) ** 0.5
    a = beta_func(9 * cin / 2, 0.5) / beta_func(cin / 2, 0.5)
    patch = unfold(diffgeom.logmap0(x, c, dim=1), 3, padding=1, stride=stride)
    source = torch.tanh(c.sqrt() * a * patch.norm(dim=1)) / c.sqrt() > max_norm
    output = y_ref.norm(dim=1) >= max_norm * (1 - 1e-6)
    return source.double().mean().item(), output.double().mean().item()


def rel_l2(cand, ref):
    return ((cand.double() - ref.double()).norm() / ref.double().norm()).item()


def errors(ref, cand, dtype):
    """Errors over the entries that are finite in both tensors."""
    finite = torch.isfinite(ref) & torch.isfinite(cand)
    ref_finite, diff = ref.double()[finite], (cand.double() - ref.double())[finite]
    ref_l2 = ref_finite.norm().item()
    max_abs = diff.abs().max().item() if diff.numel() else math.nan
    all_finite = bool(finite.all())
    return {
        "numel": ref.numel(),
        "nonfinite_ref": (~torch.isfinite(ref)).sum().item(),
        "nonfinite_factorized": (~torch.isfinite(cand)).sum().item(),
        "nonfinite_mismatch": (torch.isfinite(ref) != torch.isfinite(cand)).sum().item(),
        "ref_l2": ref_l2,
        "ref_max_abs": ref_finite.abs().max().item() if diff.numel() else math.nan,
        "max_abs_error": max_abs,
        "rel_l2_error": diff.norm().item() / ref_l2 if ref_l2 > 0 else math.nan,
        "paper_criterion_pass": all_finite and max_abs <= TOLERANCE[dtype] * (1 + ref_l2),
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
            draws = draw(case, c, device, args.seed + 100 * case_index + int(10 * c))
            for dtype_name, dtype in DTYPES.items():
                values = {key: value.to(dtype) for key, value in draws.items()}
                reference, manifold = make_layer(case, values, dtype, device)
                factorized = factorize_hconvs(nn.ModuleList([reference]))[0]
                x = ball_input(case, values["tangent"], manifold.c().detach())
                ref = run(reference, manifold, x, values["cotangent"])
                fac = run(factorized, manifold, x, values["cotangent"])
                oracle = {}
                if dtype != torch.float64:
                    exact = {key: value.double() for key, value in values.items()}
                    with float32_margin():
                        layer64, manifold64 = make_layer(case, exact, torch.float64, device)
                        oracle = run(layer64, manifold64, x.double(), exact["cotangent"])
                source_cap, output_cap = cap_fractions(case, manifold, x, ref["output"])
                setting = dict(zip(FIELDS, case, strict=True))
                setting |= {"dtype": dtype_name, "curvature": c}
                caps = {
                    "source_cap_fraction": source_cap,
                    "output_cap_fraction": output_cap,
                }
                for tensor in ref:
                    error = errors(ref[tensor], fac[tensor], dtype)
                    if oracle:
                        error["ref_rel_l2_vs_fp64"] = rel_l2(ref[tensor], oracle[tensor])
                        error["fac_rel_l2_vs_fp64"] = rel_l2(fac[tensor], oracle[tensor])
                    rows.append(setting | {"tensor": tensor} | error | caps)
                    print(
                        f"{case[0]:20s} {dtype_name} c={c:<4} {tensor:15s} "
                        f"max_abs={error['max_abs_error']:.3e} "
                        f"rel_l2={error['rel_l2_error']:.3e}",
                        flush=True,
                    )
                del reference, factorized, ref, fac, oracle
                torch.cuda.empty_cache()

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
