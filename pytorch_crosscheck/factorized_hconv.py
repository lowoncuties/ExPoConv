"""Factorized HNN++ Poincare convolution, a drop-in replacement for HypLL's HConvolution2d.

HypLL 0.1.1 (``hypll.nn.HConvolution2d``) maps each input pixel to the tangent space at the
origin, scales it by a = B(K*C_in/2, 1/2) / B(C_in/2, 1/2), unfolds the full patch P, maps P
to the ball with expmap0 and applies the Poincare fully connected layer. Because expmap0 is
radial, x_P = kappa(||P||) * P, so the hyperplane score needs P only through

    ||P||^2  = a^2 * box_filter(sum_c t_c^2)      (all-ones kernel, same stride and padding)
    z_j^T P  = a * conv2d(t, z_j)                 (native convolution with reshaped weights)

This module computes these two statistics and then evaluates HypLL's own score and output
formulas (``poincare_hyperplane_dists`` and ``poincare_fully_connected`` in
hypll/manifolds/poincare_ball/math/linalg.py) with HypLL's clamps and projection margins.
The ball-valued patch tensor of size B x K*C_in x H_out*W_out is never built.
"""

import torch
from hypll.manifolds.poincare_ball.math.diffgeom import expmap0, logmap0
from hypll.nn import HConvolution2d
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match
from hypll.utils.math import beta_func
from torch import nn
from torch.nn.functional import conv2d


class FactorizedHConvolution2d(nn.Module):
    """Drop-in for HConvolution2d with the same arguments, parameters and interface."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        manifold,
        bias=True,
        stride=1,
        padding=0,
        id_init=True,
    ):
        super().__init__()
        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.kernel_vol = kernel_size, kernel_size[0] * kernel_size[1]
        self.manifold, self.stride, self.padding = manifold, stride, padding
        self.id_init, self.has_bias = id_init, bias  # id_init is unused, as in HypLL 0.1.1
        # weights (K*C_in, C_out): rows in torch unfold order (channel, ky, kx).
        self.weights, self.bias = manifold.construct_dl_parameters(
            in_features=self.kernel_vol * in_channels, out_features=out_channels, bias=bias
        )
        manifold.reset_parameters(weight=self.weights, bias=self.bias)
        beta_patch = beta_func(self.kernel_vol * in_channels / 2, 1 / 2)
        self.beta_ratio = beta_patch / beta_func(in_channels / 2, 1 / 2)  # a in unfold

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=1, input=x)
        c = self.manifold.c()
        c_sqrt = c.sqrt()
        a = self.beta_ratio
        t = logmap0(x.tensor, c, dim=1)  # (B, C_in, H, W)

        # Patch statistics by native convolution: z_j^T P and ||P||^2.
        z = self.weights.tensor  # (K*C_in, C_out)
        kernel = z.t().reshape(self.out_channels, self.in_channels, *self.kernel_size)
        dots = a * conv2d(t, kernel, stride=self.stride, padding=self.padding)
        energy = t.pow(2).sum(dim=1, keepdim=True)
        box = t.new_ones(1, 1, *self.kernel_size)
        radius_sq = a**2 * conv2d(energy, box, stride=self.stride, padding=self.padding)

        # Ball radius ||expmap0(P)|| from HypLL's expmap0 applied to the one-element vector
        # (||P||,), with its 1e-15 norm floor and projection margin. The 1e-30 floor matches
        # that norm floor and keeps sqrt differentiable for an all-zero patch.
        radius = radius_sq.clamp_min(1e-30).sqrt()
        ball_radius = expmap0(radius.unsqueeze(-1), c, dim=-1).squeeze(-1)
        ball_dots = ball_radius / radius * dots  # z_j^T expmap0(P)

        # poincare_hyperplane_dists with x^T z -> ball_dots and ||x||^2 -> ball_radius^2.
        # Same operation order as HypLL; without bias HypLL also omits the offset terms.
        lam = 2 / (1 - c * ball_radius.pow(2))
        z_norm = z.norm(dim=0).clamp_min(1e-15).view(1, -1, 1, 1)
        arg = c_sqrt * lam / z_norm * ball_dots
        if self.bias is not None:
            two_csqrt_r = (2.0 * c_sqrt * self.bias).view(1, -1, 1, 1)
            arg = arg * two_csqrt_r.cosh() - (lam - 1) * two_csqrt_r.sinh()
        v = 2 * z_norm / c_sqrt * torch.asinh(arg)

        # poincare_fully_connected output map followed by PoincareBall.project.
        y = (c_sqrt * v).sinh() / c_sqrt
        y = y / (1 + (1 + c * y.pow(2).sum(dim=1, keepdim=True)).sqrt())
        return self.manifold.project(ManifoldTensor(y, manifold=self.manifold, man_dim=1))


def factorize_hconvs(module: nn.Module) -> nn.Module:
    """Replace every HConvolution2d inside ``module`` in place, sharing its parameters."""
    for name, child in module.named_children():
        if isinstance(child, HConvolution2d):
            new = FactorizedHConvolution2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.manifold,
                bias=child.has_bias,
                stride=child.stride,
                padding=child.padding,
                id_init=child.id_init,
            )
            new.weights, new.bias = child.weights, child.bias
            setattr(module, name, new)
        else:
            factorize_hconvs(child)
    return module
