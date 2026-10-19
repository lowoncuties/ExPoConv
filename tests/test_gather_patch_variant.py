"""Gather-built patch controls Rg/Mg against the convolution-built R/M paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.ad_checkpoint import _optimization_barrier

from poincare_exec.config.constants import EXECUTION_VARIANTS, HYPERBOLIC_VARIANTS
from poincare_exec.core import hnnpp
from poincare_exec.core.hnnpp import conv2d, dense, init_conv2d, init_dense
from poincare_exec.core.operations import transpose_conv2d
from poincare_exec.models import init_resnet20, init_unet, resnet20_forward, unet_forward

jax.config.update("jax_enable_x64", True)

DTYPES = (jnp.float32, jnp.float64)
# R and Rg build bitwise-identical patches, but XLA fuses a convolution-plus-
# transpose producer and a concatenation producer differently into the patch
# reductions, and reverse mode accumulates overlapping-patch cotangents in a
# different order. Agreement is therefore bounded per leaf by k machine epsilons
# of the leaf's largest reference magnitude; scalar leaves are full reductions.
ARRAY_EPSILONS = 8.0
SCALAR_EPSILONS = 64.0
# Through a complete model these rounding differences propagate through every
# layer; small CPU models stayed within 14 epsilons of the largest logit.
MODEL_EPSILONS = 32.0


def _bits(value: jax.Array) -> np.ndarray:
    array = np.asarray(value)
    return array.view(np.dtype(f"u{array.dtype.itemsize}"))


def _assert_bitwise_equal(reference, candidate) -> None:
    reference_leaves = jax.tree_util.tree_leaves(reference)
    candidate_leaves = jax.tree_util.tree_leaves(candidate)
    assert len(reference_leaves) == len(candidate_leaves)
    for reference_leaf, candidate_leaf in zip(
        reference_leaves, candidate_leaves, strict=True
    ):
        assert reference_leaf.dtype == candidate_leaf.dtype
        assert reference_leaf.shape == candidate_leaf.shape
        np.testing.assert_array_equal(_bits(candidate_leaf), _bits(reference_leaf))


def _assert_within_epsilons(
    reference, candidate, *, array_epsilons: float = ARRAY_EPSILONS
) -> None:
    """Apply max|candidate - reference| <= k * eps(dtype) * max|reference| per leaf."""

    for reference_leaf, candidate_leaf in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(candidate),
        strict=True,
    ):
        reference_array = np.asarray(reference_leaf)
        candidate_array = np.asarray(candidate_leaf)
        assert reference_array.dtype == candidate_array.dtype
        assert reference_array.shape == candidate_array.shape
        epsilons = SCALAR_EPSILONS if reference_array.size == 1 else array_epsilons
        scale = np.max(np.abs(reference_array), initial=0.0)
        error = np.max(np.abs(candidate_array - reference_array), initial=0.0)
        assert error <= epsilons * np.finfo(reference_array.dtype).eps * scale


def _patches(tangent, kernel_shape, *, stride, padding, dilation=1):
    options = {
        "stride": (stride, stride),
        "padding": padding,
        "dilation": (dilation, dilation),
    }
    reference = jax.jit(
        lambda value: hnnpp._extract_patches(value, kernel_shape, **options)
    )
    gathered = jax.jit(lambda value: hnnpp._gather_patches(value, kernel_shape, **options))
    return reference(tangent), gathered(tangent)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize(
    ("shape", "kernel_shape"),
    [((2, 7, 7, 3), (3, 3)), ((2, 8, 8, 5), (3, 3)), ((2, 8, 6, 4), (1, 1))],
)
def test_gathered_patch_equals_convolution_patch_bitwise(
    dtype, stride: int, shape: tuple[int, ...], kernel_shape: tuple[int, int]
) -> None:
    tangent = jax.random.normal(jax.random.PRNGKey(3), shape, dtype=dtype)

    reference, gathered = _patches(tangent, kernel_shape, stride=stride, padding="SAME")

    _assert_bitwise_equal(reference, gathered)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize(("padding", "dilation"), [("VALID", 1), ("SAME", 2)])
def test_gathered_patch_follows_valid_padding_and_dilation(
    stride: int, padding: str, dilation: int
) -> None:
    tangent = jax.random.normal(jax.random.PRNGKey(5), (2, 9, 8, 3), dtype=jnp.float64)

    reference, gathered = _patches(
        tangent, (2, 3), stride=stride, padding=padding, dilation=dilation
    )

    _assert_bitwise_equal(reference, gathered)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gathered_patch_keeps_negative_zero_that_the_convolution_drops(dtype) -> None:
    tangent = jax.random.normal(jax.random.PRNGKey(7), (1, 5, 5, 2), dtype=dtype)
    tangent = tangent.at[0, 2, 2, 0].set(-0.0)

    reference, gathered = _patches(tangent, (3, 3), stride=1, padding="SAME")
    reference = np.asarray(reference)
    gathered = np.asarray(gathered)

    np.testing.assert_array_equal(gathered, reference)
    reference_negative_zero = np.signbit(reference) & (reference == 0.0)
    gathered_negative_zero = np.signbit(gathered) & (gathered == 0.0)
    assert not reference_negative_zero.any()
    assert int(gathered_negative_zero.sum()) == 9
    assert np.array_equal(_bits(gathered) != _bits(reference), gathered_negative_zero)


LAYER_CASES = [(1, "SAME"), (2, "SAME"), (2, "VALID")]


def _layer_inputs(dtype):
    key_params, key_input, key_cotangent = jax.random.split(jax.random.PRNGKey(31), 3)
    params = init_conv2d(
        key_params, 3, 5, 6, dtype=dtype, weight_scale=1.0, offset_scale=0.05
    )
    tangent = 0.3 * jax.random.normal(key_input, (2, 8, 8, 5), dtype=dtype)
    return params, tangent, jnp.asarray(0.7, dtype=dtype), key_cotangent


def _layer_forward(variant: str, stride: int, padding: str):
    def forward(params, tangent, curvature):
        return conv2d(
            params, tangent, curvature, variant=variant, stride=stride, padding=padding
        )

    return jax.jit(forward)


def _layer_output_and_gradients(variant: str, dtype, stride: int, padding: str):
    params, tangent, curvature, key_cotangent = _layer_inputs(dtype)
    output, pullback = jax.vjp(
        _layer_forward(variant, stride, padding), params, tangent, curvature
    )
    cotangent = jax.random.normal(key_cotangent, output.shape, dtype=dtype)
    return output, pullback(cotangent)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("stride", "padding"), LAYER_CASES)
def test_mg_matches_m_outputs_and_parameter_gradients_bitwise(
    dtype, stride: int, padding: str
) -> None:
    reference_output, reference_gradients = _layer_output_and_gradients(
        "M", dtype, stride, padding
    )
    candidate_output, candidate_gradients = _layer_output_and_gradients(
        "Mg", dtype, stride, padding
    )
    reference_params, reference_tangent, reference_curvature = reference_gradients
    candidate_params, candidate_tangent, candidate_curvature = candidate_gradients

    _assert_bitwise_equal(reference_output, candidate_output)
    _assert_bitwise_equal(reference_params, candidate_params)
    _assert_bitwise_equal(reference_curvature, candidate_curvature)
    _assert_within_epsilons(reference_tangent, candidate_tangent)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("stride", "padding"), LAYER_CASES)
def test_rg_matches_r_outputs_and_gradients_within_rounding(
    dtype, stride: int, padding: str
) -> None:
    reference = _layer_output_and_gradients("R", dtype, stride, padding)
    candidate = _layer_output_and_gradients("Rg", dtype, stride, padding)

    _assert_within_epsilons(reference, candidate)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("stride", "padding"), LAYER_CASES)
def test_rg_matches_r_bitwise_when_the_patch_is_materialized(
    monkeypatch, dtype, stride: int, padding: str
) -> None:
    """Pin the R/Rg rounding difference on fusion, not on the patch values."""

    for name in ("_extract_patches", "_gather_patches"):
        builder = getattr(hnnpp, name)
        monkeypatch.setattr(
            hnnpp,
            name,
            lambda *args, builder=builder, **kwargs: _optimization_barrier(
                builder(*args, **kwargs)
            ),
        )
    params, tangent, curvature, _ = _layer_inputs(dtype)
    reference_output = _layer_forward("R", stride, padding)(params, tangent, curvature)
    candidate_output = _layer_forward("Rg", stride, padding)(params, tangent, curvature)

    _assert_bitwise_equal(reference_output, candidate_output)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_variants_execute_dense_parts_as_their_sources(dtype) -> None:
    params = init_dense(jax.random.PRNGKey(5), 3, 2 * 2 * 4, dtype=dtype, offset_scale=0.05)
    tangent = 0.1 * jax.random.normal(jax.random.PRNGKey(7), (2, 3, 3, 3), dtype)
    curvature = jnp.asarray(0.7, dtype=dtype)

    def apply_dense(variant: str):
        return dense(params, tangent, curvature, variant=variant)

    def apply_transpose(variant: str):
        return transpose_conv2d(
            params, tangent, curvature, output_channels=4, variant=variant
        )

    _assert_bitwise_equal(apply_dense("R"), apply_dense("Rg"))
    _assert_bitwise_equal(apply_dense("F"), apply_dense("Mg"))
    _assert_bitwise_equal(apply_transpose("R"), apply_transpose("Rg"))
    _assert_bitwise_equal(apply_transpose("F"), apply_transpose("Mg"))


@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_variants_match_complete_model_logits(dtype) -> None:
    """Forward logits only; full-size update steps are compared on the GPU by E31."""

    unet = init_unet(
        jax.random.PRNGKey(13), classes=2, initial_width=2, depth=2, dtype=dtype
    )
    images = jax.random.uniform(jax.random.PRNGKey(17), (2, 16, 16, 3), dtype=dtype)
    resnet = init_resnet20(
        jax.random.PRNGKey(7),
        channels=(2, 4, 8),
        blocks_per_group=(1, 1, 1),
        classes=5,
        dtype=dtype,
    )
    inputs = 0.02 * jax.random.normal(jax.random.PRNGKey(11), (2, 32, 32, 3), dtype)
    curvature = jnp.asarray(0.1, dtype=dtype)

    def unet_logits(variant: str):
        return jax.jit(lambda: unet_forward(unet, images, variant=variant))()

    def resnet_logits(variant: str):
        return jax.jit(
            lambda: resnet20_forward(resnet, inputs, curvature, variant=variant)
        )()

    for logits in (unet_logits, resnet_logits):
        for reference, candidate in (("R", "Rg"), ("M", "Mg")):
            _assert_within_epsilons(
                logits(reference), logits(candidate), array_epsilons=MODEL_EPSILONS
            )


def test_patch_builders_are_routed_by_variant(monkeypatch) -> None:
    params = init_conv2d(jax.random.PRNGKey(1), 3, 2, 3, dtype=jnp.float64)
    tangent = 0.1 * jax.random.normal(jax.random.PRNGKey(2), (1, 5, 5, 2), jnp.float64)
    curvature = jnp.asarray(0.5, dtype=jnp.float64)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected patch builder")

    with monkeypatch.context() as patch:
        patch.setattr(hnnpp, "_gather_patches", forbidden)
        for variant in ("R", "M", "F", "T", "E"):
            conv2d(params, tangent, curvature, variant=variant)
    with monkeypatch.context() as patch:
        patch.setattr(hnnpp, "_extract_patches", forbidden)
        for variant in ("Rg", "Mg", "F", "T", "E"):
            conv2d(params, tangent, curvature, variant=variant)
        with pytest.raises(AssertionError, match="unexpected patch builder"):
            conv2d(params, tangent, curvature, variant="R")


def test_gather_variants_are_benchmark_only_controls() -> None:
    params = init_conv2d(jax.random.PRNGKey(1), 3, 2, 3, dtype=jnp.float64)
    tangent = jnp.zeros((1, 4, 4, 2), dtype=jnp.float64)

    assert HYPERBOLIC_VARIANTS == ("R", "F", "T")
    assert EXECUTION_VARIANTS == ("R", "F", "T", "M", "E", "Rg", "Mg")
    with pytest.raises(ValueError, match="variant must be one of"):
        conv2d(params, tangent, jnp.asarray(0.5), variant="G")
