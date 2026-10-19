# Upstream convolution execution audit

Audit date: 2026-10-16

This audit records the version-specific source evidence used to position the
standalone JAX reference path (R). It compares patch extraction and score
evaluation only; it does not claim that R reproduces the complete upstream
frameworks, training loops, or every public implementation.

## Pinned sources

| Source | Immutable identity | Relevant evidence |
|---|---|---|
| Hyperbolic U-Net | `bec504e2f4d198ee4c438579b86d72f9d8186d78` | [`unet_parts.py` lines 37--60](https://github.com/swastishreya/Hyperbolic-U-Net/blob/bec504e2f4d198ee4c438579b86d72f9d8186d78/unet/hyperbolic_unet/unet_parts.py#L37-L60) instantiates HypLL `HConvolution2d`; [`environment.yml` lines 36--43](https://github.com/swastishreya/Hyperbolic-U-Net/blob/bec504e2f4d198ee4c438579b86d72f9d8186d78/environment.yml#L36-L43) pins `hypll==0.1.1`. |
| HypLL | `99a6d89e15be7169da5c339fc647d6115b063e98` | [`convolution.py` lines 101--107](https://github.com/maxvanspengler/hyperbolic_learning_library/blob/99a6d89e15be7169da5c339fc647d6115b063e98/hypll/nn/modules/convolution.py#L101-L107) calls `manifold.unfold` before `fully_connected`. [`manifold.py` lines 244--274](https://github.com/maxvanspengler/hyperbolic_learning_library/blob/99a6d89e15be7169da5c339fc647d6115b063e98/hypll/manifolds/poincare_ball/manifold.py#L244-L274) applies the origin log map and beta scaling, materializes the complete patch with PyTorch `unfold`, and applies the origin exponential map. |
| Poincare ResNet | `fbbbbf890571115444b89b114f9ed52d128d5aa3` | [`convolution.py` lines 35--40](https://github.com/maxvanspengler/poincare-resnet/blob/fbbbbf890571115444b89b114f9ed52d128d5aa3/models/nn/modules/convolution.py#L35-L40) constructs `nn.Unfold`; [lines 94--109](https://github.com/maxvanspengler/poincare-resnet/blob/fbbbbf890571115444b89b114f9ed52d128d5aa3/models/nn/modules/convolution.py#L94-L109) execute beta scaling, unfolding, `expmap0`, and `fully_connected` in that order. |
| Local HNN++ reference | `28737e22822562ac18d5ea03f8c3e3929e945a83` | Frozen by `protocol.yaml`; used to resolve the supported score definition. |

The [PyPI HypLL 0.1.1 source distribution](https://pypi.org/project/hypll/0.1.1/#files)
inspected for the Hyperbolic U-Net pin has SHA-256
`6d76aa088c9ad33f48ec0dbec5cf3ec5bb4bfa60b64104e740aa0d7a24a087ec`.
Its relevant layer sources match the audited HypLL revision above.

## Finding and scope

Both named official execution paths explicitly materialize the beta-scaled
receptive-field vector, map that complete vector to the Poincare ball, and
then evaluate the HNN++ fully connected score. R preserves this
patch-extraction and score-evaluation order in standalone JAX. F instead
computes directional products and patch energy without constructing the
ball-valued patch tensor.

Neither audited official revision contains the patch-statistics
factorization. This supports a comparison against those pinned upstream
paths, not an exhaustive priority claim over forks, unpublished code, or
independent implementations.
