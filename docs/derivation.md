# Derivation and implementation map

## Coordinate contract

Let $c>0$ be the curvature magnitude and let $t\in T_0\mathbb{D}_c^d$.
With the source convention,

\[
 \exp_0^c(t)=\frac{\tanh(\sqrt c\lVert t\rVert)}
 {\sqrt c\lVert t\rVert}t,
 \qquad
 \log_0^c(x)=\frac{\operatorname{atanh}(\sqrt c\lVert x\rVert)}
 {\sqrt c\lVert x\rVert}x.
\]

Every external activation in this repository is the tangent coordinate
$t=\log_0^c(x)$. R materializes $x$, F removes patch-point materialization,
and T additionally removes the output-point round trip. All variants retain the
same source projection before any downstream operation.

## Patch factorization

For one convolutional receptive field, flatten the tangent patch after its
dimension-beta scale into $q\in\mathbb{R}^D$. The projected origin exponential
has the radial form $x=a_c(\lVert q\rVert^2)q$, where $a_c$ includes the
source's dtype-specific ball projection. The HNN++ hyperplane score for output
orientation $z_j$, norm-floored at $10^{-15}$, and signed offset $r_j$
depends on the patch only through

\[
 s=\lVert q\rVert^2,
 \qquad
 p_j=\langle q,z_j\rangle.
\]

Let $n_j=\max(\lVert z_j\rVert,10^{-15})$,
$\lambda_x=2/(1-c\lVert x\rVert^2)$,
$w_j=\langle x,z_j\rangle/n_j$, and $\delta_j=2\sqrt c\,r_j$. The literal
source score implemented by R is

\[
 h_c(x,z_j,r_j)=\frac{2n_j}{\sqrt c}\operatorname{asinh}\!\left(
 \sqrt c\,\lambda_x w_j\cosh(\delta_j)
 -(\lambda_x-1)\sinh(\delta_j)\right).
\]

Let $m$ be the source projection margin and define the clipped normalized
radius, conformal factor, and dimensionless oriented product by

\[
 \rho(s)=\min\{\tanh(\sqrt{cs}),1-m\},\qquad
 \lambda(s)=\frac{2}{1-\rho(s)^2},\qquad
 u_j(s,p_j)=\frac{\rho(s)p_j}{\sqrt{s}\,n_j},
\]

where $u_j(0,0)=0$ is the continuous value extension and
$\rho(s)/\sqrt{s}\to\sqrt c$ as $s\downarrow0$. The unchanged source score
becomes

\[
 v_j=\frac{2n_j}{\sqrt c}\operatorname{asinh}\!\left(
 \lambda(s)u_j(s,p_j)\cosh(\delta_j)
 -(\lambda(s)-1)\sinh(\delta_j)\right).
\]

`safe_norm` uses the zero derivative convention at an exact zero vector. The
$n_j=\max(\lVert z_j\rVert,10^{-15})$ floor is continuous but has a branch
kink at the threshold; the derivation does not assert a unique derivative at
that kink.

Thus F computes $p_j$ with one native convolution and computes $s$ with a
single-channel convolution of squared inputs by an all-ones spatial kernel.
It then evaluates the unchanged score. This is algebraically the same map as R,
including stride, padding, dilation, beta scaling, norm floors, and projection.

## Direct tangent output

For the score vector $v$, define $y=\sinh(\sqrt c\,v)$. Before source
projection, the HNN++ output has direction $y/\lVert y\rVert$ and tangent
radius

\[
 r_T=\frac{\operatorname{asinh}(\lVert y\rVert)}{2\sqrt c}.
\]

Therefore T returns

\[
 \log_0^c(x_{out}) = r_T\frac{y}{\lVert y\rVert},
\]

with $r_T$ clipped to the exact tangent radius corresponding to the source
ball-projection margin. For ordinary scores, R/F reconstruct the published
point with the rationalized expression

\[
 x(v)=\frac{y}{\sqrt c\left(1+\sqrt{1+\lVert y\rVert^2}\right)}.
\]

This form and T's direct expression have the continuous extension
$\Phi_c(v)=v/2+O(\lVert v\rVert^3)$, so
$D\Phi_c(0)=I/2$. Squared-norm series are used near zero to avoid
differentiating a normalized-zero direction, while a scaled exponential form
retains large-score stability. Projection uses the identity branch at and
inside the cap and the radial cap outside it; its derivative at the origin is
therefore the identity.

## Dense, ReLU, and beta concatenation

For a dense input $t\in\mathbb R^{d}$, no spatial patch is materialized. The
same score transformation uses

\[
 s=\lVert t\rVert^2,\qquad p_j=z_j^{\mathsf T}t,
\]

implemented by a reduction and native matrix product in
`core/hnnpp.py:dense`. Under the inspected network convention, ReLU is already
an origin-tangent operation,

\[
 t' = \operatorname{ReLU}(t)
 \quad\text{or, in ball notation,}\quad
 x'=\exp_0^c(\operatorname{ReLU}(\log_0^c x)),
\]

and is called directly at the model boundaries. For tangent blocks $t_i$ of
dimensions $d_i$ and $D=\sum_i d_i$, source beta concatenation is

\[
 \operatorname{concat}_{\beta}(t_1,\ldots,t_k)
 =\operatorname{concat}_i\!\left(\frac{\beta_D}{\beta_{d_i}}t_i\right),
 \qquad \beta_d=B(d/2,1/2),
\]

implemented by `core/hnnpp.py:beta_concat_tangent`.

## Supported composition boundaries

| Operator | R | F | T direct-tangent ablation | Preserved boundary/state |
|---|---:|---:|---:|---|
| HNN++ dense | yes | yes | yes | weights, offsets, curvature, projection |
| HNN++ spatial convolution | yes | yes | yes | kernel geometry and beta scale |
| source transpose convolution | yes | yes | partial | FC plus Poincaré fold remains explicit |
| midpoint Poincaré BN | yes | yes | explicit tangent boundary | learned mean/variance; current-batch statistics |
| Möbius residual | yes | yes | explicit tangent boundary | operand order `fr` |
| beta concatenation | yes | yes | yes | dimension-beta factors |
| tangent ReLU/max/average pool | yes | yes | yes | source coordinate operation |

No cancellation is claimed across midpoint BN, Möbius addition, transpose-fold,
or a curvature change. Those operations remain explicit and are included in
complete-model measurements.

## Complexity contract

Let $n=B H_o W_o$ and $K=k_hk_w$. The native dot product in both R and F has
arithmetic cost $O(nKC_iC_o)$, so factorization does not change the leading
convolution order. R additionally computes patch norms and projections in
$O(nKC_i)$ and materializes actual tangent-patch and ball-patch arrays with
$nKC_i$ elements each. Its score and reconstructed-output arrays have
$nC_o$ elements.

F replaces those patch-side arrays with an input-channel squared-norm reduction
of cost $O(BH_iW_iC_i)$, a scalar spatial convolution of cost $O(nK)$, filter
norms of cost $O(KC_iC_o)$, and feature-sized score/reconstruction work of cost
$O(nC_o)$. Its persistent factorization-side values are the scalar input/radius
fields and the native $nC_o$ convolution output. T has the same factorized
patch side as F and avoids the $nC_o$ ball-output round trip at supported
tangent boundaries, without changing the asymptotic feature-sized order.
Native convolution workspaces, compiler fusion, and transient allocation remain
implementation-dependent; E03, E05, E08, and E11 measure them rather than
treating this accounting as a speed or peak-memory guarantee.

## Equation-to-code map and numerical coverage

The table links each equation to its implementation and to the numerical
evidence that exercises it. "Cases" names existing experiment or unit-test
evidence and does not turn a diagnostic row into an acceptance row. The
constructed exact-cap point is nondifferentiable, and T's 33 finite structured
gradient misses remain failures rather than being explained by the derivation.

| Source equation or boundary | Transformed equation | Assumptions and zero/clipping limits | Code function | Existing numerical cases and gradients |
|---|---|---|---|---|
| Origin maps $x=\exp_0^c(t)$, $t=\log_0^c(x)$ and radial source projection $P_{c,m}$ | $\log_0^c(P_{c,m}(\exp_0^c(t)))$ is a tangent radial clamp at $\operatorname{atanh}(1-m)/\sqrt c$ | $c>0$ and $0<m<1$; callers must honor the margin domain because `resolve_projection_margin` checks scalar shape but not range. Implementation defaults are $m=0.004$ in FP32 and $10^{-5}$ in FP64, while accepted E02-REPAIR fixed $m=0.004$ in both precisions; identity branch at and inside the cap; origin derivative is the identity; the exact cap is a constructed kink diagnostic | `core/geometry.py:expmap0`, `logmap0`, `project_ball`, `project_tangent`, `resolve_projection_margin` | `configs/numerics/e02_repair.yaml:projection` differentiates `project_ball` at zero, near-zero, inside, outside, and exact-cap radii in radial/tangential directions. `results/numerics/summary.json` records 11/12 exact-cap diagnostic passes. `tests/test_geometry.py` additionally compares direct tangent projection with the ball round trip for primal, JVP, and reverse-mode derivatives strictly inside/outside the cap and checks the identity origin derivative. |
| Published HNN++ score $h_c(x,z_j,r_j)$ with $\lambda_x=2/(1-c\lVert x\rVert^2)$ | The displayed $v_j(s,p_j,n_j,r_j,c,m)$ equation above, including $\rho(s)$ and the norm floor | $c>0$; $n_j\ge10^{-15}$; $u_j(0,0)=0$; input clipping is represented by the same normalized-radius cap as R | `core/hnnpp.py:hyperplane_scores_ball`, `_scores_from_radius_products`; stable scalar limits in `core/numerics.py` | E02 repaired N088 and structured convolution-stack/small-U-Net rows compare outputs and reverse-mode gradients with respect to inputs, weights, offsets, and curvature in FP32/FP64. All 1,360 F structured comparisons pass; T has 33 reported FP32 small-U-Net gradient misses. There is no isolated factorized input-clipping gradient case or norm-floor gradient oracle. |
| Patch statistics $R_p^2=a^2\sum_{q\in\mathcal P_p}\lVert t_q\rVert^2$ and $D_{pj}=a\sum_{q\in\mathcal P_p}z_{qj}^{\mathsf T}t_q$ | One scalar convolution of $\sum_i t_i^2$ produces $R_p^2$; native convolution produces every $D_{pj}$ | Exact same stride, padding, dilation, HWC patch order, beta ratio, weights, offsets, curvature, norm floor, and projection margin as R | `core/hnnpp.py:_factorized_conv_scores`, `_scores_from_radius_products`, `conv2d` | `configs/numerics/e02_repair_structured.yaml`: eight-layer convolution stack and small U-Net, ten realizations in each precision; gradients cover convolution and transpose-convolution parameters, normalization parameters, inputs, and curvature. Accepted E02 cases use unit stride/dilation. `tests/test_hnnpp.py` separately compares R/F outputs and parameter/input/curvature gradients for nondefault stride and nondefault dilation. |
| Published score-to-ball output $x(v)$ followed by source projection | $y=\sinh(\sqrt c\,v)$ and $x(v)=y/[\sqrt c(1+\sqrt{1+\lVert y\rVert^2})]$ | Continuous extension $x(0)=0$; squared-norm series avoid a normalized-zero derivative; the scaled branch is selected only when $\operatorname{stop\_gradient}(\max_j\lvert\sqrt c\,v_j\rvert)>20$ (equality takes the direct branch); source projection is retained | `core/hnnpp.py:_reconstruct_scores`, `scores_to_ball`; `core/numerics.py:sinhc`, `asinhc_from_square`, `asinh_exp` | `scalar_zero`, `vector_zero`, and `vector_near_zero` in `configs/numerics/e02_repair.yaml` exercise outputs and JVPs; N083/N084 retain clipping-kink and cancellation diagnostics rather than entering the smooth gate. Accepted bounded rows that record `scaled_reconstruction_count` report zero. `tests/test_hnnpp.py` intentionally selects the scaled branch at large positive/negative scores and checks the one-output closed form and JVP. |
| Composition $\log_0^c(x(v))$ at a supported tangent boundary | $t_{out}=\operatorname{asinh}(\lVert y\rVert)y/(2\sqrt c\lVert y\rVert)$, followed by the exactly corresponding tangent-radius clamp | $t_{out}(0)=0$ and $Dt_{out}(0)=I/2$; no cancellation is claimed across midpoint normalization, M\"obius addition, transpose-fold, or curvature change | `core/hnnpp.py:_reconstruct_scores`, `scores_to_projected_tangent`, `_variant_output`; retained boundaries in `core/operations.py` | E02 zero/structured cases test T indirectly through outputs and model gradients. The accepted result is bounded: 1,327/1,360 structured T comparisons pass and all 33 misses are retained. Direct `project_tangent` derivative coverage is a unit-test regression check, not an additional accepted experiment row. |
| Dense score evaluation | $s=\lVert t\rVert^2$ and $p_j=z_j^{\mathsf T}t$ feed the same $v_j$ and output reconstruction | Same curvature, norm floor, projection margin, weights, and offsets as R; no patch beta scaling applies to either dense path | `core/hnnpp.py:dense`, `_scores_from_radius_products`, `_variant_output` | `tests/test_hnnpp.py` checks FP64 primal R/F/T agreement at four curvatures. E05 times backward execution but is not a numerical-equivalence oracle; no accepted isolated dense-gradient comparison exists. |
| Tangent ReLU and source beta concatenation | $t'=\operatorname{ReLU}(t)$ and $\operatorname{concat}_{\beta}(t_1,\ldots,t_k)=\operatorname{concat}_i[(\beta_D/\beta_{d_i})t_i]$ | ReLU stays in the origin tangent space; concatenated dimensions and beta ratios match the source | `jax.nn.relu` in `models/common.py` and `models/resnet.py`; `core/hnnpp.py:beta_concat_tangent` called by `models/unet.py` | Structured small-U-Net comparisons exercise these operations end to end, but there is no isolated accepted numerical row for either operation. |
| Complete controlled U-Net/ResNet boundaries and state | R/F/T share tangent external coordinates while unsupported geometric/stateful operations remain explicit | Identical parameters and batches are required for compatibility checks; fixed-weight agreement does not imply equal accumulated training trajectories | `models/unet.py`, `models/resnet.py`; `core/operations.py` | E06 replays retained U-Net checkpoints through R/F/T for outputs and selected activations. E07-J2 compares logits, loss, full loss-gradient trees, one Adam update and moments, and a 100-update continuation with repeated-R and one-ULP controls. |

Not separately validated by an accepted experiment row:

- factorized-score gradients that intentionally select the input-clipping branch;
- an isolated exact-zero sufficient-statistics score value/JVP case for the
  $u_j(0,0)$ extension;
- an isolated gradient oracle at and around the orientation norm floor; and
- an isolated dense-gradient equivalence comparison.

The equations establish expression equivalence under the stated conventions.
Floating-point agreement, including derivatives at branches and projections,
is established empirically by E02, E06 and E07-J2 within their tested scope
and is not assumed from the derivation.
