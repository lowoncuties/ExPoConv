"""Small, explicit statistics requested during manuscript review.

The functions in this module are intentionally independent of manuscript
rendering.  They operate on native-scale paired observations and on arrays
that are still resident during numerical experiments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import t as student_t

ELEMENTWISE_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)


def elementwise_relative_error_summary(
    reference: Any,
    candidate: Any,
    denominator_floor: float,
    *,
    quantiles: Sequence[float] = ELEMENTWISE_QUANTILES,
) -> dict[str, Any]:
    """Summarize ``abs(candidate-reference) / max(abs(reference), floor)``.

    The denominator floor makes zero and near-zero reference entries finite.
    Its influence is not hidden: the returned count and fraction identify how
    many denominators were floored.  Quantiles use NumPy's linear definition.
    """

    if not math.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")
    requested = tuple(float(value) for value in quantiles)
    if not requested or any(not 0.0 <= value <= 1.0 for value in requested):
        raise ValueError("quantiles must be nonempty probabilities in [0, 1]")
    if tuple(sorted(set(requested))) != requested:
        raise ValueError("quantiles must be strictly increasing")

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("reference and candidate shapes differ")
    if reference_array.size == 0:
        raise ValueError("elementwise summaries require at least one element")
    finite = bool(np.isfinite(reference_array).all() and np.isfinite(candidate_array).all())
    prefix = "elementwise_relative_error"
    if not finite:
        return {
            f"{prefix}_definition": (
                "abs(candidate-reference)/max(abs(reference),denominator_floor)"
            ),
            f"{prefix}_quantile_method": "linear",
            f"{prefix}_denominator_floor": denominator_floor,
            f"{prefix}_elements": int(reference_array.size),
            f"{prefix}_denominator_floored_elements": None,
            f"{prefix}_denominator_floored_fraction": None,
            **{f"{prefix}_{_quantile_label(value)}": None for value in requested},
            f"{prefix}_max": None,
        }

    absolute_reference = np.abs(reference_array).ravel()
    floored = absolute_reference < denominator_floor
    relative_error = np.abs(candidate_array - reference_array).ravel() / np.maximum(
        absolute_reference, denominator_floor
    )
    return {
        f"{prefix}_definition": (
            "abs(candidate-reference)/max(abs(reference),denominator_floor)"
        ),
        f"{prefix}_quantile_method": "linear",
        f"{prefix}_denominator_floor": denominator_floor,
        f"{prefix}_elements": int(relative_error.size),
        f"{prefix}_denominator_floored_elements": int(np.count_nonzero(floored)),
        f"{prefix}_denominator_floored_fraction": float(np.mean(floored)),
        **{
            f"{prefix}_{_quantile_label(probability)}": float(
                np.quantile(relative_error, probability, method="linear")
            )
            for probability in requested
        },
        f"{prefix}_max": float(np.max(relative_error)),
    }


def _quantile_label(probability: float) -> str:
    value = probability * 100.0
    if value.is_integer():
        return f"p{int(value)}"
    return f"p{str(value).replace('.', '_')}"


def paired_tost(
    differences: Sequence[float],
    margin: float,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run a paired two-one-sided t test for bounds ``(-margin, margin)``.

    ``differences`` are candidate minus reference on the metric's native
    zero-to-one scale.  Equivalence is concluded only when both one-sided null
    hypotheses are rejected at ``alpha``; equivalently, the ``1-2*alpha``
    confidence interval must lie strictly inside the stated bounds.
    """

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired TOST requires at least two one-dimensional pairs")
    if not np.isfinite(values).all():
        raise ValueError("paired TOST differences must be finite")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("TOST margin must be finite and positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("TOST alpha must be between zero and one half")

    pairs = len(values)
    degrees_of_freedom = pairs - 1
    mean = float(np.mean(values))
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(pairs)
    if standard_error == 0.0:
        lower_t = math.inf if mean > -margin else -math.inf
        upper_t = -math.inf if mean < margin else math.inf
        lower_p = 0.0 if mean > -margin else 1.0
        upper_p = 0.0 if mean < margin else 1.0
        lower = mean
        upper = mean
    else:
        lower_t = (mean + margin) / standard_error
        upper_t = (mean - margin) / standard_error
        lower_p = float(student_t.sf(lower_t, df=degrees_of_freedom))
        upper_p = float(student_t.cdf(upper_t, df=degrees_of_freedom))
        critical = float(student_t.ppf(1.0 - alpha, df=degrees_of_freedom))
        lower = mean - critical * standard_error
        upper = mean + critical * standard_error
    tost_p = max(lower_p, upper_p)
    equivalent = bool(lower_p < alpha and upper_p < alpha)
    return {
        "pairs": pairs,
        "mean_difference": mean,
        "difference_sample_sd": sample_sd,
        "difference_standard_error": standard_error,
        "equivalence_lower_bound": -margin,
        "equivalence_upper_bound": margin,
        "alpha": alpha,
        "confidence_level": 1.0 - 2.0 * alpha,
        "tost_lower_t": lower_t,
        "tost_lower_pvalue": lower_p,
        "tost_upper_t": upper_t,
        "tost_upper_pvalue": upper_p,
        "tost_pvalue": tost_p,
        "equivalence_ci_lower": lower,
        "equivalence_ci_upper": upper,
        "degrees_of_freedom": degrees_of_freedom,
        "equivalent_at_alpha": equivalent,
    }


def paired_mean_interval(
    differences: Sequence[float], *, confidence: float = 0.95
) -> dict[str, float | int]:
    """Return a conventional paired t interval for descriptive summaries."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired interval requires at least two finite differences")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(len(values))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    mean = float(np.mean(values))
    return {
        "pairs": len(values),
        "mean_difference": mean,
        "difference_sample_sd": sample_sd,
        "difference_standard_error": standard_error,
        "paired_t_interval_confidence": confidence,
        "paired_t_interval_lower": mean - critical * standard_error,
        "paired_t_interval_upper": mean + critical * standard_error,
        "paired_t_degrees_of_freedom": len(values) - 1,
    }
