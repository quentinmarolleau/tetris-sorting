"""Uncertainties on the success rates the benchmark measures.

A success rate is a binomial proportion: `successes` kept loadings out
of `samples` independent draws.
"""

import numpy as np

__all__ = ["standard_error", "wilson_interval"]


def standard_error(successes: int, samples: int) -> float:
    """Standard error of a binomial proportion, sqrt(p (1 - p) / n).

    This is the uncertainty on the rate p, not the standard deviation of
    the individual Bernoulli draws, which is larger by sqrt(n) and says
    nothing about how well p is known. It collapses to zero at p = 0 or
    p = 1; use `wilson_interval` near those limits.

    Args:
        successes: Number of successes.
        samples: Number of trials n.

    Returns:
        The standard error, or NaN when there are no trials.
    """
    if samples <= 0:
        return float("nan")
    rate = successes / samples
    return float(np.sqrt(rate * (1.0 - rate) / samples))


def wilson_interval(
    successes: int, samples: int, z: float = 1.0
) -> tuple[float, float]:
    """Wilson score interval on a binomial proportion.

    The success rates here run to within a few parts in ten thousand of
    one, where the symmetric interval p +/- z sigma would reach past one
    and understate how much room is left below. The Wilson interval stays
    inside [0, 1] and keeps its coverage at those extremes.

    Args:
        successes: Number of successes.
        samples: Number of trials.
        z: Half-width in standard deviations; 1 gives about 68 % coverage.

    Returns:
        (low, high) bounds, always containing the observed rate, or a pair
        of NaN when there are no trials.
    """
    if samples <= 0:
        return float("nan"), float("nan")
    rate = successes / samples
    denominator = 1.0 + z**2 / samples
    centre = (rate + z**2 / (2 * samples)) / denominator
    spread = (
        z * np.sqrt(rate * (1 - rate) / samples + z**2 / (4 * samples**2)) / denominator
    )
    low = float(max(0.0, centre - spread))
    high = float(min(1.0, centre + spread))
    # at a rate of exactly zero or one the analytic bound lands on the
    # observed rate, and rounding can put it a fraction on the wrong
    # side; an interval that excludes its own point estimate is worse
    # than one that is a few ulps wide
    return min(low, rate), max(high, rate)
