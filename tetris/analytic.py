"""Closed forms for two questions the benchmark would otherwise have to
sample.

How many atoms does a loading waste? A row hands at most one atom to
each target column, so a row carrying more atoms than the target is wide
leaves the surplus where it lies. The count is a binomial tail.

How many discarded loadings held enough atoms to have filled the target
anyway? An accepted loading always does, so the share of rejections that
were salvageable follows from the measured success rate and a second
binomial tail, over the whole array this time. Nothing has to be drawn.

`total >= sites` is what an assembler free to move any atom anywhere
would need. A row-then-column scheme is more constrained than that, so
the share computed here is an upper bound on what any change to the
packing rule could recover.

Every function takes the loading array side L and a filling probability
p, each site being loaded independently, so a row holds K ~ Bin(L, p)
atoms and the whole array Bin(L^2, p).
"""

from functools import cache
from math import comb, erfc, exp, lgamma, log, sqrt

__all__ = [
    "EXACT_TAIL_LIMIT",
    "enough_atoms_probability",
    "expected_wasted_atoms",
    "expected_wasted_atoms_per_row",
    "overfull_row_probability",
    "salvageable_share",
]


def _check(draws: int, threshold: int) -> None:
    if draws < 1:
        raise ValueError(f"a binomial needs at least one draw, got {draws}")
    if threshold < 0:
        raise ValueError(f"a count cannot be negative, got {threshold}")


# above this many draws the exact tail gives way to the approximation.
# The exact sum costs one big-integer multiply-divide per draw above the
# threshold, on integers that are draws bits wide: 20 ms at the 10^4
# sites of an L = 100 loading, 1.9 s at L = 316, and hours at the 9x10^6
# of L = 3000. The limit sits exactly at L = 100 so that every figure
# drawn before this sweep existed keeps the arithmetic it was drawn
# with, and the approximation, already accurate to 3e-6 there and
# better above, covers the rest.
EXACT_TAIL_LIMIT = 10_000


def _upper_tail_normal(draws: int, threshold: int, p: float = 0.5) -> float:
    """P(X >= threshold) for X ~ Bin(draws, p), through the normal
    approximation with a continuity correction.

    At p = 1/2 the binomial is symmetric, so the leading Edgeworth
    correction, the one carrying the skewness, is identically zero and
    the error falls as 1/draws rather than as its square root. At the
    nine million draws of an L = 3000 loading that is far below the
    sampling error of anything it gets compared against.
    """
    if threshold <= 0:
        return 1.0
    if threshold > draws:
        return 0.0
    mean = draws * p
    deviation = sqrt(draws * p * (1.0 - p))
    return 0.5 * erfc((threshold - 0.5 - mean) / (deviation * sqrt(2.0)))


@cache
def _upper_tail(draws: int, threshold: int, p: float = 0.5) -> float:
    """P(X >= threshold) for X ~ Bin(draws, p).

    At p = 1/2 the tail is a sum of binomial coefficients over a power of
    two, so it is accumulated in exact integers and divided once. The
    coefficients come from the recurrence rather than from `comb` per
    term: C(n, j+1) = C(n, j) (n-j) / (j+1) divides exactly, and one
    multiply-divide beats recomputing a factorial ratio when the sum runs
    to thousands of terms. Any other p goes through logs.
    """
    if threshold <= 0:
        return 1.0
    if threshold > draws:
        return 0.0

    if p == 0.5:
        term = comb(draws, threshold)
        total = term
        for j in range(threshold, draws):
            term = term * (draws - j) // (j + 1)
            total += term
        return total / 2**draws

    log_p, log_q = log(p), log(1.0 - p)
    log_choose = lgamma(draws + 1)
    return sum(
        exp(
            log_choose
            - lgamma(j + 1)
            - lgamma(draws - j + 1)
            + j * log_p
            + (draws - j) * log_q
        )
        for j in range(threshold, draws + 1)
    )


def overfull_row_probability(
    loading_array_size: int, columns: int, p: float = 0.5
) -> float:
    """Probability that one row holds more atoms than the target has
    columns, P(K > columns) for K ~ Bin(L, p).

    Such a row is overfull: its surplus atoms cannot be placed in the
    target and stay where they are.

    Args:
        loading_array_size: Side L of the loading array.
        columns: Number of target columns.
        p: Filling probability of a site.

    Returns:
        The probability.

    Raises:
        ValueError: If L is below one or `columns` is negative.
    """
    _check(loading_array_size, columns)
    return _upper_tail(loading_array_size, columns + 1, p)


def expected_wasted_atoms_per_row(
    loading_array_size: int, columns: int, p: float = 0.5
) -> float:
    """Mean surplus of one row, E[(K - columns)+] for K ~ Bin(L, p).

    Splitting the expectation at the threshold gives
    E[(K-m)+] = E[K 1{K>m}] - m P(K>m), and the first term is
    L p P(Bin(L-1, p) >= m). Two tails instead of a sum over the tail.

    Args:
        loading_array_size: Side L of the loading array.
        columns: Number of target columns m.
        p: Filling probability of a site.

    Returns:
        The expected number of atoms the row leaves outside the target.

    Raises:
        ValueError: If L is below one or `columns` is negative.
    """
    _check(loading_array_size, columns)
    if columns >= loading_array_size:
        return 0.0
    carried = loading_array_size * p * _upper_tail(loading_array_size - 1, columns, p)
    return carried - columns * _upper_tail(loading_array_size, columns + 1, p)


def expected_wasted_atoms(
    loading_array_size: int, columns: int, p: float = 0.5
) -> float:
    """Mean number of atoms a whole loading leaves outside the target,
    L times `expected_wasted_atoms_per_row`.

    Args:
        loading_array_size: Side L of the loading array.
        columns: Number of target columns.
        p: Filling probability of a site.

    Returns:
        The expected surplus summed over the L rows.

    Raises:
        ValueError: If L is below one or `columns` is negative.
    """
    return loading_array_size * expected_wasted_atoms_per_row(
        loading_array_size, columns, p
    )


def enough_atoms_probability(
    loading_array_size: int, target_sites: int, p: float = 0.5
) -> float:
    """Probability that a loading holds at least as many atoms as the
    target has sites, P(N >= target_sites) for N ~ Bin(L^2, p).

    Any rearrangement needs that many atoms, so this bounds the success
    rate of the Tetris acceptance rule from above. The tail is exact up to
    `EXACT_TAIL_LIMIT` sites and uses the normal approximation with a
    continuity correction above.

    Args:
        loading_array_size: Side L of the loading array.
        target_sites: Number of target sites.
        p: Filling probability of a site.

    Returns:
        The probability.

    Raises:
        ValueError: If L is below one or `target_sites` is negative.
    """
    _check(loading_array_size, target_sites)
    draws = loading_array_size**2
    if draws > EXACT_TAIL_LIMIT:
        return _upper_tail_normal(draws, target_sites, p)
    return _upper_tail(draws, target_sites, p)


def salvageable_share(success_rate: float, enough_atoms: float) -> float:
    """Share of the discarded loadings that held enough atoms to have
    filled the target.

    An accepted loading always holds enough, so the two events nest and
    the joint probability is the difference of the marginals. A measured
    rate sitting a hair above its own bound is sampling noise on a rate
    of 0.9997, not a negative share, so the result is clamped.

    Args:
        success_rate: Measured fraction of loadings kept.
        enough_atoms: `enough_atoms_probability` for the same target.

    Returns:
        (enough_atoms - success_rate) / (1 - success_rate), clamped to
        [0, 1]; zero when no loading was discarded.
    """
    rejected = 1.0 - success_rate
    if rejected <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (enough_atoms - success_rate) / rejected))
