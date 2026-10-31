"""The closed forms in `tetris.analytic`, against enumeration, direct
summation and a seeded sample."""

import itertools
from math import comb, isclose

import numpy as np
import pytest

from tetris.analytic import (
    enough_atoms_probability,
    expected_wasted_atoms,
    expected_wasted_atoms_per_row,
    overfull_row_probability,
    salvageable_share,
)


def _direct_waste(loading_array_size: int, columns: int, p: float) -> float:
    """E[(K - columns)+] summed term by term, the definition the closed
    form replaces."""
    return sum(
        (k - columns)
        * comb(loading_array_size, k)
        * p**k
        * (1 - p) ** (loading_array_size - k)
        for k in range(columns + 1, loading_array_size + 1)
    )


@pytest.mark.parametrize("size", [2, 3])
def test_enough_atoms_probability_matches_enumeration(size: int) -> None:
    sites = size * size
    for target_sites in range(sites + 1):
        counted = (
            sum(
                1
                for bits in itertools.product((0, 1), repeat=sites)
                if sum(bits) >= target_sites
            )
            / 2**sites
        )
        assert isclose(
            enough_atoms_probability(size, target_sites), counted, abs_tol=1e-12
        )


def test_overfull_row_probability_matches_enumeration() -> None:
    size = 6
    for columns in range(size + 1):
        counted = (
            sum(
                1
                for bits in itertools.product((0, 1), repeat=size)
                if sum(bits) > columns
            )
            / 2**size
        )
        assert isclose(overfull_row_probability(size, columns), counted, abs_tol=1e-12)


@pytest.mark.parametrize("size", [12, 20])
@pytest.mark.parametrize("p", [0.5, 0.3, 0.75])
def test_expected_waste_matches_direct_summation(size: int, p: float) -> None:
    for columns in range(size + 1):
        assert isclose(
            expected_wasted_atoms_per_row(size, columns, p),
            _direct_waste(size, columns, p),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )


def test_expected_waste_is_the_per_row_figure_times_the_rows() -> None:
    for size in (8, 30, 100):
        columns = int(size / 2**0.5) - 1
        assert isclose(
            expected_wasted_atoms(size, columns),
            size * expected_wasted_atoms_per_row(size, columns),
            rel_tol=1e-12,
        )


def test_expected_waste_matches_a_seeded_sample() -> None:
    size, columns = 12, 8
    rng = np.random.default_rng(20260910)
    draws = rng.integers(0, 2, size=(40_000, size, size)).sum(axis=2)
    surplus = np.clip(draws - columns, 0, None).sum(axis=1)
    error = surplus.std(ddof=1) / np.sqrt(surplus.size)
    assert abs(surplus.mean() - expected_wasted_atoms(size, columns)) < 4 * error


def test_a_row_narrower_than_the_target_wastes_nothing() -> None:
    for size in (4, 17, 60):
        assert expected_wasted_atoms_per_row(size, size) == 0.0
        assert overfull_row_probability(size, size) == 0.0


def test_overfull_row_probability_falls_as_the_target_widens() -> None:
    size = 40
    probabilities = [overfull_row_probability(size, c) for c in range(size + 1)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(probabilities))
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_a_target_nobody_could_fill_is_never_reachable() -> None:
    assert enough_atoms_probability(5, 26) == 0.0
    assert enough_atoms_probability(5, 0) == 1.0


def test_salvageable_share_is_zero_when_the_rate_meets_the_bound() -> None:
    assert salvageable_share(0.8, 0.8) == 0.0
    # a measured rate above its own bound is sampling noise, not a
    # negative share
    assert salvageable_share(0.999664, 0.999624) == 0.0


def test_salvageable_share_splits_the_rejections() -> None:
    assert isclose(salvageable_share(0.5, 0.75), 0.5)
    assert isclose(salvageable_share(0.0, 1.0), 1.0)
    assert salvageable_share(1.0, 1.0) == 0.0


@pytest.mark.parametrize("size, columns", [(0, 1), (-1, 2), (10, -1)])
def test_degenerate_arguments_are_rejected(size: int, columns: int) -> None:
    with pytest.raises(ValueError):
        overfull_row_probability(size, columns)


# --- the large-array tail --------------------------------------------
# The exact tail sums one big-integer term per draw above the threshold,
# which is fine at L = 100 and hopeless at L = 3000, where the loading
# holds nine million sites. A normal approximation takes over. At p = 1/2
# the binomial is symmetric, so its skewness vanishes and the
# continuity-corrected approximation is accurate to O(1/n) rather than
# the usual O(1/sqrt(n)).


@pytest.mark.parametrize("size", [40, 60, 80, 100])
@pytest.mark.parametrize("reshape", [0, -1, -2])
def test_the_approximation_agrees_with_the_exact_tail(size: int, reshape: int) -> None:
    from math import floor, sqrt

    from tetris.analytic import _upper_tail, _upper_tail_normal

    sites = (floor(size / sqrt(2)) + reshape) ** 2
    exact = _upper_tail(size**2, sites)
    assert isclose(_upper_tail_normal(size**2, sites), exact, abs_tol=5e-4)


def test_the_exact_path_is_kept_where_it_is_affordable() -> None:
    from tetris.analytic import EXACT_TAIL_LIMIT, _upper_tail, _upper_tail_normal

    # inside the limit the answer is the exact one, so a small array
    # keeps the arithmetic the rest of the notebook is checked against
    draws = EXACT_TAIL_LIMIT
    assert enough_atoms_probability(int(draws**0.5), draws // 2) == _upper_tail(
        int(draws**0.5) ** 2, draws // 2
    )
    assert _upper_tail(100, 60) != _upper_tail_normal(100, 60)


def test_a_three_thousand_site_array_is_answered_at_all() -> None:
    from math import floor, sqrt
    from time import monotonic

    # nine million draws: the exact sum would run for hours
    started = monotonic()
    value = enough_atoms_probability(3000, floor(3000 / sqrt(2)) ** 2)
    assert monotonic() - started < 1.0
    assert 0.0 < value < 1.0


def test_the_approximation_is_the_continuity_corrected_normal() -> None:
    from math import floor, sqrt
    from statistics import NormalDist

    # against an independent implementation of the same integral rather
    # than against a remembered number
    size = 3000
    sites = floor(size / sqrt(2)) ** 2
    draws = size**2
    normal = NormalDist(draws / 2, sqrt(draws) / 2)
    assert isclose(
        enough_atoms_probability(size, sites),
        1.0 - normal.cdf(sites - 0.5),
        rel_tol=1e-12,
    )


@pytest.mark.parametrize("size", [200, 1000, 3000])
def test_a_target_the_loading_cannot_fill_stays_impossible(size: int) -> None:
    assert enough_atoms_probability(size, size**2) == pytest.approx(0.0, abs=1e-12)


def test_a_target_of_no_sites_is_always_reachable() -> None:
    # the degenerate ends of the tail, which the approximation must
    # answer exactly rather than through its integral
    from tetris.analytic import _upper_tail_normal

    assert _upper_tail_normal(9_000_000, 0) == 1.0
    assert _upper_tail_normal(9_000_000, 9_000_001) == 0.0
