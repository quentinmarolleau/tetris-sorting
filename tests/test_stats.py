"""Interval estimates."""

import numpy as np
import pytest

from tetris.stats import standard_error, wilson_interval


def test_standard_error_shrinks_with_the_sample() -> None:
    assert standard_error(500, 1000) > standard_error(5000, 10000)
    assert standard_error(500, 1000) == pytest.approx(0.5 / np.sqrt(1000))


def test_wilson_stays_inside_the_unit_interval() -> None:
    low, high = wilson_interval(4000, 4000, z=3.0)
    assert 0.0 <= low < 1.0
    assert high <= 1.0


def test_wilson_brackets_the_rate_for_a_balanced_sample() -> None:
    low, high = wilson_interval(500, 1000)
    assert low < 0.5 < high
    assert high - low == pytest.approx(2 * standard_error(500, 1000), rel=0.05)


def test_wilson_is_asymmetric_near_one() -> None:
    rate = 3999 / 4000
    low, high = wilson_interval(3999, 4000)
    assert rate - low > high - rate


def test_empty_sample_is_not_an_error() -> None:
    assert np.isnan(standard_error(0, 0))
    assert all(np.isnan(bound) for bound in wilson_interval(0, 0))


@pytest.mark.parametrize("successes", [0, 1, 1999, 2000])
def test_interval_always_contains_the_observed_rate(successes: int) -> None:
    low, high = wilson_interval(successes, 2000)
    assert low <= successes / 2000 <= high
