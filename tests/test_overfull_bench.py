"""Smoke tests for the overfull-row benchmark: structure of the result,
and that its invariants (the gap is never negative, keys are omitted
rather than empty when a regime draws no row) hold on a small run."""

import numpy as np
import pytest

from tetris.acceptance import target_geometry
from tetris.overfull_bench import (
    MARGIN_RESHAPE,
    _mean_and_error,
    rows_by_regime,
    run_overfull_benchmark,
)


def test_the_benchmark_uses_the_margin_target() -> None:
    """`k = floor(L / sqrt(2)) - 1`, the `(-1, -1)` reshape."""
    assert MARGIN_RESHAPE == (-1, -1)
    for size in (10, 37, 100):
        assert (
            target_geometry(size, reshape=MARGIN_RESHAPE).columns
            == target_geometry(size).columns - 1
        )


def test_small_sizes_draw_both_regimes() -> None:
    """At L = 4 a half-loaded row often meets or exceeds the target
    window; the benchmark should not report either regime empty there."""
    window = target_geometry(4)
    rng = np.random.default_rng(0)
    exact, overfull = rows_by_regime(rng, 4, window, configs=200)
    assert len(exact) > 0
    assert len(overfull) > 0


def test_benchmark_result_shape_and_invariants() -> None:
    results = run_overfull_benchmark(
        sizes=(4, 6, 300), configs_per_size=200, entropy=42, timing_repeats=3
    )
    assert [r["size"] for r in results] == [4, 6, 300]
    saw_missing_regime = False
    for r in results:
        assert {"size", "exact_rows", "overfull_rows"} <= set(r)
        if r["exact_rows"] > 0:
            assert r["exact_seconds_per_row"] > 0
        else:
            assert "exact_seconds_per_row" not in r
        if r["overfull_rows"] > 0:
            assert r["overfull_seconds_per_row"] > 0
            assert r["mean_absolute_gap"] >= 0
            assert r["mean_relative_gap"] >= 0
            assert 0 <= r["agreement_fraction"] <= 1
            assert r["max_absolute_gap"] >= 0
            assert r["mean_absolute_leftmost_gap"] >= 0
            assert r["mean_relative_leftmost_gap"] >= 0
            assert 0 <= r["leftmost_agreement_fraction"] <= 1
            assert r["max_absolute_leftmost_gap"] >= 0
        else:
            saw_missing_regime = True
            assert "overfull_seconds_per_row" not in r
            assert "agreement_fraction" not in r
            assert "leftmost_agreement_fraction" not in r
    assert saw_missing_regime


def test_overfull_rows_get_rare_at_large_sizes() -> None:
    """Sanity-checks the module's claim: comparable sampling effort
    at a much larger size finds far fewer, or no, overfull rows."""
    window_small = target_geometry(10)
    window_large = target_geometry(200)
    rng = np.random.default_rng(1)
    _, small_overfull = rows_by_regime(rng, 10, window_small, configs=500)
    _, large_overfull = rows_by_regime(rng, 200, window_large, configs=500)
    assert len(large_overfull) < len(small_overfull)


def test_mean_and_error_of_a_sample() -> None:
    mean, error = _mean_and_error([1.0, 2.0, 3.0, 6.0])
    assert mean == 3.0
    assert error == np.std([1.0, 2.0, 3.0, 6.0], ddof=1) / 2


def test_a_single_value_has_no_spread() -> None:
    assert _mean_and_error([4.0]) == (4.0, 0.0)


def test_gaps_carry_standard_errors_and_the_differing_mean() -> None:
    """The mean over the rows that differ is measured directly, and it is
    the overall mean divided by the share of rows that differ."""
    results = run_overfull_benchmark(
        sizes=(10, 20), configs_per_size=300, entropy=7, timing_repeats=1
    )
    for r in results:
        for prefix in ("", "leftmost_"):
            fraction = r[f"{prefix}agreement_fraction"]
            overall = r[f"mean_relative_{prefix}gap"]
            assert r[f"relative_{prefix}gap_standard_error"] >= 0
            if fraction < 1:
                differing = r[f"mean_relative_{prefix}gap_differing"]
                assert differing * (1 - fraction) == pytest.approx(overall)
                assert r[f"relative_{prefix}gap_differing_standard_error"] >= 0
            else:
                assert f"mean_relative_{prefix}gap_differing" not in r
