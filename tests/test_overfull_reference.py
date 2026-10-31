"""`movers_naive` agrees with `_movers` under any run selection;
`movers_sum_optimal` does not agree with the default one,
because minimising the bottleneck and minimising the total are different
objectives on the same set of admissible runs. `movers_leftmost` is
neither: it does not search at all, and can only cost as much as or more
than the bottleneck-optimal run at `_movers`'s own metric.
"""

from itertools import permutations

import numpy as np

from tests.test_run_selection import admissible_range, random_line
from tetris.config import AtomsConfiguration
from tetris.fast import (
    RUN_CRITERIA,
    RunCriterion,
    _movers,
    _run_cumulated_displacement_cost,
    _run_displacement_bottleneck_cost,
)
from tetris.overfull_reference import (
    movers_leftmost,
    movers_naive,
    movers_sum_optimal,
    run_sum_cost,
)

# every ordering of every subset of the criteria, the empty one included
EVERY_RUN_SELECTION = [
    order
    for length in range(len(RUN_CRITERIA) + 1)
    for order in permutations(RUN_CRITERIA, length)
]


def test_naive_matches_the_closed_form_bottleneck() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        naive_first = movers_naive(positions, window_start, count)
        fast_first = _movers(positions, window_start, count)
        assert naive_first == fast_first
        assert _run_displacement_bottleneck_cost(
            positions, window_start, count, naive_first
        ) == _run_displacement_bottleneck_cost(
            positions, window_start, count, fast_first
        )


def test_naive_picks_the_same_run_under_every_order() -> None:
    """Index for index, not only cost for cost: the closed-form search,
    the pair it hands on and the full scan must agree whatever the order
    of the criteria, since each order is its own answer."""
    rng = np.random.default_rng(11)
    with_a_choice = 0
    for _ in range(1500):
        positions, window_start, count = random_line(rng)
        with_a_choice += len(admissible_range(positions, window_start, count)) > 1
        for run_selection in EVERY_RUN_SELECTION:
            assert _movers(
                positions, window_start, count, run_selection
            ) == movers_naive(positions, window_start, count, run_selection), (
                run_selection
            )
    assert with_a_choice > 100, with_a_choice


def test_single_criteria_reproduce_the_reference_rules() -> None:
    rng = np.random.default_rng(12)
    for _ in range(5000):
        positions, window_start, count = random_line(rng)
        assert _movers(
            positions, window_start, count, ("LEFTMOST",)
        ) == movers_leftmost(positions, window_start, count)
        assert _movers(
            positions, window_start, count, ("CUMULATED",)
        ) == movers_sum_optimal(positions, window_start, count)


def test_run_sum_cost_is_the_cumulated_criterion() -> None:
    rng = np.random.default_rng(13)
    for _ in range(2000):
        positions, window_start, count = random_line(rng)
        for first in admissible_range(positions, window_start, count):
            assert run_sum_cost(
                positions, window_start, count, first
            ) == _run_cumulated_displacement_cost(positions, window_start, count, first)


def test_the_class_moves_differently_but_no_slower_under_another_order() -> None:
    """A different order changes which atoms move, never the longest move
    of any step: the rows and columns pay the same bottleneck."""
    rng = np.random.default_rng(14)
    differs = 0
    for _ in range(40):
        matrix = rng.random((24, 24)) < 0.5
        runs = []
        selections: tuple[tuple[RunCriterion, ...], ...] = (
            ("BOTTLENECK", "LEFTMOST"),
            ("BOTTLENECK", "CUMULATED"),
        )
        for run_selection in selections:
            configuration = AtomsConfiguration(
                occupation_matrix=matrix,
                reshape_target=(-1, -1),
                run_selection=run_selection,
            )
            configuration.tetrimino_construct()
            if not configuration.configuration_kept:
                break
            configuration.tetrimino_eliminate()
            assert configuration.tetriminoes_motions is not None
            assert configuration.eliminated_motions is not None
            motions = (
                configuration.tetriminoes_motions + configuration.eliminated_motions
            )
            runs.append(motions)
        if len(runs) < 2:
            continue
        bottlenecks = [
            [max((abs(s - d) for s, d in step), default=0) for step in motions]
            for motions in runs
        ]
        assert bottlenecks[0] == bottlenecks[1]
        differs += runs[0] != runs[1]
    assert differs > 0


def test_sum_optimal_is_the_cheapest_admissible_run_by_total() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        chosen = movers_sum_optimal(positions, window_start, count)
        sums = [
            run_sum_cost(positions, window_start, count, first)
            for first in admissible_range(positions, window_start, count)
        ]
        assert run_sum_cost(positions, window_start, count, chosen) == min(sums)


def test_bottleneck_optimal_can_cost_more_in_total() -> None:
    """The counter-example: `_movers` takes the run with the smaller
    largest displacement (2 against 3), but it is not the run with the
    smaller total displacement (5 against 4)."""
    positions = np.array([0, 1, 2, 4, 8, 9, 10, 11])
    window_start, count = 3, 3

    bottleneck_first = _movers(positions, window_start, count)
    sum_first = movers_sum_optimal(positions, window_start, count)

    assert bottleneck_first != sum_first
    assert _run_displacement_bottleneck_cost(
        positions, window_start, count, bottleneck_first
    ) < _run_displacement_bottleneck_cost(positions, window_start, count, sum_first)
    assert run_sum_cost(
        positions, window_start, count, bottleneck_first
    ) > run_sum_cost(positions, window_start, count, sum_first)


def test_leftmost_is_the_lowest_admissible_start() -> None:
    rng = np.random.default_rng(4)
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        chosen = movers_leftmost(positions, window_start, count)
        assert chosen == min(admissible_range(positions, window_start, count))


def test_leftmost_never_beats_the_bottleneck_optimal_run() -> None:
    """Wang et al.'s rule does not search, so it can only tie or lose
    against `_movers` at the metric `_movers` minimises."""
    rng = np.random.default_rng(5)
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        leftmost_first = movers_leftmost(positions, window_start, count)
        bottleneck_first = _movers(positions, window_start, count)
        assert _run_displacement_bottleneck_cost(
            positions, window_start, count, leftmost_first
        ) >= _run_displacement_bottleneck_cost(
            positions, window_start, count, bottleneck_first
        )


def test_bottleneck_optimal_run_is_often_also_sum_optimal() -> None:
    """The counter-example exists but is not typical: over random lines,
    the bottleneck-optimal run usually is the sum-optimal one too."""
    rng = np.random.default_rng(3)
    agreements = 0
    trials = 20000
    for _ in range(trials):
        positions, window_start, count = random_line(rng)
        bottleneck_first = _movers(positions, window_start, count)
        sum_first = movers_sum_optimal(positions, window_start, count)
        agreements += run_sum_cost(
            positions, window_start, count, bottleneck_first
        ) == run_sum_cost(positions, window_start, count, sum_first)
    assert agreements > trials * 0.5, agreements
