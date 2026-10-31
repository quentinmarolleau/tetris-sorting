"""Which run of atoms an overfull line hands to the target window.

A line carrying more atoms than the window has sites gets a choice. All
of its atoms move at once, so the move is collision-free exactly when the
final order is the initial order, which forces the atoms landing in the
window to be a run of consecutive loaded ones. The only freedom left is
where that run starts, and the runs are not equally cheap.
"""

import numpy as np
import pytest

from tetris.config import AtomsConfiguration
from tetris.fast import (
    DEFAULT_RUN_SELECTION,
    RUN_CRITERIA,
    RunCriterion,
    _movers,
    _run_cumulated_displacement_cost,
    _run_displacement_bottleneck_cost,
    _run_moved_atoms_cost,
    _window_displacement,
    validate_run_selection,
)

# The row at index 5 of the cartoon drawn at seed 3238: seven atoms in a
# row of nine, symmetric about site 4, against a window of five columns
# starting at column 2. Centring the run on the window costs one move of
# one site; either end costs two.
CARTOON_ROW = np.array([0, 1, 2, 4, 6, 7, 8])
CARTOON_START = 2
CARTOON_COLUMNS = 5


def admissible_range(positions: np.ndarray, window_start: int, count: int) -> range:
    """Run starts that leave every surplus atom outside the window."""
    lowest = max(0, int(np.searchsorted(positions, window_start + count)) - count)
    highest = min(positions.size - count, int(np.searchsorted(positions, window_start)))
    return range(lowest, highest + 1)


def scan_costs(positions: np.ndarray, window_start: int, count: int) -> list[int]:
    """Brute-force cost of every admissible run, straight from the
    definition rather than from the closed form under test."""
    destinations = np.arange(window_start, window_start + count)
    return [
        int(np.abs(positions[first : first + count] - destinations).max())
        for first in admissible_range(positions, window_start, count)
    ]


def random_line(rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """A loaded line with at least as many atoms as the window it feeds,
    and the window that feeds off it."""
    while True:
        size = int(rng.integers(4, 60))
        count = int(rng.integers(1, size + 1))
        positions = np.flatnonzero(rng.random(size) < rng.uniform(0.2, 0.95))
        if positions.size < count:
            continue
        return positions, int(rng.integers(0, size - count + 1)), count


def test_the_cartoon_row_centres_its_run() -> None:
    """The example that raised the question: the shipped rule took the
    rightmost run and paid two sites of travel where one was enough."""
    assert scan_costs(CARTOON_ROW, CARTOON_START, CARTOON_COLUMNS) == [2, 1, 2]
    assert _movers(CARTOON_ROW, CARTOON_START, CARTOON_COLUMNS) == 1
    assert _window_displacement(CARTOON_ROW, CARTOON_START, CARTOON_COLUMNS) == 1


def test_a_line_with_no_surplus_has_one_admissible_run() -> None:
    """With exactly as many atoms as sites there is nothing to choose, so
    no run selection can change the answer."""
    positions = np.array([0, 3, 5, 6])
    selections: tuple[tuple[RunCriterion, ...], ...] = (
        (),
        ("LEFTMOST",),
        ("ATOMS", "CUMULATED"),
        DEFAULT_RUN_SELECTION,
    )
    for run_selection in selections:
        assert _movers(positions, 1, 4, run_selection) == 0
    assert _window_displacement(positions, 1, 4) == 2


# Three lines against the same window, sites 10 to 12, each with two
# admissible runs that tie on the bottleneck. They were found by brute
# force over short lines, one per stage of the default order, and in the
# first two the later stage picks the right-hand run, so that the leftmost
# fallback cannot pass for it.
TIE_WINDOW_START = 10
TIE_K = 3
TIE_BROKEN_BY_CUMULATED = np.array([1, 2, 3, 21])
TIE_BROKEN_BY_ATOMS = np.array([0, 10, 13, 22])
TIE_SURVIVING_EVERY_CRITERION = np.array([1, 2, 13, 21])


def criteria_of(positions: np.ndarray) -> list[tuple[int, int, int]]:
    """(bottleneck, cumulated, moved atoms) of runs 0 and 1."""
    return [
        (
            _run_displacement_bottleneck_cost(positions, TIE_WINDOW_START, TIE_K, j),
            _run_cumulated_displacement_cost(positions, TIE_WINDOW_START, TIE_K, j),
            _run_moved_atoms_cost(positions, TIE_WINDOW_START, TIE_K, j),
        )
        for j in (0, 1)
    ]


def test_a_bottleneck_tie_goes_to_the_smaller_cumulated_displacement() -> None:
    assert criteria_of(TIE_BROKEN_BY_CUMULATED) == [(9, 27, 3), (9, 25, 3)]
    assert _movers(TIE_BROKEN_BY_CUMULATED, TIE_WINDOW_START, TIE_K) == 1


def test_a_cumulated_tie_goes_to_fewer_moved_atoms() -> None:
    assert criteria_of(TIE_BROKEN_BY_ATOMS) == [(10, 12, 3), (10, 12, 2)]
    assert _movers(TIE_BROKEN_BY_ATOMS, TIE_WINDOW_START, TIE_K) == 1


def test_a_tie_on_every_criterion_goes_to_the_leftmost_run() -> None:
    assert criteria_of(TIE_SURVIVING_EVERY_CRITERION) == [(9, 19, 3), (9, 19, 3)]
    assert _movers(TIE_SURVIVING_EVERY_CRITERION, TIE_WINDOW_START, TIE_K) == 0


def test_the_bottleneck_alone_settles_a_tie_leftmost() -> None:
    """Without further criteria nothing distinguishes the two runs, and
    the leftmost wins, where the old strict comparison took the right."""
    assert (
        _movers(TIE_BROKEN_BY_CUMULATED, TIE_WINDOW_START, TIE_K, ("BOTTLENECK",)) == 0
    )


def test_the_default_order_is_every_criterion_once() -> None:
    assert sorted(DEFAULT_RUN_SELECTION) == sorted(RUN_CRITERIA)
    assert DEFAULT_RUN_SELECTION[0] == "BOTTLENECK"
    assert DEFAULT_RUN_SELECTION[-1] == "LEFTMOST"


@pytest.mark.parametrize(
    ("run_selection", "error"),
    [
        (("BOTTLENECK", "FASTEST"), ValueError),
        (("ATOMS", "BOTTLENECK", "ATOMS"), ValueError),
        ("BOTTLENECK", TypeError),
    ],
)
def test_a_malformed_run_selection_is_refused(run_selection, error) -> None:
    with pytest.raises(error):
        validate_run_selection(run_selection)
    with pytest.raises(error):
        AtomsConfiguration(loading_array_size=8, run_selection=run_selection)


def test_an_empty_run_selection_is_valid() -> None:
    assert validate_run_selection([]) == ()
    assert validate_run_selection(["LEFTMOST", "ATOMS"]) == ("LEFTMOST", "ATOMS")


def test_the_chosen_run_is_admissible() -> None:
    """A run that left a surplus atom inside the window would put two
    atoms on one site."""
    rng = np.random.default_rng(4242)
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        first = _movers(positions, window_start, count)
        assert first in admissible_range(positions, window_start, count)


def test_the_chosen_run_is_the_cheapest_admissible_one() -> None:
    rng = np.random.default_rng(909)
    with_a_choice = 0
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        costs = scan_costs(positions, window_start, count)
        with_a_choice += len(costs) > 1
        assert _window_displacement(positions, window_start, count) == min(costs)
    assert with_a_choice > 1000, with_a_choice


def test_every_order_that_leads_with_the_bottleneck_keeps_its_cost() -> None:
    """The later criteria only ever choose between runs of equal
    bottleneck, so they cannot make the move any slower."""
    rng = np.random.default_rng(3131)
    selections: tuple[tuple[RunCriterion, ...], ...] = (
        ("BOTTLENECK",),
        ("BOTTLENECK", "ATOMS"),
        DEFAULT_RUN_SELECTION,
    )
    for _ in range(5000):
        positions, window_start, count = random_line(rng)
        cheapest = min(scan_costs(positions, window_start, count))
        for run_selection in selections:
            assert (
                _window_displacement(positions, window_start, count, run_selection)
                == cheapest
            )


def test_the_closed_form_cost_matches_the_definition() -> None:
    """Only the two ends of a run matter: the displacement profile along
    it is monotone, so the largest travel is at one end or the other."""
    rng = np.random.default_rng(1717)
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        destinations = np.arange(window_start, window_start + count)
        for first in admissible_range(positions, window_start, count):
            moving = positions[first : first + count]
            assert _run_displacement_bottleneck_cost(
                positions, window_start, count, first
            ) == int(np.abs(moving - destinations).max())


def test_the_cost_profile_is_unimodal() -> None:
    """The rule binary-searches the profile, which is only sound because
    the costs fall and then rise."""
    rng = np.random.default_rng(2024)
    for _ in range(20000):
        positions, window_start, count = random_line(rng)
        costs = scan_costs(positions, window_start, count)
        bottom = costs.index(min(costs))
        assert costs[: bottom + 1] == sorted(costs[: bottom + 1], reverse=True)
        assert costs[bottom:] == sorted(costs[bottom:])


def test_a_line_too_short_for_the_window_is_refused() -> None:
    with pytest.raises(ValueError):
        _window_displacement(np.array([0, 2]), 0, 3)
