"""Reference implementations to check `_movers` against.

`movers_naive` picks the same run `_movers` does, under any run
selection, but by scanning every admissible run and comparing the
criteria one after another instead of binary-searching the bottleneck
profile. It exists to cross-check `_movers`'s choice, index for index,
against a brute-force scan.

`movers_sum_optimal` picks the run with the smallest *total*
displacement and ignores the bottleneck. It exists to measure how far
the bottleneck-optimal choice sits from the sum-optimal one.

`movers_leftmost` is Wang et al.'s own rule: "the atoms left are again
arranged to the leftmost target sites," with no search over where the
run starts. It exists to measure what that rule costs, at the paper's
own bottleneck metric, against the centred run `_movers` picks instead.
"""

import numpy as np

from tetris.fast import (
    _RUN_COSTS,
    DEFAULT_RUN_SELECTION,
    RunCriterion,
    _admissible_runs,
    _run_cumulated_displacement_cost,
)

__all__ = ["movers_leftmost", "movers_naive", "movers_sum_optimal", "run_sum_cost"]


def movers_naive(
    positions: np.ndarray,
    window_start: int,
    k: int,
    run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
) -> int:
    """Run chosen by `run_selection`, found by brute force.

    Every admissible run is scored on every criterion from scratch, and
    the lexicographic minimum wins, with the run's start as the last key
    so that a complete tie goes to the leftmost run. This is what
    `_movers` replaces with one `searchsorted` when BOTTLENECK leads.

    Args:
        positions: Sorted indices of the loaded sites along the line.
        window_start: Index of the first target site.
        k: Number of target sites, and so of atoms in the run.
        run_selection: Criteria applied in order. Not validated.

    Returns:
        Index into `positions` of the first atom of the chosen run.
    """
    lowest, highest = _admissible_runs(positions, window_start, k)
    return min(
        range(lowest, highest + 1),
        key=lambda j: (
            *(
                _RUN_COSTS[name](positions, window_start, k, j)
                for name in run_selection
            ),
            j,
        ),
    )


def run_sum_cost(positions: np.ndarray, window_start: int, k: int, j: int) -> int:
    """Total displacement of a run, every atom counted.

    Args:
        positions: Sorted indices of the loaded sites along the line.
        window_start: Index of the first target site.
        k: Number of target sites, and so of atoms in the run.
        j: Index into `positions` of the first atom of the run.

    Returns:
        Sum over the run's atoms of the distance to the site each is
        assigned, rather than the largest of those distances.
    """
    return _run_cumulated_displacement_cost(positions, window_start, k, j)


def movers_leftmost(positions: np.ndarray, window_start: int, k: int) -> int:
    """Earliest admissible run, with no search.

    What Wang et al. describe, against which `_movers` centres the run
    instead.

    Args:
        positions: Sorted indices of the loaded sites along the line.
        window_start: Index of the first target site.
        k: Number of target sites, and so of atoms in the run.

    Returns:
        Index into `positions` of the first atom of the leftmost run that
        leaves every surplus atom outside the window.
    """
    lowest, _ = _admissible_runs(positions, window_start, k)
    return lowest


def movers_sum_optimal(positions: np.ndarray, window_start: int, k: int) -> int:
    """Run minimising total displacement, found by brute force.

    The bottleneck and the total do not agree in general: see
    `test_overfull_reference` for a counter-example. On a tie in the total
    the leftmost run wins.

    Args:
        positions: Sorted indices of the loaded sites along the line.
        window_start: Index of the first target site.
        k: Number of target sites, and so of atoms in the run.

    Returns:
        Index into `positions` of the first atom of the admissible run with
        the smallest summed displacement.
    """
    lowest, highest = _admissible_runs(positions, window_start, k)
    return min(
        range(lowest, highest + 1),
        key=lambda j: run_sum_cost(positions, window_start, k, j),
    )
