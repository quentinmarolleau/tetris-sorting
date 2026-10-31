"""Rearrangement cost of a kept loading, computed from its occupation
matrix alone.

The Tetris rearrangement has two steps. In the construct, every loading
row moves its atoms horizontally, all at once, onto a block of target
columns. In the elimination, every target column then moves its atoms
vertically onto the target rows. A line holding more atoms than the
target needs is overfull: it hands a run of consecutive atoms to the
target and leaves the rest outside, and a run selection decides which
run. Each parallel move lasts as long as its longest single displacement,
its bottleneck.

`AtomsConfiguration` builds a full occupation matrix, a per-row motion
table and a Rich rendering path for every configuration it examines. A
benchmark examines millions of them and needs none of that, so what
rearranging a kept loading costs is recomputed here directly from the
loaded matrix. Whether a loading is kept lives in `tetris.acceptance`.
About sixteen times faster than the class at L = 30 and twenty-five
times at L = 100, the gap widening because the class pays per site and
this pays per row.

The two implementations are held equal by the tests.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

from tetris.acceptance import TargetWindow, _row_columns, configuration_kept

__all__ = [
    "DEFAULT_RUN_SELECTION",
    "RUN_CRITERIA",
    "RearrangementMetrics",
    "RunCriterion",
    "parallel_displacements",
    "rearrangement_metrics",
    "validate_run_selection",
]

# How an overfull line chooses which run of atoms fills the window, when
# more than one run leaves the surplus atoms outside it. BOTTLENECK
# minimises the longest single displacement, which sets how long the
# parallel move lasts; CUMULATED the displacement summed over the run;
# ATOMS how many of the run's atoms change site; LEFTMOST the start of the
# run, which is Wang et al.'s own rule. A run selection applies them in
# order, and whatever is still tied at the end goes to the leftmost run.
RunCriterion = Literal["BOTTLENECK", "CUMULATED", "ATOMS", "LEFTMOST"]
RUN_CRITERIA: tuple[RunCriterion, ...] = get_args(RunCriterion)
DEFAULT_RUN_SELECTION: tuple[RunCriterion, ...] = (
    "BOTTLENECK",
    "CUMULATED",
    "ATOMS",
    "LEFTMOST",
)


def _admissible_runs(
    positions: np.ndarray, window_start: int, k: int
) -> tuple[int, int]:
    """First and last run start that leave every surplus atom outside the
    window. A run further left would strand an atom the window needs to
    land on; a run further right would do the same at the other end."""
    # one call rather than two: both bounds search the same array
    highest, past = np.searchsorted(positions, (window_start, window_start + k))
    return (
        max(0, int(past) - k),
        min(positions.size - k, int(highest)),
    )


def validate_run_selection(run_selection: Iterable[str]) -> tuple[RunCriterion, ...]:
    """Check a run selection and return it as a tuple.

    Args:
        run_selection: Criterion names from `RUN_CRITERIA`, applied in order
            when an overfull line has more than one admissible run. Each
            criterion only breaks the ties the previous ones left. An empty
            selection is allowed and picks the leftmost run.

    Returns:
        The same criteria as a tuple, usable as a default argument and as
        part of a cache fingerprint.

    Raises:
        TypeError: If `run_selection` is a bare string rather than a
            sequence of strings.
        ValueError: If a criterion is not one of `RUN_CRITERIA`, or if one
            appears more than once.
    """
    if isinstance(run_selection, str):
        raise TypeError(
            f"a run selection is a sequence of criteria, such as "
            f"({run_selection!r},), not a bare string"
        )
    selection = tuple(run_selection)
    unknown = [criterion for criterion in selection if criterion not in RUN_CRITERIA]
    if unknown:
        raise ValueError(
            f"unknown run criteria {unknown}, expected some of {list(RUN_CRITERIA)}"
        )
    repeated = sorted(
        {criterion for criterion in selection if selection.count(criterion) > 1}
    )
    if repeated:
        raise ValueError(f"run criteria given more than once: {repeated}")
    return selection  # type: ignore[return-value]


def _run_displacement_bottleneck_cost(
    positions: np.ndarray, window_start: int, k: int, j: int
) -> int:
    """Largest distance travelled by the run of `k` atoms starting at
    index `j`, in constant time.

    Successive displacements differ by `1 - gap` with `gap >= 1`, so the
    profile along a run never increases and its extremes are the two
    ends. The leading atom is pulled right by `window_start - p_first`
    and the trailing one left by `p_last - (window_start + k - 1)`;
    since the run spans at least `k - 1` sites, they cannot both be
    negative."""
    window_end = window_start + k - 1
    return int(
        max(
            window_start - positions[j],
            positions[j + k - 1] - window_end,
        )
    )


def _run_cumulated_displacement_cost(
    positions: np.ndarray, window_start: int, k: int, j: int
) -> int:
    """Distance travelled by the run of `k` atoms starting at index `j`,
    summed over its atoms rather than maximised."""
    destinations = np.arange(window_start, window_start + k)
    return int(np.abs(destinations - positions[j : j + k]).sum())


def _run_moved_atoms_cost(
    positions: np.ndarray, window_start: int, k: int, j: int
) -> int:
    """How many atoms of the run starting at index `j` change site. An
    atom already on the site it is assigned does not move."""
    destinations = np.arange(window_start, window_start + k)
    return int(np.count_nonzero(positions[j : j + k] != destinations))


def _run_start_cost(positions: np.ndarray, window_start: int, k: int, j: int) -> int:
    return j


_RUN_COSTS: dict[str, Callable[[np.ndarray, int, int, int], int]] = {
    "BOTTLENECK": _run_displacement_bottleneck_cost,
    "CUMULATED": _run_cumulated_displacement_cost,
    "ATOMS": _run_moved_atoms_cost,
    "LEFTMOST": _run_start_cost,
}


def _narrow(
    positions: np.ndarray,
    window_start: int,
    k: int,
    candidates: list[int],
    criteria: tuple[str, ...],
) -> int:
    """The candidate left once each criterion in turn has kept only its
    minimisers. Candidates come in increasing order, so a tie that
    survives every criterion goes to the leftmost run."""
    for criterion in criteria:
        if len(candidates) == 1:
            break
        cost = _RUN_COSTS[criterion]
        costs = [cost(positions, window_start, k, j) for j in candidates]
        best = min(costs)
        candidates = [j for j, value in zip(candidates, costs) if value == best]
    return candidates[0]


def _movers(
    positions: np.ndarray,
    window_start: int,
    k: int,
    run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
) -> int:
    """Index of the first of `k` consecutive atoms to assign to a
    window starting at `window_start`. Picking a consecutive run keeps
    the assignment non-crossing; among the runs that keep the atoms left
    behind outside the window, `run_selection` decides. It is not
    validated here: this runs once per line of every configuration a
    benchmark examines.

    With BOTTLENECK first the search is closed-form. The two terms of
    `_run_displacement_bottleneck_cost` move in opposite directions as
    the run slides right, so the cost falls and then rises and the
    minimum sits where they cross. Both `p_first` and `p_last` grow with
    the run, so the crossing is where their sum reaches
    `window_start + window_end`, which is to say where the run is centred
    on the window. Each term changes by at least one site per step, so
    only the two runs either side of the crossing can share the minimum,
    and only they go on to the remaining criteria. Any other leading
    criterion scans the whole admissible interval."""
    lowest, highest = _admissible_runs(positions, window_start, k)
    if lowest == highest:
        return lowest
    if not run_selection or run_selection[0] == "LEFTMOST":
        return lowest
    if run_selection[0] != "BOTTLENECK":
        return _narrow(
            positions, window_start, k, list(range(lowest, highest + 1)), run_selection
        )

    # the sums are increasing, so one searchsorted over the admissible
    # interval locates the crossing; the interval is surplus + 1 wide
    ends = positions[lowest : highest + 1] + positions[lowest + k - 1 : highest + k]
    threshold = 2 * window_start + k - 1
    j = min(lowest + int(np.searchsorted(ends, threshold)), highest)
    if j == lowest:
        return j

    # the crossing lands between two runs, so the one just short of it
    # can be the cheaper of the pair, or cost exactly the same
    before = _run_displacement_bottleneck_cost(positions, window_start, k, j - 1)
    after = _run_displacement_bottleneck_cost(positions, window_start, k, j)
    if before != after:
        return j - 1 if before < after else j
    # a pair needs no candidate lists, and ties here are a third of the
    # calls on the columns of a kept configuration
    for criterion in run_selection[1:]:
        cost = _RUN_COSTS[criterion]
        before = cost(positions, window_start, k, j - 1)
        after = cost(positions, window_start, k, j)
        if before != after:
            return j - 1 if before < after else j
    return j - 1


def _window_displacement(
    positions: np.ndarray,
    window_start: int,
    k: int,
    run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
) -> int:
    """Largest distance any atom travels when `k` of the atoms
    at `positions` are compressed into the window. Atoms move in
    parallel, so the largest of them is what the move costs."""
    if positions.size < k:
        raise ValueError("not enough atoms to fill the window")
    j = _movers(positions, window_start, k, run_selection)
    return _run_displacement_bottleneck_cost(positions, window_start, k, j)


@dataclass(frozen=True)
class RearrangementMetrics:
    """What rearranging one kept configuration costs, step by step.

    The construct moves every row once and the elimination every target
    column once. For each step three quantities are summed over its
    moves, all in lattice sites or atoms.

    Attributes:
        construct_bottleneck: Sum over the row moves of the longest single
            displacement in each. A parallel move lasts as long as its
            longest displacement, so this is the construct's share of the
            procedure time, and Wang et al.'s cost.
        construct_cumulated: Sum over the row moves of every atom's
            displacement.
        construct_moved_atoms: Number of atoms whose site changes, summed
            over the row moves.
        elimination_bottleneck: As `construct_bottleneck`, over the column
            moves.
        elimination_cumulated: As `construct_cumulated`, over the column
            moves.
        elimination_moved_atoms: As `construct_moved_atoms`, over the
            column moves. An atom moved in both steps counts in both.
    """

    construct_bottleneck: int
    construct_cumulated: int
    construct_moved_atoms: int
    elimination_bottleneck: int
    elimination_cumulated: int
    elimination_moved_atoms: int

    @property
    def total_bottleneck(self) -> int:
        """Both steps' bottlenecks: the paper's parallel displacements."""
        return self.construct_bottleneck + self.elimination_bottleneck

    @property
    def total_cumulated(self) -> int:
        """Every displacement of both steps, summed."""
        return self.construct_cumulated + self.elimination_cumulated

    @property
    def total_moved_atoms(self) -> int:
        """Moved atoms of both steps, an atom moved twice counted twice."""
        return self.construct_moved_atoms + self.elimination_moved_atoms


type Move = tuple[np.ndarray, np.ndarray]


def _move_costs(moving: np.ndarray, destinations: np.ndarray) -> tuple[int, int, int]:
    """(longest, summed, nonzero) displacement of one parallel move."""
    if moving.size == 0:
        return 0, 0, 0
    travel = moving - destinations
    np.abs(travel, out=travel)
    return int(travel.max()), int(travel.sum()), int(np.count_nonzero(travel))


def _plan_construct(
    matrix: np.ndarray,
    window: TargetWindow,
    run_selection: tuple[RunCriterion, ...],
) -> tuple[np.ndarray, list[Move]]:
    """Where every row's atoms go, and what the target columns hold after.

    This is the decision a controller makes for the row step, and nothing
    else: no displacement is measured here, so the benchmark can time it
    apart from the scoring.

    Returns:
        The occupancy of the target columns, loading rows by target
        columns, and one (sources, destinations) pair of column indices
        per row, in row order. A row with no atom gives two empty arrays.
    """
    columns = window.columns
    column_destinations = window.column_start + np.arange(columns)
    occupancy = np.zeros((matrix.shape[0], columns), dtype=bool)
    moves: list[Move] = []
    shift = 0
    for index, row in enumerate(matrix):
        positions = np.flatnonzero(row)
        atoms_in_row = positions.size
        filled = np.fromiter(
            (
                (column % columns)
                for column in _row_columns(atoms_in_row, shift, columns)
            ),
            dtype=np.intp,
            count=min(atoms_in_row, columns),
        )
        occupancy[index, filled] = True
        if atoms_in_row >= columns:
            j = _movers(positions, window.column_start, columns, run_selection)
            moves.append((positions[j : j + columns], column_destinations))
        else:
            moves.append((positions, window.column_start + np.sort(filled)))
            shift = (shift + atoms_in_row) % columns
    return occupancy, moves


def _plan_elimination(
    occupancy: np.ndarray,
    window: TargetWindow,
    run_selection: tuple[RunCriterion, ...],
) -> list[Move]:
    """Where every target column's atoms go in the column step.

    Returns:
        One (sources, destinations) pair of row indices per target
        column, in column order.
    """
    rows = window.rows
    row_destinations = window.row_start + np.arange(rows)
    moves: list[Move] = []
    for column in range(window.columns):
        positions = np.flatnonzero(occupancy[:, column])
        j = _movers(positions, window.row_start, rows, run_selection)
        moves.append((positions[j : j + rows], row_destinations))
    return moves


def _score(moves: list[Move]) -> list[int]:
    """[bottleneck, cumulated, moved atoms] summed over one step's moves."""
    totals = [0, 0, 0]
    for moving, destinations in moves:
        totals = [
            total + cost
            for total, cost in zip(totals, _move_costs(moving, destinations))
        ]
    return totals


def rearrangement_metrics(
    matrix: np.ndarray,
    window: TargetWindow,
    run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
) -> RearrangementMetrics:
    """Cost of rearranging a kept configuration, without building it.

    Reproduces the moves `AtomsConfiguration` records in its two motion
    tables, and adds them up, from the loaded matrix alone. The row moves
    are measured across the window's columns and the column moves down
    its rows, which differ as soon as the target is not square.

    Args:
        matrix: Square boolean loading, True where a site holds an atom.
        window: Target window inside that loading.
        run_selection: Criteria choosing the run of atoms an overfull row
            or column hands to the window. See `RUN_CRITERIA`.

    Returns:
        The bottleneck, cumulated displacement and moved-atom count of the
        construct and of the elimination.

    Raises:
        ValueError: If the configuration would have been discarded, or if
            `run_selection` names an unknown or repeated criterion.
        TypeError: If `run_selection` is a bare string.
    """
    run_selection = validate_run_selection(run_selection)
    if not configuration_kept(matrix, window):
        raise ValueError("configuration would have been discarded")
    return _rearrangement_metrics(matrix, window, run_selection)


def _rearrangement_metrics(
    matrix: np.ndarray,
    window: TargetWindow,
    run_selection: tuple[RunCriterion, ...],
) -> RearrangementMetrics:
    """`rearrangement_metrics` for a caller that has already validated the
    run selection and seen the configuration kept, as the benchmark has."""
    occupancy, construct = _plan_construct(matrix, window, run_selection)
    elimination = _plan_elimination(occupancy, window, run_selection)
    return RearrangementMetrics(*_score(construct), *_score(elimination))


def parallel_displacements(
    matrix: np.ndarray,
    window: TargetWindow,
    run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
) -> int:
    """Number of parallel displacements needed to reach the target array.

    The quantity Wang et al. 2023 report against atom number: every atom
    in a row moves at once, so a row move costs the largest single
    displacement in it, and likewise for the column compression that
    follows. The total is the sum over the row moves and the
    `window.columns` column moves.

    Args:
        matrix: Square boolean loading, True where a site holds an atom.
        window: Target window inside that loading.
        run_selection: Criteria choosing the run of atoms an overfull row
            or column hands to the window. Any order led by BOTTLENECK gives
            the same number.

    Returns:
        `rearrangement_metrics(...).total_bottleneck`.

    Raises:
        ValueError: If the configuration would have been discarded, or if
            `run_selection` names an unknown or repeated criterion.
        TypeError: If `run_selection` is a bare string.
    """
    return rearrangement_metrics(matrix, window, run_selection).total_bottleneck
