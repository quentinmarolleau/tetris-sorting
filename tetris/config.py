"""The Tetris rearrangement of Wang et al. 2023, step by step, on one
loading.

`AtomsConfiguration` holds the occupation of a square loading array and
runs the two steps of the scheme on it, recording the matrix and the
atom moves after each: the construct, where every row moves its atoms
horizontally towards the target columns, and the elimination, where
every target column then moves its atoms vertically into the target
rows. It is the readable implementation, used for the demonstrations and
the figures. The benchmarks use `tetris.fast`, which computes the same
moves without building the intermediate matrices.

Type aliases:
    OccupationMatrix: A square loading, as a boolean NumPy array or a list
        of lists of booleans, True where a site holds an atom.
"""

from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rich.console import Console

from tetris import display, plotting
from tetris.acceptance import TargetWindow, target_geometry
from tetris.fast import (
    DEFAULT_RUN_SELECTION,
    RearrangementMetrics,
    RunCriterion,
    _movers,
    validate_run_selection,
)

__all__ = ["AtomsConfiguration", "OccupationMatrix"]

type OccupationMatrix = NDArray[np.bool_] | list[list[bool]]


class AtomsConfiguration:
    """One loading of the tweezer array and its Tetris rearrangement.

    The target array has a natural side of `loading_array_size / sqrt(2)`,
    centred in the loading array, so that it holds about half the sites
    and therefore about as many sites as the 50 percent loading provides
    atoms. `reshape_target` adjusts that side one axis at a time, so
    (0, -1) drops a column and (-1, -1) drops a row and a column, which
    is the convention Wang et al. use in their own simulations.
    `target_shape` names the two sides outright instead.

    Args:
        loading_array_size: Side of a square loading array to fill at
            random, each site loaded with probability one half. Exclusive
            with `occupation_matrix`.
        occupation_matrix: An existing square loading, as a boolean array
            or a list of lists of booleans. Exclusive with
            `loading_array_size`.
        reshape_target: (rows, columns) added to the natural target side.
        target_shape: (rows, columns) of the target, given outright.
            Exclusive with a non-zero `reshape_target`.
        rng: Generator for the random loading. A fresh unseeded one if
            omitted.
        run_selection: Criteria, applied in order, that choose which run of
            atoms fills the window when a row or a column holds more atoms
            than the target needs. See `tetris.fast.RUN_CRITERIA`.

    Raises:
        ValueError: If both or neither of `loading_array_size` and
            `occupation_matrix` are given, if the target does not fit or
            combines `target_shape` with a non-zero `reshape_target`, or
            if `run_selection` names an unknown or repeated criterion.
        TypeError: If `occupation_matrix` is neither a boolean array nor a
            list of lists, or if `run_selection` is a bare string.

    Attributes:
        rng: Generator the loading was drawn from.
        loading_array_size: Side L of the loading array, in sites.
        occupation_matrix: L x L boolean loading, True where a site holds
            an atom. Row index first, counted from the top.
        window: Where the target sits in the loading array.
        run_selection: Criteria choosing the run of an overfull line.
        contraction_ratio: Target sites over loading sites.
        tetriminoes_matrix: Occupation after the construct, or None before
            `tetrimino_construct` runs.
        tetriminoes_motions: One list per loading row of (source,
            destination) column pairs, one pair per moved atom, or None
            before `tetrimino_construct` runs.
        configuration_kept: Whether every target column holds enough atoms
            after the construct, or None before `tetrimino_construct` runs.
        eliminated_matrix: Occupation after the elimination, or None before
            `tetrimino_eliminate` runs.
        eliminated_motions: One list per loading column of (source,
            destination) row pairs, empty outside the target, or None
            before `tetrimino_eliminate` runs.
    """

    rng: np.random.Generator
    loading_array_size: int
    occupation_matrix: np.ndarray
    window: TargetWindow
    run_selection: tuple[RunCriterion, ...]
    contraction_ratio: float
    tetriminoes_matrix: np.ndarray | None
    tetriminoes_motions: list[list[tuple[int, int]]] | None
    configuration_kept: bool | None
    eliminated_matrix: np.ndarray | None
    eliminated_motions: list[list[tuple[int, int]]] | None

    def __init__(
        self,
        loading_array_size: int | None = None,
        occupation_matrix: OccupationMatrix | None = None,
        reshape_target: tuple[int, int] = (0, 0),
        target_shape: tuple[int, int] | None = None,
        rng: np.random.Generator | None = None,
        run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.run_selection = validate_run_selection(run_selection)

        match (loading_array_size, occupation_matrix):
            case (int() as size, None):
                matrix = self._generate_occupation_matrix(size)

            case (None, np.ndarray() as matrix) if matrix.dtype == bool:
                pass

            case (None, list() as matrix):
                matrix = np.asarray(matrix, dtype=bool)

            case (int(), np.ndarray() | list()):
                raise ValueError(
                    "loading_array_size and occupation_matrix are mutually exclusive"
                )

            case (None, None):
                raise ValueError(
                    "Either loading_array_size or occupation_matrix must be provided"
                )

            case _:
                raise TypeError(
                    "occupation_matrix must be a boolean NumPy array "
                    "or a list of lists of booleans"
                )

        self.occupation_matrix = matrix
        self.loading_array_size = matrix.shape[0]
        self.window = target_geometry(
            self.loading_array_size, reshape_target, target_shape
        )
        self.contraction_ratio = self.window.sites / self.loading_array_size**2

        # Populated by tetrimino_construct()
        self.tetriminoes_matrix = None
        self.tetriminoes_motions = None
        self.configuration_kept = None

        # Populated by tetrimino_eliminate()
        self.eliminated_matrix = None
        self.eliminated_motions = None

    def _generate_occupation_matrix(self, size: int) -> np.ndarray:
        return self.rng.integers(low=0, high=2, size=(size, size)).astype(bool)

    @staticmethod
    def _pack_row(
        loaded_row: np.ndarray,
        shift: int,
        columns: int,
        column_start: int,
        run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
    ) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
        """Rearrange one row's atoms towards the target columns. Returns
        the packed row, the shift to carry into the next one, and the
        (source, destination) column pairs for the atoms that move.

        Sources and destinations are both in ascending order, so the
        atoms never cross and can all be moved in a single parallel
        step."""
        positions = np.flatnonzero(loaded_row)
        atoms_in_row = positions.size
        packed = np.zeros_like(loaded_row)

        if atoms_in_row < columns:
            # Place the atoms on a contiguous block of columns starting
            # at the running shift, which spreads successive rows over the
            # columns that are furthest behind. Wrapping round the window
            # is deliberate: it is what makes the staircase of tetriminoes.
            filled = column_start + (shift + np.arange(atoms_in_row)) % columns
            packed[filled] = True
            destinations = np.sort(filled)
            moving = positions
            next_shift = (shift + atoms_in_row) % columns
        else:
            # The row already has enough atoms to serve every target
            # column. Fill the window and leave the surplus atoms where
            # they are rather than dragging them along; a row can only
            # ever give one atom per column, so nothing is lost.
            destinations = column_start + np.arange(columns)
            j = _movers(positions, column_start, columns, run_selection)
            moving = positions[j : j + columns]
            packed[destinations] = True
            packed[positions[:j]] = True
            packed[positions[j + columns :]] = True
            # a row that serves every column hands on no staircase offset
            next_shift = shift

        motions = [
            (int(source), int(destination))
            for source, destination in zip(moving, destinations)
        ]
        return packed, next_shift, motions

    @staticmethod
    def _find_deficient_columns(
        matrix: np.ndarray, window: TargetWindow
    ) -> list[tuple[int, int]]:
        """(column index, atom count) for each target column holding
        fewer atoms than the target has rows. Indices are relative to the
        target window. A configuration is kept only if this list is
        empty."""
        target_columns = matrix[
            :, window.column_start : window.column_start + window.columns
        ]
        column_counts = target_columns.sum(axis=0)
        deficient = np.flatnonzero(column_counts < window.rows)
        return [(int(column), int(column_counts[column])) for column in deficient]

    def tetrimino_construct(self, verbose: bool = False) -> None:
        """Run the construct, the first of the paper's two steps.

        Every row moves its atoms horizontally, in one parallel move. A
        row with fewer atoms than the target has columns fills a block of
        target columns starting where the previous row's block ended,
        wrapping round the window, so that successive rows build a
        staircase of tetriminoes and the columns fill evenly. A row with
        at least as many atoms fills every target column and leaves its
        surplus atoms in place. Afterwards the loading is kept if every
        target column holds at least `window.rows` atoms.

        Sets `tetriminoes_matrix`, `tetriminoes_motions` and
        `configuration_kept`.

        Args:
            verbose: Print a row-by-row account of the step with Rich.
        """
        shift = 0
        tetriminoes_matrix = np.empty_like(self.occupation_matrix, dtype=bool)
        tetriminoes_motions: list[list[tuple[int, int]]] = []
        total_atoms = 0
        total_motions = 0
        console = Console(force_terminal=True) if verbose else None

        if console:
            display.print_header(console, self.loading_array_size, self.window)

        for index, loaded_row in enumerate(self.occupation_matrix):
            packed, shift, motions = self._pack_row(
                loaded_row,
                shift,
                self.window.columns,
                self.window.column_start,
                self.run_selection,
            )
            tetriminoes_matrix[index] = packed
            tetriminoes_motions.append(motions)
            total_atoms += int(loaded_row.sum())
            if console:
                total_motions += sum(
                    1 for source, destination in motions if source != destination
                )
                display.print_row(
                    console,
                    index,
                    loaded_row,
                    packed,
                    motions,
                    shift,
                    self.window,
                )

        deficient_columns = self._find_deficient_columns(
            tetriminoes_matrix, self.window
        )
        kept = not deficient_columns

        if console:
            display.print_summary(
                console,
                total_atoms,
                total_motions,
                kept,
                deficient_columns,
            )

        self.tetriminoes_matrix = tetriminoes_matrix
        self.tetriminoes_motions = tetriminoes_motions
        self.configuration_kept = kept

    @staticmethod
    def _compact_column(
        packed_column: np.ndarray,
        rows: int,
        row_start: int,
        run_selection: tuple[RunCriterion, ...] = DEFAULT_RUN_SELECTION,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Let one target column fall into the window. Returns the column
        after the fall and its (source, destination) row pairs.

        This is `_pack_row` turned on its side, minus the staircase: the
        atoms that land in the window are a run of consecutive ones, the
        atoms the run passes over stay where the construct left them, and
        the run is the cheapest that leaves those atoms clear of the
        window. A column never has too few atoms once the configuration
        has been kept, so the underfilled branch of the row step has no
        counterpart here.
        """
        positions = np.flatnonzero(packed_column)
        if positions.size < rows:
            raise ValueError(
                f"a column holds {positions.size} atoms, too few for a "
                f"window {rows} rows deep"
            )
        compacted = np.zeros_like(packed_column)
        destinations = row_start + np.arange(rows)
        j = _movers(positions, row_start, rows, run_selection)
        moving = positions[j : j + rows]
        compacted[destinations] = True
        compacted[positions[:j]] = True
        compacted[positions[j + rows :]] = True
        motions = [
            (int(source), int(destination))
            for source, destination in zip(moving, destinations)
        ]
        return compacted, motions

    def tetrimino_eliminate(self) -> None:
        """Run the elimination, the second of the paper's two steps.

        Every target column moves a run of its atoms vertically onto the
        target rows, in one parallel move, and leaves the atoms outside
        that run in place. Sets `eliminated_matrix` and
        `eliminated_motions`, and runs `tetrimino_construct` first if it
        has not run.

        `eliminated_motions` carries one list per column of the *loading*
        array, the way `tetriminoes_motions` carries one per row, so the
        two tables are indexed alike. Columns the target does not reach
        get an empty list: nothing in the scheme ever brings those atoms
        in, and they are left exactly where the construct put them.

        Raises:
            ValueError: If the construct discarded the configuration.
        """
        if self.tetriminoes_matrix is None:
            self.tetrimino_construct()
        if not self.configuration_kept:
            raise ValueError("configuration would have been discarded")
        # set by the construct above; the attribute type cannot say so
        tetriminoes = self.tetriminoes_matrix
        assert tetriminoes is not None

        window = self.window
        eliminated = tetriminoes.copy()
        eliminated_motions: list[list[tuple[int, int]]] = []
        for column in range(self.loading_array_size):
            if not (
                window.column_start <= column < window.column_start + window.columns
            ):
                eliminated_motions.append([])
                continue
            compacted, motions = self._compact_column(
                tetriminoes[:, column],
                window.rows,
                window.row_start,
                self.run_selection,
            )
            eliminated[:, column] = compacted
            eliminated_motions.append(motions)

        self.eliminated_matrix = eliminated
        self.eliminated_motions = eliminated_motions

    def rearrangement_metrics(self) -> RearrangementMetrics:
        """Cost of the rearrangement, read off the two motion tables.

        Runs `tetrimino_eliminate` first if it has not run, so the numbers
        and the matrices describe the same moves. The row moves are
        measured across the window's columns and the column moves down its
        rows, which differ as soon as the target is not square.

        Returns:
            The bottleneck, cumulated displacement and moved-atom count of
            the construct and of the elimination, as
            `tetris.fast.rearrangement_metrics` computes them without the
            class.

        Raises:
            ValueError: If the configuration was discarded.
        """
        if self.eliminated_motions is None:
            self.tetrimino_eliminate()
        # both set by the elimination above
        assert self.tetriminoes_motions is not None
        assert self.eliminated_motions is not None

        def step(table: list[list[tuple[int, int]]]) -> tuple[int, int, int]:
            travels = [
                [abs(source - destination) for source, destination in motions]
                for motions in table
            ]
            return (
                sum(max(travel, default=0) for travel in travels),
                sum(sum(travel) for travel in travels),
                sum(sum(1 for distance in travel if distance) for travel in travels),
            )

        return RearrangementMetrics(
            *step(self.tetriminoes_motions), *step(self.eliminated_motions)
        )

    def count_parallel_displacements(self) -> int:
        """Parallel displacements needed to reach the target array.

        The sum, over the row moves and then the column moves, of the
        largest single atom displacement in each. This is the cost Wang et
        al. report against atom number, since it is what the rearrangement
        time is dominated by.

        Returns:
            `rearrangement_metrics().total_bottleneck`.

        Raises:
            ValueError: If the configuration was discarded.
        """
        return self.rearrangement_metrics().total_bottleneck

    def plot_configuration(
        self,
        which: Literal["loading", "tetriminoes", "eliminated", "all"] = "loading",
        atom_radius: float = 0.32,
        size: float = 5.0,
    ) -> Figure:
        """Plot one stage of the rearrangement, or all three.

        Runs whichever step the requested stage needs. See
        `tetris.plotting.plot_configuration` for the layout.

        Args:
            which: "loading", "tetriminoes" (after the construct),
                "eliminated" (after the elimination), or "all".
            atom_radius: Radius of an atom's disc, in lattice sites.
            size: Width and height of one panel, in inches.

        Returns:
            The figure.

        Raises:
            ValueError: If "eliminated" is asked for a discarded
                configuration.
        """
        return plotting.plot_configuration(self, which, atom_radius, size)
