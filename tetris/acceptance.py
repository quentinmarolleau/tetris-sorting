"""Target geometry and the acceptance rule.

A loading is an L x L grid of sites, each holding an atom or not. The
target is a smaller rectangle of sites, centred in the loading array,
that the rearrangement must fill without a hole. This module places that
rectangle and decides whether a given loading can fill it; a loading
that cannot is discarded before any tweezer moves.

Whether a loading is kept depends only on how many atoms each row holds
and on the target window, never on which run of atoms an overfull line
hands over. Keeping this apart from `tetris.fast` gives the acceptance
benchmark a source digest that editing the run selection does not move.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["TargetWindow", "configuration_kept", "target_geometry"]


@dataclass(frozen=True)
class TargetWindow:
    """Where the target array sits inside the loading array.

    Rows and columns are tracked separately. They are equal for the
    square targets the paper uses, but the two play different parts:
    `columns` is how many atoms a row can place, `rows` is how many
    atoms each column has to supply.

    Indices count from zero at the top-left corner of the loading array.

    Attributes:
        rows: Number of target rows, the atoms each target column needs.
        columns: Number of target columns, the atoms a row can place.
        row_start: Index of the first target row in the loading array.
        column_start: Index of the first target column in the loading
            array.
    """

    rows: int
    columns: int
    row_start: int
    column_start: int

    @property
    def sites(self) -> int:
        """Number of target sites, `rows * columns`: the atom count a
        defect-free target holds."""
        return self.rows * self.columns


def target_geometry(
    loading_array_size: int,
    reshape: tuple[int, int] = (0, 0),
    shape: tuple[int, int] | None = None,
) -> TargetWindow:
    """The target window inscribed in a loading array of the given size.

    The natural side is `L / sqrt(2)` rounded down, which asks for about
    half the loading sites and so for about as many atoms as a 50 percent
    loading provides. `reshape` adds to that side, one entry per axis, so
    that (0, -1) drops a column and (-1, -1) drops a row and a column,
    the convention Wang et al. use. `shape` names the two sides outright
    instead, and cannot be combined with a non-zero `reshape`.

    The window is centred on each axis independently once its sides are
    settled. When the leftover sites on an axis are odd, the extra site
    goes after the window.

    Args:
        loading_array_size: Side L of the square loading array, in sites.
        reshape: (rows, columns) added to the natural side.
        shape: (rows, columns) of the target, replacing the natural side.

    Returns:
        The target window.

    Raises:
        ValueError: If L is below one, if `shape` is given with a non-zero
            `reshape`, or if either side comes out below one or larger than
            L.
    """
    if loading_array_size < 1:
        raise ValueError(
            f"a loading array needs at least one site, got {loading_array_size}"
        )

    if shape is not None:
        if tuple(reshape) != (0, 0):
            raise ValueError(
                "a target shape and a reshape cannot both be given: "
                f"got shape {tuple(shape)} and reshape {tuple(reshape)}"
            )
        rows, columns = (int(side) for side in shape)
        because = "the requested target shape"
    else:
        side = int(loading_array_size / np.sqrt(2))
        rows, columns = (side + int(delta) for delta in reshape)
        because = (
            "the natural target side"
            if tuple(reshape) == (0, 0)
            else f"the natural target side reshaped by {tuple(reshape)}"
        )

    for axis, extent in (("rows", rows), ("columns", columns)):
        if extent < 1:
            raise ValueError(
                f"a loading array of size {loading_array_size} leaves no "
                f"target array to fill: {because} gives {extent} {axis}"
            )
        if extent > loading_array_size:
            raise ValueError(
                f"a target of {rows} rows by {columns} columns does not fit "
                f"in a loading array of size {loading_array_size}: "
                f"{because} gives {extent} {axis}"
            )

    return TargetWindow(
        rows=rows,
        columns=columns,
        row_start=(loading_array_size - rows) // 2,
        column_start=(loading_array_size - columns) // 2,
    )


def _row_columns(atoms_in_row: int, shift: int, columns: int) -> range:
    """Target columns, relative to the window, that a row carrying
    `atoms_in_row` atoms fills: a contiguous block from the running
    shift, taken modulo the column count. For a uniform target this is
    the same assignment as the paper's rule of always serving the least
    filled columns first."""
    return range(shift, shift + min(atoms_in_row, columns))


def configuration_kept(matrix: np.ndarray, window: TargetWindow) -> bool:
    """Whether a loading holds enough atoms, in the right rows, to fill
    the target.

    Each row hands its atoms, up to `window.columns` of them, to a block of
    target columns starting where the previous row's block stopped and
    wrapping round the window. The loading is kept when every target
    column then holds at least `window.rows` atoms, which is the condition
    for the vertical compression to fill the target. Only the atom count
    of each row matters, not where in the row the atoms sit.

    The occupancy is accumulated as a difference array, so the cost is
    constant per row instead of proportional to the target width.

    Args:
        matrix: L x L occupation matrix, non-zero where a site holds an
            atom.
        window: Target window inside that loading array.

    Returns:
        True if the loading is kept, False if it is discarded. An empty
        window is never kept.
    """
    columns, rows = window.columns, window.rows
    if columns < 1 or rows < 1:
        return False

    # occupancy is accumulated as a difference array over the target
    # columns, so a row costs O(1) instead of O(columns)
    boundaries = np.zeros(columns + 1, dtype=np.int32)
    full_rows = 0
    shift = 0
    for atoms_in_row in matrix.sum(axis=1).tolist():
        if atoms_in_row >= columns:
            full_rows += 1
            continue
        end = shift + atoms_in_row
        if end <= columns:
            boundaries[shift] += 1
            boundaries[end] -= 1
        else:
            boundaries[shift] += 1
            boundaries[columns] -= 1
            boundaries[0] += 1
            boundaries[end - columns] -= 1
        shift = end % columns

    occupancy = np.cumsum(boundaries[:columns]) + full_rows
    return bool(occupancy.min() >= rows)
