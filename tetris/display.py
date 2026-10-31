"""Text rendering of the construct, row by row, with Rich.

Used by `AtomsConfiguration.tetrimino_construct(verbose=True)` for the
notebook's step-by-step walkthrough. Nothing here is used by the
benchmarks.
"""

import numpy as np
from rich.console import Console
from rich.table import Table

from tetris.acceptance import TargetWindow

__all__ = [
    "format_motions",
    "format_row",
    "print_header",
    "print_row",
    "print_summary",
]


def format_row(row: np.ndarray, window: TargetWindow) -> str:
    """One row of sites as Rich markup.

    A bullet per atom, green inside the target columns and yellow outside
    them, a dot per empty site, and bars on either side of the target
    columns.

    Args:
        row: Boolean occupation of the row.
        window: Target window, for its column range.

    Returns:
        The markup string.
    """
    start = window.column_start
    end = start + window.columns
    symbols = []
    for index, occupied in enumerate(row):
        if not occupied:
            symbols.append("[dim]·[/dim]")
        elif start <= index < end:
            symbols.append("[green]●[/green]")
        else:
            symbols.append("[yellow]●[/yellow]")
    symbols.insert(end, "[bold cyan]|[/bold cyan]")
    symbols.insert(start, "[bold cyan]|[/bold cyan]")
    return "  ".join(symbols)


def format_motions(motions: list[tuple[int, int]]) -> str:
    """The moves of one row as `source→destination` column pairs.

    Args:
        motions: (source, destination) column pairs, one per atom handed
            to the target.

    Returns:
        Rich markup listing the atoms that change site, or a dash when
        none does.
    """
    moved = [
        (source, destination)
        for source, destination in motions
        if source != destination
    ]
    if not moved:
        return "[dim]—[/dim]"
    return "  ".join(
        f"[cyan]{source}[/cyan][dim]→[/dim][magenta]{destination}[/magenta]"
        for source, destination in moved
    )


def print_header(
    console: Console, loading_array_size: int, window: TargetWindow
) -> None:
    """Print the loading array size and the target's shape and position.

    Args:
        console: Rich console to print to.
        loading_array_size: Side L of the loading array.
        window: Target window.
    """
    info = Table(show_header=False, box=None, pad_edge=False)
    info.add_column(style="bold")
    info.add_row("Loading array size", str(loading_array_size))
    info.add_row("Target shape", f"{window.rows} x {window.columns}")
    info.add_row("Target start", f"{window.row_start}, {window.column_start}")
    console.print(info)


def print_row(
    console: Console,
    index: int,
    loaded_row: np.ndarray,
    packed_row: np.ndarray,
    motions: list[tuple[int, int]],
    shift: int,
    window: TargetWindow,
) -> None:
    """Print one row of the construct: before, after, and the moves.

    Args:
        console: Rich console to print to.
        index: Row index in the loading array.
        loaded_row: Occupation of the row before the construct.
        packed_row: Occupation of the row after the construct.
        motions: (source, destination) column pairs of the row's move.
        shift: Target column the next row's block starts at, relative to
            the window.
        window: Target window.
    """
    moved = sum(1 for source, destination in motions if source != destination)
    console.rule(f"[bold blue]Row {index}[/bold blue]", style="blue")
    console.print(
        f"[bold]{'Loaded':<10}[/bold] "
        f"{format_row(loaded_row, window)}"
        f"  [dim]({loaded_row.sum()} atoms)[/dim]"
    )
    console.print(
        f"[bold]{'Corrected':<10}[/bold] "
        f"{format_row(packed_row, window)}"
        f"  [dim]({moved} motions)[/dim]"
    )
    console.print(f"[bold]{'Motions':<10}[/bold] {format_motions(motions)}")
    console.print(f"[bold]{'Next shift':<10}[/bold] {shift}")


def print_summary(
    console: Console,
    total_atoms: int,
    total_motions: int,
    configuration_kept: bool,
    deficient_columns: list[tuple[int, int]],
) -> None:
    """Print the totals of the construct and whether the loading is kept.

    Args:
        console: Rich console to print to.
        total_atoms: Atoms in the loading.
        total_motions: Atoms that changed site during the construct.
        configuration_kept: Whether every target column holds enough atoms.
        deficient_columns: (column, atom count) of each target column
            short of atoms, relative to the window; listed only when the
            loading is discarded.
    """
    ratio = total_motions / total_atoms if total_atoms else 0.0
    console.rule(style="blue")
    console.print(f"[bold]Total atoms:[/bold] {total_atoms}")
    console.print(
        f"[bold]Total motions:[/bold] {total_motions}  [dim](ratio: {ratio:.3f})[/dim]"
    )
    style = "green" if configuration_kept else "red"
    console.print(
        f"[bold]Configuration kept:[/bold] [{style}]{configuration_kept}[/{style}]"
    )
    if not configuration_kept:
        console.print(
            "[bold]Deficient columns:[/bold] "
            + "  ".join(
                f"[red]{column}[/red] [dim]({count} atoms)[/dim]"
                for column, count in deficient_columns
            )
        )
