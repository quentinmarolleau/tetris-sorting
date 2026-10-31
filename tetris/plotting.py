"""Matplotlib figures of a loading and of the two rearrangement steps.

Atoms inside the target are drawn green, atoms outside it red, and the
target itself as a shaded rectangle. The figures follow the active
Matplotlib style, dark or light.
"""

from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from tetris.acceptance import TargetWindow

__all__ = ["draw_configuration", "plot_configuration"]

MAX_TICKS = 10

# Softer than matplotlib's named "green" and "red", which are loud enough
# against a grayscale bed to be the first thing the eye lands on. These
# two sit at the same lightness as each other and read on either theme.
INSIDE_COLOUR = "#3f9b58"
OUTSIDE_COLOUR = "#d2564a"

# The shaded window, as a fraction of the ink colour.
ZONE_ALPHA = 0.08


def _ink() -> str:
    """The foreground of whatever style is in force, so the figure works
    under `dark_background` as well as plainly."""
    return plt.rcParams["text.color"]


def draw_configuration(
    ax: Axes,
    matrix: np.ndarray,
    title: str,
    window: TargetWindow,
    atom_radius: float,
    show_ylabel: bool = True,
) -> None:
    """Draw one occupation matrix on existing axes.

    A filled disc per atom, green inside the target and red outside it,
    and the target itself as a shaded rectangle. Empty sites are left
    blank: a ring on each of them competes with the atoms for attention
    and, on a large array, buries them. Row 0 is drawn at the top, as the
    matrix is printed.

    Args:
        ax: Axes to draw on.
        matrix: Square boolean occupation matrix.
        title: Panel title.
        window: Target window to shade.
        atom_radius: Radius of an atom's disc, in lattice sites.
        show_ylabel: Label the vertical axis. Off for a panel whose left
            neighbour already carries it.
    """
    loading_array_size = matrix.shape[0]
    ink = _ink()
    ax.set_facecolor("none")
    ax.add_patch(
        Rectangle(
            (window.column_start - 0.5, window.row_start - 0.5),
            window.columns,
            window.rows,
            facecolor=ink,
            alpha=ZONE_ALPHA,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (window.column_start - 0.5, window.row_start - 0.5),
            window.columns,
            window.rows,
            fill=False,
            edgecolor=ink,
            linewidth=1.0,
            zorder=1,
        )
    )

    occupied_yx = np.argwhere(matrix)
    target_mask = np.zeros_like(matrix, dtype=bool)
    target_mask[
        window.row_start : window.row_start + window.rows,
        window.column_start : window.column_start + window.columns,
    ] = True
    inside = target_mask[occupied_yx[:, 0], occupied_yx[:, 1]]

    for atoms, colour in (
        (occupied_yx[inside], INSIDE_COLOUR),
        (occupied_yx[~inside], OUTSIDE_COLOUR),
    ):
        ax.add_collection(
            PatchCollection(
                [Circle((x, y), atom_radius) for y, x in atoms],
                facecolor=colour,
                edgecolor="none",
                zorder=3,
            )
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    if show_ylabel:
        ax.set_ylabel("Y")
    ax.set_xlim(-0.5, loading_array_size - 0.5)
    ax.set_ylim(loading_array_size - 0.5, -0.5)
    ax.set_aspect("equal")

    tick_step = max(1, int(np.ceil(loading_array_size / MAX_TICKS)))
    ticks = np.arange(0, loading_array_size, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


TITLES = {
    "loading": "Initial atom configuration",
    "tetriminoes": "Tetrimino-constructed configuration",
    "eliminated": "Tetrimino-eliminated configuration",
}

DISCARDED_TITLE = "Tetrimino elimination — loading discarded"


def _stage(config, which: str) -> np.ndarray:
    """The matrix one panel draws, running whichever step produces it."""
    if which == "loading":
        return config.occupation_matrix
    if which == "tetriminoes":
        if config.tetriminoes_matrix is None:
            config.tetrimino_construct()
        assert config.tetriminoes_matrix is not None
        return config.tetriminoes_matrix
    if config.eliminated_matrix is None:
        config.tetrimino_eliminate()
    assert config.eliminated_matrix is not None
    return config.eliminated_matrix


def plot_configuration(
    config,
    which: Literal["loading", "tetriminoes", "eliminated", "all"] = "loading",
    atom_radius: float = 0.32,
    size: float = 5.0,
) -> Figure:
    """Plot one stage of the algorithm, or all three.

    `all` lays the three out on a 2x2 grid, reading along the top row and
    then down the right-hand column: the loading top left, the construct
    top right, the elimination beneath it. The fourth cell stays empty,
    which is what makes the path through the figure legible.

    A discarded loading has no elimination. Asking for that stage on its
    own raises, since there is nothing to draw; `all` draws the two
    stages that did happen and says on the third panel why it is blank,
    a discarded loading being the interesting case to look at.

    Args:
        config: An `AtomsConfiguration`. The steps a stage needs are run
            on it if they have not run.
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
    if which != "all":
        fig, ax = plt.subplots(figsize=(size, size))
        draw_configuration(
            ax,
            _stage(config, which),
            TITLES[which],
            config.window,
            atom_radius,
        )
        plt.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(2 * size, 2 * size))
    axes[1][0].remove()
    draw_configuration(
        axes[0][0],
        _stage(config, "loading"),
        TITLES["loading"],
        config.window,
        atom_radius,
    )
    draw_configuration(
        axes[0][1],
        _stage(config, "tetriminoes"),
        TITLES["tetriminoes"],
        config.window,
        atom_radius,
        show_ylabel=False,
    )
    if config.configuration_kept:
        draw_configuration(
            axes[1][1],
            _stage(config, "eliminated"),
            TITLES["eliminated"],
            config.window,
            atom_radius,
        )
    else:
        axes[1][1].set_title(DISCARDED_TITLE)
        axes[1][1].set_axis_off()

    plt.tight_layout()
    return fig
