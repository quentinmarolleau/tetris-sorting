"""The Rich walkthrough and the matplotlib figures the notebook shows."""

from typing import Literal

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from tetris import AtomsConfiguration
from tetris.acceptance import TargetWindow
from tetris.display import format_motions, format_row


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_format_row_marks_the_target_region() -> None:
    row = np.array([True, False, True, False, True, False])
    rendered = format_row(
        row, TargetWindow(rows=2, columns=2, row_start=2, column_start=2)
    )
    assert rendered.count("[bold cyan]|[/bold cyan]") == 2
    assert "[green]●[/green]" in rendered
    assert "[yellow]●[/yellow]" in rendered
    assert "[dim]·[/dim]" in rendered


def test_format_motions_skips_atoms_that_stayed() -> None:
    assert format_motions([(3, 3), (5, 5)]) == "[dim]—[/dim]"
    rendered = format_motions([(1, 2), (4, 4)])
    assert "1" in rendered and "2" in rendered
    assert rendered.count("→") == 1


def test_verbose_construction_runs(capsys) -> None:
    config = AtomsConfiguration(loading_array_size=10, rng=np.random.default_rng(3))
    config.tetrimino_construct(verbose=True)
    printed = capsys.readouterr().out
    assert "Loading array size" in printed
    assert "Target shape" in printed
    assert "Configuration kept" in printed
    # Rich splits the rule title across style spans, so the
    # label and its index are not one substring
    assert "Row " in printed
    assert "Next shift" in printed


def test_verbose_construction_reports_deficient_columns(capsys) -> None:
    config = AtomsConfiguration(occupation_matrix=np.zeros((8, 8), dtype=bool))
    config.tetrimino_construct(verbose=True)
    assert "Deficient columns" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["loading", "tetriminoes", "all"])
def test_plot_configuration_produces_a_figure(
    which: Literal["loading", "tetriminoes", "all"],
) -> None:
    config = AtomsConfiguration(loading_array_size=14, rng=np.random.default_rng(4))
    figure = config.plot_configuration(which=which)
    # `all` is the three stages on a 2x2 grid, the fourth cell removed
    expected_axes = 3 if which == "all" else 1
    assert len(figure.axes) == expected_axes
    for ax in figure.axes:
        assert ax.get_title()
    assert config.tetriminoes_matrix is not None or which == "loading"


def test_a_rectangular_target_is_drawn_as_a_rectangle() -> None:
    config = AtomsConfiguration(
        loading_array_size=18,
        target_shape=(8, 12),
        rng=np.random.default_rng(6),
    )
    figure = config.plot_configuration(which="all")
    for ax in figure.axes:
        rectangle = next(patch for patch in ax.patches if isinstance(patch, Rectangle))
        assert (rectangle.get_height(), rectangle.get_width()) == (8, 12)
        assert rectangle.get_xy() == (
            config.window.column_start - 0.5,
            config.window.row_start - 0.5,
        )


def test_plot_ticks_stay_readable_for_a_large_array() -> None:
    config = AtomsConfiguration(loading_array_size=200, rng=np.random.default_rng(5))
    figure = config.plot_configuration()
    assert len(figure.axes[0].get_xticks()) <= 10
