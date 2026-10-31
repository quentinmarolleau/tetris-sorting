"""The figure the elimination earns: a third panel, and the 2x2 layout
that carries the algorithm's three stages."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from tetris import AtomsConfiguration

FULL_MARGIN = (-1, -1)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def kept_configuration(seed: int, size: int = 12) -> AtomsConfiguration:
    rng = np.random.default_rng(seed)
    while True:
        config = AtomsConfiguration(
            loading_array_size=size, reshape_target=FULL_MARGIN, rng=rng
        )
        config.tetrimino_construct()
        if config.configuration_kept:
            return config


def test_all_draws_the_three_stages_in_a_two_by_two_grid() -> None:
    config = kept_configuration(101)
    figure = config.plot_configuration(which="all")
    assert len(figure.axes) == 3
    positions = sorted(
        (round(ax.get_position().y0, 3), round(ax.get_position().x0, 3))
        for ax in figure.axes
    )
    (low_y, low_x), (top_left_y, top_left_x), (_, top_right_x) = positions
    # one panel on the bottom row, two above it, and the bottom one sits
    # under the right-hand of the pair rather than the left
    assert low_y < top_left_y
    assert top_left_x < top_right_x
    assert low_x == top_right_x


def test_all_titles_name_the_three_stages() -> None:
    config = kept_configuration(102)
    titles = [ax.get_title() for ax in config.plot_configuration(which="all").axes]
    assert len(titles) == 3
    assert all(titles)
    assert len(set(titles)) == 3


def test_all_runs_the_elimination() -> None:
    config = kept_configuration(103)
    assert config.eliminated_matrix is None
    config.plot_configuration(which="all")
    assert config.eliminated_matrix is not None


def test_eliminated_draws_one_panel() -> None:
    config = kept_configuration(104)
    figure = config.plot_configuration(which="eliminated")
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title()


def test_a_discarded_loading_still_draws_what_it_has() -> None:
    """The stage that cannot happen is the interesting one to look at, so
    `all` says so on the panel instead of refusing to draw."""
    config = AtomsConfiguration(occupation_matrix=np.zeros((10, 10), dtype=bool))
    figure = config.plot_configuration(which="all")
    assert len(figure.axes) == 3
    assert any("discard" in ax.get_title().lower() for ax in figure.axes)


def test_a_discarded_loading_cannot_be_drawn_eliminated_alone() -> None:
    config = AtomsConfiguration(occupation_matrix=np.zeros((10, 10), dtype=bool))
    with pytest.raises(ValueError):
        config.plot_configuration(which="eliminated")


def test_the_eliminated_panel_shows_a_full_target() -> None:
    """Every atom of the window is inside it, so the panel draws exactly
    `window.sites` atoms in the target colour."""
    config = kept_configuration(105)
    figure = config.plot_configuration(which="eliminated")
    inside = figure.axes[0].collections[0]
    assert len(inside.get_paths()) == config.window.sites
