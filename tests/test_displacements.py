"""Parallel displacement counting, the quantity Wang et al. 2023 report."""

import numpy as np
import pytest

from tetris import AtomsConfiguration
from tetris.acceptance import configuration_kept, target_geometry
from tetris.fast import (
    RearrangementMetrics,
    parallel_displacements,
    rearrangement_metrics,
)

FULL_MARGIN = (-1, -1)


def test_already_dense_configuration_needs_no_motion() -> None:
    """A loading array whose target window is already full and whose
    other sites are empty needs no horizontal and no vertical move."""
    window = target_geometry(12, FULL_MARGIN)
    matrix = np.zeros((12, 12), dtype=bool)
    matrix[
        window.row_start : window.row_start + window.rows,
        window.column_start : window.column_start + window.columns,
    ] = True
    config = AtomsConfiguration(occupation_matrix=matrix, reshape_target=FULL_MARGIN)
    config.tetrimino_construct()
    assert config.configuration_kept
    assert config.count_parallel_displacements() == 0


def test_single_column_shift_is_counted_once() -> None:
    """One atom per row, all one site left of the target window: every
    row move displaces by one, no column move is needed."""
    window = target_geometry(6)
    matrix = np.zeros((6, 6), dtype=bool)
    start = window.column_start
    matrix[:, start - 1 : start - 1 + window.columns] = True
    config = AtomsConfiguration(occupation_matrix=matrix)
    config.tetrimino_construct()
    assert config.configuration_kept
    # six rows each shifted by one site, then no vertical compression
    assert config.count_parallel_displacements() == 6


def test_fast_path_matches_the_class() -> None:
    rng = np.random.default_rng(31337)
    checked = 0
    for size in range(6, 51, 2):
        for _ in range(12):
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            config = AtomsConfiguration(
                occupation_matrix=matrix, reshape_target=FULL_MARGIN
            )
            config.tetrimino_construct()
            if not config.configuration_kept:
                continue
            checked += 1
            assert config.count_parallel_displacements() == (
                parallel_displacements(matrix, config.window)
            ), size
    assert checked > 100


@pytest.mark.parametrize("shape", [(9, 11), (11, 9), (8, 13), (7, 14)])
def test_fast_path_matches_the_class_on_a_rectangular_target(shape) -> None:
    """The row moves span the window's columns and the column moves span
    its rows. Those are the same number for a square target, so only a
    rectangular one can tell a swap of the two apart."""
    rng = np.random.default_rng(80808)
    checked = 0
    for _ in range(4000):
        matrix = rng.integers(0, 2, size=(14, 14)).astype(bool)
        config = AtomsConfiguration(occupation_matrix=matrix, target_shape=shape)
        config.tetrimino_construct()
        if not config.configuration_kept:
            continue
        checked += 1
        assert config.count_parallel_displacements() == (
            parallel_displacements(matrix, config.window)
        ), shape
    assert checked > 50, (shape, checked)


def test_displacements_are_only_defined_for_kept_configurations() -> None:
    matrix = np.zeros((10, 10), dtype=bool)
    config = AtomsConfiguration(occupation_matrix=matrix, reshape_target=FULL_MARGIN)
    config.tetrimino_construct()
    assert not config.configuration_kept
    with pytest.raises(ValueError):
        config.count_parallel_displacements()
    assert not configuration_kept(matrix, config.window)


def test_scaling_stays_close_to_linear_in_atom_number() -> None:
    """The paper fits N^1.03(7) for compact geometry. A coarse check that
    the implementation is in that regime rather than, say, quadratic."""
    rng = np.random.default_rng(5)
    atom_numbers, means = [], []
    for size in (20, 40, 80):
        window = target_geometry(size, FULL_MARGIN)
        samples = []
        while len(samples) < 60:
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            if configuration_kept(matrix, window):
                samples.append(parallel_displacements(matrix, window))
        atom_numbers.append(window.sites)
        means.append(float(np.mean(samples)))
    slope = np.polyfit(np.log(atom_numbers), np.log(means), 1)[0]
    assert 0.7 < slope < 1.4, (atom_numbers, means, slope)


def kept_configurations(rng, sizes, per_size, **target):
    """Configurations the class keeps, with the class already run on them."""
    for size in sizes:
        for _ in range(per_size):
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            config = AtomsConfiguration(occupation_matrix=matrix, **target)
            config.tetrimino_construct()
            if config.configuration_kept:
                yield matrix, config


@pytest.mark.parametrize(
    "run_selection",
    [
        ("BOTTLENECK", "CUMULATED", "ATOMS", "LEFTMOST"),
        ("BOTTLENECK", "LEFTMOST"),
        ("CUMULATED", "BOTTLENECK"),
        ("ATOMS",),
        ("LEFTMOST",),
        (),
    ],
)
def test_fast_metrics_match_the_class_under_any_run_selection(run_selection) -> None:
    """Every field, both steps: the fast path has to reproduce the moves
    the class records, not only their longest displacements."""
    rng = np.random.default_rng(2718)
    checked = 0
    for matrix, config in kept_configurations(
        rng, range(6, 41, 5), 10, reshape_target=FULL_MARGIN
    ):
        expected = AtomsConfiguration(
            occupation_matrix=matrix,
            reshape_target=FULL_MARGIN,
            run_selection=run_selection,
        ).rearrangement_metrics()
        assert rearrangement_metrics(matrix, config.window, run_selection) == expected
        checked += 1
    assert checked > 40, checked


@pytest.mark.parametrize("shape", [(9, 11), (11, 9)])
def test_fast_metrics_match_the_class_on_a_rectangular_target(shape) -> None:
    rng = np.random.default_rng(1618)
    checked = 0
    for matrix, config in kept_configurations(rng, (14,), 3000, target_shape=shape):
        assert (
            rearrangement_metrics(matrix, config.window)
            == config.rearrangement_metrics()
        )
        checked += 1
    assert checked > 50, checked


def test_the_run_selection_moves_atoms_differently_at_the_same_bottleneck() -> None:
    """Orders led by BOTTLENECK only choose between runs of equal longest
    move, so they share the paper's cost and differ in the rest."""
    rng = np.random.default_rng(1414)
    window = target_geometry(40, FULL_MARGIN)
    differ = 0
    for _ in range(60):
        matrix = rng.integers(0, 2, size=(40, 40)).astype(bool)
        if not configuration_kept(matrix, window):
            continue
        by_total = rearrangement_metrics(matrix, window, ("BOTTLENECK", "CUMULATED"))
        by_start = rearrangement_metrics(matrix, window, ("BOTTLENECK", "LEFTMOST"))
        assert by_total.total_bottleneck == by_start.total_bottleneck
        assert by_total.total_cumulated <= by_start.total_cumulated
        differ += by_total != by_start
    assert differ > 0


def test_totals_add_the_two_steps() -> None:
    metrics = RearrangementMetrics(1, 2, 3, 10, 20, 30)
    assert (
        metrics.total_bottleneck,
        metrics.total_cumulated,
        metrics.total_moved_atoms,
    ) == (
        11,
        22,
        33,
    )


def test_metrics_refuse_a_discarded_configuration() -> None:
    matrix = np.zeros((10, 10), dtype=bool)
    window = target_geometry(10, FULL_MARGIN)
    with pytest.raises(ValueError):
        rearrangement_metrics(matrix, window)
    with pytest.raises(ValueError):
        rearrangement_metrics(
            np.ones((10, 10), dtype=bool),
            window,
            ("FASTEST",),  # pyright: ignore[reportArgumentType]
        )


def test_the_plan_has_one_move_per_row_and_per_target_column() -> None:
    """The benchmark times the two planning passes on their own, so they
    must describe the whole rearrangement: every loading row, every target
    column, each atom sent to a site of the window."""
    from tetris.fast import (
        DEFAULT_RUN_SELECTION,
        _plan_construct,
        _plan_elimination,
        _score,
    )

    rng = np.random.default_rng(17)
    window = target_geometry(20, FULL_MARGIN)
    checked = 0
    while checked < 30:
        matrix = rng.integers(0, 2, size=(20, 20)).astype(bool)
        if not configuration_kept(matrix, window):
            continue
        checked += 1
        occupancy, rows = _plan_construct(matrix, window, DEFAULT_RUN_SELECTION)
        columns = _plan_elimination(occupancy, window, DEFAULT_RUN_SELECTION)
        assert len(rows) == 20
        assert len(columns) == window.columns
        for moving, destinations in rows:
            assert moving.size == destinations.size
            assert np.all(destinations >= window.column_start)
            assert np.all(destinations < window.column_start + window.columns)
        for moving, destinations in columns:
            assert moving.size == destinations.size == window.rows
        metrics = rearrangement_metrics(matrix, window)
        assert _score(rows) == [
            metrics.construct_bottleneck,
            metrics.construct_cumulated,
            metrics.construct_moved_atoms,
        ]
        assert _score(columns) == [
            metrics.elimination_bottleneck,
            metrics.elimination_cumulated,
            metrics.elimination_moved_atoms,
        ]
