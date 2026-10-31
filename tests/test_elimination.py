"""The Tetrimino elimination: letting each target column fall into the
window, and the motions that describe the fall."""

import numpy as np
import pytest

from tetris import AtomsConfiguration

FULL_MARGIN = (-1, -1)


def eliminate(
    config: AtomsConfiguration,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    """Run the elimination and return what the class then holds: the
    packed matrix, the eliminated one and the column motions."""
    config.tetrimino_eliminate()
    assert config.tetriminoes_matrix is not None
    assert config.eliminated_matrix is not None
    assert config.eliminated_motions is not None
    return (
        config.tetriminoes_matrix,
        config.eliminated_matrix,
        config.eliminated_motions,
    )


def kept_configuration(
    seed: int, size: int = 12, reshape: tuple[int, int] = FULL_MARGIN
) -> AtomsConfiguration:
    """A configuration the construct did not discard, so that there is an
    elimination to run at all."""
    rng = np.random.default_rng(seed)
    while True:
        config = AtomsConfiguration(
            loading_array_size=size, reshape_target=reshape, rng=rng
        )
        config.tetrimino_construct()
        if config.configuration_kept:
            return config


def window_block(config: AtomsConfiguration, matrix: np.ndarray) -> np.ndarray:
    window = config.window
    return matrix[
        window.row_start : window.row_start + window.rows,
        window.column_start : window.column_start + window.columns,
    ]


def test_the_elimination_fills_every_target_site() -> None:
    config = kept_configuration(11)
    _, eliminated, _ = eliminate(config)
    assert window_block(config, eliminated).all()


def test_the_elimination_loses_no_atom() -> None:
    config = kept_configuration(12)
    _, eliminated, _ = eliminate(config)
    assert eliminated.sum() == config.occupation_matrix.sum()


def test_the_elimination_moves_no_atom_between_columns() -> None:
    config = kept_configuration(13)
    packed, eliminated, _ = eliminate(config)
    assert (eliminated.sum(axis=0) == packed.sum(axis=0)).all()


def test_columns_the_target_does_not_use_are_left_alone() -> None:
    config = kept_configuration(14)
    packed, eliminated, motions = eliminate(config)
    window = config.window
    outside = [
        column
        for column in range(config.loading_array_size)
        if not (window.column_start <= column < window.column_start + window.columns)
    ]
    assert outside, "the window covers every column, nothing to check"
    assert np.array_equal(eliminated[:, outside], packed[:, outside])
    assert all(motions[column] == [] for column in outside)


def test_the_motions_replay_into_the_eliminated_matrix() -> None:
    """The motion table is the description of the move, so applying it by
    hand has to land on the matrix the method reports."""
    config = kept_configuration(15)
    packed_matrix, eliminated, motion_table = eliminate(config)
    for column in range(config.loading_array_size):
        packed = packed_matrix[:, column]
        motions = motion_table[column]
        replayed = np.zeros_like(packed)
        moved = {source for source, _ in motions}
        for row in np.flatnonzero(packed):
            if int(row) not in moved:
                replayed[row] = True
        for _, destination in motions:
            assert not replayed[destination], "two atoms on one site"
            replayed[destination] = True
        assert np.array_equal(replayed, eliminated[:, column])


def test_every_column_falls_without_crossing() -> None:
    """Atoms in a column move at once, so the move is collision-free
    exactly when the column ends in the order it started in."""
    config = kept_configuration(16)
    packed, _, motions = eliminate(config)
    for column in range(config.loading_array_size):
        positions = np.flatnonzero(packed[:, column])
        table = dict(motions[column])
        final = np.array([table.get(int(row), int(row)) for row in positions])
        assert np.all(np.diff(final) > 0), column


def test_the_motion_table_has_one_entry_per_loading_column() -> None:
    config = kept_configuration(17)
    _, _, motions = eliminate(config)
    assert len(motions) == config.loading_array_size


def test_the_elimination_runs_the_construct_first() -> None:
    config = kept_configuration(18)
    _, eliminated, _ = eliminate(config)
    fresh = AtomsConfiguration(
        occupation_matrix=config.occupation_matrix, reshape_target=FULL_MARGIN
    )
    assert fresh.tetriminoes_matrix is None
    fresh.tetrimino_eliminate()
    assert fresh.tetriminoes_matrix is not None
    assert fresh.eliminated_matrix is not None
    assert np.array_equal(fresh.eliminated_matrix, eliminated)


def test_a_discarded_configuration_cannot_be_eliminated() -> None:
    config = AtomsConfiguration(occupation_matrix=np.zeros((10, 10), dtype=bool))
    config.tetrimino_construct()
    assert not config.configuration_kept
    with pytest.raises(ValueError):
        config.tetrimino_eliminate()


def test_the_column_moves_agree_with_the_displacement_count() -> None:
    """The count and the matrix have to describe the same move, which is
    what the figures used to have to take on trust."""
    for seed in range(20, 32):
        config = kept_configuration(seed, size=14)
        _, _, column_motions = eliminate(config)
        assert config.tetriminoes_motions is not None
        rows = sum(
            max((abs(a - b) for a, b in motions), default=0)
            for motions in config.tetriminoes_motions
        )
        columns = sum(
            max((abs(a - b) for a, b in motions), default=0)
            for motions in column_motions
        )
        assert rows + columns == config.count_parallel_displacements()


def test_a_rectangular_target_falls_the_right_way() -> None:
    """Rows and columns are different extents for a rectangle, so a
    swapped axis survives a square target and not this one."""
    config = kept_configuration(40, size=14, reshape=(0, 0))
    config = AtomsConfiguration(
        occupation_matrix=config.occupation_matrix, target_shape=(7, 11)
    )
    config.tetrimino_construct()
    if not config.configuration_kept:
        pytest.skip("that loading does not fill a 7x11 target")
    _, eliminated, _ = eliminate(config)
    block = window_block(config, eliminated)
    assert block.shape == (7, 11)
    assert block.all()
