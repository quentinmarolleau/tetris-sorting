"""Correctness of the tetrimino construction and of the acceptance rule."""

import numpy as np
import pytest

from tetris import AtomsConfiguration
from tetris.acceptance import configuration_kept, target_geometry

RESHAPES = [(0, 0), (0, -1), (-1, -1)]


def random_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(0, 2, size=(size, size)).astype(bool)


def paper_rule_kept(matrix: np.ndarray, window) -> bool:
    """Direct transcription of the assignment rule of Wang et al. 2023:
    each row's atoms are assigned to the target columns whose smallest
    unassigned target row index is lowest, i.e. the least filled ones."""
    counts = np.zeros(window.columns, dtype=np.int64)
    for atoms_in_row in matrix.sum(axis=1).tolist():
        if atoms_in_row >= window.columns:
            counts += 1
        else:
            least_filled = np.argsort(counts, kind="stable")[:atoms_in_row]
            counts[least_filled] += 1
    return bool(counts.min() >= window.rows)


@pytest.mark.parametrize("reshape", RESHAPES)
def test_target_window_is_centred(reshape) -> None:
    for size in range(4, 101):
        window = target_geometry(size, reshape)
        left = window.column_start
        right = size - (window.column_start + window.columns)
        top = window.row_start
        bottom = size - (window.row_start + window.rows)
        assert abs(left - right) <= 1, (size, reshape, left, right)
        assert abs(top - bottom) <= 1, (size, reshape, top, bottom)


@pytest.mark.parametrize("shape", [(8, 12), (12, 8), (3, 17), (17, 17)])
def test_a_rectangular_window_is_centred_on_each_axis(shape) -> None:
    """The two axes are centred independently. They only came out equal
    while the target was square, which is what let one origin stand in
    for both."""
    for size in range(17, 41):
        window = target_geometry(size, shape=shape)
        assert (window.rows, window.columns) == shape
        assert abs(window.row_start - (size - window.rows - window.row_start)) <= 1
        assert (
            abs(window.column_start - (size - window.columns - window.column_start))
            <= 1
        )


def test_a_shape_and_a_reshape_cannot_both_be_given() -> None:
    with pytest.raises(ValueError, match="cannot both be given"):
        target_geometry(20, (0, -1), (8, 8))
    with pytest.raises(ValueError, match="cannot both be given"):
        AtomsConfiguration(
            loading_array_size=20, reshape_target=(-1, -1), target_shape=(8, 8)
        )


@pytest.mark.parametrize("shape", [(0, 8), (8, 0), (-1, 8), (21, 8), (8, 21), (21, 21)])
def test_a_target_that_cannot_exist_is_rejected(shape) -> None:
    with pytest.raises(ValueError):
        target_geometry(20, shape=shape)


@pytest.mark.parametrize("reshape", RESHAPES)
def test_full_row_covers_every_target_column(reshape) -> None:
    """A row holding exactly as many atoms as the target is wide must
    reach every target column: it is the boundary case that the
    off-centre target window used to break."""
    for size in range(4, 61):
        window = target_geometry(size, reshape)
        for offset in range(size - window.columns + 1):
            row = np.zeros(size, dtype=bool)
            row[offset : offset + window.columns] = True
            matrix = np.tile(row, (size, 1))
            config = AtomsConfiguration(
                occupation_matrix=matrix, reshape_target=reshape
            )
            config.tetrimino_construct()
            assert config.tetriminoes_matrix is not None
            packed = config.tetriminoes_matrix[
                :,
                window.column_start : window.column_start + window.columns,
            ]
            assert packed.all(), (size, reshape, offset)


@pytest.mark.parametrize("shape", [(6, 9), (9, 6)])
def test_a_full_row_covers_a_rectangular_window_too(shape) -> None:
    size = 14
    window = target_geometry(size, shape=shape)
    for offset in range(size - window.columns + 1):
        row = np.zeros(size, dtype=bool)
        row[offset : offset + window.columns] = True
        matrix = np.tile(row, (size, 1))
        config = AtomsConfiguration(occupation_matrix=matrix, target_shape=shape)
        config.tetrimino_construct()
        assert config.tetriminoes_matrix is not None
        packed = config.tetriminoes_matrix[
            :, window.column_start : window.column_start + window.columns
        ]
        assert packed.all(), (shape, offset)


@pytest.mark.parametrize("reshape", RESHAPES)
def test_matches_the_paper_assignment_rule(reshape) -> None:
    rng = np.random.default_rng(20260909)
    for size in range(4, 71):
        for _ in range(20):
            matrix = random_matrix(rng, size)
            config = AtomsConfiguration(
                occupation_matrix=matrix, reshape_target=reshape
            )
            config.tetrimino_construct()
            assert config.configuration_kept == paper_rule_kept(
                matrix, config.window
            ), (size, reshape)


@pytest.mark.parametrize("reshape", RESHAPES)
def test_fast_path_matches_the_class(reshape) -> None:
    rng = np.random.default_rng(4242)
    for size in range(4, 71):
        for _ in range(20):
            matrix = random_matrix(rng, size)
            config = AtomsConfiguration(
                occupation_matrix=matrix, reshape_target=reshape
            )
            config.tetrimino_construct()
            assert config.configuration_kept == configuration_kept(
                matrix, config.window
            ), (size, reshape)


@pytest.mark.parametrize("shape", [(9, 11), (11, 9), (8, 13), (7, 14)])
def test_a_rectangular_target_agrees_across_all_three_rules(shape) -> None:
    """The acceptance rule reads the two axes differently: a row places
    `columns` atoms, a column has to supply `rows` of them. A square
    target hides a swap of the two, so the class, the fast path and the
    paper's own rule are held equal here on targets that are not."""
    rng = np.random.default_rng(1515)
    size = 14
    agreed = 0
    for _ in range(3000):
        matrix = random_matrix(rng, size)
        config = AtomsConfiguration(occupation_matrix=matrix, target_shape=shape)
        config.tetrimino_construct()
        assert config.configuration_kept == configuration_kept(matrix, config.window), (
            shape
        )
        assert config.configuration_kept == paper_rule_kept(matrix, config.window), (
            shape
        )
        agreed += bool(config.configuration_kept)
    # the shapes are chosen to accept some and reject some, or a stuck
    # implementation answering the same way every time would pass
    assert 0 < agreed < 3000, (shape, agreed)


def test_a_wide_target_is_easier_than_its_transpose() -> None:
    """Rows and columns are not interchangeable. A row can serve every
    column once, so widening the target spreads each row further while
    shortening it lowers what every column must supply. `l_r x l_c` and
    its transpose ask for the same number of atoms and are not equally
    easy, which is the property a swapped axis would destroy."""
    rng = np.random.default_rng(606)
    size = 14
    matrices = [random_matrix(rng, size) for _ in range(2000)]
    rates = {}
    for shape in ((9, 11), (11, 9)):
        window = target_geometry(size, shape=shape)
        rates[shape] = sum(
            configuration_kept(matrix, window) for matrix in matrices
        ) / len(matrices)
    assert window.sites == 99
    assert rates[(9, 11)] > rates[(11, 9)] + 0.05, rates


def test_atoms_are_conserved() -> None:
    rng = np.random.default_rng(1)
    for size in (7, 16, 31):
        for _ in range(20):
            matrix = random_matrix(rng, size)
            config = AtomsConfiguration(occupation_matrix=matrix)
            config.tetrimino_construct()
            assert config.tetriminoes_matrix is not None
            assert config.tetriminoes_matrix.sum() == matrix.sum()


def test_motions_never_cross() -> None:
    """Sources and destinations are paired in ascending order, which is
    what keeps atoms from colliding during a parallel row move."""
    rng = np.random.default_rng(2)
    for size in (9, 24, 40):
        config = AtomsConfiguration(occupation_matrix=random_matrix(rng, size))
        config.tetrimino_construct()
        assert config.tetriminoes_motions is not None
        for motions in config.tetriminoes_motions:
            sources = [source for source, _ in motions]
            destinations = [destination for _, destination in motions]
            assert sources == sorted(sources)
            assert destinations == sorted(destinations)


def test_rejects_ambiguous_construction_arguments() -> None:
    with pytest.raises(ValueError):
        AtomsConfiguration()
    with pytest.raises(ValueError):
        AtomsConfiguration(loading_array_size=8, occupation_matrix=[[True]])


def test_generator_is_injectable_and_reproducible() -> None:
    first = AtomsConfiguration(loading_array_size=12, rng=np.random.default_rng(7))
    second = AtomsConfiguration(loading_array_size=12, rng=np.random.default_rng(7))
    third = AtomsConfiguration(loading_array_size=12, rng=np.random.default_rng(8))
    assert np.array_equal(first.occupation_matrix, second.occupation_matrix)
    assert not np.array_equal(first.occupation_matrix, third.occupation_matrix)


@pytest.mark.parametrize(
    "size, expected",
    [(8, 1.0000), (30, 0.9995)],
)
def test_full_margin_success_rate_regression(size: int, expected: float) -> None:
    """Sizes where the target window used to sit half a site off centre.
    The old code reported 0.9133 at L=8 and 0.9905 at L=30."""
    rng = np.random.default_rng(90909)
    window = target_geometry(size, (-1, -1))
    samples = 4000
    kept = sum(
        configuration_kept(random_matrix(rng, size), window) for _ in range(samples)
    )
    rate = kept / samples
    # Wilson half-width at three sigma, widened for the small counts
    tolerance = 3 * np.sqrt(max(expected * (1 - expected), 1e-4) / samples)
    assert abs(rate - expected) < tolerance + 0.002, rate


@pytest.mark.parametrize("size, reshape", [(1, (0, 0)), (1, (-1, -1)), (2, (-1, -1))])
def test_rejects_a_loading_array_with_no_target(size, reshape) -> None:
    with pytest.raises(ValueError, match="no target array"):
        AtomsConfiguration(loading_array_size=size, reshape_target=reshape)


def test_smallest_usable_arrays_are_accepted() -> None:
    for size, reshape in ((2, (0, 0)), (4, (-1, -1))):
        config = AtomsConfiguration(loading_array_size=size, reshape_target=reshape)
        assert config.window.rows >= 1
        assert config.window.columns >= 1
        config.tetrimino_construct()
        assert config.configuration_kept in (True, False)
