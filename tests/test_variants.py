"""Names given to the target reshapes the benchmark sweeps over."""

import pytest

from tetris.variants import variant_key, variant_label

SWEPT = [(0, 0), (0, -1), (-1, -1)]


@pytest.mark.parametrize(
    "reshape, key",
    [((0, 0), "r0c0"), ((0, -1), "r0c-1"), ((-1, -1), "r-1c-1")],
)
def test_keys_are_what_the_caches_are_filed_under(reshape, key) -> None:
    assert variant_key(reshape) == key


def test_keys_distinguish_the_two_axes() -> None:
    """(-1, 0) and (0, -1) ask for different targets and must not share
    a key, or one sweep would overwrite the other in the cache."""
    assert variant_key((-1, 0)) != variant_key((0, -1))


def test_every_swept_variant_reads_as_a_phrase() -> None:
    for reshape in SWEPT + [(-1, 0)]:
        label = variant_label(reshape)
        assert label[0].isupper()
        assert not any(character.isdigit() for character in label)


def test_an_unnamed_reshape_falls_back_to_its_numbers() -> None:
    assert variant_label((2, -3)) == "+2 rows, -3 columns"
    assert variant_label((0, 4)) == "+0 rows, +4 columns"
