"""TOML literals and layout used by the readable cache rendering."""

import math
import tomllib

import pytest

from tetris.toml_format import (
    LINE_LENGTH,
    aligned,
    array,
    exponent,
    key,
    round_to_error,
    significant,
    value,
)


@pytest.mark.parametrize(
    ("mean", "error", "expected"),
    [
        (2.0020375508638617, 0.0030031076170657155, ("2.0020", "0.0030")),
        (0.9997141632128055, 0.00020208824221771814, ("0.99971", "0.00020")),
        (2452.6523230879197, 0.622365105137672, ("2452.7", "0.6")),
        (55118.66690493209, 10.984735028842287, ("55119.0", "11.0")),
        (1.0, 0.354, ("1.00", "0.35")),
        (1.0, 0.355, ("1.0", "0.4")),
        (1.0, 0.949, ("1.0", "0.9")),
        (2.0, 0.0961, ("2.00", "0.10")),
        (123456.0, 996.0, ("123500.0", "1000.0")),
    ],
)
def test_mean_follows_the_pdg_rounding_rule(mean, error, expected) -> None:
    assert round_to_error(mean, error) == expected


@pytest.mark.parametrize("error", [0.0, float("nan"), float("inf")])
def test_mean_without_a_usable_error_is_left_whole(error) -> None:
    mean, shown_error = round_to_error(0.123456789, error)
    assert mean == "0.123456789"
    parsed = tomllib.loads(f"e = {shown_error}")["e"]
    assert parsed == error or (math.isnan(parsed) and math.isnan(error))


def test_a_nan_mean_stays_nan() -> None:
    assert round_to_error(float("nan"), 0.1) == ("nan", "0.1")


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (60.00339426699793, "60.0"),
        (326.89709875661265, "326.9"),
        (8249.776354538957, "8250.0"),
        (0.0625, "0.0625"),
        (float("nan"), "nan"),
        (float("-inf"), "-inf"),
    ],
)
def test_significant_digits_without_padding(number, expected) -> None:
    assert significant(number, 4) == expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [(-7.5e-5, "-7.5e-5"), (1.4285e-4, "1.4e-4"), (0.0, "0.0"), (float("inf"), "inf")],
)
def test_exponent_is_short(number, expected) -> None:
    assert exponent(number, 2) == expected


def test_keys_are_bare_only_when_they_can_be() -> None:
    assert key("r-1c-1") == "r-1c-1"
    assert key("100") == "100"
    assert key("BOTTLENECK,LEFTMOST") == '"BOTTLENECK,LEFTMOST"'


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (True, "true"),
        (7, "7"),
        (2**100, '"1267650600228229401496703205376"'),
        ("a", '"a"'),
        ([1, [2, "b"]], '[1, [2, "b"]]'),
    ],
)
def test_values(item, expected) -> None:
    assert value(item) == expected


def test_none_has_no_value() -> None:
    with pytest.raises(TypeError):
        value(None)


def test_a_long_array_wraps() -> None:
    lines = array("sizes", list(range(4, 101)))
    assert 2 < len(lines) < 10
    assert all(len(line) <= LINE_LENGTH for line in lines)
    assert tomllib.loads("\n".join(lines))["sizes"] == list(range(4, 101))
    assert array("short", [1, 2]) == ["short = [1, 2]"]


def test_pairs_are_aligned() -> None:
    assert aligned([("a", "1"), ("long", "2")]) == ["a    = 1", "long = 2"]
