"""TOML literals and layout for `tetris.toml_export`.

Numbers are written for a reader: a mean to the digits its standard
error supports, a timing to four significant digits, a small offset in
short exponent form. Tables are key-aligned and arrays wrap at the line
length instead of taking a line per element.
"""

import json
import math
import re
from collections.abc import Iterable, Sequence

__all__ = [
    "LINE_LENGTH",
    "aligned",
    "array",
    "comment",
    "exponent",
    "key",
    "round_to_error",
    "significant",
    "table",
    "value",
]

LINE_LENGTH = 88
INDENT = "  "

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _float(number: float) -> str:
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return repr(float(number))


def _fixed(number: float, decimals: int) -> str:
    """`number` rounded to `decimals` places, always written as a float."""
    rounded = round(number, decimals)
    return f"{rounded:.{decimals}f}" if decimals > 0 else f"{rounded:.1f}"


def significant(number: float, digits: int) -> str:
    """At most `digits` significant digits, without padding zeros.

    Args:
        number: The value.
        digits: Significant digits kept.

    Returns:
        A TOML float literal; 0.0625 stays exact, 8249.8 at four digits
        reads 8250.0.
    """
    if not math.isfinite(number):
        return _float(number)
    return _float(float(f"{number:.{digits}g}"))


def exponent(number: float, digits: int) -> str:
    """Short exponent form, `-7.5e-5` rather than `-7.5e-05`.

    Args:
        number: The value.
        digits: Significant digits kept.

    Returns:
        A TOML float literal; zero and non-finite values in plain form.
    """
    if number == 0 or not math.isfinite(number):
        return _float(number)
    mantissa, power = f"{number:.{digits - 1}e}".split("e")
    return f"{mantissa}e{int(power)}"


def round_to_error(mean: float, error: float) -> tuple[str, str]:
    """Format a mean and its standard error by the Particle Data Group rule.

    The three leading digits of the error decide how many it keeps: 100
    to 354 keep two, 355 to 949 keep one, and 950 to 999 round up to 1000
    and keep two. The mean is cut at the same decimal place. An error of
    zero or one that is not finite says nothing about the precision, so
    the mean is then written whole.

    Args:
        mean: The mean.
        error: Its standard error.

    Returns:
        The mean and the error as TOML float literals.
    """
    if error == 0 or not math.isfinite(error) or not math.isfinite(mean):
        return _float(mean), _float(error)
    power = math.floor(math.log10(abs(error)))
    leading = round(abs(error) / 10.0 ** (power - 2))
    # 950 and above round up to the next decade, where the place that
    # keeps one digit of the error keeps two of the rounded one
    decimals = 1 - power if leading <= 354 else -power
    return _fixed(mean, decimals), _fixed(error, decimals)


def key(name: str) -> str:
    """A TOML key: bare when it can be, quoted otherwise.

    Args:
        name: The key.

    Returns:
        `name` itself, or `name` as a quoted string.
    """
    return name if _BARE_KEY.match(name) else json.dumps(name)


def value(item) -> str:
    """A TOML literal for a JSON value.

    Args:
        item: A bool, int, float, string, or a list of those.

    Returns:
        The literal. An integer outside 64 bits, such as a seed entropy,
        becomes a string.

    Raises:
        TypeError: For any other type, None included.
    """
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, int):
        return str(item) if -(2**63) <= item < 2**63 else json.dumps(str(item))
    if isinstance(item, float):
        return _float(item)
    if isinstance(item, str):
        return json.dumps(item)
    if isinstance(item, list | tuple):
        return f"[{', '.join(value(element) for element in item)}]"
    raise TypeError(f"no TOML form for {type(item).__name__}")


def array(name: str, items: Sequence) -> list[str]:
    """`name = [...]` on one line if it fits, else items packed into lines.

    Args:
        name: The key, already formatted.
        items: The array elements.

    Returns:
        The lines.
    """
    single = f"{name} = {value(list(items))}"
    if len(single) <= LINE_LENGTH:
        return [single]
    lines, current = [f"{name} = ["], INDENT
    for item in items:
        text = f"{value(item)},"
        if len(current) + len(text) + 1 > LINE_LENGTH and current.strip():
            lines.append(current.rstrip())
            current = INDENT
        current += text + " "
    lines.append(current.rstrip())
    return [*lines, "]"]


def aligned(pairs: Iterable[tuple[str, str]]) -> list[str]:
    """`key = value` lines with the equals signs in one column.

    Args:
        pairs: Formatted keys and literals.

    Returns:
        The lines.
    """
    pairs = list(pairs)
    width = max((len(name) for name, _ in pairs), default=0)
    return [f"{name.ljust(width)} = {literal}" for name, literal in pairs]


def table(header: str, pairs: Iterable[tuple[str, str]]) -> list[str]:
    """A blank line, `[header]`, and its aligned pairs.

    Args:
        header: The dotted table name, already formatted.
        pairs: Formatted keys and literals.

    Returns:
        The lines.
    """
    return ["", f"[{header}]", *aligned(pairs)]


def comment(text: str) -> list[str]:
    """Every line of `text` as a TOML comment.

    Args:
        text: The comment, possibly several lines.

    Returns:
        The lines.
    """
    return [f"# {line}".rstrip() for line in text.splitlines()]
