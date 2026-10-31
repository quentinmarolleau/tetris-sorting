"""Names for the target reshapes the benchmark sweeps over.

A reshape is a (rows, columns) pair added to the natural target side.
`variant_key` turns one into the string a cached result is filed under;
`variant_label` turns it into the phrase a plot legend or a paragraph
uses. Neither decides anything the benchmark measures.
"""

__all__ = ["variant_key", "variant_label"]

_LABELS = {
    (0, 0): "No margin",
    (0, -1): "One column off",
    (-1, 0): "One row off",
    (-1, -1): "One row and one column off",
}


def variant_key(reshape: tuple[int, int]) -> str:
    """Key a reshape's results are filed under in a cache.

    A string survives a round trip through JSON, where a tuple key would
    not, and `r0c-1` stays readable in a diff of the cache.

    Args:
        reshape: (rows, columns) added to the natural target side.

    Returns:
        The key, e.g. `r0c-1` for (0, -1).
    """
    rows, columns = reshape
    return f"r{int(rows)}c{int(columns)}"


def variant_label(reshape: tuple[int, int]) -> str:
    """Readable name of a reshape, for plot legends and prose.

    Args:
        reshape: (rows, columns) added to the natural target side.

    Returns:
        A fixed name such as "One column off" for the four reshapes the
        notebook uses, otherwise the signed offsets, e.g.
        "+1 rows, -2 columns".
    """
    rows, columns = int(reshape[0]), int(reshape[1])
    named = _LABELS.get((rows, columns))
    if named is not None:
        return named
    return f"{rows:+d} rows, {columns:+d} columns"
