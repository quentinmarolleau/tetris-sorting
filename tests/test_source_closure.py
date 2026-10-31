"""The set of sources the digest covers, derived from the imports rather
than written down.

A hand-written list drifts in both directions: it had picked up
`config.py`, which no sweep executes, and missed the benchmark driver
and `stats.py`, which decide the sampling and three of the cached fields.
"""

from pathlib import Path

import pytest

from tetris.cache import algorithm_digest, source_closure

PACKAGE = Path(__file__).resolve().parents[1] / "tetris"
BENCHMARKS = ("tetris.acceptance_bench", "tetris.rearrangement_bench")


def names(paths) -> set[str]:
    return {path.stem for path in paths}


@pytest.mark.parametrize("root", BENCHMARKS)
def test_the_closure_holds_the_root_itself(root) -> None:
    """The sampling loop and the per-task seeding live in the root, so
    leaving it out would miss the module most able to move a number."""
    assert PACKAGE / f"{root.split('.')[-1]}.py" in set(source_closure(root))


@pytest.mark.parametrize("root", BENCHMARKS)
def test_the_class_and_its_rendering_stay_out(root) -> None:
    """The benchmarks never run the pedagogical path, so editing it must
    not cost a rerun."""
    assert names(source_closure(root)).isdisjoint(
        {"config", "display", "plotting", "analytic", "cache"}
    )


def test_the_package_init_is_not_followed() -> None:
    """`tetris/__init__.py` re-exports the class, so walking through it
    would drag in everything the re-export touches and undo the point."""
    for root in BENCHMARKS:
        assert "__init__" not in names(source_closure(root))


def test_every_path_exists_and_is_absolute() -> None:
    for path in source_closure("tetris.rearrangement_bench"):
        assert path.is_absolute(), path
        assert path.is_file(), path


def test_the_closure_is_sorted_and_free_of_repeats() -> None:
    paths = source_closure("tetris.rearrangement_bench")
    assert list(paths) == sorted(paths)
    assert len(set(paths)) == len(paths)


def test_a_different_root_gives_a_different_closure() -> None:
    assert names(source_closure("tetris.acceptance")) == {"acceptance"}
    assert names(source_closure("tetris.fast")) == {"acceptance", "fast"}
    assert names(source_closure("tetris.config")) == {
        "acceptance",
        "config",
        "display",
        "fast",
        "plotting",
    }


def test_an_unknown_root_is_refused() -> None:
    with pytest.raises(ValueError, match="no such module"):
        source_closure("tetris.nonexistent")


def test_the_digest_accepts_the_closure() -> None:
    paths = source_closure("tetris.acceptance_bench")
    digest = algorithm_digest(*paths)
    assert len(digest) == 16
    assert digest == algorithm_digest(*reversed(paths))


def test_the_acceptance_closure_leaves_the_run_selection_out() -> None:
    """Editing how an overfull line picks its run cannot move a success
    rate, so it must not invalidate the acceptance cache."""
    assert names(source_closure("tetris.acceptance_bench")) == {
        "acceptance",
        "acceptance_bench",
        "stats",
        "sweep",
        "variants",
    }


def test_the_rearrangement_closure() -> None:
    assert names(source_closure("tetris.rearrangement_bench")) == {
        "acceptance",
        "fast",
        "rearrangement_bench",
        "sweep",
        "variants",
    }


def write_module(directory: Path, source: str) -> Path:
    directory.mkdir()
    path = directory / "module.py"
    path.write_text(source)
    return path


CODE = '''"""Module docstring."""


class Window:
    """Class docstring."""

    def width(self) -> int:
        """Method docstring."""
        return 3


def kept(counts):
    """Function docstring."""
    return min(counts) >= 3
'''


def test_the_digest_ignores_docstrings_comments_and_layout(tmp_path) -> None:
    """Documentation cannot move a measured number, so rewording it must
    not cost a rerun of the sweeps."""
    reference = algorithm_digest(write_module(tmp_path / "a", CODE))
    reworded = (
        CODE.replace("Module docstring.", "Reworded, over\ntwo lines.")
        .replace('"""Class docstring."""\n', "")
        .replace("Function docstring.", "Args: counts.")
        .replace("return 3", "return 3  # a comment")
        .replace("min(counts) >= 3", "min(\n        counts\n    ) >= 3")
    )
    assert algorithm_digest(write_module(tmp_path / "b", reworded)) == reference


def test_the_digest_follows_a_change_of_code(tmp_path) -> None:
    reference = algorithm_digest(write_module(tmp_path / "a", CODE))
    edited = CODE.replace(">= 3", "> 3")
    assert algorithm_digest(write_module(tmp_path / "b", edited)) != reference


def test_a_string_statement_after_the_first_still_counts(tmp_path) -> None:
    """Only the leading string of a body is a docstring."""
    reference = algorithm_digest(write_module(tmp_path / "a", CODE))
    edited = CODE.replace("return 3", '"not a docstring"\n        return 3')
    assert algorithm_digest(write_module(tmp_path / "b", edited)) != reference
