"""The rearrangement benchmark, exercised at counts small enough for CI."""

import math

import pytest

from tetris.fast import DEFAULT_RUN_SELECTION
from tetris.rearrangement_bench import (
    DRAWS_PER_CONFIGURATION,
    REARRANGEMENT_METRICS,
    build_rearrangement_tasks,
    run_rearrangement_benchmark,
    selection_key,
)

SIZES = (6, 9, 12)
VARIANTS = ((0, 0), (0, -1), (-1, -1))
DEFAULT_KEY = "BOTTLENECK,CUMULATED,ATOMS,LEFTMOST"
BOTTLENECKS = ("construct_bottleneck", "elimination_bottleneck", "total_bottleneck")


def paired(run_selections, count=60, size=14, entropy=5):
    return run_rearrangement_benchmark(
        (size,),
        ((-1, -1),),
        configurations_per_task=count,
        workers=1,
        entropy=entropy,
        run_selections=run_selections,
    )[size]["r-1c-1"]


def test_selection_key() -> None:
    assert selection_key(DEFAULT_RUN_SELECTION) == DEFAULT_KEY
    assert selection_key(["LEFTMOST"]) == "LEFTMOST"
    assert selection_key(()) == ""
    assert selection_key(("ATOMS", "CUMULATED")) != selection_key(
        ("CUMULATED", "ATOMS")
    )


def test_tasks_cover_every_combination_once() -> None:
    tasks = build_rearrangement_tasks(SIZES, VARIANTS, entropy=1)
    assert sorted((size, reshape) for size, reshape, _ in tasks) == sorted(
        (size, reshape) for size in SIZES for reshape in VARIANTS
    )


def test_a_run_is_well_formed() -> None:
    results = run_rearrangement_benchmark(
        SIZES,
        VARIANTS,
        configurations_per_task=20,
        workers=2,
        entropy=11,
        run_selections=(DEFAULT_RUN_SELECTION, ("LEFTMOST",)),
    )
    assert list(results) == sorted(SIZES)
    for size, variants in results.items():
        assert set(variants) == {"r0c0", "r0c-1", "r-1c-1"}
        for entry in variants.values():
            assert entry["measured_configurations"] == 20
            assert entry["target_number_of_atoms"] == (
                entry["target_rows"] * entry["target_columns"]
            )
            assert list(entry["selections"]) == [DEFAULT_KEY, "LEFTMOST"]
            reference = entry["selections"][DEFAULT_KEY]
            other = entry["selections"]["LEFTMOST"]
            assert "difference_to_reference" not in reference
            assert set(other["difference_to_reference"]) == set(REARRANGEMENT_METRICS)
            for selection in (reference, other):
                assert set(selection["metrics"]) == set(REARRANGEMENT_METRICS)
                assert selection["construct_microseconds_per_row"] == pytest.approx(
                    selection["construct_cpu_seconds"] / (20 * size) * 1e6
                )
                assert selection[
                    "elimination_microseconds_per_column"
                ] == pytest.approx(
                    selection["elimination_cpu_seconds"]
                    / (20 * entry["target_columns"])
                    * 1e6
                )
                assert selection["construct_cpu_seconds"] > 0
                assert selection["elimination_cpu_seconds"] > 0


def test_totals_add_up() -> None:
    metrics = paired((DEFAULT_RUN_SELECTION,))["selections"][DEFAULT_KEY]["metrics"]
    for quantity in ("bottleneck", "cumulated", "moved_atoms"):
        assert metrics[f"total_{quantity}"]["mean"] == pytest.approx(
            metrics[f"construct_{quantity}"]["mean"]
            + metrics[f"elimination_{quantity}"]["mean"]
        )


def test_selections_are_measured_on_the_same_loadings() -> None:
    """A selection led by BOTTLENECK cannot change any move's bottleneck,
    so on shared loadings the difference is zero on every configuration,
    and its standard error with it. Two separate samples would not give
    exactly zero."""
    entry = paired((DEFAULT_RUN_SELECTION, ("BOTTLENECK", "LEFTMOST")))
    other = entry["selections"]["BOTTLENECK,LEFTMOST"]
    reference = entry["selections"][DEFAULT_KEY]
    for name in BOTTLENECKS:
        assert other["metrics"][name] == reference["metrics"][name]
        assert other["difference_to_reference"][name] == {
            "mean": 0.0,
            "standard_error": 0.0,
        }


def test_the_difference_is_taken_against_the_reference() -> None:
    entry = paired((DEFAULT_RUN_SELECTION, ("LEFTMOST",)))
    reference = entry["selections"][DEFAULT_KEY]["metrics"]
    other = entry["selections"]["LEFTMOST"]
    for name in REARRANGEMENT_METRICS:
        assert other["difference_to_reference"][name]["mean"] == pytest.approx(
            other["metrics"][name]["mean"] - reference[name]["mean"]
        )
    # the default minimises every move's bottleneck, which the leftmost
    # run can only match or exceed
    assert other["difference_to_reference"]["total_bottleneck"]["mean"] >= 0


def test_the_same_entropy_measures_the_same_loadings() -> None:
    def metrics(entropy):
        entry = paired((DEFAULT_RUN_SELECTION,), count=30, entropy=entropy)
        return entry["selections"][DEFAULT_KEY]["metrics"]

    assert metrics(9) == metrics(9)
    assert metrics(9) != metrics(10)


def test_a_target_no_loading_fills_ends_short() -> None:
    """A 10 by 10 target in a 10 by 10 loading needs every site filled,
    about one draw in 2**100. The task stops at its draw limit."""
    entry = run_rearrangement_benchmark(
        (10,), ((3, 3),), configurations_per_task=1, workers=1, entropy=2
    )[10]["r3c3"]
    assert DRAWS_PER_CONFIGURATION >= 1
    assert entry["measured_configurations"] == 0
    selection = entry["selections"][DEFAULT_KEY]
    assert math.isnan(selection["construct_microseconds_per_row"])
    assert math.isnan(selection["elimination_microseconds_per_column"])
    assert math.isnan(selection["metrics"]["total_bottleneck"]["mean"])


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"run_selections": (("LEFTMOST",), ("LEFTMOST",))}, ValueError),
        ({"run_selections": (("FASTEST",),)}, ValueError),
        ({"run_selections": ("LEFTMOST",)}, TypeError),
        ({"run_selections": ()}, ValueError),
        ({"configurations_per_task": 0}, ValueError),
        ({"workers": 0}, ValueError),
        ({"sizes": (1,)}, ValueError),
    ],
)
def test_malformed_arguments_are_refused_before_forking(overrides, error) -> None:
    arguments = {
        "sizes": (6,),
        "reshape_variants": ((0, 0),),
        "configurations_per_task": 1,
        "workers": 1,
        "entropy": 1,
        **overrides,
    }
    with pytest.raises(error):
        run_rearrangement_benchmark(**arguments)
