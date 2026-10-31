"""The acceptance benchmark, exercised at a budget small enough for CI."""

import pytest

from tetris.acceptance_bench import build_acceptance_tasks, run_acceptance_benchmark

SIZES = (6, 9, 12)
VARIANTS = ((0, 0), (0, -1), (-1, -1))
KEYS = {"r0c0", "r0c-1", "r-1c-1"}
FIELDS = {
    "target_rows",
    "target_columns",
    "target_number_of_atoms",
    "contraction_ratio",
    "kept",
    "samples",
    "success_rate",
    "standard_error",
    "wilson_low",
    "wilson_high",
    "acceptance_cpu_seconds",
    "microseconds_per_configuration",
    "wall_seconds",
    "min_concurrency",
}


def test_tasks_cover_every_combination_once() -> None:
    tasks = build_acceptance_tasks(SIZES, VARIANTS, entropy=1)
    assert sorted((size, reshape) for size, reshape, _ in tasks) == sorted(
        (size, reshape) for size in SIZES for reshape in VARIANTS
    )


def test_task_order_is_shuffled_but_reproducible() -> None:
    sizes = tuple(range(4, 40))

    def order(entropy):
        return [task[:2] for task in build_acceptance_tasks(sizes, VARIANTS, entropy)]

    assert order(7) == order(7)
    assert order(7) != [(size, reshape) for size in sizes for reshape in VARIANTS]


def test_the_loadings_of_a_default_sweep_are_kept() -> None:
    """The one-driver cache recorded its entropy and spawned seeds in
    (size, reshape) order. The same entropy must still hand each
    (size, reshape) the same stream."""
    tasks = build_acceptance_tasks(SIZES, VARIANTS, entropy=21)
    seeds = {(size, reshape): seed.spawn_key for size, reshape, seed in tasks}
    combinations = [(size, reshape) for size in SIZES for reshape in VARIANTS]
    assert [seeds[combination] for combination in combinations] == [
        (index,) for index in range(len(combinations))
    ]


def test_a_run_is_well_formed() -> None:
    results = run_acceptance_benchmark(
        SIZES, VARIANTS, seconds_per_task=0.3, workers=2, entropy=11
    )
    assert list(results) == sorted(SIZES)
    for size, variants in results.items():
        assert set(variants) == KEYS
        for entry in variants.values():
            assert set(entry) == FIELDS
            assert 0 < entry["samples"]
            assert entry["kept"] <= entry["samples"]
            assert entry["wilson_low"] <= entry["success_rate"] <= entry["wilson_high"]
            assert entry["target_number_of_atoms"] == (
                entry["target_rows"] * entry["target_columns"]
            )
            assert entry["contraction_ratio"] == pytest.approx(
                entry["target_number_of_atoms"] / size**2
            )
            assert 1 <= entry["min_concurrency"] <= 2


def test_cost_comes_from_measured_cpu_time() -> None:
    """The nominal budget must not be reused as the measured cost: that
    is what made an under-subscribed batch look like a faster algorithm."""
    budget = 0.3
    results = run_acceptance_benchmark(
        SIZES, ((-1, -1),), seconds_per_task=budget, workers=2, entropy=12
    )
    measured = []
    for variants in results.values():
        entry = variants["r-1c-1"]
        measured.append(entry["acceptance_cpu_seconds"])
        assert entry["microseconds_per_configuration"] == pytest.approx(
            entry["acceptance_cpu_seconds"] / entry["samples"] * 1e6
        )
        # the loop tests its deadline before each draw, so it overruns
        assert entry["wall_seconds"] >= budget
        assert entry["acceptance_cpu_seconds"] > 0
    # three tasks, three different CPU times: the number is measured, not
    # the budget played back
    assert len(set(measured)) == len(measured)


@pytest.mark.parametrize(
    "overrides",
    [
        {"workers": 0},
        {"sizes": (1,)},
        {"reshape_variants": ((0, 50),)},
    ],
)
def test_malformed_arguments_are_refused_before_forking(overrides) -> None:
    arguments = {
        "sizes": (6,),
        "reshape_variants": ((0, 0),),
        "seconds_per_task": 0.01,
        "workers": 1,
        "entropy": 1,
        **overrides,
    }
    with pytest.raises(ValueError):
        run_acceptance_benchmark(**arguments)
