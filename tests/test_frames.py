"""Flat DataFrames built from the nested benchmark caches."""

import math

import pytest

from tetris.frames import (
    acceptance_frame,
    against_reference,
    overfull_columns_frame,
    overfull_rows_frame,
    rearrangement_frame,
    with_full_load,
)

RESHAPES = ((0, 0), (-1, -1))
DEFAULT = ("BOTTLENECK", "CUMULATED", "ATOMS", "LEFTMOST")
SELECTIONS = (DEFAULT, ("LEFTMOST",))
METRICS = (
    "construct_bottleneck",
    "elimination_bottleneck",
    "total_bottleneck",
)


def _acceptance_entry(size: int, rate: float, concurrency: int = 23) -> dict:
    return {
        "target_rows": size - 3,
        "target_columns": size - 3,
        "success_rate": rate,
        "wilson_low": rate - 0.01,
        "wilson_high": rate + 0.01,
        "min_concurrency": concurrency,
    }


def _acceptance_stats() -> dict[int, dict]:
    # variants deliberately out of sweep order, as the JSON files hold them
    return {
        10: {"r-1c-1": _acceptance_entry(10, 0.9), "r0c0": _acceptance_entry(10, 0.5)},
        11: {
            "r0c0": _acceptance_entry(11, 0.6, 20),
            "r-1c-1": _acceptance_entry(11, 0.95),
        },
    }


def _metrics(base: float) -> dict:
    return {
        name: {"mean": base + index, "standard_error": 0.1 * (index + 1)}
        for index, name in enumerate(METRICS)
    }


def _rearrangement_entry(size: int, base: float) -> dict:
    return {
        "target_number_of_atoms": (size - 3) ** 2,
        "min_concurrency": 23,
        "selections": {
            "LEFTMOST": {
                "construct_microseconds_per_row": 20.0,
                "metrics": _metrics(base + 10),
                "difference_to_reference": _metrics(10),
            },
            ",".join(DEFAULT): {
                "construct_microseconds_per_row": 30.0,
                "metrics": _metrics(base),
            },
        },
    }


def _rearrangement_stats() -> dict[int, dict]:
    return {
        size: {
            "r0c0": _rearrangement_entry(size, 100.0),
            "r-1c-1": _rearrangement_entry(size, 200.0),
        }
        for size in (10, 20)
    }


def test_acceptance_has_one_row_per_size_and_reshape() -> None:
    frame = acceptance_frame(_acceptance_stats(), RESHAPES)
    assert len(frame) == 4
    assert list(frame["size"]) == [10, 10, 11, 11]


def test_acceptance_variants_follow_the_sweep_order() -> None:
    frame = acceptance_frame(_acceptance_stats(), RESHAPES)
    assert list(frame["variant"].cat.categories) == ["r0c0", "r-1c-1"]
    assert list(frame["variant"]) == ["r0c0", "r-1c-1", "r0c0", "r-1c-1"]
    groups = [name for name, _ in frame.groupby("variant", observed=True)]
    assert groups == ["r0c0", "r-1c-1"]


def test_acceptance_carries_the_reshape_and_its_label() -> None:
    frame = acceptance_frame(_acceptance_stats(), RESHAPES)
    margin = frame[frame["variant"] == "r-1c-1"].iloc[0]
    assert (margin["reshape_rows"], margin["reshape_columns"]) == (-1, -1)
    assert margin["label"] == "One row and one column off"


def test_acceptance_keeps_every_cached_field() -> None:
    frame = acceptance_frame(_acceptance_stats(), RESHAPES)
    row = frame[(frame["size"] == 11) & (frame["variant"] == "r0c0")].iloc[0]
    assert row["success_rate"] == 0.6
    assert row["wilson_high"] == pytest.approx(0.61)
    assert row["min_concurrency"] == 20


def test_acceptance_drops_variants_not_asked_for() -> None:
    frame = acceptance_frame(_acceptance_stats(), ((0, 0),))
    assert set(frame["variant"]) == {"r0c0"}


def test_acceptance_fails_on_a_missing_variant() -> None:
    with pytest.raises(KeyError):
        acceptance_frame(_acceptance_stats(), ((0, -1),))


def test_rearrangement_has_one_row_per_size_reshape_and_selection() -> None:
    frame = rearrangement_frame(_rearrangement_stats(), RESHAPES, SELECTIONS)
    assert len(frame) == 2 * 2 * 2
    assert list(frame["selection"].cat.categories) == [",".join(DEFAULT), "LEFTMOST"]
    assert frame["is_reference"].to_numpy().sum() == 4


def test_rearrangement_flattens_the_metrics() -> None:
    frame = rearrangement_frame(_rearrangement_stats(), RESHAPES, SELECTIONS)
    row = frame[
        (frame["size"] == 20) & (frame["variant"] == "r-1c-1") & frame["is_reference"]
    ].iloc[0]
    assert row["total_bottleneck"] == 202.0
    assert row["total_bottleneck_error"] == pytest.approx(0.3)
    assert row["construct_microseconds_per_row"] == 30.0
    assert row["target_number_of_atoms"] == 289
    assert math.isnan(row["total_bottleneck_difference"])


def test_rearrangement_keeps_the_paired_difference() -> None:
    frame = rearrangement_frame(_rearrangement_stats(), RESHAPES, SELECTIONS)
    row = frame[(frame["size"] == 10) & (frame["selection"] == "LEFTMOST")].iloc[0]
    assert row["elimination_bottleneck_difference"] == 11.0
    assert row["elimination_bottleneck_difference_error"] == pytest.approx(0.2)


def test_against_reference_divides_by_the_reference_mean() -> None:
    frame = rearrangement_frame(_rearrangement_stats(), RESHAPES, SELECTIONS)
    relative = against_reference(frame, "total_bottleneck")
    assert len(relative) == 4
    assert set(relative["selection"]) == {"LEFTMOST"}
    row = relative[(relative["size"] == 10) & (relative["variant"] == "r0c0")].iloc[0]
    # difference 12 on a reference of 102
    assert row["relative_difference"] == pytest.approx(100 * 12 / 102)
    assert row["relative_difference_error"] == pytest.approx(100 * 0.3 / 102)


def test_overfull_columns_names_the_three_statistics() -> None:
    results = [
        {
            "size": 10,
            "reshape": [-1, -1],
            "mean": [0.9, 2.0, 0.2],
            "error": [0.01, 0.02, 0.03],
        },
        {
            "size": 10,
            "reshape": [0, 0],
            "mean": [0.5, 1.0, 0.1],
            "error": [0.04, 0.05, 0.06],
        },
    ]
    frame = overfull_columns_frame(results, RESHAPES)
    assert list(frame["variant"]) == ["r0c0", "r-1c-1"]
    row = frame.iloc[1]
    assert (row["overfull_share"], row["surplus"], row["bottleneck_saving"]) == (
        0.9,
        2.0,
        0.2,
    )
    assert row["bottleneck_saving_error"] == 0.03


def test_overfull_rows_turns_missing_keys_into_nan() -> None:
    results = [
        {"size": 20, "agreement_fraction": 0.8},
        {"size": 10, "exact_seconds_per_row": 1e-6},
    ]
    frame = overfull_rows_frame(results)
    assert list(frame["size"]) == [10, 20]
    assert math.isnan(frame.iloc[0]["agreement_fraction"])


def test_full_load_keeps_the_busiest_tasks_only() -> None:
    frame = with_full_load(acceptance_frame(_acceptance_stats(), RESHAPES))
    assert len(frame) == 3
    assert set(frame["min_concurrency"]) == {23}
