"""Benchmark cache metadata and its invalidation."""

import json

import pytest

from tetris.cache import (
    AcceptanceParameters,
    RearrangementParameters,
    load_results,
    save_results,
)


def acceptance(**overrides) -> AcceptanceParameters:
    defaults = {
        "sizes": (4, 5, 6),
        "reshape_variants": ((0, 0), (0, -1), (-1, -1)),
        "seconds_per_task": 60.0,
        "workers": 7,
        "algorithm_digest": "abc123",
    }
    return AcceptanceParameters(**{**defaults, **overrides})


def rearrangement(**overrides) -> RearrangementParameters:
    defaults = {
        "sizes": (4, 5, 6),
        "reshape_variants": ((0, 0), (0, -1), (-1, -1)),
        "configurations_per_task": 2000,
        "workers": 7,
        "algorithm_digest": "abc123",
    }
    return RearrangementParameters(**{**defaults, **overrides})


def test_round_trip(tmp_path) -> None:
    path = tmp_path / "stats.json"
    params = acceptance()
    save_results(path, params, entropy=999, results={4: {"r0c0": 1.0}})
    results, reason = load_results(path, params)
    assert reason == ""
    assert results == {4: {"r0c0": 1.0}}


def test_missing_file_is_not_an_error(tmp_path) -> None:
    results, reason = load_results(tmp_path / "absent.json", acceptance())
    assert results is None
    assert "no cache" in reason


def test_worker_count_does_not_invalidate(tmp_path) -> None:
    """The worker count is recorded for provenance but does not change
    the sampled distribution, so it must not throw the cache away."""
    path = tmp_path / "stats.json"
    save_results(path, acceptance(), entropy=1, results={4: {}})
    results, _ = load_results(path, acceptance(workers=3))
    assert results is not None


def test_corrupt_file_is_reported(tmp_path) -> None:
    path = tmp_path / "stats.json"
    path.write_text("{not json")
    results, reason = load_results(path, acceptance())
    assert results is None
    assert "unreadable" in reason


def test_entropy_is_recorded(tmp_path) -> None:
    path = tmp_path / "stats.json"
    save_results(path, acceptance(), entropy=4242, results={4: {}})
    meta = json.loads(path.read_text())["meta"]
    assert meta["seed_entropy"] == 4242
    assert "cpu_count" in meta


def test_reshape_variants_survive_the_json_round_trip(tmp_path) -> None:
    """A reshape is a tuple going in and a list coming back, so the
    fingerprint has to be computed over a form that does not depend on
    which side of the file it is read from."""
    path = tmp_path / "stats.json"
    params = acceptance()
    save_results(path, params, entropy=1, results={4: {}})
    meta = json.loads(path.read_text())["meta"]
    assert meta["reshape_variants"] == [[0, 0], [0, -1], [-1, -1]]
    results, reason = load_results(path, params)
    assert results is not None, reason


@pytest.mark.parametrize(
    ("build", "override"),
    [
        (acceptance, {"sizes": (4, 5)}),
        (acceptance, {"reshape_variants": ((-1, -1),)}),
        (acceptance, {"seconds_per_task": 30.0}),
        (acceptance, {"algorithm_digest": "deadbeef"}),
        (rearrangement, {"sizes": (4, 5)}),
        (rearrangement, {"reshape_variants": ((-1, -1),)}),
        (rearrangement, {"configurations_per_task": 1000}),
        (rearrangement, {"algorithm_digest": "deadbeef"}),
        (rearrangement, {"run_selections": (("LEFTMOST",),)}),
    ],
)
def test_a_changed_benchmark_parameter_invalidates(tmp_path, build, override) -> None:
    path = tmp_path / "stats.json"
    save_results(path, build(), entropy=1, results={4: {}})
    results, reason = load_results(path, build(**override))
    assert results is None
    assert "parameters" in reason


@pytest.mark.parametrize("build", [acceptance, rearrangement])
def test_a_benchmark_cache_round_trips(tmp_path, build) -> None:
    path = tmp_path / "stats.json"
    save_results(path, build(), entropy=5, results={4: {"r0c0": {}}})
    results, reason = load_results(path, build(workers=2))
    assert results == {4: {"r0c0": {}}}, reason
    meta = json.loads(path.read_text())["meta"]
    assert meta["workers"] == 7
    assert meta["seed_entropy"] == 5


def test_each_benchmark_names_itself_in_the_meta(tmp_path) -> None:
    path = tmp_path / "stats.json"
    save_results(path, rearrangement(), entropy=1, results={})
    meta = json.loads(path.read_text())["meta"]
    assert meta["benchmark"] == "rearrangement"
    assert meta["configurations_per_task"] == 2000
    assert meta["run_selections"] == [["BOTTLENECK", "CUMULATED", "ATOMS", "LEFTMOST"]]


def test_the_two_benchmarks_never_share_a_fingerprint() -> None:
    assert acceptance().fingerprint() != rearrangement().fingerprint()


def test_rearrangement_selections_survive_the_json_round_trip() -> None:
    assert (
        rearrangement(run_selections=[["BOTTLENECK", "LEFTMOST"]]).fingerprint()
        == rearrangement(run_selections=(("BOTTLENECK", "LEFTMOST"),)).fingerprint()
    )


def test_cpu_model_is_readable() -> None:
    from tetris.cache import cpu_model

    name = cpu_model()
    assert name
    assert name != "x86_64"


# --- measurements the notebook cannot recompute -----------------------
# The long sweeps run from a console over hours and land in files the
# notebook only reads. There is no fingerprint to check them against,
# because there is no parameter set the notebook would rerun them from;
# what does have to hold is that the algorithm they measured is the one
# in the tree.


def measurement_file(tmp_path, **meta_overrides):
    meta = {
        "algorithm_digest": "d96b341ea3c952bb",
        "seconds_per_task": 600.0,
        "band": "log",
        **meta_overrides,
    }
    path = tmp_path / "measurements.json"
    path.write_text(
        json.dumps({"meta": meta, "results": {"100": {"r0c0": {"kept": 3}}}})
    )
    return path


def test_measurements_come_back_keyed_by_integer_size(tmp_path) -> None:
    from tetris.cache import load_measurements

    results, meta = load_measurements(measurement_file(tmp_path), "d96b341ea3c952bb")
    assert results == {100: {"r0c0": {"kept": 3}}}
    assert meta["band"] == "log"


def test_a_measurement_of_another_algorithm_is_refused(tmp_path) -> None:
    from tetris.cache import load_measurements

    with pytest.raises(ValueError, match="digest"):
        load_measurements(measurement_file(tmp_path), "0000000000000000")


def test_a_missing_measurement_file_says_which_one(tmp_path) -> None:
    from tetris.cache import load_measurements

    with pytest.raises(FileNotFoundError, match="absent.json"):
        load_measurements(tmp_path / "absent.json", "d96b341ea3c952bb")


def test_a_truncated_measurement_file_is_refused(tmp_path) -> None:
    from tetris.cache import load_measurements

    path = tmp_path / "half.json"
    path.write_text('{"meta": {"algorithm_digest": "d96b341ea3c952bb"}}')
    with pytest.raises(ValueError, match="results"):
        load_measurements(path, "d96b341ea3c952bb")


def test_bands_merge_into_one_series(tmp_path) -> None:
    from tetris.cache import merge_measurements

    assert merge_measurements({4: {"r0c0": {}}}, {100: {"r0c0": {}}}) == {
        4: {"r0c0": {}},
        100: {"r0c0": {}},
    }


def test_bands_that_overlap_are_refused(tmp_path) -> None:
    from tetris.cache import merge_measurements

    # two bands measuring the same size under different budgets would
    # put two different numbers on one point of a curve
    with pytest.raises(ValueError, match="100"):
        merge_measurements({100: {"r0c0": {}}}, {100: {"r0c0": {}}})
