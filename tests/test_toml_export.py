"""The readable TOML rendering of the two benchmark caches."""

import json
import math
import tomllib

import pytest

from tetris.rearrangement_bench import REARRANGEMENT_METRICS
from tetris.toml_export import convert, main, to_toml

DEFAULT_KEY = "BOTTLENECK,CUMULATED,ATOMS,LEFTMOST"
ENTROPY = 115203644748788470937368678040190589606
NAN = float("nan")


def meta(benchmark: str, **fields) -> dict:
    return {
        "date": "2026-09-16T23:00:36.407908+00:00",
        "fingerprint": "1221cd4c47df65e9",
        "benchmark": benchmark,
        "sizes": list(range(4, 101)),
        "reshape_variants": [[0, 0], [-1, -1]],
        "algorithm_digest": "041c31e52ebd666a",
        "workers": 23,
        "seed_entropy": ENTROPY,
        "cpu_count": 24,
        "cpu_model": "AMD Ryzen 9 3900X 12-Core Processor",
        **fields,
    }


def acceptance_entry(**overrides) -> dict:
    return {
        "target_rows": 70,
        "target_columns": 69,
        "target_number_of_atoms": 4830,
        "contraction_ratio": 0.483,
        "kept": 6995,
        "samples": 6997,
        "success_rate": 0.9997141632128055,
        "standard_error": 0.00020208824221771814,
        "wilson_low": 0.9994284353486659,
        "wilson_high": 0.99985707479709,
        "acceptance_cpu_seconds": 2.2872990000000186,
        "microseconds_per_configuration": 326.89709875661265,
        "wall_seconds": 60.00339426699793,
        "min_concurrency": 23,
        **overrides,
    }


def acceptance_payload(**meta_fields) -> dict:
    return {
        "meta": meta("acceptance", seconds_per_task=60.0, **meta_fields),
        "results": {
            "100": {"r-1c-1": acceptance_entry(), "r0c0": acceptance_entry()},
            "4": {"r0c0": acceptance_entry()},
        },
    }


def metrics(mean=2.0020375508638617, error=0.0030031076170657155) -> dict:
    return {
        name: {"mean": mean, "standard_error": error} for name in REARRANGEMENT_METRICS
    }


def rearrangement_entry(measured=2000) -> dict:
    nan_metrics = metrics(NAN, NAN)
    return {
        "target_rows": 70,
        "target_columns": 69,
        "target_number_of_atoms": 4830,
        "measured_configurations": measured,
        "wall_seconds": 31.5,
        "min_concurrency": 23,
        "selections": {
            DEFAULT_KEY: {
                "construct_cpu_seconds": 9.12345,
                "elimination_cpu_seconds": 6.54321,
                "construct_microseconds_per_row": 45.61725,
                "elimination_microseconds_per_column": 47.4145652,
                "metrics": metrics() if measured else nan_metrics,
            },
            "LEFTMOST": {
                "construct_cpu_seconds": 8.0,
                "elimination_cpu_seconds": 6.0,
                "construct_microseconds_per_row": 40.0,
                "elimination_microseconds_per_column": 43.47826,
                "metrics": metrics() if measured else nan_metrics,
                "difference_to_reference": metrics(0.5, 0.0123)
                if measured
                else nan_metrics,
            },
        },
    }


def rearrangement_payload() -> dict:
    return {
        "meta": meta(
            "rearrangement",
            configurations_per_task=2000,
            run_selections=[DEFAULT_KEY.split(","), ["LEFTMOST"]],
        ),
        "results": {
            "100": {"r-1c-1": rearrangement_entry(), "r0c0": rearrangement_entry(0)}
        },
    }


def parsed(payload: dict) -> dict:
    return tomllib.loads(to_toml(payload))


def test_both_caches_render_as_valid_toml() -> None:
    for payload in (acceptance_payload(), rearrangement_payload()):
        assert set(parsed(payload)) == {"meta", "results"}


def test_every_line_fits() -> None:
    for payload in (acceptance_payload(), rearrangement_payload()):
        assert all(len(line) <= 88 for line in to_toml(payload).splitlines())


def test_an_unknown_benchmark_is_refused() -> None:
    payload = acceptance_payload()
    payload["meta"]["benchmark"] = "legacy"
    with pytest.raises(ValueError, match="legacy"):
        to_toml(payload)
    del payload["meta"]["benchmark"]
    with pytest.raises(ValueError):
        to_toml(payload)


def test_meta_is_kept() -> None:
    document = parsed(rearrangement_payload())["meta"]
    assert document["benchmark"] == "rearrangement"
    assert document["sizes"] == list(range(4, 101))
    assert document["run_selections"] == [DEFAULT_KEY.split(","), ["LEFTMOST"]]
    assert document["seed_entropy"] == str(ENTROPY)
    assert document["configurations_per_task"] == 2000


def test_extra_meta_is_carried_and_none_left_out() -> None:
    document = parsed(acceptance_payload(note="rerun", nothing=None))["meta"]
    assert document["note"] == "rerun"
    assert "nothing" not in document


def test_sizes_are_ascending_and_tasks_follow_the_sweep_order() -> None:
    text = to_toml(acceptance_payload())
    headers = ["[results.4.r0c0]", "[results.100.r0c0]", "[results.100.r-1c-1]"]
    positions = [text.index(header) for header in headers]
    assert positions == sorted(positions)


def test_acceptance_fields_are_grouped() -> None:
    task = parsed(acceptance_payload())["results"]["100"]["r0c0"]
    assert set(task) == {
        "wall_seconds",
        "min_concurrency",
        "target_geometry",
        "success_rate",
        "acceptance_cost",
    }
    assert task["target_geometry"] == {
        "target_rows": 70,
        "target_columns": 69,
        "target_number_of_atoms": 4830,
        "contraction_ratio": 0.483,
    }
    assert task["acceptance_cost"] == {
        "acceptance_cpu_seconds": 2.287,
        "microseconds_per_configuration": 326.9,
    }
    assert task["wall_seconds"] == 60.0


def test_success_rate_carries_a_range_and_a_span() -> None:
    success = parsed(acceptance_payload())["results"]["100"]["r0c0"]["success_rate"]
    assert success["success_rate"] == 0.99971
    assert success["standard_error"] == 0.0002
    assert success["wilson_range"] == [0.9994284353486659, 0.99985707479709]
    assert success["wilson_span"] == [-2.9e-4, 1.4e-4]
    assert "wilson_low" not in success


def test_unknown_task_fields_are_not_lost() -> None:
    payload = acceptance_payload()
    payload["results"]["4"]["r0c0"]["extra_field"] = 1.5
    assert parsed(payload)["results"]["4"]["r0c0"]["extra_field"] == 1.5


def test_unknown_reshapes_come_last() -> None:
    payload = acceptance_payload()
    payload["results"]["4"]["r5c5"] = acceptance_entry()
    text = to_toml(payload)
    assert text.index("[results.4.r0c0]") < text.index("[results.4.r5c5]")


def test_rearrangement_selections_are_tables() -> None:
    task = parsed(rearrangement_payload())["results"]["100"]["r-1c-1"]
    assert task["measured_configurations"] == 2000
    assert list(task["selections"]) == [DEFAULT_KEY, "LEFTMOST"]
    assert task["target_geometry"]["target_number_of_atoms"] == 4830
    reference = task["selections"][DEFAULT_KEY]
    other = task["selections"]["LEFTMOST"]
    assert "difference_to_reference" not in reference
    assert reference["construct_cpu_seconds"] == 9.123
    assert reference["construct_microseconds_per_row"] == 45.62
    assert reference["elimination_microseconds_per_column"] == 47.41
    for name in REARRANGEMENT_METRICS:
        assert reference[name] == {"mean": 2.002, "standard_error": 0.003}
        assert other["difference_to_reference"][name] == {
            "mean": 0.5,
            "standard_error": 0.012,
        }


def test_selection_keys_are_quoted_and_metrics_rounded() -> None:
    text = to_toml(rearrangement_payload())
    assert f'[results.100.r-1c-1.selections."{DEFAULT_KEY}"]' in text
    assert "[results.100.r-1c-1.selections.LEFTMOST.difference_to_reference]" in text
    assert "{ mean = 2.0020, standard_error = 0.0030 }" in text
    assert "{ mean = 0.500, standard_error = 0.012 }" in text


def test_unmeasured_metrics_are_nan() -> None:
    task = parsed(rearrangement_payload())["results"]["100"]["r0c0"]
    selection = task["selections"][DEFAULT_KEY]
    assert math.isnan(selection["total_bottleneck"]["mean"])


def test_header_documents_each_benchmark() -> None:
    acceptance = to_toml(acceptance_payload()).split("\n[meta]")[0]
    rearrangement = to_toml(rearrangement_payload()).split("\n[meta]")[0]
    for field in ("wilson_span", "acceptance_cpu_seconds", "Particle Data Group"):
        assert field in acceptance
    assert "difference_to_reference" not in acceptance
    for field in (
        "The first is the reference",
        "difference_to_reference",
        "moved_atoms",
        "max_a d(m, a)",
        "construct_microseconds_per_row",
        "no motion is simulated",
        "2 cov(a, b)",
    ):
        assert field in rearrangement
    assert "wilson_span" not in rearrangement


def test_convert_writes_beside_the_json(tmp_path) -> None:
    source = tmp_path / "acceptance_stats.json"
    source.write_text(json.dumps(acceptance_payload()))
    written = convert(source)
    assert written == tmp_path / "acceptance_stats.toml"
    assert tomllib.loads(written.read_text())["meta"]["workers"] == 23
    assert "acceptance_stats.json" in written.read_text().splitlines()[1]


def test_main_takes_an_output_path(tmp_path, capsys) -> None:
    source = tmp_path / "rearrangement_stats.json"
    source.write_text(json.dumps(rearrangement_payload()))
    target = tmp_path / "out" / "readable.toml"
    assert main([str(source), "-o", str(target)]) == 0
    assert target.exists()
    assert str(target) in capsys.readouterr().out
