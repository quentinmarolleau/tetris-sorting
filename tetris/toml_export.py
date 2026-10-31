"""Readable TOML rendering of a benchmark cache.

A JSON cache is written for `load_results`: every float at full
precision, one line per list element, no word on what a field means.
This module turns an acceptance or a rearrangement cache into a file
meant for reading. Fields are grouped by what they measure, each mean is
cut to the digits its standard error supports, and a header explains
every field. The JSON stays the source of truth; nothing reads the TOML
back.

    uv run python -m tetris.toml_export acceptance_stats.json [-o out.toml]
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from tetris import toml_format as fmt
from tetris.rearrangement_bench import REARRANGEMENT_METRICS
from tetris.variants import variant_key

__all__ = ["convert", "main", "to_toml"]

TIMING_DIGITS = 4
SPAN_DIGITS = 2

_META_ORDER = (
    "benchmark",
    "date",
    "fingerprint",
    "algorithm_digest",
    "seed_entropy",
    "sizes",
    "reshape_variants",
    "seconds_per_task",
    "configurations_per_task",
    "run_selections",
    "workers",
    "cpu_count",
    "cpu_model",
)
_TIMINGS = frozenset(
    {
        "wall_seconds",
        "acceptance_cpu_seconds",
        "microseconds_per_configuration",
        "construct_cpu_seconds",
        "elimination_cpu_seconds",
        "construct_microseconds_per_row",
        "elimination_microseconds_per_column",
        "contraction_ratio",
    }
)

_INTRODUCTION = """\
{benchmark} benchmark of the Tetris sorting algorithm, rendered from
{source} for reading. Regenerate with
  uv run python -m tetris.toml_export {source}
rather than editing this file; nothing reads it back.

[meta] describes the run. `fingerprint` hashes every parameter that
moves the numbers, `algorithm_digest` the sources this benchmark runs.
`seed_entropy` is a string when it exceeds TOML's 64-bit integers.

[results.<L>.<reshape>] holds one task: loading size L and the target
reshape `r<rows>c<columns>` added to the natural side.
  wall_seconds     elapsed time of the task
  min_concurrency  fewest busy workers seen at the start or end of the
                   task; lower at the tail of a sweep
"""

_ACCEPTANCE = """
.target_geometry
  target_rows, target_columns  target window shape
  target_number_of_atoms       target_rows * target_columns
  contraction_ratio            target_number_of_atoms / L^2

.success_rate: loadings drawn at p = 1/2 until the task's time budget
runs out, each tested for acceptance
  kept, samples    accepted loadings, drawn loadings
  success_rate     kept / samples
  standard_error   sqrt(rate * (1 - rate) / samples)
  wilson_range     Wilson score interval at one standard deviation
  wilson_span      wilson_range minus success_rate, two digits; the
                   interval is not symmetric near 1

.acceptance_cost: CPU time of the worker process
  acceptance_cpu_seconds          drawing and testing every loading
  microseconds_per_configuration  acceptance_cpu_seconds / samples
  wall_seconds exceeds acceptance_cpu_seconds by the time the process
  spent off the CPU, and by the one draw that crosses the deadline.
"""

_REARRANGEMENT = """
The task draws loadings until configurations_per_task are kept, and
measures each kept one under every run selection, so the selections are
compared on the same loadings.
  measured_configurations  kept loadings measured; below
                           configurations_per_task only when the task
                           hit its draw limit

.target_geometry
  target_rows, target_columns  target window shape
  target_number_of_atoms       target_rows * target_columns

.selections.<criteria>: one run selection, criteria joined by commas, in
the order of meta.run_selections. The first is the reference.

  Timing: process CPU time of the decision only, that is reading each
  line's atom positions, assigning target sites, choosing the run of an
  overfull line and writing the move down. Drawing and testing loadings,
  scoring moves and statistics are not counted.
  construct_cpu_seconds                row step, over all measured loadings
  elimination_cpu_seconds              column step, likewise
  construct_microseconds_per_row       construct_cpu_seconds
                                       / (measured_configurations * L)
  elimination_microseconds_per_column  elimination_cpu_seconds
                                       / (measured_configurations
                                          * target_columns)
  These are Python and NumPy timings: a line costs a few NumPy calls of
  about a microsecond each, flat in L, so they bound a compiled
  controller's cost from above. With one worker per hardware thread they
  also read about twice what a lone process measures.

  Metrics, per loading, then mean and standard error over the loadings.
  A move shifts the atoms of one line in parallel: the construct makes
  one per loading row (L moves), the elimination one per target column
  (target_columns moves). For atom a of move m, d(m, a) is the distance
  in lattice sites between its source and destination site along that
  line. Every value follows from the planned sources and destinations;
  no motion is simulated.
    <step>_bottleneck   = sum over the step's moves m of max_a d(m, a).
                          A parallel move lasts as long as its longest
                          displacement, so this is proportional to the
                          step's duration. total_bottleneck is Wang et
                          al.'s number of parallel displacements.
    <step>_cumulated    = sum over moves of sum_a d(m, a): total travel
                          of all atoms, in sites.
    <step>_moved_atoms  = sum over moves of the number of atoms with
                          d(m, a) > 0. An atom moved in both steps counts
                          once in each, so twice in the total.
    step: construct, elimination, or total = construct + elimination
    of the same loading.
  standard_error is that of the mean, sqrt(s^2 / n) with s^2 the sample
  variance over the n = measured_configurations loadings. nan when
  nothing was measured.

.selections.<criteria>.difference_to_reference: for each loading, this
selection's value minus the reference's, then mean and standard error
of those differences. Because both are measured on the same loadings,
the error is the paired one, sqrt((var a + var b - 2 cov(a, b)) / n),
usually far below either mean's error. Absent on the reference. The
difference between two non-reference selections is not stored: its
mean follows by subtraction, its error does not.
"""

_PRECISION = """
Precision. A standard error follows the Particle Data Group rule: its
three leading digits 100-354 keep two significant digits, 355-949 keep
one, 950-999 round up to 1000 and keep two. The mean is cut at the same
decimal place: 2.00204 +/- 0.00300 reads 2.0020 and 0.0030. Timings
and contraction_ratio keep four significant digits; counts and
wilson_range are exact.
"""


def _field(name: str, item) -> str:
    if name in _TIMINGS and isinstance(item, float):
        return fmt.significant(item, TIMING_DIGITS)
    return fmt.value(item)


def _pairs(entry: dict, names: Sequence[str]) -> list[tuple[str, str]]:
    return [(name, _field(name, entry[name])) for name in names if name in entry]


def _others(entry: dict, rendered: set[str]) -> list[tuple[str, str]]:
    """Fields a renderer does not know, kept rather than dropped."""
    return [
        (fmt.key(name), _field(name, item))
        for name, item in entry.items()
        if name not in rendered and item is not None and not isinstance(item, dict)
    ]


def _metric(metric: dict) -> str:
    mean, error = fmt.round_to_error(metric["mean"], metric["standard_error"])
    return f"{{ mean = {mean}, standard_error = {error} }}"


def _metrics(metrics: dict) -> list[tuple[str, str]]:
    return [
        (name, _metric(metrics[name]))
        for name in REARRANGEMENT_METRICS
        if name in metrics
    ]


_ACCEPTANCE_TOP = ("wall_seconds", "min_concurrency")
_GEOMETRY = (
    "target_rows",
    "target_columns",
    "target_number_of_atoms",
    "contraction_ratio",
)
_SUCCESS = (
    "kept",
    "samples",
    "success_rate",
    "standard_error",
    "wilson_low",
    "wilson_high",
)
_ACCEPTANCE_COST = ("acceptance_cpu_seconds", "microseconds_per_configuration")


def _acceptance_task(header: str, entry: dict) -> list[str]:
    rendered = {*_ACCEPTANCE_TOP, *_GEOMETRY, *_SUCCESS, *_ACCEPTANCE_COST}
    lines = fmt.table(header, _pairs(entry, _ACCEPTANCE_TOP) + _others(entry, rendered))
    lines += fmt.table(f"{header}.target_geometry", _pairs(entry, _GEOMETRY))

    rate, error = entry["success_rate"], entry["standard_error"]
    low, high = entry["wilson_low"], entry["wilson_high"]
    shown_rate, shown_error = fmt.round_to_error(rate, error)
    span = [
        fmt.exponent(low - rate, SPAN_DIGITS),
        fmt.exponent(high - rate, SPAN_DIGITS),
    ]
    lines += fmt.table(
        f"{header}.success_rate",
        [
            ("kept", fmt.value(entry["kept"])),
            ("samples", fmt.value(entry["samples"])),
            ("success_rate", shown_rate),
            ("standard_error", shown_error),
            ("wilson_range", fmt.value([low, high])),
            ("wilson_span", f"[{', '.join(span)}]"),
        ],
    )
    lines += fmt.table(f"{header}.acceptance_cost", _pairs(entry, _ACCEPTANCE_COST))
    return lines


_REARRANGEMENT_TOP = (
    "measured_configurations",
    "wall_seconds",
    "min_concurrency",
)
_SELECTION_COST = (
    "construct_cpu_seconds",
    "elimination_cpu_seconds",
    "construct_microseconds_per_row",
    "elimination_microseconds_per_column",
)


def _rearrangement_task(header: str, entry: dict) -> list[str]:
    rendered = {*_REARRANGEMENT_TOP, *_GEOMETRY, "selections"}
    lines = fmt.table(
        header, _pairs(entry, _REARRANGEMENT_TOP) + _others(entry, rendered)
    )
    lines += fmt.table(f"{header}.target_geometry", _pairs(entry, _GEOMETRY))
    for name, selection in entry.get("selections", {}).items():
        selection_header = f"{header}.selections.{fmt.key(name)}"
        lines += fmt.table(
            selection_header,
            _pairs(selection, _SELECTION_COST) + _metrics(selection.get("metrics", {})),
        )
        if "difference_to_reference" in selection:
            lines += fmt.table(
                f"{selection_header}.difference_to_reference",
                _metrics(selection["difference_to_reference"]),
            )
    return lines


def _task_order(meta: dict, tasks: dict) -> list[str]:
    """Task keys in the order the sweep declares them, unknown keys last."""
    declared = [
        variant_key(tuple(reshape)) for reshape in meta.get("reshape_variants", [])
    ]
    known = [name for name in declared if name in tasks]
    return known + sorted(name for name in tasks if name not in known)


def _meta(meta: dict) -> list[str]:
    lines = ["[meta]"]
    names = [name for name in _META_ORDER if name in meta]
    names += [name for name in meta if name not in _META_ORDER]
    for name in names:
        item = meta[name]
        if item is None:
            continue
        if name == "sizes" and item:
            lines.append(f"# {len(item)} sizes, {min(item)} to {max(item)}")
        if isinstance(item, list | tuple):
            lines += fmt.array(fmt.key(name), item)
        else:
            lines.append(f"{fmt.key(name)} = {fmt.value(item)}")
    return lines


_RENDERERS: dict[str, tuple[str, Callable[[str, dict], list[str]]]] = {
    "acceptance": (_ACCEPTANCE, _acceptance_task),
    "rearrangement": (_REARRANGEMENT, _rearrangement_task),
}


def to_toml(payload: dict, source: str = "stats.json") -> str:
    """Render a benchmark cache as commented, grouped TOML.

    Args:
        payload: The cache as `tetris.cache.save_results` writes it, with
            `meta` naming the benchmark and `results` keyed by size as a
            string, then by reshape.
        source: File name quoted in the header.

    Returns:
        The TOML document.

    Raises:
        ValueError: If `meta.benchmark` is missing or names no known
            benchmark.
    """
    meta = payload.get("meta", {})
    benchmark = meta.get("benchmark")
    if benchmark not in _RENDERERS:
        raise ValueError(
            f"{source} names benchmark {benchmark!r}, expected one of "
            f"{sorted(_RENDERERS)}"
        )
    fields, render = _RENDERERS[benchmark]
    header = _INTRODUCTION.format(benchmark=benchmark.capitalize(), source=source)
    header += fields + _PRECISION
    lines = [*fmt.comment(header), "", *_meta(meta)]
    results = payload.get("results", {})
    for size in sorted(results, key=int):
        lines += ["", "", f"# {'-' * 30} L = {size} {'-' * 30}"]
        tasks = results[size]
        for name in _task_order(meta, tasks):
            lines += render(f"results.{fmt.key(size)}.{fmt.key(name)}", tasks[name])
    return "\n".join(lines) + "\n"


def convert(source: Path, target: Path | None = None) -> Path:
    """Write the TOML rendering of a JSON cache.

    Args:
        source: The JSON cache.
        target: Output file; defaults to `source` with a `.toml` suffix.

    Returns:
        The path written.

    Raises:
        ValueError: If the cache names no known benchmark.
    """
    source = Path(source)
    target = source.with_suffix(".toml") if target is None else Path(target)
    payload = json.loads(source.read_text())
    document = to_toml(payload, source=source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments, without the program name; None reads `sys.argv`.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("source", type=Path, help="JSON benchmark cache")
    parser.add_argument("-o", "--output", type=Path, help="TOML file to write")
    arguments = parser.parse_args(argv)
    print(convert(arguments.source, arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
