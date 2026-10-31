"""Rearrangement cost of kept loadings, at a fixed count per task.

A task is one (size, reshape). It draws loadings until
`configurations_per_task` of them are kept, and measures each kept one
under every run selection swept. A fixed count rather than a time
budget: the metrics are means of quantities whose spread is small next
to their size, so a few thousand configurations settle them, and a
fixed count keeps that precision the same at every size. How often a
loading is kept is `tetris.acceptance_bench`'s job.

Metrics. For each kept loading and each selection, `fast._plan_construct`
decides one move per loading row and `fast._plan_elimination` one move
per target column. A move is a parallel displacement of the atoms of one
line, given as the source and destination index of each atom along that
line, in lattice sites. Each move is then scored by `fast._move_costs`:
its longest displacement, its summed displacement, and how many of its
atoms change site. Summed over the moves of a step, these are
`<step>_bottleneck`, `<step>_cumulated` and `<step>_moved_atoms`, for the
construct (row moves) and the elimination (column moves); `total_*` is
their sum for the same loading. They come from the assignment in closed
form, without simulating the atoms' motion.

Statistics. Every metric is accumulated as a running count n, sum and
sum of squares, so no sample is kept. The mean is the sum over n, the
sample variance s2 = (sum of squares - n mean^2) / (n - 1), clipped at
zero against rounding, and the standard error sqrt(s2 / n): the
uncertainty of the mean, not the spread of single loadings.

Pairing. Every selection is measured on the same loadings, and for every
selection but the first, the reference, the per-loading difference
d = b - a against the reference is accumulated the same way. Its
standard error is that of a paired sample, sqrt(var(d) / n) with
var(d) = var(a) + var(b) - 2 cov(a, b). Loadings that are costly under
one selection are costly under the other, so the covariance is large and
positive, and the difference is known far better than either mean: at
L = 100 with one row and one column off, LEFTMOST minus the default
order is about 100 parallel displacements known to within 0.8, where
two separate samples of the same size would give about 2.1. A difference
between two non-reference selections is not stored; its mean follows by
subtraction, but its standard error needs the covariance, which is not
kept.

Timing. Only the decision is timed: `_plan_construct` and
`_plan_elimination`, per selection, each with the worker's process CPU
time. That covers reading each line's atom positions, assigning target
sites, choosing the run of an overfull line and writing the move down.
Drawing and testing the loading, scoring the moves and accumulating the
statistics are outside it. The per-row and per-column figures are those
CPU times over the rows or columns processed. They are Python and NumPy
timings: each line costs a few NumPy calls of about a microsecond, which
hides most of the per-site work, so they bound what a compiled
controller would spend from above rather than estimate it. They also
depend on the load: the sweep runs one worker per hardware thread less
one, and on the 12-core, 24-thread machine it was first run on a whole
rearrangement took about twice as long as in a lone process.
"""

from collections.abc import Iterable
from time import monotonic, process_time

import numpy as np

from tetris.acceptance import configuration_kept, target_geometry
from tetris.fast import (
    DEFAULT_RUN_SELECTION,
    RearrangementMetrics,
    RunCriterion,
    _plan_construct,
    _plan_elimination,
    _score,
    validate_run_selection,
)
from tetris.sweep import (
    BusyWorkers,
    draw_loading,
    run_tasks,
    shuffled_tasks,
    task_generator,
)
from tetris.variants import variant_key

__all__ = [
    "DRAWS_PER_CONFIGURATION",
    "REARRANGEMENT_METRICS",
    "build_rearrangement_tasks",
    "run_rearrangement_benchmark",
    "selection_key",
]

# draws allowed per configuration asked for, so that a target almost no
# loading can fill ends its task short instead of never ending it
DRAWS_PER_CONFIGURATION = 1000

# the fields of `RearrangementMetrics` a result entry averages, totals
# included, in the order they are reported
REARRANGEMENT_METRICS: tuple[str, ...] = tuple(
    f"{step}_{quantity}"
    for step in ("construct", "elimination", "total")
    for quantity in ("bottleneck", "cumulated", "moved_atoms")
)


def selection_key(run_selection: Iterable[str]) -> str:
    """Key a run selection's results are filed under.

    Args:
        run_selection: Criteria, in the order they are applied.

    Returns:
        The criteria joined by commas, such as
        `BOTTLENECK,CUMULATED,ATOMS,LEFTMOST`. Order matters; an empty
        selection gives an empty key.
    """
    return ",".join(run_selection)


def build_rearrangement_tasks(
    sizes: tuple[int, ...],
    reshape_variants: tuple[tuple[int, int], ...],
    entropy: int,
) -> list[tuple]:
    """Shuffled (size, reshape, seed) tasks.

    Run selections are not part of a task: every selection is measured on
    every kept loading of it.

    Args:
        sizes: Loading array sizes.
        reshape_variants: (rows, columns) reshapes of the target.
        entropy: Root entropy, for the seeds and for the shuffle.

    Returns:
        One task per combination, in shuffled order.
    """
    combinations = [
        (int(size), tuple(reshape)) for size in sizes for reshape in reshape_variants
    ]
    return shuffled_tasks(combinations, entropy)


class _Moments:
    """Running count, sum and sum of squares of the metrics, enough for a
    mean and its standard error without keeping every sample."""

    def __init__(self) -> None:
        self.count = 0
        self.sums = np.zeros(len(REARRANGEMENT_METRICS))
        self.squares = np.zeros(len(REARRANGEMENT_METRICS))

    def add(self, values: np.ndarray) -> None:
        self.count += 1
        self.sums += values
        self.squares += values * values

    def summary(self) -> dict[str, dict[str, float]]:
        if self.count == 0:
            nan = {"mean": float("nan"), "standard_error": float("nan")}
            return {name: dict(nan) for name in REARRANGEMENT_METRICS}
        means = self.sums / self.count
        if self.count > 1:
            variances = (self.squares - self.count * means * means) / (self.count - 1)
            errors = np.sqrt(np.clip(variances, 0.0, None) / self.count)
        else:
            errors = np.full(len(REARRANGEMENT_METRICS), np.nan)
        return {
            name: {"mean": float(mean), "standard_error": float(error)}
            for name, mean, error in zip(REARRANGEMENT_METRICS, means, errors)
        }


def _metric_values(metrics) -> np.ndarray:
    return np.array(
        [getattr(metrics, name) for name in REARRANGEMENT_METRICS], dtype=float
    )


def _rearrangement_worker(
    task,
    active,
    configurations_per_task: int,
    run_selections: tuple[tuple[RunCriterion, ...], ...],
) -> dict:
    size, reshape, seed = task
    window = target_geometry(size, reshape)
    rng = task_generator(seed)
    metrics = [_Moments() for _ in run_selections]
    differences = [_Moments() for _ in run_selections[1:]]
    construct_cpu = [0.0] * len(run_selections)
    elimination_cpu = [0.0] * len(run_selections)
    kept = draws = 0

    with BusyWorkers(active) as busy:
        wall_start = monotonic()
        while (
            kept < configurations_per_task
            and draws < DRAWS_PER_CONFIGURATION * configurations_per_task
        ):
            matrix = draw_loading(rng, size)
            draws += 1
            if not configuration_kept(matrix, window):
                continue
            kept += 1
            values = []
            for index, run_selection in enumerate(run_selections):
                # the decision alone is timed, one step at a time; scoring
                # the moves is bookkeeping a controller would not do
                started = process_time()
                occupancy, rows = _plan_construct(matrix, window, run_selection)
                planned = process_time()
                columns = _plan_elimination(occupancy, window, run_selection)
                construct_cpu[index] += planned - started
                elimination_cpu[index] += process_time() - planned
                measured = RearrangementMetrics(*_score(rows), *_score(columns))
                values.append(_metric_values(measured))
                metrics[index].add(values[index])
            for index, difference in enumerate(differences, start=1):
                difference.add(values[index] - values[0])
        wall_seconds = monotonic() - wall_start

    return {
        "size": size,
        "reshape": list(reshape),
        "measured_configurations": kept,
        "wall_seconds": wall_seconds,
        "min_concurrency": busy.minimum,
        "target_rows": window.rows,
        "target_columns": window.columns,
        "construct_cpu_seconds": construct_cpu,
        "elimination_cpu_seconds": elimination_cpu,
        "metrics": [moments.summary() for moments in metrics],
        "differences": [moments.summary() for moments in differences],
    }


def _summarise(
    record: dict, run_selections: tuple[tuple[RunCriterion, ...], ...]
) -> tuple[int, str, dict]:
    measured = record["measured_configurations"]
    # every loading row is a construct move, every target column an
    # elimination move
    rows_processed = measured * record["size"]
    columns_processed = measured * record["target_columns"]
    selections = {}
    for index, run_selection in enumerate(run_selections):
        construct = record["construct_cpu_seconds"][index]
        elimination = record["elimination_cpu_seconds"][index]
        selection = {
            "construct_cpu_seconds": construct,
            "elimination_cpu_seconds": elimination,
            "construct_microseconds_per_row": (
                construct / rows_processed * 1e6 if measured else float("nan")
            ),
            "elimination_microseconds_per_column": (
                elimination / columns_processed * 1e6 if measured else float("nan")
            ),
            "metrics": record["metrics"][index],
        }
        if index > 0:
            selection["difference_to_reference"] = record["differences"][index - 1]
        selections[selection_key(run_selection)] = selection
    return (
        record["size"],
        variant_key(tuple(record["reshape"])),
        {
            "target_rows": record["target_rows"],
            "target_columns": record["target_columns"],
            "target_number_of_atoms": record["target_rows"] * record["target_columns"],
            "measured_configurations": measured,
            "wall_seconds": record["wall_seconds"],
            "min_concurrency": record["min_concurrency"],
            "selections": selections,
        },
    )


def run_rearrangement_benchmark(
    sizes: tuple[int, ...],
    reshape_variants: tuple[tuple[int, int], ...],
    configurations_per_task: int,
    workers: int,
    entropy: int,
    run_selections: tuple[tuple[RunCriterion, ...], ...] = (DEFAULT_RUN_SELECTION,),
) -> dict[int, dict]:
    """Rearrangement cost over a sweep of tasks, paired across selections.

    Args:
        sizes: Loading array sizes.
        reshape_variants: (rows, columns) reshapes of the target.
        configurations_per_task: Kept loadings measured per task. A task
            gives up after `DRAWS_PER_CONFIGURATION` times as many draws,
            and then reports how many it measured.
        workers: Worker processes to spread the tasks over.
        entropy: Root entropy; the same entropy draws the same loadings.
        run_selections: Run selections measured on every kept loading.
            The first is the reference the differences are taken against.

    Returns:
        Results keyed by size, then by `variant_key(reshape)`. Each entry
        carries the target geometry, `measured_configurations`, and under
        `selections[selection_key]`, in the order `run_selections` gives:
        the CPU time spent deciding the construct and the elimination,
        the same per row and per target column in microseconds, and under
        `metrics` the mean and standard error of every name in
        `REARRANGEMENT_METRICS`. Every selection but the reference also
        carries `difference_to_reference`: the mean and standard error of
        its metrics minus the reference's, configuration by configuration.

    Raises:
        ValueError: If a run selection is malformed or given twice, if
            there is none, if `configurations_per_task` or `workers` is
            below one, or if a size and reshape leave no target that fits.
        TypeError: If a run selection is a bare string.
    """
    selections = tuple(
        validate_run_selection(selection) for selection in run_selections
    )
    if not selections:
        raise ValueError("at least one run selection is needed")
    if len(set(selections)) != len(selections):
        raise ValueError("a run selection is given more than once")
    if configurations_per_task < 1:
        raise ValueError(
            f"configurations_per_task must be at least 1, got {configurations_per_task}"
        )
    if workers < 1:
        raise ValueError(f"at least one worker is needed, got {workers}")
    # refused here rather than inside a forked worker, where it would hang
    # the queue instead of reaching the caller
    for size in sizes:
        for reshape in reshape_variants:
            target_geometry(size, reshape)

    tasks = build_rearrangement_tasks(sizes, reshape_variants, entropy)
    records = run_tasks(
        _rearrangement_worker, tasks, workers, configurations_per_task, selections
    )
    results: dict[int, dict] = {}
    for record in records:
        size, key, summary = _summarise(record, selections)
        results.setdefault(size, {})[key] = summary
    return dict(sorted(results.items()))
