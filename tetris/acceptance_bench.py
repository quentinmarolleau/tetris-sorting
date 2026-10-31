"""Success rate of the acceptance rule, under a time budget per task.

The success rate is the fraction of random loadings that
`tetris.acceptance.configuration_kept` keeps. A task is one (size, reshape). It draws loadings until its budget runs
out and tests every one; nothing else runs inside it, so the CPU time it
reports is the cost of drawing and deciding alone. What rearranging a
kept loading costs is `tetris.rearrangement_bench`'s job, measured apart
so that neither eats into the other's sample.
"""

from time import monotonic, process_time

from tetris.acceptance import configuration_kept, target_geometry
from tetris.stats import standard_error, wilson_interval
from tetris.sweep import (
    BusyWorkers,
    draw_loading,
    run_tasks,
    shuffled_tasks,
    task_generator,
)
from tetris.variants import variant_key

__all__ = ["build_acceptance_tasks", "run_acceptance_benchmark"]


def build_acceptance_tasks(
    sizes: tuple[int, ...],
    reshape_variants: tuple[tuple[int, int], ...],
    entropy: int,
) -> list[tuple]:
    """Shuffled (size, reshape, seed) tasks.

    Running in size order lets any slow drift over the run, a laptop
    heating up for instance, masquerade as a trend against array size.
    Shuffling turns it into noise. Seeds are spawned in (size, reshape)
    order, which is the order a sweep of the default run selection used
    before the two benchmarks were separated, so the same entropy draws
    the same loadings.

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


def _acceptance_worker(task, active, seconds_per_task: float) -> dict:
    size, reshape, seed = task
    window = target_geometry(size, reshape)
    rng = task_generator(seed)
    kept = samples = 0

    with BusyWorkers(active) as busy:
        cpu_start, wall_start = process_time(), monotonic()
        deadline = wall_start + seconds_per_task
        # the deadline is tested before each draw, so a task always
        # overruns its budget rather than stopping short of it
        while monotonic() < deadline:
            kept += configuration_kept(draw_loading(rng, size), window)
            samples += 1
        cpu_seconds = process_time() - cpu_start
        wall_seconds = monotonic() - wall_start

    return {
        "size": size,
        "reshape": list(reshape),
        "kept": kept,
        "samples": samples,
        "acceptance_cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "min_concurrency": busy.minimum,
        "target_rows": window.rows,
        "target_columns": window.columns,
    }


def _summarise(record: dict) -> tuple[int, str, dict]:
    size, kept, samples = record["size"], record["kept"], record["samples"]
    sites = record["target_rows"] * record["target_columns"]
    low, high = wilson_interval(kept, samples)
    return (
        size,
        variant_key(tuple(record["reshape"])),
        {
            "target_rows": record["target_rows"],
            "target_columns": record["target_columns"],
            "target_number_of_atoms": sites,
            "contraction_ratio": sites / size**2,
            "kept": kept,
            "samples": samples,
            "success_rate": kept / samples if samples else float("nan"),
            "standard_error": standard_error(kept, samples),
            "wilson_low": low,
            "wilson_high": high,
            "acceptance_cpu_seconds": record["acceptance_cpu_seconds"],
            "microseconds_per_configuration": (
                record["acceptance_cpu_seconds"] / samples * 1e6
                if samples
                else float("nan")
            ),
            "wall_seconds": record["wall_seconds"],
            "min_concurrency": record["min_concurrency"],
        },
    )


def run_acceptance_benchmark(
    sizes: tuple[int, ...],
    reshape_variants: tuple[tuple[int, int], ...],
    seconds_per_task: float,
    workers: int,
    entropy: int,
) -> dict[int, dict]:
    """Success rate of the acceptance rule over a sweep of tasks.

    Args:
        sizes: Loading array sizes.
        reshape_variants: (rows, columns) reshapes of the target.
        seconds_per_task: Wall-clock budget of each task.
        workers: Worker processes to spread the tasks over.
        entropy: Root entropy; the same entropy draws the same loadings.

    Returns:
        Results keyed by size, then by `variant_key(reshape)`. Each entry
        carries the target geometry, the success rate with its standard
        error and Wilson interval at one standard deviation, the samples
        behind it, and `acceptance_cpu_seconds`, the CPU time spent
        drawing and testing them.

    Raises:
        ValueError: If `workers` is below one, or if a size and reshape
            leave no target that fits.
    """
    if workers < 1:
        raise ValueError(f"at least one worker is needed, got {workers}")
    # refused here rather than inside a forked worker, where it would hang
    # the queue instead of reaching the caller
    for size in sizes:
        for reshape in reshape_variants:
            target_geometry(size, reshape)

    tasks = build_acceptance_tasks(sizes, reshape_variants, entropy)
    results: dict[int, dict] = {}
    for record in run_tasks(_acceptance_worker, tasks, workers, seconds_per_task):
        size, key, summary = _summarise(record)
        results.setdefault(size, {})[key] = summary
    return dict(sorted(results.items()))
