"""How the run choice of an overfull row behaves on random loadings.

A row is overfull when it holds more atoms than the target has columns,
and `_movers` then chooses which run of consecutive atoms fills the
window. Three questions, on the rows of random loadings rather than on
synthetic lines: how much `_movers`'s own time depends on whether the row
is exactly filled or overfull, how far its bottleneck-optimal choice sits
from the sum-optimal one that `movers_sum_optimal` finds by brute force,
and how far it sits from Wang et al.'s own leftmost rule.

The target window uses the `(-1, -1)` margin reshape throughout, the
variant the long sweep and the salvageable-share figures already use, so
`k = floor(L / sqrt(2)) - 1`.

Runs single-process. A row's worth of work is microseconds, so the
worker pool in `tetris.sweep` would cost more than it saves; what
matters here is drawing enough overfull rows, and overfull rows get rare
fast as the loading array grows, since the window asks for more atoms
than a half-loaded row supplies on average.
"""

from time import perf_counter

import numpy as np

from tetris.acceptance import TargetWindow, target_geometry
from tetris.fast import _movers, _run_displacement_bottleneck_cost
from tetris.overfull_reference import movers_leftmost, movers_sum_optimal, run_sum_cost
from tetris.sweep import task_generator, task_seeds

MARGIN_RESHAPE = (-1, -1)

__all__ = ["MARGIN_RESHAPE", "rows_by_regime", "run_overfull_benchmark"]


def rows_by_regime(
    rng: np.random.Generator, size: int, window: TargetWindow, configs: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Atom positions of the rows that `_movers` is called on, by regime.

    A row with exactly `window.columns` atoms has a single admissible run,
    and `_movers` returns without searching. A row with more is overfull,
    and `_movers` searches. Rows with fewer atoms never reach `_movers`
    and are dropped. Every row of every loading counts, whether the
    loading would be kept or not.

    Args:
        rng: Generator the loadings are drawn from, at filling one half.
        size: Loading array side L.
        window: Target window; only its column count is used.
        configs: Number of loadings to draw.

    Returns:
        (exact, overfull): two lists of sorted column indices, one array
        per row.
    """
    exact: list[np.ndarray] = []
    overfull: list[np.ndarray] = []
    for _ in range(configs):
        matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
        counts = matrix.sum(axis=1)
        for row_index in np.flatnonzero(counts == window.columns):
            exact.append(np.flatnonzero(matrix[row_index]))
        for row_index in np.flatnonzero(counts > window.columns):
            overfull.append(np.flatnonzero(matrix[row_index]))
    return exact, overfull


def _mean_and_error(values) -> tuple[float, float]:
    """Mean of `values` and its standard error, sd / sqrt(n); a single
    value has no spread to estimate, and gets an error of zero."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _relative_gap_fields(name: str, relative_gaps: list[float]) -> dict:
    """Standard error of the mean relative gap, and the mean and its error
    over the rows that differ, where the gap is not zero. Measured
    directly on those rows rather than derived from the overall mean, so
    their error does not have to be propagated through the agreement
    share. The last two are omitted when every row agrees."""
    gaps = np.asarray(relative_gaps, dtype=float)
    _, error = _mean_and_error(gaps)
    fields = {f"relative_{name}_standard_error": error}
    differing = gaps[gaps > 0]
    if differing.size:
        mean, error = _mean_and_error(differing)
        fields[f"mean_relative_{name}_differing"] = mean
        fields[f"relative_{name}_differing_standard_error"] = error
    return fields


def _time_calls(
    fn, rows: list[np.ndarray], column_start: int, columns: int, repeats: int
) -> float:
    """Seconds per call, best of `repeats` passes over every row —
    the standard `timeit` defence against scheduler noise, applied to a
    batch since one call is too short to time on its own reliably."""
    best = float("inf")
    for _ in range(repeats):
        start = perf_counter()
        for positions in rows:
            fn(positions, column_start, columns)
        elapsed = perf_counter() - start
        best = min(best, elapsed)
    return best / len(rows)


def run_overfull_benchmark(
    sizes: tuple[int, ...],
    configs_per_size: int,
    entropy: int,
    timing_repeats: int = 50,
) -> list[dict]:
    """How `_movers` behaves on the loading rows it is actually given.

    A row is overfull when it holds more atoms than the target has
    columns: several runs of atoms could then fill the window and
    `_movers` has to choose. A row holding exactly as many atoms has no
    choice. For each size the benchmark draws `configs_per_size` loadings,
    kept or not, and splits their rows into the two regimes. It times
    `_movers` on each, and on the overfull rows compares its choice with
    the run of smallest summed displacement and with Wang et al.'s
    leftmost run.

    Two runs "agree" when they cost the same on the metric compared, even
    if they start at different atoms. The mean gaps are averaged over
    every overfull row, agreeing rows included at zero; divide by
    `1 - agreement_fraction` for the mean over the rows that differ.

    Args:
        sizes: Loading array sizes L. The target uses `MARGIN_RESHAPE`.
        configs_per_size: Loadings drawn per size.
        entropy: Root entropy; each size draws from its own child stream.
        timing_repeats: Passes over the rows per timing; the fastest pass
            is kept.

    Returns:
        One dict per size, in the order of `sizes`, with the keys:

        - `size`, `exact_rows`, `overfull_rows`: L and the row counts of
          each regime.
        - `exact_seconds_per_row`, `overfull_seconds_per_row`: mean time
          of one `_movers` call in that regime. Omitted when no row of the
          regime was drawn.
        - `agreement_fraction`: share of overfull rows where `_movers`'s
          run has the smallest summed displacement.
        - `mean_absolute_gap`, `max_absolute_gap`: summed displacement of
          `_movers`'s run minus the smallest one, in sites.
        - `mean_relative_gap`: the same gap over the smallest summed
          displacement, floored at one site.
        - `leftmost_agreement_fraction`: share of overfull rows where the
          leftmost run has the same bottleneck as `_movers`'s.
        - `mean_absolute_leftmost_gap`, `max_absolute_leftmost_gap`:
          bottleneck of the leftmost run minus that of `_movers`'s run, in
          sites.
        - `mean_relative_leftmost_gap`: the same gap over `_movers`'s
          bottleneck, floored at one site.
        - `relative_gap_standard_error`,
          `relative_leftmost_gap_standard_error`: standard errors of the
          two mean relative gaps.
        - `mean_relative_gap_differing`,
          `relative_gap_differing_standard_error`, and the same with
          `leftmost_gap`: mean relative gap over the rows that differ
          only, and its standard error. Omitted when every row agrees.

        The keys from `agreement_fraction` on are omitted when no
        overfull row was drawn.
        The floor at one site only matters for a run already on the
        window; it can understate a relative gap, never overstate it.
    """
    seeds = task_seeds(entropy, len(sizes))
    results = []
    for size, seed in zip(sizes, seeds):
        rng = task_generator(seed)
        window = target_geometry(size, reshape=MARGIN_RESHAPE)
        exact, overfull = rows_by_regime(rng, size, window, configs_per_size)
        column_start, columns = window.column_start, window.columns

        entry: dict = {
            "size": size,
            "exact_rows": len(exact),
            "overfull_rows": len(overfull),
        }
        if exact:
            entry["exact_seconds_per_row"] = _time_calls(
                _movers, exact, column_start, columns, timing_repeats
            )
        if overfull:
            entry["overfull_seconds_per_row"] = _time_calls(
                _movers, overfull, column_start, columns, timing_repeats
            )

            absolute_gaps = []
            relative_gaps = []
            agreements = 0
            leftmost_absolute_gaps = []
            leftmost_relative_gaps = []
            leftmost_agreements = 0
            for positions in overfull:
                bottleneck_first = _movers(positions, column_start, columns)
                optimal_first = movers_sum_optimal(positions, column_start, columns)
                bottleneck_sum = run_sum_cost(
                    positions, column_start, columns, bottleneck_first
                )
                optimal_sum = run_sum_cost(
                    positions, column_start, columns, optimal_first
                )
                gap = bottleneck_sum - optimal_sum
                absolute_gaps.append(gap)
                relative_gaps.append(gap / max(optimal_sum, 1))
                agreements += gap == 0

                leftmost_first = movers_leftmost(positions, column_start, columns)
                bottleneck_cost = _run_displacement_bottleneck_cost(
                    positions, column_start, columns, bottleneck_first
                )
                leftmost_cost = _run_displacement_bottleneck_cost(
                    positions, column_start, columns, leftmost_first
                )
                leftmost_gap = leftmost_cost - bottleneck_cost
                leftmost_absolute_gaps.append(leftmost_gap)
                leftmost_relative_gaps.append(leftmost_gap / max(bottleneck_cost, 1))
                leftmost_agreements += leftmost_gap == 0

            entry.update(
                {
                    "agreement_fraction": agreements / len(overfull),
                    "mean_absolute_gap": float(np.mean(absolute_gaps)),
                    "mean_relative_gap": float(np.mean(relative_gaps)),
                    "max_absolute_gap": int(np.max(absolute_gaps)),
                    "leftmost_agreement_fraction": leftmost_agreements / len(overfull),
                    "mean_absolute_leftmost_gap": float(
                        np.mean(leftmost_absolute_gaps)
                    ),
                    "mean_relative_leftmost_gap": float(
                        np.mean(leftmost_relative_gaps)
                    ),
                    "max_absolute_leftmost_gap": int(np.max(leftmost_absolute_gaps)),
                    **_relative_gap_fields("gap", relative_gaps),
                    **_relative_gap_fields("leftmost_gap", leftmost_relative_gaps),
                }
            )
        results.append(entry)
    return results
