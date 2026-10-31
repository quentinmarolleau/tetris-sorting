"""The benchmark caches as flat tables, one row per measurement.

The caches nest their results by size, then by reshape variant, then, for
the rearrangement sweep, by run selection and by metric. That shape suits
the JSON file and the TOML rendering; it makes every plot rebuild its
arrays with nested comprehensions. The functions here flatten each cache
once, so a plot selects rows and reads named columns instead.

Nothing in a benchmark imports this module, so editing it never changes
an algorithm digest and never invalidates a cached run.
"""

from collections.abc import Iterable, Sequence

import pandas as pd

from tetris.variants import variant_key, variant_label

__all__ = [
    "acceptance_frame",
    "against_reference",
    "overfull_columns_frame",
    "overfull_rows_frame",
    "rearrangement_frame",
    "with_full_load",
]

# the order of the three statistics `overfull_columns_stats.json` stores
# positionally in `mean` and `error`
_COLUMN_STATISTICS = ("overfull_share", "surplus", "bottleneck_saving")


def _variant_columns(frame: pd.DataFrame, reshapes: Sequence[tuple[int, int]]):
    """Insert the reshape, its label and an ordered `variant` category
    after `size`, and sort rows by size then by sweep order."""
    keys = [variant_key(reshape) for reshape in reshapes]
    by_key = dict(zip(keys, reshapes))
    variant = pd.Categorical(frame.pop("variant"), categories=keys, ordered=True)
    frame.insert(1, "variant", variant)
    frame.insert(2, "reshape_rows", [by_key[key][0] for key in frame["variant"]])
    frame.insert(3, "reshape_columns", [by_key[key][1] for key in frame["variant"]])
    frame.insert(4, "label", [variant_label(by_key[key]) for key in frame["variant"]])
    return frame.sort_values(["size", "variant"], ignore_index=True)


def acceptance_frame(
    stats: dict[int, dict], reshapes: Sequence[tuple[int, int]]
) -> pd.DataFrame:
    """The acceptance sweep, one row per size and reshape.

    Args:
        stats: Results keyed by integer size, then by variant key, as
            `load_results` returns them.
        reshapes: The reshapes to keep, in the order groups should come
            out of a `groupby("variant")`.

    Returns:
        Columns `size`, `variant` (ordered category), `reshape_rows`,
        `reshape_columns`, `label`, then every field the cache holds for
        a point (`success_rate`, `wilson_low`, `samples`, ...).

    Raises:
        KeyError: A size has no entry for one of `reshapes`.
    """
    records = [
        {
            "size": size,
            "variant": variant_key(reshape),
            **variants[variant_key(reshape)],
        }
        for size, variants in stats.items()
        for reshape in reshapes
    ]
    return _variant_columns(pd.DataFrame.from_records(records), reshapes)


def rearrangement_frame(
    stats: dict[int, dict],
    reshapes: Sequence[tuple[int, int]],
    selections: Sequence[Iterable[str]],
) -> pd.DataFrame:
    """The rearrangement sweep, one row per size, reshape and run selection.

    Each metric becomes four columns: `<metric>` (the mean),
    `<metric>_error`, and the paired difference to the reference
    selection `<metric>_difference` and `<metric>_difference_error`,
    which are NaN on the reference rows.

    Args:
        stats: Results keyed by integer size, then by variant key, as
            `load_results` returns them.
        reshapes: The reshapes to keep, in sweep order.
        selections: Run selections as criteria tuples, the reference
            first.

    Returns:
        Columns `size`, `variant`, `reshape_rows`, `reshape_columns`,
        `label`, the per-task fields (`target_number_of_atoms`,
        `min_concurrency`, ...), `selection` (ordered category of
        selection keys), `is_reference`, the per-selection timings and
        the metric columns.

    Raises:
        KeyError: A size lacks one of `reshapes` or `selections`.
    """
    names = [",".join(selection) for selection in selections]
    records = []
    for size, variants in stats.items():
        for reshape in reshapes:
            entry = variants[variant_key(reshape)]
            task = {key: value for key, value in entry.items() if key != "selections"}
            for name in names:
                measured = entry["selections"][name]
                record = {"size": size, "variant": variant_key(reshape), **task}
                record["selection"] = name
                record["is_reference"] = name == names[0]
                for key, value in measured.items():
                    if key not in ("metrics", "difference_to_reference"):
                        record[key] = value
                for metric, value in measured["metrics"].items():
                    record[metric] = value["mean"]
                    record[f"{metric}_error"] = value["standard_error"]
                    difference = measured.get("difference_to_reference", {}).get(metric)
                    record[f"{metric}_difference"] = (
                        difference["mean"] if difference else float("nan")
                    )
                    record[f"{metric}_difference_error"] = (
                        difference["standard_error"] if difference else float("nan")
                    )
                records.append(record)
    frame = pd.DataFrame.from_records(records)
    frame["selection"] = pd.Categorical(
        frame["selection"], categories=names, ordered=True
    )
    frame = _variant_columns(frame, reshapes)
    return frame.sort_values(["size", "variant", "selection"], ignore_index=True)


def against_reference(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired difference of every non-reference selection, in percent of
    the reference selection's mean on the same size and reshape.

    Args:
        frame: A frame from `rearrangement_frame`.
        metric: One metric name, such as `total_bottleneck`.

    Returns:
        The non-reference rows with two more columns,
        `relative_difference` and `relative_difference_error`, in
        percent.
    """
    reference = frame.loc[frame["is_reference"], ["size", "variant", metric]]
    others = frame[~frame["is_reference"]].merge(
        reference, on=["size", "variant"], suffixes=("", "_reference")
    )
    base = others[f"{metric}_reference"]
    return others.assign(
        relative_difference=100 * others[f"{metric}_difference"] / base,
        relative_difference_error=100 * others[f"{metric}_difference_error"] / base,
    )


def overfull_columns_frame(
    results: list[dict], reshapes: Sequence[tuple[int, int]]
) -> pd.DataFrame:
    """The overfull-column measurement, one row per size and reshape.

    Args:
        results: The cached list, each item with `size`, `reshape`, and
            `mean` and `error` triples in the order share of overfull
            columns, surplus per column, and the fraction by which the
            centred run shortens the summed longest column moves against
            the naive run.
        reshapes: The reshapes measured, in sweep order.

    Returns:
        Columns `size`, `variant`, `reshape_rows`, `reshape_columns`,
        `label`, `overfull_share`, `surplus`, `bottleneck_saving`, and the
        three matching `_error` columns.
    """
    records = [
        {
            "size": item["size"],
            "variant": variant_key(item["reshape"]),
            **dict(zip(_COLUMN_STATISTICS, item["mean"])),
            **{
                f"{name}_error": error
                for name, error in zip(_COLUMN_STATISTICS, item["error"])
            },
        }
        for item in results
    ]
    return _variant_columns(pd.DataFrame.from_records(records), reshapes)


def overfull_rows_frame(results: list[dict]) -> pd.DataFrame:
    """The overfull-row study, one row per size.

    Args:
        results: The cached list of `run_overfull_benchmark`, one dict per
            size. A size that drew no row of a regime lacks the fields
            measured on it, and one where every overfull row agrees lacks
            the `_differing` fields.

    Returns:
        The keys of `run_overfull_benchmark` as columns, sorted by size:
        `size`, the row counts `exact_rows` and `overfull_rows`, the
        timings `exact_seconds_per_row` and `overfull_seconds_per_row`,
        the agreement shares `agreement_fraction` and
        `leftmost_agreement_fraction`, the gaps `mean_absolute_gap`,
        `max_absolute_gap`, `mean_relative_gap` and their `leftmost`
        counterparts, and the standard errors of the mean relative gaps
        with the means over the differing rows. Fields a size lacks are
        NaN.
    """
    return pd.DataFrame.from_records(results).sort_values("size", ignore_index=True)


def with_full_load(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows measured while every worker was busy.

    A task that started or ended at the tail of a sweep shared its
    physical core less and reads cheaper for the same work.

    Args:
        frame: Any frame with a `min_concurrency` column.

    Returns:
        The rows whose `min_concurrency` is the frame's maximum.
    """
    return frame.loc[frame["min_concurrency"] == frame["min_concurrency"].max()]
