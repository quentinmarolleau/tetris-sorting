"""Worker plumbing shared by the benchmarks.

Every task holds its own random stream, tasks are drawn from one queue
in a shuffled order, and each worker reports its own CPU time. Shuffling
keeps anything about the machine's scheduling or its thermal behaviour
from lining up with the array size being measured.
"""

import multiprocessing as mp
import random
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np

__all__ = [
    "BusyWorkers",
    "draw_loading",
    "run_tasks",
    "shuffled_tasks",
    "task_generator",
    "task_seeds",
]

_SENTINEL = None


def task_seeds(entropy: int, count: int) -> list[np.random.SeedSequence]:
    """One independent child stream per task.

    The failure this replaces: forked workers inherited a single global
    NumPy state, so workers on the same array size drew identical
    configurations.

    Args:
        entropy: Root entropy of the run.
        count: Number of tasks.

    Returns:
        `count` child seed sequences spawned from `entropy`.
    """
    return np.random.SeedSequence(entropy).spawn(count)


def task_generator(seed: np.random.SeedSequence) -> np.random.Generator:
    """Random generator of one task.

    Args:
        seed: The task's own seed sequence, from `task_seeds`.

    Returns:
        A generator drawing from that stream only.
    """
    return np.random.default_rng(seed)


def draw_loading(rng: np.random.Generator, size: int) -> np.ndarray:
    """One loading, every site filled independently with probability
    one half.

    Args:
        rng: The task's generator.
        size: Loading array side.

    Returns:
        A `size` by `size` boolean matrix, True where a site holds an atom.
    """
    return rng.integers(0, 2, size=(size, size)).astype(bool)


def shuffled_tasks(combinations: Sequence[tuple], entropy: int) -> list[tuple]:
    """Each combination with its own seed appended, in shuffled order.

    Seeds are spawned in the order `combinations` is given, then the list
    is shuffled, both from `entropy`, so a run stays reproducible from the
    entropy it records.

    Args:
        combinations: One tuple of task parameters per task.
        entropy: Root entropy, for the seeds and for the shuffle.

    Returns:
        `(*combination, seed)` tuples in shuffled order.
    """
    seeds = task_seeds(entropy, len(combinations))
    tasks = [(*combination, seed) for combination, seed in zip(combinations, seeds)]
    random.Random(entropy).shuffle(tasks)
    return tasks


class BusyWorkers:
    """Context manager counting the workers busy on a task while this one
    runs.

    The count is read when the task starts and when it ends, and the
    smaller of the two is kept. It drops at the tail of a sweep, when the
    queue empties and fewer tasks compete for the machine, so a timing
    taken there is not comparable with the rest.

    Args:
        active: Shared integer counter with a lock, such as a
            `multiprocessing.Value("i", 0)`, common to every worker.

    Attributes:
        minimum: Smaller of the busy-worker counts read on entry and on
            exit, this worker included. Zero before the block is entered.
    """

    def __init__(self, active) -> None:
        self._active = active
        self.minimum = 0

    def __enter__(self) -> Self:
        with self._active.get_lock():
            self._active.value += 1
            self.minimum = self._active.value
        return self

    def __exit__(self, *_) -> None:
        with self._active.get_lock():
            self.minimum = min(self.minimum, self._active.value)
            self._active.value -= 1


def _serve(worker: Callable, tasks, results, active, args: tuple) -> None:
    while True:
        task = tasks.get()
        if task is _SENTINEL:
            return
        results.put(worker(task, active, *args))


def run_tasks(
    worker: Callable[..., dict], tasks: Sequence[tuple], workers: int, *args: Any
) -> list[dict]:
    """Run every task on forked worker processes.

    Args:
        worker: Called as `worker(task, active, *args)` for each task, with
            `active` the shared counter `BusyWorkers` takes; returns the
            task's record.
        tasks: The tasks, in the order they are queued.
        workers: Number of worker processes.
        *args: Passed to every call of `worker`.

    Returns:
        One record per task, in completion order.
    """
    context = mp.get_context("fork")
    queue: mp.Queue = context.Queue()
    results: mp.Queue = context.Queue()
    active = context.Value("i", 0)
    for task in tasks:
        queue.put(task)
    for _ in range(workers):
        queue.put(_SENTINEL)

    processes = [
        context.Process(target=_serve, args=(worker, queue, results, active, args))
        for _ in range(workers)
    ]
    for process in processes:
        process.start()
    collected = [results.get() for _ in tasks]
    for process in processes:
        process.join()
    return collected
