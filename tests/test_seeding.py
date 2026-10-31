"""Independence and reproducibility of the per-task random streams."""

import multiprocessing as mp

import numpy as np

from tetris import AtomsConfiguration
from tetris.sweep import task_generator, task_seeds


def test_spawned_streams_differ() -> None:
    seeds = task_seeds(entropy=12345, count=4)
    matrices = [
        AtomsConfiguration(
            loading_array_size=16, rng=np.random.default_rng(seed)
        ).occupation_matrix
        for seed in seeds
    ]
    for index in range(1, len(matrices)):
        assert not np.array_equal(matrices[0], matrices[index])


def test_same_entropy_reproduces_the_run() -> None:
    first = task_seeds(entropy=777, count=3)
    second = task_seeds(entropy=777, count=3)
    matrices = [
        [
            AtomsConfiguration(
                loading_array_size=10, rng=np.random.default_rng(seed)
            ).occupation_matrix
            for seed in seeds
        ]
        for seeds in (first, second)
    ]
    for left, right in zip(*matrices):
        assert np.array_equal(left, right)


def _draw(queue: "mp.Queue", seed) -> None:
    matrix = AtomsConfiguration(
        loading_array_size=20, rng=task_generator(seed)
    ).occupation_matrix
    queue.put(matrix.tobytes())


def test_forked_workers_do_not_share_a_stream() -> None:
    """The failure this replaces: forked workers inherited one global
    NumPy state, so every worker drew the same configurations."""
    context = mp.get_context("fork")
    queue = context.Queue()
    seeds = task_seeds(entropy=2024, count=4)
    workers = [context.Process(target=_draw, args=(queue, seed)) for seed in seeds]
    for worker in workers:
        worker.start()
    drawn = [queue.get() for _ in workers]
    for worker in workers:
        worker.join()
    assert len(set(drawn)) == len(drawn)
