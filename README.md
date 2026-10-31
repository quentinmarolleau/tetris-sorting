# tetris-sorting

This package implements the rearrangement scheme of Wang *et al.*,
[*Accelerating the Assembly of Defect-Free Atomic Arrays with Maximum
Parallelisms*][paper], Phys. Rev. Applied **19**, 054032 (2023), and
benchmarks its success rate, its rearrangement cost and its decision time.

Individual atoms load into an array of optical tweezers with probability one
half per site, so a dense, defect-free target array has to be assembled by
moving the atoms that did load. The paper's scheme does it row by row, which
suits a camera streaming its rows one at a time, and packs each row against
the ones before it the way a Tetris player packs a well.

The study built on this package (the algorithm step by step, the benchmarks
against the paper, and why the success rate follows a sawtooth in the array
size) is published as a blog post, with the marimo notebook behind it
available for download:
[Tetris algorithm for atomic arrays active sorting][post].

[paper]: https://doi.org/10.1103/PhysRevApplied.19.054032
[post]: https://marolleau.me/posts/tetris-sorting/

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and needs
Python 3.13 or later.

```sh
git clone https://github.com/quentinmarolleau/tetris-sorting.git
cd tetris-sorting
uv sync
```

## Usage

`AtomsConfiguration` follows one loading through the algorithm, keeps every
intermediate array and move, and can print and plot each step:

```python
import numpy as np
from tetris import AtomsConfiguration

config = AtomsConfiguration(
    loading_array_size=18,
    reshape_target=(-1, -1),
    rng=np.random.default_rng(0),
)
config.tetrimino_construct(verbose=True)
if config.configuration_kept:
    config.tetrimino_eliminate()
    print(config.count_parallel_displacements())
config.plot_configuration()
```

For benchmarking, plain functions work on a bare boolean matrix and return
only the verdict or the costs:

```python
import numpy as np
from tetris import configuration_kept, rearrangement_metrics, target_geometry

matrix = np.random.default_rng(0).random((100, 100)) < 0.5
window = target_geometry(100, reshape=(-1, -1))
if configuration_kept(matrix, window):
    print(rearrangement_metrics(matrix, window))
```

The target has a natural side of ⌊*L*/√2⌋, which asks for about as many atoms
as a half-filled loading provides. `reshape` (`reshape_target` for the class)
adds a signed offset to each side, and `shape` (`target_shape`) sets the
target outright:

```python
target_geometry(18)  # (0, 0), the full natural square
target_geometry(18, reshape=(0, -1))  # one column off
target_geometry(18, reshape=(-1, 0))  # one row off
target_geometry(18, shape=(8, 12))  # 8 rows, 12 columns
```

Rows and columns are not interchangeable. A loaded row gives at most one atom
to each target column, so the width limits what a row can contribute while the
height sets what each column must supply. At *L* = 14, a 9 × 11 target is
accepted 46.5 % of the time and its transpose, asking for the same 99 atoms,
36.3 %.

The sweeps run on forked workers, one task per size and reshape, each seeded
from one recorded entropy value:

```python
from tetris.acceptance_bench import run_acceptance_benchmark

stats = run_acceptance_benchmark(
    sizes=(20, 30),
    reshape_variants=((0, 0), (-1, -1)),
    seconds_per_task=5,
    workers=4,
    entropy=2024,
)
```

`tetris.cache` saves such results with a fingerprint of their parameters and a
digest of the modules the benchmark runs, so that a stale result is never
served silently, and `python -m tetris.toml_export` renders a saved cache as
commented TOML.

## Layout

| Module | Description |
|---|:---|
| [`config`](tetris/config.py) | Defines `AtomsConfiguration`, which generates a loading, packs it into tetriminoes and renders it. It favours readability over speed. |
| [`acceptance`](tetris/acceptance.py) | Computes the target window and applies the acceptance rule at constant cost per row, which makes the sweeps affordable. |
| [`fast`](tetris/fast.py) | Selects the runs and computes the rearrangement metrics of a kept loading from the loaded matrix alone. |
| [`sweep`](tetris/sweep.py) | Provides the worker plumbing both benchmarks share: a seed per task, a shuffled task order and forked workers. |
| [`acceptance_bench`](tetris/acceptance_bench.py) | Measures the success rate under a time budget per task, along with the CPU time that drawing and testing took. |
| [`rearrangement_bench`](tetris/rearrangement_bench.py) | Measures the rearrangement cost on a fixed number of kept loadings per task, every run selection being applied to the same loadings. |
| [`overfull_reference`](tetris/overfull_reference.py) | Implements the sum-optimal and leftmost run choices for an overfull line, to compare `_movers` against. Nothing in the algorithm calls them. |
| [`overfull_bench`](tetris/overfull_bench.py) | Runs the overfull-row study: it draws the rows the construct produces, times the run choice and compares it with the reference choices. |
| [`cache`](tetris/cache.py) | Stores benchmark results on disk, fingerprinted against their parameters and the algorithm source. |
| [`analytic`](tetris/analytic.py) | Gives closed forms for the atoms a loading wastes and for the bound every rearrangement obeys. |
| [`variants`](tetris/variants.py) | Names the target reshapes that the sweeps compare. |
| [`stats`](tetris/stats.py) | Computes Wilson score intervals for the measured rates. |
| [`display`](tetris/display.py) | Renders a tetrimino construction row by row with Rich. |
| [`plotting`](tetris/plotting.py) | Draws a loading array and its packed counterpart with Matplotlib. |
| [`frames`](tetris/frames.py) | Flattens benchmark results into pandas tables, one row per measurement. |
| [`toml_export`](tetris/toml_export.py), [`toml_format`](tetris/toml_format.py) | Render a benchmark cache as commented TOML, every mean cut to the digits its error supports. |

## Results

The table below compares the results with the paper, on the reshape it uses
(one row and one column off):

| Quantity | Here | Wang *et al.* |
|---|---|---|
| Parallel displacements against atom number | *N*<sup>0.973(2)</sup> | *N*<sup>1.03(7)</sup>, and *N*<sup>1.6(1)</sup> for Hungarian |
| Acceptance over the sweep | 0.9997 | 0.9982 |

Over loading sizes 4 to 100, the mean success rate depends on the reshape as
follows:

| Reshape | Mean | Worst |
|---|---|---|
| (0, 0), no margin | 0.8607 | 0.5097 at *L* = 99 |
| (0, −1), one column off | 0.9868 | 0.9240 at *L* = 17 |
| (−1, 0), one row off | 0.9877 | 0.9249 at *L* = 99 |
| (−1, −1), one row and one column off | 0.9997 | 0.9972 at *L* = 17 |

The blog post derives the sawtooth behind the worst cases and discusses the
rest of the measurements.

## Development

```sh
uv run pytest --cov      # tests, with coverage
uv run ruff check        # lint
uv run ruff format       # format
uv run basedpyright      # type check, standard mode
```

CI runs each of these as its own job, the tests on Python 3.13, 3.14 and
3.15 with a 95 % coverage floor, and also checks that `uv.lock` is current
and that the built wheel carries the package alone.

## Licence

The package is released under the MIT licence, see [LICENSE](LICENSE).
