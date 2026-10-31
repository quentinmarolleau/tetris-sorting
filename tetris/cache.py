"""On-disk caches for the benchmarks, with enough provenance that a
stale file cannot be mistaken for a fresh one.

The acceptance sweep takes about a quarter of an hour and the
rearrangement sweep a few minutes, so their results are kept on disk and
reused. Reuse keyed on the file merely existing would silently serve a
changed size range or a changed algorithm from an old run. Everything
that can move the numbers is folded into a fingerprint instead, the
algorithm through a digest of the sources the benchmark runs, and a
mismatch reports why.
"""

import ast
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tetris.fast import DEFAULT_RUN_SELECTION

PACKAGE_NAME = __name__.split(".")[0]

__all__ = [
    "AcceptanceParameters",
    "RearrangementParameters",
    "algorithm_digest",
    "cpu_model",
    "load_measurements",
    "load_results",
    "merge_measurements",
    "save_results",
    "source_closure",
]


@dataclass(frozen=True)
class AcceptanceParameters:
    """Everything an acceptance benchmark run is characterised by.

    Mirrors the arguments of
    `tetris.acceptance_bench.run_acceptance_benchmark`, plus the digest of
    the code that ran. `workers` is recorded for provenance but stays out
    of the fingerprint: it changes how long the run takes, not the
    distribution it samples.

    Attributes:
        sizes: Loading array sizes swept.
        reshape_variants: (rows, columns) reshapes of the target swept.
        seconds_per_task: Wall-clock budget of each task.
        workers: Worker processes the run used.
        algorithm_digest: `algorithm_digest` of the benchmark's sources.
    """

    sizes: tuple[int, ...]
    reshape_variants: tuple[tuple[int, int], ...]
    seconds_per_task: float
    workers: int
    algorithm_digest: str

    def fields(self) -> dict:
        """Every parameter that can change the results.

        Returns:
            The parameters as JSON values, `workers` left out, with
            `benchmark` naming the benchmark.
        """
        return {
            "benchmark": "acceptance",
            "sizes": list(self.sizes),
            "reshape_variants": [list(reshape) for reshape in self.reshape_variants],
            "seconds_per_task": self.seconds_per_task,
            "algorithm_digest": self.algorithm_digest,
        }

    def fingerprint(self) -> str:
        """Hash of `fields()`, the key a cached run is matched on.

        Returns:
            Sixteen hexadecimal digits.
        """
        return _fingerprint(self.fields())


@dataclass(frozen=True)
class RearrangementParameters:
    """Everything a rearrangement benchmark run is characterised by.

    Mirrors the arguments of
    `tetris.rearrangement_bench.run_rearrangement_benchmark`, plus the
    digest of the code that ran. `workers` stays out of the fingerprint.

    Attributes:
        sizes: Loading array sizes swept.
        reshape_variants: (rows, columns) reshapes of the target swept.
        configurations_per_task: Kept loadings measured per task.
        workers: Worker processes the run used.
        algorithm_digest: `algorithm_digest` of the benchmark's sources.
        run_selections: Run selections measured, the reference first.
    """

    sizes: tuple[int, ...]
    reshape_variants: tuple[tuple[int, int], ...]
    configurations_per_task: int
    workers: int
    algorithm_digest: str
    run_selections: tuple[tuple[str, ...], ...] = (DEFAULT_RUN_SELECTION,)

    def fields(self) -> dict:
        """Every parameter that can change the results.

        Returns:
            The parameters as JSON values, `workers` left out, with
            `benchmark` naming the benchmark.
        """
        return {
            "benchmark": "rearrangement",
            "sizes": list(self.sizes),
            "reshape_variants": [list(reshape) for reshape in self.reshape_variants],
            "configurations_per_task": self.configurations_per_task,
            "run_selections": [list(selection) for selection in self.run_selections],
            "algorithm_digest": self.algorithm_digest,
        }

    def fingerprint(self) -> str:
        """Hash of `fields()`, the key a cached run is matched on.

        Returns:
            Sixteen hexadecimal digits.
        """
        return _fingerprint(self.fields())


type CacheParameters = AcceptanceParameters | RearrangementParameters


def _fingerprint(fields: dict) -> str:
    """Sixteen hexadecimal digits over `fields`, the same whether tuples
    or the lists a JSON round trip gives back were passed in."""
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cpu_model() -> str:
    """Readable name of the CPU, recorded beside every timing.

    `platform.processor()` answers `x86_64` on Linux, which says nothing
    about the machine a timing was taken on, so `/proc/cpuinfo` is read
    first when it exists.

    Returns:
        The model name, such as "AMD Ryzen 9 3900X 12-Core Processor", or
        the platform's processor or machine string as a fallback.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _package_directory() -> Path:
    return Path(__file__).resolve().parent


def _imported_submodules(source: str) -> set[str]:
    """Names of this package's modules that one source file imports,
    covering both `from tetris.x import y` and `from tetris import x`."""
    package = _package_directory()
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == PACKAGE_NAME:
            # `from tetris import display` names submodules, not objects,
            # whenever a file of that name exists beside this one
            found.update(
                alias.name
                for alias in node.names
                if (package / f"{alias.name}.py").is_file()
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == PACKAGE_NAME and len(parts) > 1:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE_NAME and len(parts) > 1:
                    found.add(parts[1])
    return found


def source_closure(root: str) -> tuple[Path, ...]:
    """Every module of this package that `root` reaches, itself included,
    as sorted absolute paths.

    `root` is a benchmark module, such as `tetris.acceptance_bench`; each
    benchmark gets its own closure, so editing what only one of them runs
    leaves the other's cache valid.

    This is what `algorithm_digest` is meant to be given. Deriving it
    from the imports keeps it honest where a hand-written list drifts:
    the list this replaced named `config.py`, which no sweep executes, so
    editing the class forced a rerun of numbers it cannot reach, and it
    omitted the benchmark driver and `stats.py`, which hold the sampling
    loop and three of the fields the cache records.

    `tetris/__init__.py` is deliberately not followed. It re-exports the
    class for convenience, so walking through it would pull in every
    module that re-export touches and undo the point of the exercise.
    Imports of the package's modules are what a source depends on;
    importing the package is not.

    Args:
        root: Dotted module name. Only its last component is used.

    Returns:
        Absolute paths of the modules reached, sorted.

    Raises:
        ValueError: If `root`, or a module it imports from the package,
            has no source file in the package.
    """
    package = _package_directory()
    pending = [root.split(".")[-1]]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = package / f"{name}.py"
        if not path.is_file():
            raise ValueError(f"no such module in {PACKAGE_NAME}: {name}")
        seen.add(name)
        pending.extend(_imported_submodules(path.read_text()))
    return tuple(sorted(package / f"{name}.py" for name in seen))


def _without_docstrings(tree: ast.Module) -> ast.Module:
    """`tree` with the leading string of every module, class and function
    body removed. The tree is freshly parsed and owned by the caller."""
    for node in ast.walk(tree):
        if (
            isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and ast.get_docstring(node, clean=False) is not None
        ):
            node.body = node.body[1:]
    return tree


def algorithm_digest(*paths: Path) -> str:
    """Hash of the sources that decide what a benchmark measures.

    Editing the algorithm changes the hash and so invalidates the cached
    numbers. The hash is taken over the syntax tree with docstrings
    removed, so rewording documentation, editing comments or reformatting
    leaves it unchanged: none of these can move a measured number. The
    tree dump depends on the Python version, so moving to another minor
    version may change every digest once.

    Args:
        paths: Source files, in any order. Pass `source_closure(root)`
            rather than a list written out by hand.

    Returns:
        Sixteen hexadecimal digits.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        tree = _without_docstrings(ast.parse(Path(path).read_text()))
        digest.update(ast.dump(tree).encode())
    return digest.hexdigest()[:16]


def save_results(
    path: Path,
    parameters: CacheParameters,
    entropy: int,
    results: dict,
    **extra_meta,
) -> None:
    """Write benchmark results with the provenance `load_results` checks.

    Args:
        path: File to write, replaced if it exists.
        parameters: The parameters the results were measured with. Their
            fingerprint is stored, and each field spelled out beside it.
        entropy: Seed entropy of the run, so it can be reproduced.
        results: Results keyed by loading array size.
        **extra_meta: Further provenance to store in the meta block.
    """
    payload = {
        "meta": {
            "date": datetime.now(UTC).isoformat(),
            "fingerprint": parameters.fingerprint(),
            **parameters.fields(),
            "workers": parameters.workers,
            "seed_entropy": entropy,
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(),
            **extra_meta,
        },
        "results": {str(size): variants for size, variants in sorted(results.items())},
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_results(path: Path, parameters: CacheParameters) -> tuple[dict | None, str]:
    """Cached results, if they were measured with these parameters.

    Args:
        path: File `save_results` wrote.
        parameters: The parameters the caller would run with now.

    Returns:
        The results keyed by integer size and an empty string, or None and
        the reason the file could not be used: missing, unreadable, or
        measured with a different fingerprint.
    """
    path = Path(path)
    if not path.exists():
        return None, f"no cache at {path.name}"
    try:
        payload = json.loads(path.read_text())
        meta = payload["meta"]
        results = payload["results"]
    except (json.JSONDecodeError, KeyError, OSError) as error:
        return None, f"cache unreadable: {error}"

    if meta.get("fingerprint") != parameters.fingerprint():
        return None, (
            "cache parameters differ from the current ones "
            f"({meta.get('fingerprint')} against "
            f"{parameters.fingerprint()})"
        )
    return {int(size): variants for size, variants in results.items()}, ""


def load_measurements(path: Path, digest: str) -> tuple[dict[int, dict], dict]:
    """Results the notebook reads but cannot produce, with their meta.

    `load_results` answers "may I reuse this, or must I rerun it", and
    returns a reason instead of raising because rerunning is the normal
    other branch. A sweep that took a console seven hours has no other
    branch: the notebook either reads the file or has nothing to draw,
    so a problem with it is an error rather than a fallback.

    The fingerprint is not checked, since there is no parameter set the
    notebook would rerun these from. The algorithm digest is: numbers
    measured on a different packing rule do not belong on the same axes
    as the rest.

    Args:
        path: File `save_results` wrote.
        digest: `algorithm_digest` of the sources the notebook runs now.

    Returns:
        The results keyed by integer size, and the meta block.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no meta and results pair, or was
            measured with a different algorithm digest.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no measurements at {path.name}")
    payload = json.loads(path.read_text())
    meta = payload.get("meta")
    results = payload.get("results")
    if meta is None or results is None:
        raise ValueError(f"{path.name} has no meta and results pair")
    recorded = meta.get("algorithm_digest")
    if recorded != digest:
        raise ValueError(
            f"{path.name} measured algorithm digest {recorded}, the tree is at {digest}"
        )
    return {int(size): variants for size, variants in results.items()}, meta


def merge_measurements(*bands: dict[int, dict]) -> dict[int, dict]:
    """One series from several bands, which must not overlap.

    Each band carries its own budget per point, so a size measured twice
    would offer two different numbers for one point of a curve with
    nothing to choose between them.

    Args:
        *bands: Results keyed by size, as `load_measurements` returns them.

    Returns:
        Every band's results in one dict, sorted by size.

    Raises:
        ValueError: If a size appears in more than one band.
    """
    merged: dict[int, dict] = {}
    for band in bands:
        shared = sorted(set(band) & set(merged))
        if shared:
            raise ValueError(
                f"bands overlap at {', '.join(str(size) for size in shared)}"
            )
        merged = {**merged, **band}
    return dict(sorted(merged.items()))
