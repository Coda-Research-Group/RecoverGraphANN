"""Turning an experiment specification into rows of a results CSV.

One `ExperimentSpec` is one (dataset, backend, normalization, build parameters) combination.
Building is done once and then searched at every value of the sweep, because that is both
faster and the only way a recall-vs-QPS curve means anything: all points on a curve must come
from the same graph.

Every row carries its own provenance — git commit, hostname, CPU model, thread count, seed —
so a CSV that has travelled away from the run that produced it can still be traced back.
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rgann.indexes import get_backend
from rgann.metrics import construction_statistics, recall_at_k, summarize_search
from rgann.transform import Normalization, apply_normalization

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rgann.datasets import Dataset

__all__ = ['ExperimentSpec', 'append_rows', 'atomic_output', 'provenance', 'run_spec']


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Yield a scratch path, and move it onto ``path`` only once the block completes.

    ``run_all_experiments.sh`` skips a stage whose output already exists, which is what makes
    an interrupted multi-day run resumable. Appending directly to the real path defeats that:
    a run killed halfway leaves a partial CSV that the next attempt accepts as finished, and
    the missing rows show up as a figure with curves quietly absent. Writing to a sibling and
    renaming means the real path only ever appears when it is complete.
    """
    partial = path.with_name(path.name + '.partial')
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    yield partial
    partial.replace(path)


@dataclass(frozen=True)
class ExperimentSpec:
    """One index, built once, searched across a sweep."""

    dataset: str
    backend: str
    normalization: Normalization
    build_params: dict[str, Any]
    search_values: Sequence[int]
    k: int = 10
    threads: int = 1
    seed: int = 100
    label: str = ''
    """Free-text tag written to the CSV, e.g. which paper artifact this row belongs to."""

    extra_columns: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        params = ' '.join(
            f'{name}={value}' for name, value in sorted(self.build_params.items()) if name != 'learn_table'
        )
        return f'{self.dataset} {self.backend} norm={self.normalization} {params}'


def _git(*args: str) -> str:
    try:
        return subprocess.run(  # noqa: S603
            ['git', *args],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ''


def _cpu_model() -> str:
    """Best-effort CPU name. Recorded because every QPS number is specific to it."""
    try:
        with Path('/proc/cpuinfo').open(encoding='utf-8') as cpuinfo:
            for line in cpuinfo:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return platform.processor() or 'unknown'


_provenance_cache: dict[str, str] | None = None


def provenance() -> dict[str, str]:
    """Where and from what this row came. Computed once per process.

    ``hostname`` is the machine's own name by default, which is what you want while running
    your own experiments. Set ``RGANN_HOSTNAME`` to publish under a label instead: results
    committed here were produced on a machine whose real name identifies private
    infrastructure, and once that is in a public repository it cannot be taken back. The
    label still says *which* machine, which is all the column is for; the CPU model, in its
    own column, is the part that explains a throughput number.
    """
    global _provenance_cache  # noqa: PLW0603
    if _provenance_cache is not None:
        return _provenance_cache

    dirty = _git('status', '--porcelain')
    _provenance_cache = {
        'run_timestamp_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': str(bool(dirty)),
        'hostname': os.environ.get('RGANN_HOSTNAME') or platform.node(),
        'cpu_model': _cpu_model(),
        'platform': platform.platform(),
        'python': platform.python_version(),
    }
    return _provenance_cache


def run_spec(spec: ExperimentSpec, dataset: Dataset, *, verbose: bool = True) -> Iterator[dict[str, Any]]:
    """Build one index and yield one row per search value.

    Rows are yielded rather than returned so a caller can flush each to disk as it arrives —
    a build that takes an hour should not lose its sweep to an interruption at the end.
    """
    backend = get_backend(spec.backend)
    X, Q = apply_normalization(dataset.X, dataset.Q, spec.normalization)

    if verbose:
        print(f'build   {spec.describe()}  d={X.shape[1]}', flush=True)

    built = backend.build(X, spec.build_params, threads=spec.threads, seed=spec.seed)
    construction = construction_statistics(built.mutual_connect_sizes, built.links_per_node)

    if verbose:
        print(
            f'        built in {built.build_time_ms / 1000:.1f}s  '
            f'acc={construction.accepted_candidates_avg:.2f}  deg={construction.avg_node_degree:.2f}',
            flush=True,
        )

    base = {
        'dataset_name': dataset.name,
        'algorithm': backend.name,
        'normalization': str(spec.normalization),
        'dimension': int(X.shape[1]),
        'database_rows': dataset.n,
        'subsampled': str(dataset.subsampled),
        'k': spec.k,
        'threads': spec.threads,
        'label': spec.label,
        'build_time_ms': built.build_time_ms,
        'search_param_name': backend.search_param_name,
        **construction.to_csv(),
        **built.metadata,
        **spec.extra_columns,
        **provenance(),
    }

    for search_value in spec.search_values:
        result = backend.search(built, Q, spec.k, search_value, threads=spec.threads)
        per_query = recall_at_k(result.ids, dataset.GT, spec.k)
        row = {
            **base,
            'search_param_value': search_value,
            **summarize_search(per_query, result.elapsed_seconds),
        }
        if verbose:
            print(
                f'        {backend.search_param_name}={search_value:<6} '
                f'recall={row["recall_avg"] * 100:6.2f}%  qps={row["queries_per_sec"]:9.1f}',
                flush=True,
            )
        yield row


def append_rows(path: Path | str, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows to a CSV, writing the header if the file is new.

    The union of keys across ``rows`` becomes the header, so backends with different
    parameters can share one file. Appending to an existing file keeps that file's header and
    drops keys it does not have, which is deliberate: a partially-rerun CSV should stay
    readable rather than growing a second header mid-file.
    """
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open(newline='') as handle:
            fieldnames = next(csv.reader(handle), [])
        if fieldnames:
            with path.open('a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
                writer.writerows(rows)
            return

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
