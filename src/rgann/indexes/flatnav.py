"""FlatNav, the single-layer graph index.

The paper gives it the edge budget equivalent to HNSW's ``M`` (``max_edges_per_node``) and
sweeps ``ef_search``. FlatNav is used unmodified — the pinned submodule commit is exactly
upstream's ``v0.1.2-rc1`` tag. The only local change is a build-configuration patch for CPUs
without AVX2+FMA, applied by ``scripts/install_flatnav.sh``.

``distance_type='angular'`` is FlatNav's name for inner product, not for cosine; see its
`create` docstring.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rgann.indexes.base import BuildResult, SearchResult, timed

__all__ = ['FlatNavBackend']

_import_checked: bool | None = None


def _cpu_has_avx2_fma() -> bool:
    """True unless /proc/cpuinfo is readable and says otherwise (so: optimistic elsewhere)."""
    try:
        with Path('/proc/cpuinfo').open(encoding='utf-8') as cpuinfo:
            for line in cpuinfo:
                if line.startswith('flags'):
                    flags = line.split(':', 1)[1]
                    return 'avx2' in flags and 'fma' in flags
    except OSError:
        return True
    return False


def ensure_importable() -> None:
    """Import flatnav in a subprocess first.

    A wheel built for AVX2 running on a pre-AVX2 CPU dies with SIGILL, which cannot be caught
    in-process — the interpreter simply disappears, taking any partial results with it. So
    the check happens somewhere expendable, and the error names the fix.
    """
    global _import_checked  # noqa: PLW0603
    if _import_checked:
        return

    result = subprocess.run(  # noqa: S603
        [sys.executable, '-c', 'import flatnav'],
        check=False,
        capture_output=True,
        timeout=60,
    )
    if result.returncode == 0:
        _import_checked = True
        return

    scalar_hint = '' if _cpu_has_avx2_fma() else '  # this CPU lacks AVX2+FMA, so it will build scalar'
    msg = (
        f'FlatNav failed to import (exit {result.returncode}), usually an illegal instruction '
        f'from a wheel built for a different CPU. Build it from the pinned submodule:\n'
        f'    bash scripts/install_flatnav.sh{scalar_hint}'
    )
    raise RuntimeError(msg)


class FlatNavBackend:
    """`flatnav` in the results CSV."""

    name = 'flatnav'
    search_param_name = 'ef_search'

    def build(
        self,
        X: np.ndarray,
        params: dict[str, Any],
        *,
        threads: int,
        seed: int = 0,
    ) -> BuildResult:
        ensure_importable()
        from flatnav.data_type import DataType  # noqa: PLC0415
        from flatnav.index import create  # noqa: PLC0415

        index = create(
            distance_type='angular',
            dim=int(X.shape[1]),
            dataset_size=int(X.shape[0]),
            max_edges_per_node=int(params['m']),
            index_data_type=DataType.float32,
        )
        index.set_num_threads(threads)

        vectors = np.ascontiguousarray(X, dtype=np.float32)
        labels = np.arange(X.shape[0], dtype=np.int32)

        with timed() as t:
            index.add(data=vectors, ef_construction=int(params['ef_construction']), labels=labels)

        return BuildResult(
            index=index,
            build_time_ms=t.ms,
            metadata={
                'm': int(params['m']),
                'ef_construction': int(params['ef_construction']),
                # FlatNav's create() exposes no seed. Recorded as requested-but-not-applied so
                # the CSV does not imply a control that does not exist.
                'build_seed': int(seed),
                'seed_applied': False,
            },
        )

    def search(
        self,
        built: BuildResult,
        Q: np.ndarray,
        k: int,
        search_value: int,
        *,
        threads: int,
    ) -> SearchResult:
        index = built.index
        index.set_num_threads(threads)
        queries = np.ascontiguousarray(Q, dtype=np.float32)

        with timed() as t:
            _, ids = index.search(queries=queries, K=k, ef_search=int(search_value))

        return SearchResult(ids=np.asarray(ids, dtype=np.int64), elapsed_seconds=t.seconds)
