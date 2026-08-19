"""NSG via FAISS `IndexNSGFlat`.

The paper builds with ``R = 48`` and ``L = C = 500`` and sweeps ``search_L``. NSG has no
incremental add — the whole database goes in one call, from which FAISS builds a k-NN graph
and then prunes it — so there is no per-insertion instrumentation to collect and NSG does
not appear in Table 1. It appears in Figure 3.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from rgann.indexes.base import BuildResult, SearchResult, timed

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ['FaissNsgBackend']


@contextmanager
def faiss_threads(n_threads: int) -> Iterator[None]:
    """Hold FAISS's OpenMP thread count for the duration of a block, then put it back.

    FAISS's thread count is process-global, so leaving it changed would silently alter every
    later timing in the same run.
    """
    import faiss  # noqa: PLC0415

    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(n_threads)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


class FaissNsgBackend:
    """`nsg-faiss` in the results CSV."""

    name = 'nsg-faiss'
    search_param_name = 'search_l'

    def build(
        self,
        X: np.ndarray,
        params: dict[str, Any],
        *,
        threads: int,
        seed: int = 0,
    ) -> BuildResult:
        import faiss  # noqa: PLC0415

        index = faiss.IndexNSGFlat(int(X.shape[1]), int(params['r']), faiss.METRIC_INNER_PRODUCT)
        index.nsg.L = int(params['l'])
        index.nsg.C = int(params['c'])

        vectors = np.ascontiguousarray(X, dtype=np.float32)
        with faiss_threads(threads), timed() as t:
            index.add(vectors)

        return BuildResult(
            index=index,
            build_time_ms=t.ms,
            metadata={
                'r': int(params['r']),
                'l': int(params['l']),
                'c': int(params['c']),
                # FAISS seeds NSG's internal RNG from its own global generator. We record the
                # requested seed for provenance but do not claim it makes the build
                # bit-reproducible; see the determinism note in the README.
                'build_seed': int(seed),
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
        index.nsg.search_L = int(search_value)
        queries = np.ascontiguousarray(Q, dtype=np.float32)

        with faiss_threads(threads), timed() as t:
            _, ids = index.search(queries, k)

        return SearchResult(ids=np.asarray(ids, dtype=np.int64), elapsed_seconds=t.seconds)
