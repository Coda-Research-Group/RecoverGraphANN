"""HNSW via the instrumented hnswlib fork.

This is the backend Table 1 is about, and the only one that reports what the
neighbor-selection heuristic did. Two things come back from the fork that stock hnswlib does
not provide:

- ``add_items`` returns, among other counters, one ``(candidate pool size, accepted count)``
  pair per ``mutuallyConnectNewElement`` call — Table 1's ``acc``.
- ``get_all_links`` returns each node's per-level neighbour lists — Table 1's ``deg``.

`enable_pruning=False` switches the heuristic off entirely. That is not an experiment the
paper reports, but it is how the diagnosis was confirmed: with pruning off, `acc` jumps to
``M`` and the sparsity disappears, which is what pins the failure on the heuristic rather
than on the data alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rgann.indexes.base import BuildResult, SearchResult, timed

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['HnswlibBackend']

#: hnswlib's own default. Pinned explicitly because layer assignment is randomised, and
#: Table 1's acc/deg do not reproduce across seeds.
DEFAULT_SEED = 100


def _require_instrumented(index: object) -> None:
    if not hasattr(index, 'get_all_links'):
        msg = (
            'this hnswlib build has no get_all_links, so it is the stock package rather than '
            'the instrumented fork, and cannot produce Table 1. Install the pinned submodule:\n'
            '    pip install ./third_party/hnswlib'
        )
        raise RuntimeError(msg)


class HnswlibBackend:
    """`hnsw-hnswlib` in the results CSV — the paper's primary index."""

    name = 'hnsw-hnswlib'
    search_param_name = 'ef_search'

    def build(
        self,
        X: np.ndarray,
        params: dict[str, Any],
        *,
        threads: int,
        seed: int = DEFAULT_SEED,
    ) -> BuildResult:
        """Index ``X`` with ``M`` and ``ef_construction`` from ``params``.

        ``params['enable_pruning']`` defaults to True, i.e. stock HNSW behaviour.
        """
        import hnswlib  # noqa: PLC0415

        index = hnswlib.Index(space='ip', dim=int(X.shape[1]))
        _require_instrumented(index)

        index.init_index(
            max_elements=int(X.shape[0]),
            ef_construction=int(params['ef_construction']),
            M=int(params['m']),
            random_seed=int(seed),
        )
        index.set_num_threads(threads)

        enable_pruning = bool(params.get('enable_pruning', True))
        # Row order is insertion order. Never shuffle: the graph, and therefore acc and deg,
        # depend on it.
        labels = np.arange(X.shape[0], dtype=np.int64)

        with timed() as t:
            returned = index.add_items(
                np.ascontiguousarray(X, dtype=np.float32),
                labels,
                num_threads=threads,
                enable_pruning=enable_pruning,
            )

        mutual_connect_sizes = _extract_mutual_connect_sizes(returned)

        return BuildResult(
            index=index,
            build_time_ms=t.ms,
            mutual_connect_sizes=mutual_connect_sizes,
            links_per_node=index.get_all_links(),
            metadata={
                'm': int(params['m']),
                'ef_construction': int(params['ef_construction']),
                'enable_pruning': enable_pruning,
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
        index.set_ef(int(search_value))
        index.set_num_threads(threads)
        queries = np.ascontiguousarray(Q, dtype=np.float32)

        with timed() as t:
            ids, _ = index.knn_query(queries, k=k, num_threads=threads)

        return SearchResult(ids=np.asarray(ids, dtype=np.int64), elapsed_seconds=t.seconds)


def _extract_mutual_connect_sizes(returned: object) -> Sequence[tuple[int, int]]:
    """Pull the ``(pool, accepted)`` pairs out of the patched ``add_items`` return value.

    The fork returns ``(candidates_visited, candidates_pruned, mutual_connect_sizes,
    heuristic_good_not_good_counts)``. Only the third element feeds Table 1; the others are
    kept in the fork because they were used to localise the problem, and are cheap.
    """
    expected_arity = 4
    if not isinstance(returned, tuple) or len(returned) != expected_arity:
        msg = (
            f'expected the instrumented add_items to return {expected_arity} values, got '
            f'{type(returned).__name__} of length {len(returned) if isinstance(returned, tuple) else "n/a"}'
        )
        raise RuntimeError(msg)
    return [(int(before), int(after)) for before, after in returned[2]]
