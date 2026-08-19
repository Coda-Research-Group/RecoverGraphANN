r"""RoarGraph, adapted to the query-agnostic setting.

**This is a methodological choice, not a detail.** RoarGraph is a query-*aware* index: it
shapes connectivity from a bipartite graph between the database and a sample of training
queries drawn from the real query distribution. The paper's setting is query-agnostic — the
index may see only ``X`` — so that sample does not exist.

To compare it at all, we substitute the database for the query side: for every
:math:`x_i \in X` we compute its exact top-``k`` neighbours *within* ``X`` and hand those to
RoarGraph as its bipartite input (`database_side_learn_table`). This measures what the
bipartite construction contributes when no query information is available. It is emphatically
not the setting RoarGraph was designed for, and the resulting numbers characterise our
adaptation rather than the published method.

RoarGraph's native mode, which consumes VIBE's ``learn_neighbors``, is deliberately not
implemented here. It is a different setting, this artifact evaluates one, and VIBE itself
benchmarks RoarGraph that way (`vibe/algorithms/roargraph/config.yml`, ``ood: true``).

The pinned fork adds Python bindings and a portable build and changes nothing about the
algorithm, so what runs here is upstream RoarGraph — which is what reproducing the paper
requires.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rgann.indexes.base import BuildResult, SearchResult, timed

__all__ = [
    'RoarGraphBackend',
    'database_side_learn_table',
    'drop_self_hits',
    'sampled_row_order',
]

#: VIBE's ``learn_neighbors`` are 100 columns wide; the database-side substitute matches it.
DEFAULT_LEARN_K = 100

#: The paper's seed for the database-side sampling.
DEFAULT_LEARN_SEED = 42


def database_side_learn_table(
    X: np.ndarray,
    k: int = DEFAULT_LEARN_K,
    seed: int = DEFAULT_LEARN_SEED,
) -> np.ndarray:
    """Every database row's exact top-``k`` neighbours inside ``X``, excluding itself.

    Computed with a FAISS flat inner-product index, so it is exact and independent of any
    graph index.

    ``seed`` fixes the row order of the resulting table; the paper used 42. That order is not
    cosmetic — RoarGraph consumes the rows in sequence while building its bipartite graph, so
    a different permutation of the same rows produces a different index. See
    `sampled_row_order`.

    Returns:
        uint32 array of shape ``(n, k)``.

    """
    import faiss  # noqa: PLC0415

    n = int(X.shape[0])
    k = min(k, n - 1)
    vectors = np.ascontiguousarray(X, dtype=np.float32)

    index = faiss.IndexFlatIP(int(X.shape[1]))
    index.add(vectors)
    # k+1 columns because each row retrieves itself first.
    _, neighbours = index.search(vectors, k + 1)

    table = drop_self_hits(neighbours, k)
    return np.ascontiguousarray(table[sampled_row_order(n, seed)])


def sampled_row_order(n: int, seed: int) -> np.ndarray:
    """The order the learn-table rows are fed to RoarGraph in.

    Must stay `Generator.choice(n, size=n, replace=False)`. That is what the harness the
    paper's numbers came from used, and — despite both being "a seeded permutation of n" —
    it does **not** agree with `Generator.permutation(n)` from the same seed. The two consume
    the bit stream differently and return different orders.

    Getting this wrong is silent: the table holds the same rows either way, every recall
    number stays plausible, and RoarGraph's raw-space curve simply lands 1.7 to 4.4 points
    below the published one.
    """
    return np.random.default_rng(seed).choice(n, size=n, replace=False)


def drop_self_hits(neighbours: np.ndarray, k: int) -> np.ndarray:
    """Remove each row's own id from its ``(n, k+1)`` neighbour list, returning ``(n, k)``.

    A row that does not retrieve itself — possible when several database vectors tie exactly
    — drops its last column instead, so the table stays rectangular either way.
    """
    n = int(neighbours.shape[0])
    if neighbours.shape[1] != k + 1:
        msg = f'expected {k + 1} columns for k={k}, got {neighbours.shape[1]}'
        raise ValueError(msg)

    is_self = neighbours == np.arange(n, dtype=neighbours.dtype)[:, None]
    # Drop exactly one column per row — the self-hit where there is one, the last column
    # otherwise — so the result is rectangular by construction rather than by luck.
    drop_column = np.where(is_self.any(axis=1), np.argmax(is_self, axis=1), k)
    keep = np.ones(neighbours.shape, dtype=bool)
    keep[np.arange(n), drop_column] = False
    return neighbours[keep].reshape(n, k).astype(np.uint32)


class RoarGraphBackend:
    """`roargraph` in the results CSV."""

    name = 'roargraph'
    search_param_name = 'l_pq'

    def build(
        self,
        X: np.ndarray,
        params: dict[str, Any],
        *,
        threads: int,
        seed: int = DEFAULT_LEARN_SEED,
    ) -> BuildResult:
        """Build from ``X`` plus a bipartite learn table.

        ``params['learn_table']`` must be the ``(rows, k)`` uint32 table, built with
        `database_side_learn_table`.
        """
        from RoarGraph import IndexRoarGraph, Metric  # noqa: PLC0415

        dimension = int(X.shape[1])
        learn_table = np.ascontiguousarray(params['learn_table'], dtype=np.uint32)
        vectors = np.ascontiguousarray(X, dtype=np.float32)

        index = IndexRoarGraph(dimension, int(X.shape[0]) + int(learn_table.shape[0]), Metric.IP)
        index.setThreads(threads)

        with timed() as t:
            index.build(
                int(learn_table.shape[0]),
                int(learn_table.shape[1]),
                int(X.shape[0]),
                int(params['m_sq']),
                int(params['m_pjbp']),
                int(params['l_pjpq']),
                threads,
                learn_table,
                vectors,
            )

        return BuildResult(
            index=index,
            build_time_ms=t.ms,
            metadata={
                'm_sq': int(params['m_sq']),
                'm_pjbp': int(params['m_pjbp']),
                'l_pjpq': int(params['l_pjpq']),
                'learn_source': 'database',
                'learn_rows': int(learn_table.shape[0]),
                'learn_k': int(learn_table.shape[1]),
                'build_seed': int(seed),
                'dimension': dimension,
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
        index.setThreads(threads)
        queries = np.ascontiguousarray(Q, dtype=np.float32)

        ids = np.empty((queries.shape[0], k), dtype=np.uint32)
        distances = np.empty((queries.shape[0], k), dtype=np.float32)

        with timed() as t:
            index.search(queries, k, int(search_value), ids, distances, queries.shape[0], threads, False)

        return SearchResult(ids=ids.astype(np.int64, copy=False), elapsed_seconds=t.seconds)
