"""Recall, and the graph-construction statistics behind Table 1.

Table 1 prints two abbreviated columns. They have full names here, because a reader of the
CSV should not have to guess:

======================  ==============  ==============================================================
Table 1 column          this module     what it is
======================  ==============  ==============================================================
``acc``                 ``accepted_candidates_avg``  average number of candidates the HNSW
                                        neighbor-selection heuristic *accepts* per insertion, out of
                                        an ``efConstruction`` pool. A count, not a ratio.
``deg``                 ``avg_node_degree``          average out-degree of a node, summed over all
                                        layers.
======================  ==============  ==============================================================

Both come from the instrumented hnswlib fork (``patches/hnswlib-*.patch``): ``add_items``
returns, per ``mutuallyConnectNewElement`` call, the pair *(candidate pool size on entry,
selected-neighbour count)*, and ``get_all_links`` exposes each node's per-level neighbour
lists. Stock hnswlib reports neither, which is why the fork exists.

`candidate_pool_avg` is the pool size those acceptances are drawn from — it stays pinned near
``efConstruction`` in every run, which is what makes ``acc`` interpretable: the heuristic sees
~500 candidates either way and the transformation changes only how many survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    'ConstructionStatistics',
    'RecallStatistics',
    'construction_statistics',
    'degree_statistics',
    'recall_at_k',
    'summarize_search',
]


def recall_at_k(found: np.ndarray, ground_truth: np.ndarray, k: int) -> np.ndarray:
    """Per-query recall@k.

    Args:
        found: Returned neighbour ids, shape ``(q, >=k)``.
        ground_truth: Exact neighbour ids, shape ``(q, >=k)``.
        k: Neighbours to score.

    Returns:
        float32 array of shape ``(q,)``, each in ``[0, 1]``.

    Notes:
        Ground truth with a repeated id (ties at the k-th distance do occur in VIBE) would
        otherwise let one returned id score twice, so duplicates within a ground-truth row
        are counted once. Ported unchanged from the original harness, since every published
        recall number came out of it.

    """
    if k <= 0:
        msg = f'k must be positive, got {k}'
        raise ValueError(msg)
    if found.shape[0] != ground_truth.shape[0]:
        msg = f'query count mismatch: found has {found.shape[0]}, ground truth has {ground_truth.shape[0]}'
        raise ValueError(msg)
    if found.shape[1] < k or ground_truth.shape[1] < k:
        msg = f'need at least {k} columns, got found={found.shape[1]} ground_truth={ground_truth.shape[1]}'
        raise ValueError(msg)

    found_k = found[:, :k]
    sorted_gt = np.sort(ground_truth[:, :k], axis=1)

    first_occurrence = np.empty_like(sorted_gt, dtype=bool)
    first_occurrence[:, 0] = True
    first_occurrence[:, 1:] = sorted_gt[:, 1:] != sorted_gt[:, :-1]

    in_result = (sorted_gt[:, :, None] == found_k[:, None, :]).any(axis=2)
    return (np.sum(in_result & first_occurrence, axis=1) / k).astype(np.float32)


@dataclass(frozen=True)
class RecallStatistics:
    """Recall@k summarised over queries. `avg` is the number the paper reports."""

    avg: float
    min: float
    p50: float
    p95: float
    p99: float
    std: float
    frac_at_0: float
    frac_at_1: float

    @classmethod
    def from_per_query(cls, recalls: np.ndarray) -> RecallStatistics:
        return cls(
            avg=float(np.mean(recalls)),
            min=float(np.min(recalls)),
            p50=float(np.percentile(recalls, 50)),
            p95=float(np.percentile(recalls, 95)),
            p99=float(np.percentile(recalls, 99)),
            std=float(np.std(recalls)),
            frac_at_0=float(np.mean(recalls == 0.0)),
            frac_at_1=float(np.mean(recalls == 1.0)),
        )

    def to_csv(self, prefix: str = 'recall') -> dict[str, float]:
        return {f'{prefix}_{name}': value for name, value in vars(self).items()}


@dataclass(frozen=True)
class ConstructionStatistics:
    """What the neighbor-selection heuristic did while the graph was being built.

    `accepted_candidates_avg` is Table 1's ``acc`` and `avg_node_degree` is its ``deg``.
    """

    accepted_candidates_avg: float
    accepted_candidates_median: float
    candidate_pool_avg: float
    candidate_pool_median: float
    avg_node_degree: float
    median_node_degree: float
    max_node_degree: int
    avg_node_degree_level0: float
    median_node_degree_level0: float
    max_node_degree_level0: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_csv(self) -> dict[str, float | int]:
        return {name: value for name, value in vars(self).items() if name != 'extra'}


def _summarize(values: np.ndarray) -> tuple[float, float, float]:
    """``(mean, median, max)``; zeros for an empty input rather than a nan."""
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.median(values)), float(np.max(values))


def degree_statistics(links_per_node: dict[int, Sequence[Sequence[int]]]) -> dict[str, float | int]:
    """Out-degree over all layers and at level 0, from ``hnswlib.Index.get_all_links()``.

    ``links_per_node`` maps a node label to its per-level neighbour lists, outermost level
    last. Table 1's ``deg`` is the all-layers figure; the level-0 figure is reported
    alongside because it is the one a reader is likely to assume, and it differs by roughly
    0.4 on these datasets.
    """
    total = np.array(
        [sum(len(level) for level in levels) for levels in links_per_node.values()],
        dtype=np.int64,
    )
    level0 = np.array(
        [len(levels[0]) if len(levels) > 0 else 0 for levels in links_per_node.values()],
        dtype=np.int64,
    )
    total_avg, total_median, total_max = _summarize(total)
    l0_avg, l0_median, l0_max = _summarize(level0)
    return {
        'avg_node_degree': total_avg,
        'median_node_degree': total_median,
        'max_node_degree': int(total_max),
        'avg_node_degree_level0': l0_avg,
        'median_node_degree_level0': l0_median,
        'max_node_degree_level0': int(l0_max),
    }


def construction_statistics(
    mutual_connect_sizes: Sequence[tuple[int, int]] | None,
    links_per_node: dict[int, Sequence[Sequence[int]]] | None,
) -> ConstructionStatistics:
    """Assemble Table 1's row from what the instrumented build returned.

    Args:
        mutual_connect_sizes: One ``(candidate_pool_size, accepted_count)`` pair per
            ``mutuallyConnectNewElement`` call, as returned by the patched
            ``hnswlib.Index.add_items``.
        links_per_node: ``hnswlib.Index.get_all_links()`` output.

    Backends other than the patched hnswlib expose neither, so both are optional and the
    corresponding columns come back as zeros — which is why Table 1 is HNSW-only.

    """
    pairs = list(mutual_connect_sizes or [])
    pool = np.array([before for before, _ in pairs], dtype=np.int64)
    accepted = np.array([after for _, after in pairs], dtype=np.int64)

    pool_avg, pool_median, _ = _summarize(pool)
    accepted_avg, accepted_median, _ = _summarize(accepted)
    degrees = degree_statistics(links_per_node or {})

    return ConstructionStatistics(
        accepted_candidates_avg=accepted_avg,
        accepted_candidates_median=accepted_median,
        candidate_pool_avg=pool_avg,
        candidate_pool_median=pool_median,
        **degrees,  # type: ignore[arg-type]
    )


def summarize_search(
    per_query_recall: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, float]:
    """Recall summary plus throughput, as reported in Figure 3.

    QPS is machine-specific by construction — see the hardware note in the README.
    """
    if elapsed_seconds <= 0.0:
        msg = f'elapsed_seconds must be positive, got {elapsed_seconds}'
        raise ValueError(msg)
    stats = RecallStatistics.from_per_query(per_query_recall)
    return {
        **stats.to_csv(),
        'num_queries': int(per_query_recall.shape[0]),
        'search_time_ms': elapsed_seconds * 1000.0,
        'queries_per_sec': per_query_recall.shape[0] / elapsed_seconds,
    }
