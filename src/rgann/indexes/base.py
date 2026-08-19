"""What every index backend has to look like.

The paper evaluates four graph indexes under one protocol: build on ``X`` alone, then sweep
a single search-effort knob. Each backend names that knob itself (`search_param_name`),
because they disagree — HNSW calls it ``efSearch``, NSG ``search_L``, RoarGraph ``L_pq`` —
and a results CSV that flattened them all to "ef" would be lying about three of the four.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = ['BuildResult', 'IndexBackend', 'SearchResult', 'timed']


@dataclass
class BuildResult:
    """A built index, plus whatever it was willing to say about how it got built.

    `mutual_connect_sizes` and `links_per_node` are populated only by the instrumented
    hnswlib backend; they are what Table 1's ``acc`` and ``deg`` are computed from. Every
    other backend leaves them empty, which is why Table 1 covers HNSW only.
    """

    index: Any
    build_time_ms: float
    mutual_connect_sizes: Sequence[tuple[int, int]] | None = None
    links_per_node: dict[int, Sequence[Sequence[int]]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Returned neighbour ids and how long the whole batch took."""

    ids: np.ndarray
    elapsed_seconds: float


class IndexBackend(Protocol):
    """Build once, then search at several effort levels."""

    name: str
    """Value written to the ``algorithm`` column, matching the paper's naming."""

    search_param_name: str
    """This backend's search-effort knob, e.g. ``ef_search``."""

    def build(
        self,
        X: np.ndarray,
        params: dict[str, Any],
        *,
        threads: int,
        seed: int,
    ) -> BuildResult:
        """Index ``X``. Insertion order is the row order of ``X`` and is never shuffled."""
        ...

    def search(
        self,
        built: BuildResult,
        Q: np.ndarray,
        k: int,
        search_value: int,
        *,
        threads: int,
    ) -> SearchResult:
        """Return the top-``k`` ids for every row of ``Q``."""
        ...


class timed:  # noqa: N801  (used as a context manager, reads as a verb at call sites)
    """Wall-clock timer. ``with timed() as t: ...`` then ``t.ms`` / ``t.seconds``."""

    __slots__ = ('_start', 'seconds')

    def __init__(self) -> None:
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.seconds = time.perf_counter() - self._start

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0
