"""The four graph indexes the paper evaluates.

Three are query-agnostic by construction — HNSW, NSG, FlatNav — and one, RoarGraph, is
query-aware and is adapted to this one; see `rgann.indexes.roargraph`.

Look up a backend by the name it uses in the results CSV::

    from rgann.indexes import get_backend
    backend = get_backend('hnsw-hnswlib')

Every backend is imported lazily, so a machine missing one of them (macOS has no RoarGraph)
can still run everything else.
"""

from __future__ import annotations

from rgann.indexes.base import BuildResult, IndexBackend, SearchResult

__all__ = ['BACKEND_NAMES', 'BuildResult', 'IndexBackend', 'SearchResult', 'get_backend']

BACKEND_NAMES = ('hnsw-hnswlib', 'nsg-faiss', 'flatnav', 'roargraph')


def get_backend(name: str) -> IndexBackend:
    """Return the backend registered under ``name``.

    Raises:
        ValueError: If ``name`` is not one of `BACKEND_NAMES`.
        RuntimeError: Re-raised from the backend when its native extension is unavailable,
            with the command that installs it.

    """
    match name:
        case 'hnsw-hnswlib':
            from rgann.indexes.hnsw import HnswlibBackend  # noqa: PLC0415

            return HnswlibBackend()
        case 'nsg-faiss':
            from rgann.indexes.nsg import FaissNsgBackend  # noqa: PLC0415

            return FaissNsgBackend()
        case 'flatnav':
            from rgann.indexes.flatnav import FlatNavBackend  # noqa: PLC0415

            return FlatNavBackend()
        case 'roargraph':
            from rgann.indexes.roargraph import RoarGraphBackend  # noqa: PLC0415

            return RoarGraphBackend()

    known = ', '.join(BACKEND_NAMES)
    msg = f'unknown index backend {name!r}; expected one of: {known}'
    raise ValueError(msg)
