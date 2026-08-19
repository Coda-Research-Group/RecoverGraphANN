"""Database-side transformations.

The paper's method is `bachrach_transform`: scale the database globally by its largest row
norm, lift it into one extra dimension so every row lands on the unit sphere in ``d+1``, and
zero-pad the queries. It uses only ``X``, never the query distribution.

`l2_normalize` is the ablation from Section 6: it equalizes the norms the same way, but
answers a cosine query, so it distorts the inner-product ranking the benchmark scores
against. Keeping both behind one enum is what makes that comparison a flag rather than an
edit to the source.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

__all__ = ['Normalization', 'apply_normalization', 'bachrach_transform', 'l2_normalize']

_EXPECTED_NDIM = 2
"""Database and query matrices are always (rows, dimensions)."""


class Normalization(StrEnum):
    """Which database-side transformation to index and search in."""

    NONE = 'none'
    """Raw vectors, exactly as VIBE ships them."""

    BACHRACH = 'bachrach'
    """The paper's method: global scale + (d+1) lift onto the unit sphere."""

    L2 = 'l2'
    """Per-row L2 normalization. Equalizes norms but distorts the IP ranking."""


def _max_row_norm(M: np.ndarray) -> float:
    if M.shape[0] == 0:
        return 0.0
    return float(np.max(np.linalg.norm(M, axis=1)))


def bachrach_transform(X: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lift the database onto the unit sphere in ``d+1``; zero-pad the queries.

    With ``c = max_i ||x_i||``::

        P(x) = [x / c ; sqrt(max(0, 1 - ||x / c||^2))]     in R^(d+1)
        Q'(q) = [q ; 0]                                    in R^(d+1)

    so that ``<P(x), Q'(q)> = <x, q> / c``. The ranking a query induces over the database is
    therefore identical to the untransformed one; only the geometry the index sees changes.

    Queries are never scaled — the transform reads nothing but ``X``.

    Args:
        X: Database vectors, shape ``(n, d)``.
        Q: Query vectors, shape ``(m, d)``.

    Returns:
        ``(X', Q')`` with shapes ``(n, d+1)`` and ``(m, d+1)``, both float32.

    Raises:
        ValueError: If ``X`` and ``Q`` disagree on dimensionality.

    """
    if X.ndim != _EXPECTED_NDIM or Q.ndim != _EXPECTED_NDIM:
        msg = f'expected 2-D arrays, got X.ndim={X.ndim} Q.ndim={Q.ndim}'
        raise ValueError(msg)
    if X.shape[1] != Q.shape[1]:
        msg = f'dimension mismatch: X has d={X.shape[1]}, Q has d={Q.shape[1]}'
        raise ValueError(msg)

    scale = _max_row_norm(X)
    # An all-zero database has no scale to divide by; every row then lifts to the pole.
    X_scaled = X / scale if scale > 0.0 else np.zeros_like(X)

    squared_norms = np.sum(X_scaled**2, axis=1)
    # max(0, .) guards the float error that can push a max-norm row a hair above 1.
    extra = np.sqrt(np.maximum(1.0 - squared_norms, 0.0))

    X_lifted = np.column_stack([X_scaled, extra]).astype(np.float32)
    Q_padded = np.column_stack([Q, np.zeros_like(Q[:, :1])]).astype(np.float32)
    return np.ascontiguousarray(X_lifted), np.ascontiguousarray(Q_padded)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Scale every row to unit norm, leaving zero rows at zero.

    Section 6's ablation. It removes the norm spread that drives HNSW's pruning, but the
    resulting index answers a cosine query, so its ranking no longer matches the
    inner-product ground truth.
    """
    if X.ndim != _EXPECTED_NDIM:
        msg = f'expected a 2-D array, got ndim={X.ndim}'
        raise ValueError(msg)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return np.ascontiguousarray((X / safe).astype(np.float32))


def apply_normalization(
    X: np.ndarray,
    Q: np.ndarray,
    mode: Normalization | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(database, query)`` pair to index and search in.

    Args:
        X: Database vectors, shape ``(n, d)``.
        Q: Query vectors, shape ``(m, d)``.
        mode: A `Normalization` or its string value, as accepted on the command line.

    Raises:
        ValueError: If ``mode`` is not a known normalization.

    """
    try:
        normalization = Normalization(mode)
    except ValueError as exc:
        known = ', '.join(m.value for m in Normalization)
        msg = f'unknown normalization {mode!r}; expected one of: {known}'
        raise ValueError(msg) from exc

    match normalization:
        case Normalization.NONE:
            database, queries = X, Q
        case Normalization.BACHRACH:
            database, queries = bachrach_transform(X, Q)
        case Normalization.L2:
            # Queries stay raw, exactly as in the `none` and `bachrach` arms: the ablation
            # isolates the database-side norm equalization, nothing else.
            database, queries = l2_normalize(X), Q
        case _:  # pragma: no cover - exhaustive above
            msg = f'unhandled normalization {normalization!r}'
            raise ValueError(msg)

    # Insertion order is the whole basis of Table 1 reproducing: HNSW's graph, and therefore
    # `acc` and `deg`, depend on the order rows are inserted in. Every transform here is
    # row-wise, so row i must still be row i. A transform that dropped or reordered rows would
    # otherwise show up only as numbers that quietly refuse to reproduce.
    if database.shape[0] != X.shape[0]:
        msg = f'normalization {normalization.value} changed the row count: {X.shape[0]} -> {database.shape[0]}'
        raise ValueError(msg)
    if queries.shape[0] != Q.shape[0]:
        msg = f'normalization {normalization.value} changed the query count: {Q.shape[0]} -> {queries.shape[0]}'
        raise ValueError(msg)

    return database, queries
