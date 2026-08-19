"""Loading the VIBE attention datasets.

The paper uses two of them, both inner-product with ``d = 128``:

===================  =======  ===========================================================
dataset              n        source model
===================  =======  ===========================================================
``yi-128-ip``        187 843  Yi-6B-200K
``llama-128-ip``     256 921  Llama-3-8B-Instruct-262k
===================  =======  ===========================================================

VIBE's HDF5 layout, and what each field is called in the paper:

==================  ==============  ======================================================
HDF5 key            attribute       role
==================  ==============  ======================================================
``train``           ``X``           the indexed database
``test``            ``Q``           the 1 000 evaluation queries
``neighbors``       ``GT``          exact top-k ground truth, for recall
``learn``           ``Q_learn``     a sample of the query distribution
``learn_neighbors`` ``Q_neighbors`` each learn query's k-NN row ids into ``X``
==================  ==============  ======================================================

`learn` and `learn_neighbors` are the query sample. Nothing in this artifact reads them —
it evaluates the query-agnostic setting only. The paper's method
never touches them, and neither does the query-agnostic RoarGraph adaptation, which
substitutes the database for the query side (see `rgann.indexes.roargraph`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ['DATASETS', 'Dataset', 'file_sha256', 'load_dataset']

DATASETS = ('yi-128-ip', 'llama-128-ip')

DEFAULT_DATA_DIR = Path('data')

#: `--quick` subsamples the database to this many rows. Enough for every backend to build a
#: real graph and for the transformation's effect to show, small enough to finish over lunch.
QUICK_DATABASE_ROWS = 10_000


@dataclass(frozen=True)
class Dataset:
    """One VIBE dataset, as loaded. Arrays are C-contiguous float32 (ids are uint32)."""

    name: str
    X: np.ndarray
    Q: np.ndarray
    GT: np.ndarray
    Q_learn: np.ndarray
    Q_neighbors: np.ndarray
    subsampled: bool = False

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    def describe(self) -> str:
        suffix = f' (subsampled from the full dataset to {self.n} rows)' if self.subsampled else ''
        return f'{self.name}: n={self.n} d={self.d} queries={self.Q.shape[0]}{suffix}'


def _contiguous_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def load_dataset(
    name: str,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    quick: bool = False,
    quick_rows: int = QUICK_DATABASE_ROWS,
) -> Dataset:
    """Load ``{data_dir}/{name}.hdf5``.

    Args:
        name: Dataset name, e.g. ``yi-128-ip``.
        data_dir: Directory holding the HDF5 files (see ``scripts/download_data.sh``).
        quick: Take the first ``quick_rows`` database rows and recompute exact ground truth
            for them. Off by default; every published number uses the full database.
        quick_rows: Database rows to keep when ``quick`` is set.

    Raises:
        FileNotFoundError: If the HDF5 file is missing, with the download command to fix it.

    """
    # Deferred like the backend imports: reading a dataset needs h5py, but merely importing an
    # experiment module should not, so the checks that never open an HDF5 file can run without
    # it. Same reason `faiss` is imported inside the functions that use it.
    import h5py  # noqa: PLC0415

    path = Path(data_dir) / f'{name}.hdf5'
    if not path.exists():
        msg = f'{path} not found. Fetch the VIBE datasets first:\n    bash scripts/download_data.sh {name}'
        raise FileNotFoundError(msg)

    with h5py.File(path, 'r') as handle:
        X = _contiguous_float32(handle['train'][:])
        Q = _contiguous_float32(handle['test'][:])
        GT = np.ascontiguousarray(handle['neighbors'][:], dtype=np.uint32)
        Q_learn = _contiguous_float32(handle['learn'][:])
        Q_neighbors = np.ascontiguousarray(handle['learn_neighbors'][:], dtype=np.uint32)

    if not quick:
        return Dataset(name=name, X=X, Q=Q, GT=GT, Q_learn=Q_learn, Q_neighbors=Q_neighbors)

    return _subsample(name, X, Q, Q_learn, quick_rows)


def _subsample(name: str, X: np.ndarray, Q: np.ndarray, Q_learn: np.ndarray, rows: int) -> Dataset:
    """Take a prefix of the database and recompute exact ground truth against it.

    The shipped ``neighbors`` are ids into the *full* database, so they are meaningless on a
    subsample — recomputing is the only way `--quick` recall means anything. Taking a prefix
    rather than a random sample keeps insertion order identical to the full run, which is
    what makes the degree and acceptance statistics comparable.
    """
    # Imported lazily: the transformation and the plotting entry points do not need faiss,
    # and importing it costs ~1s and pulls in OpenMP.
    import faiss  # noqa: PLC0415

    rows = min(rows, X.shape[0])
    X_small = np.ascontiguousarray(X[:rows])

    flat = faiss.IndexFlatIP(X_small.shape[1])
    flat.add(X_small)
    _, GT_small = flat.search(Q, min(100, rows))

    # Kept for completeness of the Dataset record; nothing in this artifact reads them.
    keep = min(Q_learn.shape[0], rows)
    learn_small = np.ascontiguousarray(Q_learn[:keep])
    flat_learn = faiss.IndexFlatIP(X_small.shape[1])
    flat_learn.add(X_small)
    _, learn_neighbors_small = flat_learn.search(learn_small, min(100, rows))

    return Dataset(
        name=name,
        X=X_small,
        Q=Q,
        GT=np.ascontiguousarray(GT_small, dtype=np.uint32),
        Q_learn=learn_small,
        Q_neighbors=np.ascontiguousarray(learn_neighbors_small, dtype=np.uint32),
        subsampled=True,
    )


def file_sha256(path: Path | str, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed. Used to pin the downloaded HDF5 files."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
