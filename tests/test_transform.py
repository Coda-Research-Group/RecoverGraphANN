"""Tests for the database-side transformations.

The Bachrach lift is the paper's method, so its two load-bearing properties get their own
tests: transformed database rows land on the unit sphere in d+1, and the inner-product
ranking is preserved exactly up to the global factor 1/c.
"""

from __future__ import annotations

import numpy as np
import pytest

import rgann.transform
from rgann.transform import Normalization, apply_normalization, bachrach_transform, l2_normalize


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Heterogeneous row norms, like the attention workloads the paper studies."""
    X = (rng.normal(size=(64, 8)) * rng.uniform(0.1, 10.0, size=(64, 1))).astype(np.float32)
    Q = rng.normal(size=(16, 8)).astype(np.float32)
    return X, Q


class TestBachrach:
    def test_adds_exactly_one_dimension(self, data):
        X, Q = data
        Xt, Qt = bachrach_transform(X, Q)
        assert Xt.shape == (X.shape[0], X.shape[1] + 1)
        assert Qt.shape == (Q.shape[0], Q.shape[1] + 1)

    def test_database_rows_land_on_the_unit_sphere(self, data):
        X, _ = data
        Xt, _ = bachrach_transform(X, X)
        np.testing.assert_allclose(np.linalg.norm(Xt, axis=1), 1.0, atol=1e-5)

    def test_queries_are_zero_padded_and_never_scaled(self, data):
        X, Q = data
        _, Qt = bachrach_transform(X, Q)
        np.testing.assert_array_equal(Qt[:, :-1], Q)
        np.testing.assert_array_equal(Qt[:, -1], 0.0)

    def test_inner_product_is_the_original_divided_by_the_max_row_norm(self, data):
        X, Q = data
        Xt, Qt = bachrach_transform(X, Q)
        c = float(np.max(np.linalg.norm(X, axis=1)))
        np.testing.assert_allclose(Xt @ Qt.T, (X @ Q.T) / c, rtol=1e-4, atol=1e-5)

    def test_ranking_is_preserved_exactly(self, data):
        """The whole point: the index sees new geometry but answers the same question."""
        X, Q = data
        Xt, Qt = bachrach_transform(X, Q)
        np.testing.assert_array_equal(
            np.argsort(-(X @ Q.T), axis=0, kind='stable'),
            np.argsort(-(Xt @ Qt.T), axis=0, kind='stable'),
        )

    def test_all_zero_database_is_not_a_division_by_zero(self):
        X = np.zeros((4, 3), dtype=np.float32)
        Q = np.ones((2, 3), dtype=np.float32)
        Xt, _ = bachrach_transform(X, Q)
        assert np.isfinite(Xt).all()
        np.testing.assert_allclose(Xt[:, -1], 1.0, atol=1e-6)

    def test_zero_row_among_nonzero_rows_keeps_unit_norm(self, data):
        X, Q = data
        X = X.copy()
        X[0] = 0.0
        Xt, _ = bachrach_transform(X, Q)
        np.testing.assert_allclose(np.linalg.norm(Xt, axis=1), 1.0, atol=1e-5)

    def test_output_is_float32(self, data):
        X, Q = data
        Xt, Qt = bachrach_transform(X, Q)
        assert Xt.dtype == np.float32
        assert Qt.dtype == np.float32


class TestL2:
    """The §Results ablation: equalizes norms like Bachrach, but distorts the IP ranking."""

    def test_rows_have_unit_norm_without_adding_a_dimension(self, data):
        X, _ = data
        Xn = l2_normalize(X)
        assert Xn.shape == X.shape
        np.testing.assert_allclose(np.linalg.norm(Xn, axis=1), 1.0, atol=1e-5)

    def test_zero_rows_stay_zero_rather_than_nan(self):
        X = np.zeros((3, 4), dtype=np.float32)
        Xn = l2_normalize(X)
        assert np.isfinite(Xn).all()
        np.testing.assert_array_equal(Xn, 0.0)

    def test_distorts_the_inner_product_ranking(self, data):
        """If this ever stops holding, the ablation in the paper has lost its meaning."""
        X, Q = data
        raw = np.argsort(-(X @ Q.T), axis=0, kind='stable')
        normalized = np.argsort(-(l2_normalize(X) @ Q.T), axis=0, kind='stable')
        assert not np.array_equal(raw, normalized)


class TestApplyNormalization:
    def test_none_returns_the_inputs_untouched(self, data):
        X, Q = data
        Xo, Qo = apply_normalization(X, Q, Normalization.NONE)
        np.testing.assert_array_equal(Xo, X)
        np.testing.assert_array_equal(Qo, Q)

    def test_bachrach_matches_the_direct_call(self, data):
        X, Q = data
        np.testing.assert_array_equal(
            apply_normalization(X, Q, Normalization.BACHRACH)[0],
            bachrach_transform(X, Q)[0],
        )

    def test_l2_normalizes_the_database_and_leaves_queries_alone(self, data):
        X, Q = data
        Xo, Qo = apply_normalization(X, Q, Normalization.L2)
        np.testing.assert_array_equal(Xo, l2_normalize(X))
        np.testing.assert_array_equal(Qo, Q)

    def test_accepts_the_plain_string_used_on_the_command_line(self, data):
        X, Q = data
        np.testing.assert_array_equal(
            apply_normalization(X, Q, 'bachrach')[0],
            apply_normalization(X, Q, Normalization.BACHRACH)[0],
        )

    def test_rejects_an_unknown_mode(self, data):
        X, Q = data
        with pytest.raises(ValueError, match='unknown normalization'):
            apply_normalization(X, Q, 'whitening')


class TestParity:
    def test_matches_the_original_harness_implementation(self, data):
        """Byte-for-byte parity with main.py's build_asymmetric_mips_transformation.

        Every number in the paper came out of that function; if this drifts, the artifact
        stops reproducing the paper.
        """
        X, Q = data

        # Verbatim from the original harness, main.py:1342-1366.
        max_norm = float(np.max(np.linalg.norm(X, axis=1)))
        X_scaled = X / max_norm if max_norm > 0.0 else np.zeros_like(X)
        norms_x = np.sum(X_scaled**2, axis=1)
        extra_col_x = np.sqrt(np.maximum(1.0 - norms_x, 0.0))
        expected_X = np.column_stack([X_scaled, extra_col_x])
        expected_Q = np.column_stack([Q, np.zeros_like(Q[:, :1])])

        Xt, Qt = bachrach_transform(X, Q)
        np.testing.assert_array_equal(Xt, expected_X.astype(np.float32))
        np.testing.assert_array_equal(Qt, expected_Q.astype(np.float32))


class TestRowOrderIsPreserved:
    """Row i must stay row i through every normalization.

    Table 1's whole substance — `acc` and `deg` — comes out of HNSW's graph, which depends on
    the order rows are inserted in. Every transform here is row-wise, so a reordering would not
    raise anything; it would just produce numbers that quietly decline to reproduce. The README
    promises this invariant, so it gets a test rather than a comment.
    """

    @staticmethod
    def _distinct_rows(n: int = 12, d: int = 5) -> np.ndarray:
        # Rows with strictly increasing norms, so any permutation is detectable from the output
        # alone without comparing against the input positionally.
        rng = np.random.default_rng(0)
        base = rng.normal(size=(n, d)).astype(np.float32)
        scales = np.arange(1, n + 1, dtype=np.float32)[:, None]
        return np.ascontiguousarray(base / np.linalg.norm(base, axis=1, keepdims=True) * scales)

    @pytest.mark.parametrize('mode', [Normalization.NONE, Normalization.BACHRACH, Normalization.L2])
    def test_the_row_count_survives(self, mode):
        X = self._distinct_rows()
        Q = self._distinct_rows(n=4)
        database, queries = apply_normalization(X, Q, mode)
        assert database.shape[0] == X.shape[0]
        assert queries.shape[0] == Q.shape[0]

    def test_bachrach_keeps_each_row_in_place(self):
        """The lifted row i must be row i scaled, not some other row's."""
        X = self._distinct_rows()
        database, _ = apply_normalization(X, X[:4], Normalization.BACHRACH)
        scale = np.linalg.norm(X, axis=1).max()
        np.testing.assert_allclose(database[:, :-1], X / scale, rtol=1e-6, atol=1e-6)

    def test_l2_keeps_each_row_in_place(self):
        X = self._distinct_rows()
        database, _ = apply_normalization(X, X[:4], Normalization.L2)
        expected = X / np.linalg.norm(X, axis=1, keepdims=True)
        np.testing.assert_allclose(database, expected, rtol=1e-6, atol=1e-6)

    def test_none_is_the_identity(self):
        X = self._distinct_rows()
        database, queries = apply_normalization(X, X[:4], Normalization.NONE)
        np.testing.assert_array_equal(database, X)
        np.testing.assert_array_equal(queries, X[:4])

    def test_a_transform_that_drops_rows_is_refused(self, monkeypatch):
        """The guard, not the arithmetic: prove it fires when the invariant breaks."""
        monkeypatch.setattr(rgann.transform, 'l2_normalize', lambda X: X[:-1])
        with pytest.raises(ValueError, match='changed the row count'):
            apply_normalization(self._distinct_rows(), self._distinct_rows(n=4), Normalization.L2)
