"""Tests for the query-agnostic RoarGraph learn table.

`drop_self_hits` is small but load-bearing: a self-edge left in the bipartite table would
give RoarGraph a free correct answer for every database point, and the paper's whole
comparison rests on the table containing no such gift.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgann.indexes.roargraph import drop_self_hits


class TestDropSelfHits:
    def test_removes_each_rows_own_id(self):
        neighbours = np.array([[0, 5, 3], [1, 2, 9], [2, 7, 8]], dtype=np.int64)
        table = drop_self_hits(neighbours, k=2)
        np.testing.assert_array_equal(table, [[5, 3], [2, 9], [7, 8]])

    def test_removes_the_self_id_wherever_it_appears_not_just_first(self):
        """Exact ties can push a row's own id off the front."""
        neighbours = np.array([[4, 0, 3], [1, 6, 2]], dtype=np.int64)
        table = drop_self_hits(neighbours, k=2)
        np.testing.assert_array_equal(table, [[4, 3], [6, 2]])

    def test_row_without_a_self_hit_drops_its_last_column(self):
        neighbours = np.array([[7, 8, 9], [4, 5, 6]], dtype=np.int64)
        table = drop_self_hits(neighbours, k=2)
        np.testing.assert_array_equal(table, [[7, 8], [4, 5]])

    def test_output_is_always_rectangular_whatever_the_mix(self):
        neighbours = np.array([[0, 8, 9], [7, 8, 9], [2, 5, 6]], dtype=np.int64)
        assert drop_self_hits(neighbours, k=2).shape == (3, 2)

    def test_no_row_contains_its_own_id(self):
        """Faiss returns distinct ids per row, so the fixture does too."""
        rng = np.random.default_rng(0)
        n, k = 40, 5
        # Row i retrieves itself first, then k distinct others — exactly what an exact
        # inner-product search returns.
        neighbours = np.stack(
            [np.concatenate([[i], rng.permutation(np.delete(np.arange(n), i))[:k]]) for i in range(n)],
        )
        table = drop_self_hits(neighbours, k=k)
        assert not any(row_index in set(table[row_index].tolist()) for row_index in range(n))

    def test_stays_rectangular_even_if_an_id_repeats_within_a_row(self):
        """Cannot happen with faiss, but ragged output would be a confusing way to find out."""
        neighbours = np.array([[0, 0, 7]], dtype=np.int64)
        assert drop_self_hits(neighbours, k=2).shape == (1, 2)

    def test_output_is_uint32_as_roargraph_requires(self):
        neighbours = np.array([[0, 1, 2]], dtype=np.int64)
        assert drop_self_hits(neighbours, k=2).dtype == np.uint32

    def test_rejects_a_table_of_the_wrong_width(self):
        neighbours = np.array([[0, 1, 2, 3]], dtype=np.int64)
        with pytest.raises(ValueError, match='expected 3 columns'):
            drop_self_hits(neighbours, k=2)
