"""The order the RoarGraph learn table is fed in, pinned.

RoarGraph consumes the learn table row by row while building its bipartite graph, so the row
order is part of the configuration, not presentation. Two different "seeded permutations of n"
give two different indexes.

This is pinned to literal sequences rather than to an implementation, because the failure it
guards against is exactly an implementation that looks equivalent and is not:
`Generator.permutation(n)` and `Generator.choice(n, size=n, replace=False)` consume the bit
stream differently and disagree from the same seed. Comparing one against the other would only
restate the mistake.

Measured cost of getting it wrong: RoarGraph's raw-space curve lands 1.7 points low on
`yi-128-ip` and 4.4 low on `llama-128-ip`, with no other symptom.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgann.indexes.roargraph import DEFAULT_LEARN_SEED, sampled_row_order

#: `np.random.default_rng(42).choice(n, size=n, replace=False)`, written out.
EXPECTED = {
    10: [2, 9, 1, 6, 3, 8, 5, 7, 4, 0],
    20: [10, 7, 2, 17, 8, 16, 0, 1, 15, 4, 12, 13, 5, 18, 11, 14, 3, 6, 19, 9],
}


@pytest.mark.parametrize(('n', 'expected'), EXPECTED.items())
def test_the_order_is_the_one_the_paper_was_produced_with(n, expected):
    assert sampled_row_order(n, DEFAULT_LEARN_SEED).tolist() == expected


def test_it_is_not_generator_permutation():
    """The specific wrong answer this exists to rule out."""
    n = 20
    wrong = np.random.default_rng(DEFAULT_LEARN_SEED).permutation(n).tolist()
    assert sampled_row_order(n, DEFAULT_LEARN_SEED).tolist() != wrong


def test_it_is_a_permutation():
    n = 50
    order = sampled_row_order(n, DEFAULT_LEARN_SEED)
    assert sorted(order.tolist()) == list(range(n)), 'rows were dropped or duplicated'


def test_different_seeds_give_different_orders():
    n = 50
    assert sampled_row_order(n, 42).tolist() != sampled_row_order(n, 43).tolist()


def test_it_is_reproducible():
    n = 50
    assert sampled_row_order(n, 7).tolist() == sampled_row_order(n, 7).tolist()


def test_the_default_seed_is_the_paper_s():
    assert DEFAULT_LEARN_SEED == 42
