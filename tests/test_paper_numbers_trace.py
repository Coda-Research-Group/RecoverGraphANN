"""Every number printed in Table 1 must resolve to a row in the committed run logs.

The paper's `acc` and `deg` columns were originally read out of `results/logs/` by hand. This
turns that reading into something a machine rechecks, so a later change to what `acc` or `deg`
mean — or to which log the table came from — fails here rather than silently producing a
table that no longer matches the paper.

It also fixes the column mapping in one place. `acc` is `selected_neighbors_avg_*`, a count
rather than a ratio, and `deg` is `degree_avg_*` over all layers rather than
`degree_lowest_layer_avg_*`; the level-0 figure reproduces none of the printed cells, and the
test below asserts that too, so the distinction cannot quietly be lost.
"""

from __future__ import annotations

import csv
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
from paper_reference import DOUBLE_ROUNDED, TABLE_1

LOGS = Path(__file__).resolve().parent.parent / 'results' / 'logs'

#: This file's rows all say use_asymmetric_transformation=True but are the L2-normalization
#: ablation — the transform was edited in place rather than selected by a flag. Any query
#: filtering on that column has to exclude it. See results/logs/README.md.
MISLABELLED_L2_RUN = '2026-06-03_10-26-36-index-metrics.csv'


def _printed(value: float) -> float:
    """Round to one decimal, half away from zero — the way a table is rounded.

    Not `round()`, which is half-to-even: `round(1.15, 1)` is 1.1, and a table would print
    1.2. Not a tolerance either, which would put several cells on the boundary and let float
    error decide the verdict.
    """
    return float(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def _double_rounded(value: float) -> float:
    """To two decimals, then to one — what produces the two cells in DOUBLE_ROUNDED."""
    two = Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return float(two.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def _expected(dataset: str, m: int, transformed: bool, column: str, printed: float) -> float:
    """What a correctly-rounded table would print for this cell."""
    known = DOUBLE_ROUNDED.get((dataset, m, transformed, column))
    return known[1] if known else printed


def _rows() -> list[dict[str, str]]:
    rows = []
    for path in sorted(LOGS.glob('*-index-metrics.csv')):
        if path.name == MISLABELLED_L2_RUN:
            continue
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                row['_source'] = path.name
                rows.append(row)
    return rows


@pytest.fixture(scope='module')
def rows() -> list[dict[str, str]]:
    if not LOGS.is_dir():
        pytest.skip(f'{LOGS} is not present')
    found = _rows()
    if not found:
        pytest.skip(f'no run logs under {LOGS}')
    return found


def _table1_rows(rows: list[dict[str, str]], dataset: str, m: int, transformed: bool) -> list[dict[str, str]]:
    """Rows matching one cell of Table 1: hnswlib, efConstruction 500, efSearch 1000, no learn insertion."""
    return [
        row
        for row in rows
        if row.get('dataset_name') == dataset
        and row.get('algorithm') == 'hnsw-hnswlib'
        and row.get('use_asymmetric_transformation') == str(transformed)
        and row.get('use_hnsw_pruning_rule_on_X') == 'True'
        and row.get('hnsw_m') == str(m)
        and row.get('hnsw_ef_construction') == '500'
        and row.get('hnsw_ef_search') == '1000'
        and (row.get('hnsw_learn_insert_order') or 'none') == 'none'
    ]


@pytest.mark.parametrize(('key', 'printed'), list(TABLE_1.items()), ids=str)
def test_table1_cell_traces_to_a_logged_run(rows, key, printed):
    dataset, m, transformed = key
    acc, deg, recall_pct = printed
    suffix = 'X_asymmetric' if transformed else 'X'

    candidates = _table1_rows(rows, dataset, m, transformed)
    assert candidates, f'no logged run for {dataset} M={m} transformed={transformed}'

    expected_acc = _expected(dataset, m, transformed, 'acc', acc)
    expected_deg = _expected(dataset, m, transformed, 'deg', deg)

    matches = [
        row
        for row in candidates
        if _printed(float(row[f'selected_neighbors_avg_{suffix}'])) == expected_acc
        and _printed(float(row[f'degree_avg_{suffix}'])) == expected_deg
        and _printed(float(row[f'recall_{suffix}_avg_at_k']) * 100) == recall_pct
    ]
    assert matches, (
        f'Table 1 prints acc={acc} deg={deg} recall={recall_pct}% for {dataset} M={m} '
        f'transformed={transformed}, but no logged row matches. Closest logged values: '
        + '; '.join(
            f'{row["_source"]}: acc={float(row[f"selected_neighbors_avg_{suffix}"]):.2f} '
            f'deg={float(row[f"degree_avg_{suffix}"]):.2f} '
            f'recall={float(row[f"recall_{suffix}_avg_at_k"]) * 100:.2f}%'
            for row in candidates[:3]
        )
    )


@pytest.mark.parametrize(('cell', 'values'), list(DOUBLE_ROUNDED.items()), ids=str)
def test_the_known_double_rounded_cells_are_still_double_rounded(rows, cell, values):
    """Guards the exception itself: if the table is corrected, this says to drop the entry."""
    dataset, m, transformed, column = cell
    printed, correct = values
    suffix = 'X_asymmetric' if transformed else 'X'
    field = 'selected_neighbors_avg' if column == 'acc' else 'degree_avg'

    candidates = _table1_rows(rows, dataset, m, transformed)
    assert candidates, f'no logged run for {dataset} M={m} transformed={transformed}'
    logged = float(candidates[0][f'{field}_{suffix}'])

    assert _printed(logged) == correct, (
        f'{dataset} M={m} {column}: logged {logged} now rounds to {_printed(logged)}, not the recorded {correct}'
    )
    assert _double_rounded(logged) == printed, (
        f'{dataset} M={m} {column}: logged {logged} no longer double-rounds to the printed '
        f'{printed}. If Table 1 was corrected to {correct}, delete this entry from '
        f'DOUBLE_ROUNDED.'
    )


def test_deg_is_all_layers_not_level_zero(rows):
    """The level-0 degree reproduces no printed cell, which is why `deg` is the all-layer one."""
    reproduced_by_level0 = []
    for (dataset, m, transformed), (_, deg, _) in TABLE_1.items():
        suffix = 'X_asymmetric' if transformed else 'X'
        for row in _table1_rows(rows, dataset, m, transformed):
            if _printed(float(row[f'degree_lowest_layer_avg_{suffix}'])) == deg:
                reproduced_by_level0.append((dataset, m, transformed))
                break

    assert not reproduced_by_level0, (
        'degree_lowest_layer_avg reproduces printed `deg` values for '
        f'{reproduced_by_level0} — the two columns have converged, so the mapping in '
        'rgann.metrics needs rechecking'
    )


def test_the_candidate_pool_is_effectively_efconstruction(rows):
    """`acc` is only interpretable because the pool it is drawn from does not move."""
    pools = []
    for dataset, m, transformed in TABLE_1:
        suffix = 'X_asymmetric' if transformed else 'X'
        pools += [float(row[f'top_candidates_avg_{suffix}']) for row in _table1_rows(rows, dataset, m, transformed)]

    assert pools, 'no rows found to check the candidate pool size'
    # M=4 on llama dips to ~490; everything else sits within a couple of candidates of 500.
    assert min(pools) > 480.0, f'candidate pool fell to {min(pools):.1f}, well below efConstruction=500'
    assert max(pools) <= 500.0, f'candidate pool exceeded efConstruction=500 at {max(pools):.1f}'
