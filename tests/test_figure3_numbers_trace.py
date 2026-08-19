"""Every number §Results quotes must resolve to a row in the CSV the paper was written from.

Companion to `test_paper_numbers_trace.py`, which does the same for Table 1. Together they
mean the paper's numbers cannot drift away from the artifact silently: they either trace, or
the suite says which one stopped tracing.

`results/paper/efSearch.csv` is the hand-assembled file the camera-ready used, reduced to the
query-agnostic rows this artifact evaluates.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from paper_reference import (
    GRAPH_INDEXES,
    RAW_BAND,
    RAW_HNSW_TOP,
    ROARGRAPH_QUERY_AGNOSTIC,
    ROARGRAPH_TRANSFORMED,
    TABLE_1,
    TOLERANCE,
)

PAPER = Path(__file__).resolve().parent.parent / 'results' / 'paper'
EF_SEARCH_CSV = PAPER / 'efSearch.csv'
DIFFERENT_M_CSV = PAPER / 'different-m.csv'

TOP_EFFORT = '1000'


def _percent(value: str) -> float:
    return float(value.strip().rstrip('%'))


@pytest.fixture(scope='module')
def ef_rows() -> list[dict[str, str]]:
    if not EF_SEARCH_CSV.exists():
        pytest.skip(f'{EF_SEARCH_CSV} is not present')
    with EF_SEARCH_CSV.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _at_top_effort(rows, dataset: str, algorithm: str, transformed: bool) -> list[float]:
    return [
        _percent(row['recall'])
        for row in rows
        if row['dataset_name'] == dataset
        and row['algorithm'] == algorithm
        and row['use_asymmetric_transformation'] == ('TRUE' if transformed else 'FALSE')
        and row['efSearch'] == TOP_EFFORT
    ]


@pytest.mark.parametrize(('dataset', 'expected'), RAW_HNSW_TOP.items())
def test_raw_hnsw_at_top_effort(ef_rows, dataset, expected):
    found = _at_top_effort(ef_rows, dataset, 'hnsw-hnswlib', transformed=False)
    assert found, f'no raw HNSW row for {dataset} at efSearch={TOP_EFFORT}'
    assert abs(found[0] - expected) <= TOLERANCE, f'§Results quotes {expected}%, CSV has {found[0]}%'


@pytest.mark.parametrize(('dataset', 'band'), RAW_BAND.items())
def test_the_three_query_agnostic_indexes_occupy_the_quoted_raw_band(ef_rows, dataset, band):
    low, high = band
    recalls = {
        algorithm: _at_top_effort(ef_rows, dataset, algorithm, transformed=False)[0] for algorithm in GRAPH_INDEXES
    }
    assert abs(min(recalls.values()) - low) <= TOLERANCE, f'band starts at {min(recalls.values())}%, quoted {low}%'
    assert abs(max(recalls.values()) - high) <= TOLERANCE, f'band ends at {max(recalls.values())}%, quoted {high}%'


@pytest.mark.parametrize(('dataset', 'expected'), ROARGRAPH_QUERY_AGNOSTIC.items())
def test_roargraph_without_a_query_sample(ef_rows, dataset, expected):
    found = _at_top_effort(ef_rows, dataset, 'roargraph', transformed=False)
    assert found, f'no query-agnostic RoarGraph row for {dataset}'
    assert abs(found[0] - expected) <= TOLERANCE, f'§Results quotes {expected}%, CSV has {found[0]}%'


@pytest.mark.parametrize(('dataset', 'expected'), ROARGRAPH_TRANSFORMED.items())
def test_transformed_roargraph_still_falls_short(ef_rows, dataset, expected):
    """¶5's point: the transformation helps RoarGraph, but not to where HNSW gets."""
    found = _at_top_effort(ef_rows, dataset, 'roargraph', transformed=True)
    assert found, f'no transformed RoarGraph row for {dataset}'
    assert abs(found[0] - expected) <= TOLERANCE, f'§Results quotes {expected}%, CSV has {found[0]}%'


def test_different_m_agrees_with_table_1(ef_rows):  # noqa: ARG001
    """The four M values Table 1 prints must match different-m.csv, which supplied that column."""
    if not DIFFERENT_M_CSV.exists():
        pytest.skip(f'{DIFFERENT_M_CSV} is not present')

    with DIFFERENT_M_CSV.open(newline='') as handle:
        rows = list(csv.DictReader(handle))

    for (dataset, m, transformed), (_, _, recall_pct) in TABLE_1.items():
        match = [
            _percent(row['recall'])
            for row in rows
            if row['dataset_name'] == dataset
            and row['M'] == str(m)
            and row['use_asymmetric_transformation'] == ('TRUE' if transformed else 'FALSE')
        ]
        assert match, f'different-m.csv has no row for {dataset} M={m} transformed={transformed}'
        assert abs(match[0] - recall_pct) <= TOLERANCE, (
            f'{dataset} M={m} transformed={transformed}: Table 1 prints {recall_pct}%, different-m.csv has {match[0]}%'
        )
