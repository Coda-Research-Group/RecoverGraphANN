"""The README's numbers must be the paper's numbers.

The README is the front door: it opens with a headline recall figure and a summary table, and
those are the numbers most readers will ever see. They were transcribed by hand, so they get
checked against `paper_reference.py` the same way Table 1 and Figure 3 are checked against the
run logs.

This caught a real error on its first run — the intro claimed the raw-space indexes stay
"below 45%", which `yi-128-ip` at M=96 exceeds at 45.4%.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from paper_reference import TABLE_1

README = Path(__file__).resolve().parent.parent / 'README.md'

#: The four cells the "Results at a glance" table shows, at the paper's headline settings.
GLANCE_M = 48

#: `(dataset, transformed)` -> the row label used in the README table.
ROW_LABEL = {False: 'raw', True: 'transformed'}


@pytest.fixture(scope='module')
def readme() -> str:
    return README.read_text()


def _glance_rows(readme: str) -> list[list[str]]:
    section = readme.split('## Results at a glance', 1)[1].split('##', 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip().strip('*').strip('`') for c in line.strip().strip('|').split('|')]
        if len(cells) == 5 and any(c.endswith('%') for c in cells):
            rows.append(cells)
    return rows


def test_the_glance_table_has_a_row_per_dataset_and_space(readme):
    assert len(_glance_rows(readme)) == len(TABLE_1) // 4, 'expected four rows in Results at a glance'


@pytest.mark.parametrize('dataset', ['yi-128-ip', 'llama-128-ip'])
@pytest.mark.parametrize('transformed', [False, True])
def test_each_glance_cell_matches_the_paper(readme, dataset, transformed):
    acc, deg, recall = TABLE_1[(dataset, GLANCE_M, transformed)]
    label = ROW_LABEL[transformed]

    rows = _glance_rows(readme)
    # The dataset name appears on the raw row; the transformed row continues it.
    index = next(i for i, row in enumerate(rows) if row[0] == dataset)
    row = rows[index + (1 if transformed else 0)]

    assert row[1] == label, f'expected the {label} row, got {row[1]!r}'
    assert float(row[2]) == acc, f'{dataset} {label} acc: README {row[2]}, paper {acc}'
    assert float(row[3]) == deg, f'{dataset} {label} deg: README {row[3]}, paper {deg}'
    assert float(row[4].rstrip('%')) == recall, f'{dataset} {label} recall: README {row[4]}, paper {recall}%'


def test_the_headline_matches_the_paper(readme):
    """The bold line at the top, and the config sentence that pins it down."""
    _, _, raw = TABLE_1[('llama-128-ip', 48, False)]
    _, _, transformed = TABLE_1[('llama-128-ip', 48, True)]

    headline = readme.split('\n\n')[2]
    assert f'{round(raw)}%' in headline, f'headline should quote {round(raw)}%, reads: {headline}'
    assert f'{round(transformed)}%' in headline, f'headline should quote {round(transformed)}%'


def test_the_collapse_claim_is_true_of_every_raw_cell(readme):
    """ "below N%" has to hold for the worst raw-space cell, not the typical one."""
    match = re.search(r'stay below (\d+)% recall@10', readme)
    assert match, 'the intro no longer states a bound on raw-space recall'

    claimed = float(match.group(1))
    worst = max(recall for (_, _, transformed), (_, _, recall) in TABLE_1.items() if not transformed)
    assert worst < claimed, (
        f'the README claims raw-space recall stays below {claimed}%, but the highest raw cell in Table 1 is {worst}%'
    )


# --- Numbers that must come from the artifact's own committed data ---------------------------
#
# Everything above checks the README against the *paper*. These check it against what the code
# currently produces, so a rerun that moves a number cannot leave the front page describing the
# previous one. Added after an audit found the intro claiming the transformation raises
# acceptance "by an order of magnitude" when the measured range is 3.3x to 19.6x.

import csv  # noqa: E402

ROOT = README.parent


def _rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline='') as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope='module')
def table1_rows() -> list[dict[str, str]]:
    return _rows('results/table1_degree.csv')


@pytest.fixture(scope='module')
def fig3_rows() -> list[dict[str, str]]:
    return _rows('results/fig3_recall_qps.csv')


def _cell(rows, dataset: str, m: int, normalization: str, column: str) -> float:
    for row in rows:
        if row['dataset_name'] == dataset and row['m'] == str(m) and row['normalization'] == normalization:
            return float(row[column])
    msg = f'no row for {dataset} M={m} {normalization}'
    raise AssertionError(msg)


def _top_recall(rows, dataset: str, algorithm: str, normalization: str) -> float:
    hits = [
        float(row['recall_avg'])
        for row in rows
        if row['dataset_name'] == dataset
        and row['algorithm'] == algorithm
        and row['normalization'] == normalization
        and row['search_param_value'] == '1000'
    ]
    assert hits, f'no rows for {dataset}/{algorithm}/{normalization}'
    return max(hits)


GRAPH_BACKENDS = ('hnsw-hnswlib', 'nsg-faiss', 'flatnav')
DATASETS = ('yi-128-ip', 'llama-128-ip')


def test_the_raw_space_ceiling_the_intro_quotes_still_holds(readme, fig3_rows):
    """The intro says the graph indexes "all stay below 50% recall@10" in raw space."""
    assert 'below 50% recall@10' in readme
    worst = max(_top_recall(fig3_rows, d, a, 'none') for d in DATASETS for a in GRAPH_BACKENDS)
    assert worst < 0.50, f'a raw-space index reaches {worst * 100:.2f}%, so "below 50%" is wrong'


@pytest.mark.parametrize(
    ('dataset', 'raw', 'transformed', 'ratio'),
    [('yi-128-ip', 1.5, 13.4, '8.8x'), ('llama-128-ip', 1.1, 22.0, '19.6x')],
)
def test_the_intros_acceptance_figures_are_the_measured_ones(readme, table1_rows, dataset, raw, transformed, ratio):
    """Quoted as counts and a ratio, so both have to survive a rerun."""
    got_raw = _cell(table1_rows, dataset, GLANCE_M, 'none', 'accepted_candidates_avg')
    got_tr = _cell(table1_rows, dataset, GLANCE_M, 'bachrach', 'accepted_candidates_avg')
    assert round(got_raw, 1) == raw, f'{dataset} raw acc is {got_raw:.2f}, README says {raw}'
    assert round(got_tr, 1) == transformed, f'{dataset} transformed acc is {got_tr:.2f}, README says {transformed}'
    assert f'{got_tr / got_raw:.1f}x' == ratio, f'{dataset} ratio is {got_tr / got_raw:.1f}x, README says {ratio}'
    assert ratio in readme


def test_the_figure_3_caption_band_matches_the_plotted_data(readme, fig3_rows):
    """The caption promises a wall "around 40%" and a transformed "93%-97% band"."""
    assert '93%-97% band' in readme
    transformed = [_top_recall(fig3_rows, d, a, 'bachrach') for d in DATASETS for a in GRAPH_BACKENDS]
    assert min(transformed) >= 0.93, f'lowest transformed curve is {min(transformed) * 100:.1f}%'
    assert max(transformed) <= 0.975, f'highest transformed curve is {max(transformed) * 100:.1f}%'


def test_the_figure_1_caption_norms_match_the_committed_stats(readme):
    stats = _rows('figures/norm_distribution_stats.csv')
    means = {(r['dataset'], r['split'].split()[0]): float(r['mean']) for r in stats}
    for dataset, database, queries in (('llama-128-ip', 26.6, 20.8), ('yi-128-ip', 21.8, 17.6)):
        assert round(means[dataset, 'Database'], 1) == database
        assert round(means[dataset, 'Queries'], 1) == queries
        assert f'{database} against {queries}' in readme


def test_the_full_run_duration_is_the_sum_of_the_committed_timings(readme):
    """ "about 3.5 hours" and the per-stage table both come from results/timings.csv."""
    first: dict[str, dict[str, str]] = {}
    for row in _rows('results/timings.csv'):
        first.setdefault(row['artifact'], row)

    total_hours = sum(int(r['wall_seconds']) for r in first.values()) / 3600
    assert abs(total_hours - 3.5) < 0.15, f'timings.csv sums to {total_hours:.2f} h, README says about 3.5'

    for artifact, claimed in (('figure3-recall-qps', 2.02), ('table1-degree', 1.05), ('ablation-l2', 0.47)):
        hours = int(first[artifact]['wall_seconds']) / 3600
        assert abs(hours - claimed) < 0.005, f'{artifact} took {hours:.2f} h, README says {claimed}'
        assert f'| {claimed} h |' in readme


@pytest.mark.parametrize(('dataset', 'n'), [('yi-128-ip', 187843), ('llama-128-ip', 256921)])
def test_the_dataset_table_sizes_match_what_was_actually_loaded(readme, fig3_rows, dataset, n):
    loaded = {int(r['database_rows']) for r in fig3_rows if r['dataset_name'] == dataset}
    assert loaded == {n}, f'{dataset} rows in results: {loaded}, README says {n}'
    assert f'{n:,}'.replace(',', ' ') in readme, f'README should print {n} with thin-space grouping'

    queries = {int(r['num_queries']) for r in fig3_rows if r['dataset_name'] == dataset}
    assert queries == {1000}


def test_the_hardware_note_names_the_cpu_the_results_were_recorded_on(readme, fig3_rows):
    recorded = {r['cpu_model'] for r in fig3_rows}
    assert any('E5-2620' in cpu for cpu in recorded), f'results were recorded on {recorded}'
    assert 'E5-2620' in readme


# --- The status section, which is the part most likely to rot -------------------------------
#
# It previously said `third_party/` had no submodules and the forks were unpublished, and kept
# saying it after all three were wired — while claiming in its own last line that it "cannot go
# stale without someone noticing". `release-check` did not notice, because it checks artifacts,
# not prose. These do.


def test_the_status_section_does_not_claim_unwired_submodules(readme):
    """The exact stale sentence that survived the submodules being wired."""
    gitmodules = (ROOT / '.gitmodules').exists()
    claims_unwired = 'has no submodules' in readme or 'are not published' in readme
    assert not (gitmodules and claims_unwired), (
        '.gitmodules exists, so the README must not still say the submodules are unwired'
    )


def test_the_release_check_verdict_the_readme_quotes_is_the_real_one(readme):
    """The README quotes the tool's verdict, so the tool has to actually give it.

    Deliberately the verdict and not the score: the denominator moves whenever the README
    gains or loses a reference the checker counts, which makes a hardcoded "36/36" fail for
    reasons unrelated to readiness. It did exactly that when two platform rows were removed.
    """
    if 'reports **Ready to tag**' not in readme:
        return

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / 'scripts' / 'check_release_ready.py')],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert 'Ready to tag.' in result.stdout, (
        f'the README says release-check is ready; it reports:\n{result.stdout[-400:]}'
    )


@pytest.mark.parametrize(
    ('path', 'seconds'),
    [('docker', 192), ('docker', 324), ('ubuntu', 263), ('ubuntu', 310), ('macos', 62), ('macos', 102)],
)
def test_the_measured_timings_are_stated_as_measured(readme, path, seconds):  # noqa: ARG001
    """These came from real runs; if one is edited away the table stops being evidence."""
    assert f'{seconds} s' in readme
