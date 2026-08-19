"""`--quick` must not write where the canonical run writes.

`make quick` is the first command the README tells a reviewer to run, and its whole point is
that its numbers are *not* the paper's — a 10 000-row subsample has different geometry. So if
it writes to the canonical paths it replaces the committed results and figures with numbers
that look like the paper's and are not, in files git tracks.

Nothing about that failure is loud: the run succeeds, the CSVs stay well-formed, and the only
signal is a dirty working tree the reviewer has no reason to inspect.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent / 'experiments'


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def table1():
    return _load('table1_degree')


@pytest.fixture(scope='module')
def fig3():
    return _load('fig3_recall_qps')


class TestTable1:
    def test_quick_writes_somewhere_else(self, table1):
        assert table1.resolve_csv(None, quick=True) != table1.DEFAULT_CSV

    def test_the_canonical_run_still_writes_the_committed_path(self, table1):
        assert table1.resolve_csv(None, quick=False) == table1.DEFAULT_CSV

    def test_an_explicit_path_always_wins(self, table1):
        chosen = Path('somewhere/else.csv')
        assert table1.resolve_csv(chosen, quick=True) == chosen
        assert table1.resolve_csv(chosen, quick=False) == chosen


class TestFigure3:
    def test_quick_writes_somewhere_else(self, fig3):
        assert fig3.resolve_csv(None, quick=True) != fig3.DEFAULT_CSV

    def test_quick_plots_somewhere_else(self, fig3):
        """Otherwise the committed Figure 3 is replaced by a subsample plot."""
        assert fig3.resolve_figure_dir(None, quick=True) != fig3.DEFAULT_FIGURE_DIR

    def test_the_canonical_run_still_writes_the_committed_paths(self, fig3):
        assert fig3.resolve_csv(None, quick=False) == fig3.DEFAULT_CSV
        assert fig3.resolve_figure_dir(None, quick=False) == fig3.DEFAULT_FIGURE_DIR

    def test_an_explicit_path_always_wins(self, fig3):
        assert fig3.resolve_csv(Path('a.csv'), quick=True) == Path('a.csv')
        assert fig3.resolve_figure_dir(Path('figs'), quick=True) == Path('figs')


def test_no_quick_path_collides_with_a_committed_one(table1, fig3):
    """The two quick outputs must also not collide with each other."""
    quick_paths = {
        table1.resolve_csv(None, quick=True),
        fig3.resolve_csv(None, quick=True),
    }
    committed = {table1.DEFAULT_CSV, fig3.DEFAULT_CSV}
    assert len(quick_paths) == 2
    assert not (quick_paths & committed)


class TestQuickToleratesAMissingBackend:
    """`--quick` proves the pipeline works *here*; the canonical run proves the paper.

    RoarGraph is Linux/x86-64 only, so on macOS `make quick` has to skip it rather than die —
    it did die, with ModuleNotFoundError, which made the documented macOS path unusable. The
    canonical run must NOT skip: a Figure 3 quietly missing a curve is worse than a failure.
    """

    def test_quick_drops_a_backend_that_is_not_installed(self, fig3, monkeypatch, capsys):
        monkeypatch.setattr(fig3, 'backend_installed', lambda backend: backend != 'roargraph')
        assert fig3.usable_backends(('hnsw-hnswlib', 'roargraph'), quick=True) == ('hnsw-hnswlib',)
        assert 'skipping roargraph' in capsys.readouterr().out

    def test_the_canonical_run_refuses_instead(self, fig3, monkeypatch):
        monkeypatch.setattr(fig3, 'backend_installed', lambda backend: backend != 'roargraph')
        with pytest.raises(SystemExit, match='not installed: roargraph'):
            fig3.usable_backends(('hnsw-hnswlib', 'roargraph'), quick=False)

    def test_nothing_is_dropped_when_everything_is_present(self, fig3, monkeypatch):
        monkeypatch.setattr(fig3, 'backend_installed', lambda _backend: True)
        both = ('hnsw-hnswlib', 'roargraph')
        assert fig3.usable_backends(both, quick=True) == both
        assert fig3.usable_backends(both, quick=False) == both
