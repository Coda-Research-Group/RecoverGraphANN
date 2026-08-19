"""`atomic_output` is what makes the multi-day run safely resumable.

`run_all_experiments.sh` skips any stage whose output already exists. Appending straight to
the real path defeats that: a run killed halfway leaves a partial CSV, the next attempt sees
a file and skips, and the missing rows surface as a figure with curves quietly absent — the
worst kind of failure, because nothing reports it.
"""

from __future__ import annotations

import pytest

from rgann.runner import atomic_output


def _crash_midway(target, content: str) -> None:
    """Write some output, then die the way a killed build does."""
    with atomic_output(target) as partial:
        partial.write_text(content)
        raise RuntimeError('build died')


class TestAtomicOutput:
    def test_the_real_path_does_not_exist_until_the_block_completes(self, tmp_path):
        target = tmp_path / 'results.csv'
        with atomic_output(target) as partial:
            partial.write_text('half a file')
            assert not target.exists(), 'the real path appeared before the block finished'
        assert target.exists()

    def test_content_written_to_the_scratch_path_ends_up_at_the_target(self, tmp_path):
        target = tmp_path / 'results.csv'
        with atomic_output(target) as partial:
            partial.write_text('a,b\n1,2\n')
        assert target.read_text() == 'a,b\n1,2\n'

    def test_a_crash_leaves_the_target_absent(self, tmp_path):
        target = tmp_path / 'results.csv'
        with pytest.raises(RuntimeError, match='build died'):
            _crash_midway(target, 'one row only')
        assert not target.exists(), 'a crashed run left a file the resume logic would skip'

    def test_a_crash_leaves_an_existing_target_untouched(self, tmp_path):
        """A rerun that dies must not destroy the results it was going to replace."""
        target = tmp_path / 'results.csv'
        target.write_text('the previous complete run\n')
        with pytest.raises(RuntimeError, match='build died'):
            _crash_midway(target, 'incomplete')
        assert target.read_text() == 'the previous complete run\n'

    def test_a_stale_partial_from_an_earlier_crash_is_discarded(self, tmp_path):
        target = tmp_path / 'results.csv'
        stale = tmp_path / 'results.csv.partial'
        stale.write_text('rows from a run that died last week\n')
        with atomic_output(target) as partial:
            assert not partial.exists(), 'the stale partial was not cleared'
            partial.write_text('fresh\n')
        assert target.read_text() == 'fresh\n'

    def test_it_replaces_an_existing_target(self, tmp_path):
        target = tmp_path / 'results.csv'
        target.write_text('old\n')
        with atomic_output(target) as partial:
            partial.write_text('new\n')
        assert target.read_text() == 'new\n'

    def test_it_creates_the_parent_directory(self, tmp_path):
        target = tmp_path / 'results' / 'nested' / 'out.csv'
        with atomic_output(target) as partial:
            partial.write_text('x\n')
        assert target.read_text() == 'x\n'

    def test_no_scratch_file_survives_a_successful_run(self, tmp_path):
        target = tmp_path / 'results.csv'
        with atomic_output(target) as partial:
            partial.write_text('x\n')
        assert list(tmp_path.iterdir()) == [target]
