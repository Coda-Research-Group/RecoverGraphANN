"""Provenance columns, and the one that must not leak.

Each results row records where it came from. `hostname` defaults to the machine's own name,
which is right while someone runs their own experiments — and wrong for results published from
a machine whose name identifies private infrastructure, because that cannot be taken back.
`RGANN_HOSTNAME` overrides it.
"""

from __future__ import annotations

import platform

import pytest

from rgann import runner


@pytest.fixture(autouse=True)
def _clear_cache():
    """provenance() memoises, so each test needs a clean slate."""
    runner._provenance_cache = None  # noqa: SLF001
    yield
    runner._provenance_cache = None  # noqa: SLF001


class TestHostname:
    def test_defaults_to_the_real_machine_name(self, monkeypatch):
        monkeypatch.delenv('RGANN_HOSTNAME', raising=False)
        assert runner.provenance()['hostname'] == platform.node()

    def test_the_override_replaces_it(self, monkeypatch):
        monkeypatch.setenv('RGANN_HOSTNAME', 'reference-machine')
        assert runner.provenance()['hostname'] == 'reference-machine'

    def test_an_empty_override_falls_back_rather_than_recording_nothing(self, monkeypatch):
        """`RGANN_HOSTNAME=` should not silently blank the column."""
        monkeypatch.setenv('RGANN_HOSTNAME', '')
        assert runner.provenance()['hostname'] == platform.node()


class TestProvenance:
    def test_it_records_what_a_reader_needs_to_place_a_measurement(self, monkeypatch):
        monkeypatch.delenv('RGANN_HOSTNAME', raising=False)
        recorded = runner.provenance()
        for field in ('run_timestamp_utc', 'git_commit', 'git_dirty', 'hostname', 'cpu_model', 'platform', 'python'):
            assert field in recorded, f'{field} is not recorded'

    def test_the_cpu_model_is_never_overridden(self, monkeypatch):
        """The hostname is a label; the CPU is the thing a throughput number depends on."""
        monkeypatch.setenv('RGANN_HOSTNAME', 'reference-machine')
        assert runner.provenance()['cpu_model'] != 'reference-machine'
        assert runner.provenance()['cpu_model']

    def test_it_is_computed_once(self, monkeypatch):
        monkeypatch.setenv('RGANN_HOSTNAME', 'first')
        first = runner.provenance()
        monkeypatch.setenv('RGANN_HOSTNAME', 'second')
        assert runner.provenance() is first, 'provenance stopped being memoised'
