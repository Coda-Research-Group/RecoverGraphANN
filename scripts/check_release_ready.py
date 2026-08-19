#!/usr/bin/env python
"""Is this artifact ready to be tagged and deposited.

The README describes the finished artifact. Until the canonical run has produced its results
and the submodules are wired, some of what it describes is not there — which is fine while
that is disclosed, and not fine at the moment someone mints a DOI against it.

This is the pre-release checklist as code. It checks that every file the README points at
exists, that every `make` target and script it tells you to run is real, and that the headline
number in its first line still matches the committed data.

    python scripts/check_release_ready.py            # report, always exit 0
    python scripts/check_release_ready.py --strict   # exit 1 if anything is missing

Not wired into CI: it is expected to report gaps for most of the artifact's life. Run it
before tagging.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The claim in the README's first line, and where it has to be checkable.
HEADLINE = (('llama-128-ip', '48', 'FALSE', '21.1%'), ('llama-128-ip', '48', 'TRUE', '95.8%'))


@dataclass
class Result:
    section: str
    label: str
    ok: bool
    detail: str = ''


def _readme() -> str:
    return (ROOT / 'README.md').read_text()


def check_linked_files(readme: str) -> list[Result]:
    """Relative links in the README must resolve."""
    results = []
    for target in sorted(set(re.findall(r'\[[^\]]+\]\(([^)#][^)]*)\)', readme))):
        if target.startswith(('http', 'mailto')):
            continue
        results.append(Result('links', target, (ROOT / target).exists()))
    return results


def check_make_targets(readme: str) -> list[Result]:
    makefile = (ROOT / 'Makefile').read_text()
    return [
        Result('make targets', f'make {name}', bool(re.search(rf'^{re.escape(name)}:', makefile, re.MULTILINE)))
        for name in sorted(set(re.findall(r'^\s*make ([a-z-]+)', readme, re.MULTILINE)))
    ]


def check_scripts(readme: str) -> list[Result]:
    results = []
    for name in sorted(set(re.findall(r'bash (scripts/[a-z_]+\.sh)', readme))):
        path = ROOT / name
        detail = '' if not path.exists() or path.stat().st_mode & 0o111 else 'not executable'
        results.append(Result('scripts', name, path.exists() and not detail, detail))
    return results


def _is_gitignored(claim: str) -> bool:
    """Generated output the README names but git deliberately does not track.

    Asked of git rather than hard-coded, so adding a `.gitignore` entry cannot leave a stale
    exception behind here.
    """
    return (
        subprocess.run(  # noqa: S603
            ['git', '-C', str(ROOT), 'check-ignore', '-q', claim],  # noqa: S607
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def check_artifacts(readme: str) -> list[Result]:
    """The results and figures the README promises. Absent until the canonical run lands."""
    results = []
    for claim in sorted(set(re.findall(r'`(results/[\w./-]+|figures/[\w./{},-]+)`', readme))):
        if _is_gitignored(claim):
            continue
        if claim.endswith('/'):
            results.append(Result('artifacts', claim, (ROOT / claim).is_dir()))
        elif '{' in claim:
            stem = Path(claim.split('.{')[0]).name
            found = sorted((ROOT / 'figures').glob(stem + '.*')) if (ROOT / 'figures').is_dir() else []
            results.append(Result('artifacts', claim, bool(found)))
        else:
            results.append(Result('artifacts', claim, (ROOT / claim).exists()))
    return results


def check_submodules() -> list[Result]:
    results = [Result('submodules', '.gitmodules', (ROOT / '.gitmodules').exists())]
    for name in ('hnswlib', 'flatnav', 'RoarGraph'):
        path = ROOT / 'third_party' / name
        results.append(Result('submodules', f'third_party/{name}', (path / '.git').exists()))
    return results


def check_headline() -> list[Result]:
    """The number in the README's first line has to be a row in the committed data."""
    source = ROOT / 'results' / 'paper' / 'different-m.csv'
    if not source.exists():
        return [Result('headline', str(source.relative_to(ROOT)), False, 'missing')]

    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))

    results = []
    for dataset, m, transformed, expected in HEADLINE:
        found = [
            row['recall']
            for row in rows
            if row['dataset_name'] == dataset and row['M'] == m and row['use_asymmetric_transformation'] == transformed
        ]
        results.append(
            Result(
                'headline',
                f'{expected} at {dataset} M={m} transformed={transformed}',
                found == [expected],
                str(found),
            ),
        )
    return results


def check_release_metadata() -> list[Result]:
    results = [Result('metadata', name, (ROOT / name).exists()) for name in ('LICENSE', 'CITATION.cff', '.zenodo.json')]

    remote = subprocess.run(  # noqa: S603
        ['git', '-C', str(ROOT), 'config', '--get', 'remote.origin.url'],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    results.append(Result('metadata', 'origin remote', bool(remote), remote))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--strict', action='store_true', help='exit 1 if anything is missing')
    args = parser.parse_args()

    readme = _readme()
    results = [
        *check_linked_files(readme),
        *check_make_targets(readme),
        *check_scripts(readme),
        *check_artifacts(readme),
        *check_submodules(),
        *check_headline(),
        *check_release_metadata(),
    ]

    section = None
    for result in results:
        if result.section != section:
            section = result.section
            print(f'\n{section}')
        mark = 'ok  ' if result.ok else 'MISS'
        detail = f'  — {result.detail}' if result.detail else ''
        print(f'  {mark}  {result.label}{detail}')

    missing = [r for r in results if not r.ok]
    print(f'\n{len(results) - len(missing)}/{len(results)} ready')
    if missing:
        print('\nNot ready to tag. Outstanding:')
        for result in missing:
            print(f'  - {result.section}: {result.label}')
        return 1 if args.strict else 0

    print('Ready to tag.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
