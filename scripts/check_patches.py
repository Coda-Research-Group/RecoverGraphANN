#!/usr/bin/env python
"""Verify that `patches/` still matches the pinned submodules, byte for byte.

`patches/` deliberately duplicates the fork branches, so that the instrumentation can be read
without cloning anything, survives a fork being renamed or deleted, and travels inside a
source archive where submodules do not. Duplication rots, so it gets a guard rather than a
promise: this regenerates each patch from the commit the submodule is pinned at and fails on
any difference.

    python scripts/check_patches.py            # verify
    python scripts/check_patches.py --write    # regenerate after changing a fork

`patches/flatnav-scalar-build.patch` is exempt: FlatNav is used unmodified, and that file is a
hand-written build-configuration change against upstream's own `setup.py`, not a fork diff.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH_DIR = ROOT / 'patches'


@dataclass(frozen=True)
class Fork:
    """A submodule carrying local changes, and the upstream commit they are relative to."""

    submodule: str
    upstream_base: str
    upstream_repo: str
    patch_name: str


FORKS = (
    Fork(
        submodule='third_party/hnswlib',
        upstream_base='3f3429661187e4c24a490a0f148fc6bc89042b3d',
        upstream_repo='https://github.com/nmslib/hnswlib',
        patch_name='hnswlib-instrumentation.patch',
    ),
    Fork(
        submodule='third_party/RoarGraph',
        upstream_base='78bf05cf248195007604d9e2386c82566a1818c2',
        upstream_repo='https://github.com/matchyc/RoarGraph',
        patch_name='roargraph-pyroar-bindings.patch',
    ),
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ['git', '-C', str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def generate(fork: Fork) -> str:
    """The diff from the fork's upstream base to its pinned commit."""
    repo = ROOT / fork.submodule
    if not (repo / '.git').exists():
        msg = f'{fork.submodule} is not checked out; run:\n    git submodule update --init --recursive'
        raise SystemExit(msg)

    head = _git(repo, 'rev-parse', 'HEAD').strip()
    header = (
        f'# {fork.submodule}\n'
        f'# upstream: {fork.upstream_repo}\n'
        f'# base:     {fork.upstream_base}\n'
        f'# pinned:   {head}\n'
        f'#\n'
        f'# Regenerate with: python scripts/check_patches.py --write\n'
        f'# Apply with:      git -C {fork.submodule} apply patches/{fork.patch_name}\n\n'
    )
    diff = _git(repo, 'diff', f'{fork.upstream_base}..{head}')
    return header + diff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help='regenerate the patches instead of checking them')
    args = parser.parse_args()

    PATCH_DIR.mkdir(exist_ok=True)
    failures = []

    for fork in FORKS:
        path = PATCH_DIR / fork.patch_name
        generated = generate(fork)

        if args.write:
            path.write_text(generated)
            print(f'wrote  {path.relative_to(ROOT)}  ({len(generated.splitlines())} lines)')
            continue

        if not path.exists():
            failures.append(f'{path.relative_to(ROOT)} is missing')
            continue
        if path.read_text() != generated:
            failures.append(
                f'{path.relative_to(ROOT)} does not match {fork.submodule} at its pinned commit — '
                f'the submodule moved without the patch being regenerated, or the reverse',
            )
            continue
        print(f'ok     {path.relative_to(ROOT)}')

    if failures:
        print('\nPatch drift detected:', file=sys.stderr)
        for failure in failures:
            print(f'  - {failure}', file=sys.stderr)
        print('\nRegenerate with: python scripts/check_patches.py --write', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
