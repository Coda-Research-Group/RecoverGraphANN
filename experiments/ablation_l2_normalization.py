#!/usr/bin/env python
"""Section 6 ablation: norm equalization versus the appended coordinate.

The Bachrach lift does two things at once: it equalizes the database norms, and it appends a
coordinate that preserves the inner-product ranking exactly. Plain L2 normalization does the
first without the second — it equalizes norms, but the index then answers a cosine query
against an inner-product ground truth.

So if L2 normalization recovers `acc` and `deg` but *not* recall, the build-time recovery is
attributable to the norms alone, and the remaining recall gap is the ranking distortion. That
is the paper's argument, and this script is what measures it.

Run at the paper's matched settings: M = 48, efConstruction = 500, efSearch = 1000, k = 10,
single-threaded, both datasets, all three normalizations.

    python experiments/ablation_l2_normalization.py
    python experiments/ablation_l2_normalization.py --quick
    python experiments/ablation_l2_normalization.py --report   # table from the committed CSV

Writes `results/ablation_l2_normalization.csv`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from rgann.datasets import load_dataset
from rgann.runner import ExperimentSpec, append_rows, atomic_output, run_spec
from rgann.transform import Normalization

DATASETS = ('yi-128-ip', 'llama-128-ip')
NORMALIZATIONS = (Normalization.NONE, Normalization.BACHRACH, Normalization.L2)
M = 48
EF_CONSTRUCTION = 500
EF_SEARCH = 1000
K = 10

DEFAULT_CSV = Path('results/ablation_l2_normalization.csv')


def run(output: Path, *, quick: bool) -> None:
    with atomic_output(output) as partial:
        _run_into(partial, quick=quick)

    print(f'\nwrote {output}', flush=True)
    report(output)


def _run_into(output: Path, *, quick: bool) -> None:
    for dataset_name in DATASETS:
        dataset = load_dataset(dataset_name, quick=quick)
        print(f'\n{dataset.describe()}', flush=True)
        for normalization in NORMALIZATIONS:
            spec = ExperimentSpec(
                dataset=dataset_name,
                backend='hnsw-hnswlib',
                normalization=normalization,
                build_params={'m': M, 'ef_construction': EF_CONSTRUCTION},
                search_values=[EF_SEARCH],
                k=K,
                threads=1,
                label='ablation-l2',
            )
            append_rows(output, list(run_spec(spec, dataset)))


def report(source: Path) -> None:
    """Print the comparison the paper's paragraph makes."""
    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))

    print(f'\nM={M}, efConstruction={EF_CONSTRUCTION}, efSearch={EF_SEARCH}, k={K}, single-threaded\n')
    header = f'{"dataset":<15}{"normalization":<15}{"acc":>8}{"deg":>8}{"recall@10":>12}'
    print(header)
    print('-' * len(header))
    for dataset_name in DATASETS:
        for normalization in NORMALIZATIONS:
            match = [
                row
                for row in rows
                if row['dataset_name'] == dataset_name and row['normalization'] == str(normalization)
            ]
            if not match:
                print(f'{dataset_name:<15}{normalization!s:<15}{"--":>8}{"--":>8}{"--":>12}')
                continue
            row = match[0]
            print(
                f'{dataset_name:<15}{normalization!s:<15}'
                f'{float(row["accepted_candidates_avg"]):>8.1f}'
                f'{float(row["avg_node_degree"]):>8.1f}'
                f'{float(row["recall_avg"]) * 100:>11.1f}%',
            )

    print(
        '\nRead it as: l2 should land close to bachrach on acc and deg — the norms are what\n'
        'the heuristic responds to — while falling well short on recall, because a cosine\n'
        'index is answering an inner-product question.',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--report', action='store_true', help='print the table from an existing CSV; run nothing')
    parser.add_argument('--quick', action='store_true', help='10k-row subsample; NOT the paper numbers')
    args = parser.parse_args()

    if args.report:
        if not args.csv.exists():
            parser.error(f'{args.csv} does not exist; run without --report first')
        report(args.csv)
        return

    run(args.csv, quick=args.quick)


if __name__ == '__main__':
    main()
