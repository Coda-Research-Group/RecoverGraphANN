#!/usr/bin/env python
"""Figure 3 — recall@10 versus QPS, raw and transformed, for all four indexes.

Reproduces `fig:recall_qps`. Each panel sweeps one search-effort knob over the paper's
sixteen values, single-threaded, k = 10:

===============  ==========================  ==============================
index            build parameters            swept knob
===============  ==========================  ==============================
HNSW (hnswlib)   M=48, efConstruction=500    ``efSearch``
FlatNav          max_edges_per_node=48        ``ef_search``
NSG (FAISS)      R=48, L=C=500                ``search_L``
RoarGraph        N_q=100, M=48, L=500         ``L_pq``
===============  ==========================  ==============================

Everything here is **query-agnostic**: the index may see only the database. RoarGraph is
query-*aware* by design, so it is run with the database substituted for the query side — see
`rgann.indexes.roargraph`. Its native mode, which reads VIBE's `learn_neighbors`, is not run:
it is a different setting, and this artifact evaluates one.

    python experiments/fig3_recall_qps.py                # run the sweeps (hours)
    python experiments/fig3_recall_qps.py --quick        # 10k subsample, minutes
    python experiments/fig3_recall_qps.py --plot         # figure from the committed CSV

Writes `results/fig3_recall_qps.csv` and `figures/efsearch_recall_qps_2x2.{pdf,png}`.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from rgann.datasets import load_dataset
from rgann.runner import ExperimentSpec, append_rows, atomic_output, run_spec
from rgann.transform import Normalization, apply_normalization

DATASETS = ('yi-128-ip', 'llama-128-ip')
SEARCH_VALUES = (10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 800, 1000)
K = 10
EF_CONSTRUCTION = 500
M = 48

DEFAULT_CSV = Path('results/fig3_recall_qps.csv')
DEFAULT_FIGURE_DIR = Path('figures')
FIGURE_STEM = 'efsearch_recall_qps_2x2'

#: `--quick` numbers come from a 10k subsample and are not the paper's, so neither the CSV nor
#: the figure is written over the committed one.
QUICK_CSV = Path('results/quick/fig3_recall_qps.csv')
QUICK_FIGURE_DIR = Path('figures/quick')

_GRAPH_BACKENDS = {
    'hnsw-hnswlib': {'m': M, 'ef_construction': EF_CONSTRUCTION},
    'flatnav': {'m': M, 'ef_construction': EF_CONSTRUCTION},
    'nsg-faiss': {'r': M, 'l': EF_CONSTRUCTION, 'c': EF_CONSTRUCTION},
}
_ROARGRAPH_PARAMS = {'m_sq': 100, 'm_pjbp': M, 'l_pjpq': EF_CONSTRUCTION}


def _roargraph_learn_table(dataset, normalization: Normalization):  # noqa: ANN001, ANN202
    """Build RoarGraph's bipartite input, query-agnostically.

    Computed on the *active* vector space, because the whole point is that RoarGraph gets no
    information the transformation does not also get. VIBE's `learn_neighbors` — the real
    query sample RoarGraph is designed around — is deliberately never read here.
    """
    from rgann.indexes.roargraph import database_side_learn_table  # noqa: PLC0415

    X, _ = apply_normalization(dataset.X, dataset.Q, normalization)
    return database_side_learn_table(X)


def backend_installed(backend: str) -> bool:
    """Whether this machine actually has the backend installed."""
    module = {'hnsw-hnswlib': 'hnswlib', 'nsg-faiss': 'faiss', 'flatnav': 'flatnav', 'roargraph': 'RoarGraph'}[backend]
    return importlib.util.find_spec(module) is not None


def usable_backends(backends: tuple[str, ...], *, quick: bool) -> tuple[str, ...]:
    """In --quick, drop what is not installed; in the canonical run, refuse to.

    The smoke test exists to prove the pipeline works on *this* machine, and RoarGraph is
    Linux/x86-64 only, so on macOS it must skip rather than fail. A canonical run has no such
    licence: silently omitting a backend there would publish a Figure 3 missing a curve.
    """
    missing = [b for b in backends if not backend_installed(b)]
    if not missing:
        return backends
    if not quick:
        msg = f'not installed: {", ".join(missing)} — see the platform table in the README'
        raise SystemExit(msg)
    for backend in missing:
        print(f'skipping {backend}: not installed on this platform', flush=True)
    return tuple(b for b in backends if b not in missing)


def run(output: Path, *, quick: bool, backends: tuple[str, ...]) -> None:
    with atomic_output(output) as partial:
        _run_into(partial, quick=quick, backends=usable_backends(backends, quick=quick))

    print(f'\nwrote {output}', flush=True)


def _run_into(output: Path, *, quick: bool, backends: tuple[str, ...]) -> None:
    for dataset_name in DATASETS:
        dataset = load_dataset(dataset_name, quick=quick)
        print(f'\n{dataset.describe()}', flush=True)

        for normalization in (Normalization.NONE, Normalization.BACHRACH):
            for backend, params in _GRAPH_BACKENDS.items():
                if backend not in backends:
                    continue
                spec = ExperimentSpec(
                    dataset=dataset_name,
                    backend=backend,
                    normalization=normalization,
                    build_params=params,
                    search_values=SEARCH_VALUES,
                    k=K,
                    threads=1,
                    label='fig3',
                )
                append_rows(output, list(run_spec(spec, dataset)))

            if 'roargraph' not in backends:
                continue

            spec = ExperimentSpec(
                dataset=dataset_name,
                backend='roargraph',
                normalization=normalization,
                build_params={
                    **_ROARGRAPH_PARAMS,
                    'learn_table': _roargraph_learn_table(dataset, normalization),
                },
                search_values=SEARCH_VALUES,
                k=K,
                threads=1,
                label='fig3',
            )
            append_rows(output, list(run_spec(spec, dataset)))


# --- Plotting -------------------------------------------------------------------------------
# Layout, sizing and styling are unchanged from the script that produced the published
# figure; only the column names differ, because the results schema now spells things out.

SIGCONF_COLUMN_WIDTH_IN = 3.33
FIG_HEIGHT_IN = 3.2
TITLE_FONTSIZE = 6.3
LABEL_FONTSIZE = 6.0
TICK_FONTSIZE = 5.4
TICK_LENGTH = 2.5
LINE_WIDTH = 0.9
MARKER_SIZE = 2.4

_DATASET_ROWS = ('llama-128-ip', 'yi-128-ip')
_NORMALIZATION_COLUMNS = (Normalization.NONE, Normalization.BACHRACH)
_COLUMN_TITLES = {Normalization.NONE: 'Raw', Normalization.BACHRACH: 'Transformed'}
_ALGORITHM_LABELS = {
    'hnsw-hnswlib': 'HNSW',
    'flatnav': 'FlatNav',
    'nsg-faiss': 'NSG',
    'roargraph': 'RoarGraph',
}


def _style_axes(ax, *, show_y_labels: bool) -> None:  # noqa: ANN001
    """Log y, linear x, and minor ticks that are actually drawn."""
    from matplotlib.ticker import LogLocator, NullFormatter  # noqa: PLC0415

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=TICK_LENGTH,
        width=0.55,
        bottom=True,
        left=True,
        labelleft=show_y_labels,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Locating the minor ticks is not enough to draw them: without this they are positioned
    # and invisible, and a log axis spanning two decades reads as though it were linear.
    ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False, length=2.2, width=0.5)
    ax.grid(visible=True, alpha=0.3)


def _load_curves(csv_path: Path):  # noqa: ANN202
    """The curves to plot."""
    import pandas as pd  # noqa: PLC0415

    frame = pd.read_csv(csv_path)
    if frame.empty:
        msg = f'{csv_path} has no rows'
        raise ValueError(msg)
    return frame


def plot(csv_path: Path, output_dir: Path, formats: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    frame = _load_curves(csv_path)

    sns.set_theme(context='paper', style='whitegrid', font_scale=0.92)
    plt.rcParams.update(
        {
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'font.size': LABEL_FONTSIZE,
            'axes.labelsize': LABEL_FONTSIZE,
            'axes.titlesize': TITLE_FONTSIZE,
            'legend.fontsize': 5.1,
            'xtick.labelsize': TICK_FONTSIZE,
            'ytick.labelsize': TICK_FONTSIZE,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        },
    )

    labels = list(_ALGORITHM_LABELS.values())
    palette = dict(zip(labels, sns.color_palette(n_colors=len(labels)), strict=True))

    fig, axes = plt.subplots(
        nrows=len(_DATASET_ROWS),
        ncols=len(_NORMALIZATION_COLUMNS),
        figsize=(SIGCONF_COLUMN_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=True,
        sharey=True,
    )

    for row, dataset_name in enumerate(_DATASET_ROWS):
        for column, normalization in enumerate(_NORMALIZATION_COLUMNS):
            ax = axes[row, column]
            panel = frame[(frame['dataset_name'] == dataset_name) & (frame['normalization'] == str(normalization))]
            for algorithm, label in _ALGORITHM_LABELS.items():
                curve = panel[panel['algorithm'] == algorithm].sort_values('search_param_value')
                if curve.empty:
                    continue
                ax.plot(
                    curve['recall_avg'],
                    curve['queries_per_sec'],
                    color=palette[label],
                    marker='o',
                    markersize=MARKER_SIZE,
                    markeredgecolor='0.2',
                    markeredgewidth=0.3,
                    linewidth=LINE_WIDTH,
                    label=label,
                )

            if row == 0:
                ax.set_title(_COLUMN_TITLES[normalization], fontsize=TITLE_FONTSIZE, pad=2.0)
            if column == 0:
                ax.set_ylabel(f'{dataset_name}\nQueries per second (QPS)')
            if row == len(_DATASET_ROWS) - 1:
                ax.set_xlabel('Recall@10')

            _style_axes(ax, show_y_labels=(column == 0))

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles,
        legend_labels,
        loc='upper right',
        frameon=True,
        borderpad=0.35,
        labelspacing=0.2,
        handlelength=1.2,
        fontsize=5.0,
    )
    fig.subplots_adjust(left=0.17, right=0.995, top=0.9, bottom=0.14, wspace=0.16, hspace=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f'{FIGURE_STEM}.{fmt}'
        fig.savefig(path, format=fmt)
        print(f'wrote {path}')
    plt.close(fig)


def resolve_csv(chosen: Path | None, *, quick: bool) -> Path:
    """Where to write the results, when the caller did not say."""
    if chosen is not None:
        return chosen
    return QUICK_CSV if quick else DEFAULT_CSV


def resolve_figure_dir(chosen: Path | None, *, quick: bool) -> Path:
    """Where to write the figure, when the caller did not say."""
    if chosen is not None:
        return chosen
    return QUICK_FIGURE_DIR if quick else DEFAULT_FIGURE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'], choices=['pdf', 'png', 'svg'])
    parser.add_argument('--plot', action='store_true', help='plot from an existing CSV; run nothing')
    parser.add_argument('--quick', action='store_true', help='10k-row subsample; NOT the paper numbers')
    parser.add_argument(
        '--backends',
        nargs='+',
        default=[*_GRAPH_BACKENDS, 'roargraph'],
        help='subset to run, e.g. to skip roargraph on macOS',
    )
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv, quick=args.quick)
    figure_dir = resolve_figure_dir(args.output, quick=args.quick)

    if args.plot:
        if not csv_path.exists():
            parser.error(f'{csv_path} does not exist; run without --plot first')
        plot(csv_path, figure_dir, tuple(args.formats))
        return

    run(csv_path, quick=args.quick, backends=tuple(args.backends))
    plot(csv_path, figure_dir, tuple(args.formats))


if __name__ == '__main__':
    main()
