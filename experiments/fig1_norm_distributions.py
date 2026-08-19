r"""Figure 1 — the distribution of database and query vector norms.

Reproduces `fig:norms`: per-vector L2 norm histograms for the database (VIBE `train`) and the
evaluation queries (VIBE `test`) on both attention datasets. This is the motivating picture —
the database norms are spread over a wide range, which is the geometry the transformation
removes.

    python experiments/fig1_norm_distributions.py
    python experiments/fig1_norm_distributions.py --data-dir data --output figures

Writes `figures/norm_distribution_panels.{pdf,png}` (Figure 1), the per-dataset variants, and
`figures/norm_distribution_stats.csv`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ACM sigconf \columnwidth (241 pt)
SIGCONF_COLUMN_WIDTH_IN = 3.33
SINGLE_PANEL_HEIGHT_IN = 2.15
LEGEND_HEIGHT_IN = 0.38
# Side-by-side panels: two datasets share one sigconf column width.
PANELS_FIG_HEIGHT_IN = 1.55
PANELS_TITLE_FONTSIZE = 6.5
PANELS_LABEL_FONTSIZE = 6.0
PANELS_TICK_FONTSIZE = 5.5

_SPLITS = ('train', 'test')
_SPLIT_LABELS = {
    'train': 'Database (X)',
    'test': 'Queries (Q)',
}
_SPLIT_LABELS_SHORT = {
    'train': 'Database (X)',
    'test': 'Queries (Q)',
}
_SPLIT_PALETTE = {
    'train': '#0066FF',
    'test': '#FF5500',
}

# Side-by-side panel order (matches other sigconf panel figures).
_DATASET_ORDER = ('llama-128-ip', 'yi-128-ip')


@dataclass(frozen=True)
class LegendStyle:
    loc: str = 'upper right'
    bbox: tuple[float, float] | None = None
    ncol: int = 1
    compact: bool = False


PANEL_LEGEND = LegendStyle(loc='upper right', bbox=(1.0, 1.0), ncol=2, compact=True)


@dataclass(frozen=True)
class AxesStyle:
    ylabel: str | None = 'Density'
    title_fontsize: float | None = None
    tick_length: float = 3.0
    show_y_tick_labels: bool = True


PANEL_AXES_STYLE = AxesStyle(title_fontsize=PANELS_TITLE_FONTSIZE, tick_length=2.5)

_STATS_COLUMNS = (
    'dataset',
    'split',
    'count',
    'min',
    'max',
    'mean',
    'std',
    'median',
    'p05',
    'p25',
    'p75',
    'p95',
)


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.replace('-', '_')


def _apply_sigconf_panels_style() -> None:
    """Smaller fonts for two side-by-side panels in one column."""
    _apply_sigconf_style()
    plt.rcParams.update(
        {
            'font.size': PANELS_LABEL_FONTSIZE,
            'axes.labelsize': PANELS_LABEL_FONTSIZE,
            'axes.titlesize': PANELS_TITLE_FONTSIZE,
            'legend.fontsize': 5.0,
            'xtick.labelsize': PANELS_TICK_FONTSIZE,
            'ytick.labelsize': PANELS_TICK_FONTSIZE,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 2.5,
            'ytick.major.size': 2.5,
            'xtick.major.width': 0.55,
            'ytick.major.width': 0.55,
        },
    )


def _apply_sigconf_style() -> None:
    sns.set_theme(context='paper', style='whitegrid', font_scale=0.92)
    plt.rcParams.update(
        {
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 8,
            'legend.fontsize': 6.5,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 3.0,
            'ytick.major.size': 3.0,
            'xtick.major.width': 0.6,
            'ytick.major.width': 0.6,
        },
    )


def _split_legend_handles(*, compact: bool = False) -> list[Patch]:
    labels = _SPLIT_LABELS_SHORT if compact else _SPLIT_LABELS
    return [
        Patch(
            facecolor=_SPLIT_PALETTE[split],
            alpha=0.42,
            edgecolor='white',
            linewidth=0.4,
            label=labels[split],
        )
        for split in _SPLITS
    ]


def _style_axis_ticks(
    ax: plt.Axes,
    *,
    tick_length: float,
    show_y_tick_labels: bool = True,
) -> None:
    """Ensure visible outward tick marks on bottom and left spines."""
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=tick_length,
        width=0.55,
        bottom=True,
        left=True,
        top=False,
        right=False,
        labelbottom=True,
        labelleft=show_y_tick_labels,
    )


def norms_to_frame(norms_by_split: dict[str, np.ndarray], dataset_name: str) -> pd.DataFrame:
    """Long-form DataFrame with columns norm, split, dataset."""
    frames = [
        pd.DataFrame({'norm': norms, 'split': split, 'dataset': dataset_name})
        for split, norms in norms_by_split.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    df['split'] = pd.Categorical(df['split'], categories=list(_SPLITS), ordered=True)
    return df


def _plot_norms_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    bins: int | str,
    legend: LegendStyle | None = None,
    axes_style: AxesStyle | None = None,
) -> None:
    style = axes_style or AxesStyle()
    sns.histplot(
        data=df,
        x='norm',
        hue='split',
        hue_order=list(_SPLITS),
        palette=_SPLIT_PALETTE,
        stat='density',
        common_norm=False,
        bins=bins,
        alpha=0.6,
        element='bars',
        fill=True,
        linewidth=0.4,
        edgecolor='white',
        ax=ax,
        legend=False,
    )
    if legend is not None:
        legend_kwargs: dict = {
            'handles': _split_legend_handles(compact=legend.compact),
            'loc': legend.loc,
            'ncol': legend.ncol,
            'frameon': True,
            'borderpad': 0.4 if legend.compact else 0.5,
            'labelspacing': 0.2 if legend.compact else 0.25,
            'handlelength': 0.55 if legend.compact else 0.9,
            'handletextpad': 0.25 if legend.compact else 0.4,
            'columnspacing': 0.45 if legend.compact else 0.8,
            'fontsize': 5.5 if legend.compact else 6.5,
        }
        if legend.bbox is not None:
            legend_kwargs['bbox_to_anchor'] = legend.bbox
        ax.legend(**legend_kwargs)
    title_kwargs = {'pad': 2}
    if style.title_fontsize is not None:
        title_kwargs['fontsize'] = style.title_fontsize
    ax.set_title(title, **title_kwargs)
    if style.ylabel is not None:
        ax.set_ylabel(style.ylabel)
    _style_axis_ticks(
        ax,
        tick_length=style.tick_length,
        show_y_tick_labels=style.show_y_tick_labels,
    )
    ax.grid(True, alpha=0.3)


def _add_figure_legend_below(fig: plt.Figure) -> None:
    """Shared horizontal legend below all axes (no overlap with histograms)."""
    fig.legend(
        handles=_split_legend_handles(),
        loc='outside lower center',
        ncol=2,
        frameon=True,
        borderpad=0.5,
        columnspacing=0.5,
        handletextpad=0.4,
        handlelength=1.0,
    )


def load_split_norms(hdf5_path: Path, split: str) -> np.ndarray:
    """Load one HDF5 split and return per-row L2 norms."""
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        vectors = np.array(hdf5_file[split][:], dtype=np.float32)  # type: ignore[index]
    return np.linalg.norm(vectors, axis=1)


def summarize_norms(norms: np.ndarray, *, dataset: str, split: str) -> dict[str, str | int | float]:
    """Return summary statistics for a norm vector."""
    percentiles = np.percentile(norms, [5, 25, 50, 75, 95])
    return {
        'dataset': dataset,
        'split': split,
        'count': int(norms.size),
        'min': float(np.min(norms)),
        'max': float(np.max(norms)),
        'mean': float(np.mean(norms)),
        'std': float(np.std(norms)),
        'median': float(percentiles[2]),
        'p05': float(percentiles[0]),
        'p25': float(percentiles[1]),
        'p75': float(percentiles[3]),
        'p95': float(percentiles[4]),
    }


def write_stats_csv(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_STATS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _panel_dataset_order(all_norms: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Datasets left-to-right in the panels figure."""
    ordered = [name for name in _DATASET_ORDER if name in all_norms]
    extra = [name for name in all_norms if name not in _DATASET_ORDER]
    return ordered + extra


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out_path = output_dir / f'{stem}.{fmt}'
        fig.savefig(out_path)
        written.append(out_path)
    plt.close(fig)
    return written


def plot_norm_overlay(
    norms_by_split: dict[str, np.ndarray],
    *,
    dataset_name: str,
    output_dir: Path,
    formats: tuple[str, ...],
    bins: int | str,
) -> list[Path]:
    """Density histogram overlay for Database (X) and Queries (Q) norms (sigconf column width)."""
    _apply_sigconf_style()
    df = norms_to_frame(norms_by_split, dataset_name)
    fig, ax = plt.subplots(
        figsize=(SIGCONF_COLUMN_WIDTH_IN, SINGLE_PANEL_HEIGHT_IN + LEGEND_HEIGHT_IN),
        layout='constrained',
    )
    _plot_norms_on_ax(ax, df, title=dataset_name, bins=bins)
    ax.set_xlabel(r'$\|v\|$')
    _add_figure_legend_below(fig)
    return _save_figure(fig, output_dir, f'norm_distribution_{_dataset_slug(dataset_name)}', formats)


def plot_norm_panels(
    all_norms: dict[str, dict[str, np.ndarray]],
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    bins: int | str,
) -> list[Path]:
    """Side-by-side panels (one per dataset) within sigconf single-column width."""
    _apply_sigconf_panels_style()
    datasets = _panel_dataset_order(all_norms)
    n_cols = len(datasets)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(SIGCONF_COLUMN_WIDTH_IN, PANELS_FIG_HEIGHT_IN),
        sharey=True,
        layout='constrained',
    )
    if n_cols == 1:
        axes = [axes]

    for col_idx, dataset_name in enumerate(datasets):
        ax = axes[col_idx]
        df = norms_to_frame(all_norms[dataset_name], dataset_name)
        panel_legend = PANEL_LEGEND if col_idx == n_cols - 1 else None
        panel_axes_style = (
            PANEL_AXES_STYLE if col_idx == 0 else replace(PANEL_AXES_STYLE, ylabel=None, show_y_tick_labels=False)
        )
        _plot_norms_on_ax(
            ax,
            df,
            title=dataset_name,
            bins=bins,
            legend=panel_legend,
            axes_style=panel_axes_style,
        )
        ax.set_xlabel(r'$\|v\|$')

    return _save_figure(fig, output_dir, 'norm_distribution_panels', formats)


def process_dataset(
    dataset_name: str,
    *,
    data_dir: Path,
    output_dir: Path,
    formats: tuple[str, ...],
    bins: int | str,
    no_plot: bool,
) -> tuple[list[dict[str, str | int | float]], dict[str, np.ndarray], list[Path]]:
    hdf5_path = data_dir / f'{dataset_name}.hdf5'
    if not hdf5_path.is_file():
        raise FileNotFoundError(f'HDF5 not found: {hdf5_path}')

    norms_by_split: dict[str, np.ndarray] = {}
    stats_rows: list[dict[str, str | int | float]] = []
    for split in _SPLITS:
        norms = load_split_norms(hdf5_path, split)
        norms_by_split[split] = norms
        stats_rows.append(
            summarize_norms(norms, dataset=dataset_name, split=_SPLIT_LABELS_SHORT[split]),
        )

    figure_paths: list[Path] = []
    if not no_plot:
        figure_paths = plot_norm_overlay(
            norms_by_split,
            dataset_name=dataset_name,
            output_dir=output_dir,
            formats=formats,
            bins=bins,
        )

    return stats_rows, norms_by_split, figure_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot L2 norm distributions for VIBE HDF5 train/test splits.',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=_REPO_ROOT / 'data',
        help='Directory containing {dataset}.hdf5 files',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=_REPO_ROOT / 'figures',
        help='Output directory for figures and CSV stats',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=list(_DATASET_ORDER),
        help='Dataset names (without .hdf5 suffix)',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        # PDF by default because that is what the paper includes; PNG alongside it because
        # PNG is byte-reproducible and PDF is not (matplotlib stamps a creation date).
        default=['png', 'pdf'],
        choices=['png', 'pdf', 'svg'],
        help='Figure file formats',
    )
    parser.add_argument(
        '--bins',
        default='120',
        help='Histogram bin count or rule (e.g. 120, fd, auto)',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Write CSV stats only, skip figures',
    )
    args = parser.parse_args()

    bins: int | str = int(args.bins) if isinstance(args.bins, str) and args.bins.isdigit() else args.bins

    formats = tuple(args.formats)
    all_stats: list[dict[str, str | int | float]] = []
    all_norms: dict[str, dict[str, np.ndarray]] = {}

    written: list[Path] = []
    for dataset_name in args.datasets:
        stats_rows, norms_by_split, figure_paths = process_dataset(
            dataset_name,
            data_dir=args.data_dir,
            output_dir=args.output,
            formats=formats,
            bins=bins,
            no_plot=args.no_plot,
        )
        all_stats.extend(stats_rows)
        all_norms[dataset_name] = norms_by_split
        written.extend(figure_paths)

    stats_path = args.output / 'norm_distribution_stats.csv'
    write_stats_csv(all_stats, stats_path)
    written.append(stats_path)

    if not args.no_plot and len(all_norms) > 1:
        written.extend(
            plot_norm_panels(
                all_norms,
                output_dir=args.output,
                formats=formats,
                bins=bins,
            ),
        )

    print('Wrote:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    main()
