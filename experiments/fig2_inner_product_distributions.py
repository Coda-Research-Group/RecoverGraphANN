r"""Figure 2 — the distribution of pairwise inner products between database vectors.

Reproduces `fig:ip_before`: density histograms of :math:`\langle x_i, x_j \rangle` over
random unordered pairs of database vectors (2M pairs per dataset, seed 0, i != j). The raw
panel is the one the paper prints, and it is the diagnosis: the distribution is concentrated
at large positive values, which is what lets a single high-norm neighbour occlude everything
else during graph construction.

The transformed panel is produced alongside it for comparison; only the raw one appears in
the paper.

    python experiments/fig2_inner_product_distributions.py
    python experiments/fig2_inner_product_distributions.py --data-dir data --output figures

Writes `figures/inner_product_distribution_panels_raw.{pdf,png}` (Figure 2), the per-dataset
and transformed variants, and `figures/inner_product_distribution_stats.csv`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'src'))

from rgann.transform import bachrach_transform  # noqa: E402

# ACM sigconf \columnwidth (241 pt)
SIGCONF_COLUMN_WIDTH_IN = 3.33
SINGLE_PANEL_HEIGHT_IN = 2.15
PANELS_FIG_HEIGHT_IN = 1.55
PANELS_TITLE_FONTSIZE = 6.5
PANELS_LABEL_FONTSIZE = 6.0
PANELS_TICK_FONTSIZE = 5.5

_PAIR_CHUNK_SIZE = 250_000
_HIST_COLOR = '#0066FF'
_INNER_PRODUCT_XLABEL = r'$\langle x_i, x_j \rangle$'

_VARIANTS = ('raw', 'asymmetric')
_VARIANT_TITLE_SUFFIX = {
    'raw': '',
    'asymmetric': ' (asymmetric DB)',
}

_DATASET_ORDER = ('llama-128-ip', 'yi-128-ip')

_STATS_COLUMNS = (
    'dataset',
    'variant',
    'num_pairs',
    'seed',
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


@dataclass(frozen=True)
class AxesStyle:
    ylabel: str | None = 'Density'
    title_fontsize: float | None = None
    tick_length: float = 3.0
    show_y_tick_labels: bool = True


PANEL_AXES_STYLE = AxesStyle(title_fontsize=PANELS_TITLE_FONTSIZE, tick_length=2.5)


@dataclass(frozen=True)
class PlotConfig:
    data_dir: Path
    output_dir: Path
    formats: tuple[str, ...]
    bins: int | str
    num_pairs: int
    seed: int
    no_plot: bool


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
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        },
    )


def _style_axis_ticks(
    ax: plt.Axes,
    *,
    tick_length: float,
    show_y_tick_labels: bool = True,
) -> None:
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


def load_train_vectors(hdf5_path: Path) -> np.ndarray:
    """Load HDF5 train split (database vectors)."""
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        return np.array(hdf5_file['train'][:], dtype=np.float32)  # type: ignore[index]


def build_asymmetric_database_vectors(X: np.ndarray) -> np.ndarray:
    """Transformed database vectors; the query side of the transform is unused here."""
    placeholder_queries = np.zeros((1, X.shape[1]), dtype=np.float32)
    X_transformed, _ = bachrach_transform(X, placeholder_queries)
    return X_transformed


def _sample_pair_inner_products_chunk(
    X: np.ndarray,
    rng: np.random.Generator,
    chunk_size: int,
) -> np.ndarray:
    n = X.shape[0]
    i = rng.integers(0, n, size=chunk_size)
    j = rng.integers(0, n - 1, size=chunk_size)
    j = np.where(j >= i, j + 1, j)
    return np.einsum('ij,ij->i', X[i], X[j], dtype=np.float64)


def sample_pair_inner_products(
    X: np.ndarray,
    num_pairs: int,
    seed: int,
    *,
    chunk_size: int = _PAIR_CHUNK_SIZE,
) -> np.ndarray:
    """Sample inner products for random unordered database pairs (i != j)."""
    rng = np.random.default_rng(seed)
    inner_products = np.empty(num_pairs, dtype=np.float64)
    offset = 0
    while offset < num_pairs:
        size = min(chunk_size, num_pairs - offset)
        inner_products[offset : offset + size] = _sample_pair_inner_products_chunk(X, rng, size)
        offset += size
    return inner_products


def summarize_inner_products(
    inner_products: np.ndarray,
    *,
    dataset: str,
    variant: str,
    num_pairs: int,
    seed: int,
) -> dict[str, str | int | float]:
    percentiles = np.percentile(inner_products, [5, 25, 50, 75, 95])
    return {
        'dataset': dataset,
        'variant': variant,
        'num_pairs': num_pairs,
        'seed': seed,
        'count': int(inner_products.size),
        'min': float(np.min(inner_products)),
        'max': float(np.max(inner_products)),
        'mean': float(np.mean(inner_products)),
        'std': float(np.std(inner_products)),
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


def inner_products_to_frame(inner_products: np.ndarray, dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame({'inner_product': inner_products, 'dataset': dataset_name})


def _plot_inner_products_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    bins: int | str,
    axes_style: AxesStyle | None = None,
) -> None:
    style = axes_style or AxesStyle()
    sns.histplot(
        data=df,
        x='inner_product',
        color=_HIST_COLOR,
        stat='density',
        bins=bins,
        alpha=0.6,
        element='bars',
        fill=True,
        linewidth=0.4,
        edgecolor='white',
        ax=ax,
    )
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


def _panel_dataset_order(all_ips: dict[str, np.ndarray]) -> list[str]:
    ordered = [name for name in _DATASET_ORDER if name in all_ips]
    extra = [name for name in all_ips if name not in _DATASET_ORDER]
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


def _figure_title(dataset_name: str, variant: str) -> str:
    return f'{dataset_name}{_VARIANT_TITLE_SUFFIX[variant]}'


def plot_inner_product_overlay(
    inner_products: np.ndarray,
    *,
    dataset_name: str,
    variant: str,
    output_dir: Path,
    formats: tuple[str, ...],
    bins: int | str,
) -> list[Path]:
    _apply_sigconf_style()
    df = inner_products_to_frame(inner_products, dataset_name)
    fig, ax = plt.subplots(
        figsize=(SIGCONF_COLUMN_WIDTH_IN, SINGLE_PANEL_HEIGHT_IN),
        layout='constrained',
    )
    _plot_inner_products_on_ax(ax, df, title=_figure_title(dataset_name, variant), bins=bins)
    ax.set_xlabel(_INNER_PRODUCT_XLABEL)
    stem = f'inner_product_distribution_{_dataset_slug(dataset_name)}_{variant}'
    return _save_figure(fig, output_dir, stem, formats)


def plot_inner_product_panels(
    all_inner_products: dict[str, np.ndarray],
    *,
    variant: str,
    output_dir: Path,
    formats: tuple[str, ...],
    bins: int | str,
) -> list[Path]:
    _apply_sigconf_panels_style()
    datasets = _panel_dataset_order(all_inner_products)
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
        df = inner_products_to_frame(all_inner_products[dataset_name], dataset_name)
        panel_axes_style = (
            PANEL_AXES_STYLE if col_idx == 0 else replace(PANEL_AXES_STYLE, ylabel=None, show_y_tick_labels=False)
        )
        _plot_inner_products_on_ax(
            ax,
            df,
            title=_figure_title(dataset_name, variant),
            bins=bins,
            axes_style=panel_axes_style,
        )
        ax.set_xlabel(_INNER_PRODUCT_XLABEL)

    stem = f'inner_product_distribution_panels_{variant}'
    return _save_figure(fig, output_dir, stem, formats)


def process_dataset(
    dataset_name: str,
    config: PlotConfig,
) -> tuple[list[dict[str, str | int | float]], dict[str, dict[str, np.ndarray]], list[Path]]:
    hdf5_path = config.data_dir / f'{dataset_name}.hdf5'
    if not hdf5_path.is_file():
        raise FileNotFoundError(f'HDF5 not found: {hdf5_path}')

    X = load_train_vectors(hdf5_path)
    vectors_by_variant = {
        'raw': X,
        'asymmetric': build_asymmetric_database_vectors(X),
    }

    stats_rows: list[dict[str, str | int | float]] = []
    inner_products_by_variant: dict[str, np.ndarray] = {}
    figure_paths: list[Path] = []

    for variant in _VARIANTS:
        inner_products = sample_pair_inner_products(
            vectors_by_variant[variant],
            config.num_pairs,
            config.seed,
        )
        inner_products_by_variant[variant] = inner_products
        stats_rows.append(
            summarize_inner_products(
                inner_products,
                dataset=dataset_name,
                variant=variant,
                num_pairs=config.num_pairs,
                seed=config.seed,
            ),
        )
        if not config.no_plot:
            figure_paths.extend(
                plot_inner_product_overlay(
                    inner_products,
                    dataset_name=dataset_name,
                    variant=variant,
                    output_dir=config.output_dir,
                    formats=config.formats,
                    bins=config.bins,
                ),
            )

    return stats_rows, {dataset_name: inner_products_by_variant}, figure_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot inner product distributions for random database vector pairs.',
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
        '--num-pairs',
        type=int,
        default=2_000_000,
        help='Number of random unordered database pairs to sample per variant',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed for pair sampling',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Write CSV stats only, skip figures',
    )
    args = parser.parse_args()

    bins: int | str = int(args.bins) if isinstance(args.bins, str) and args.bins.isdigit() else args.bins
    config = PlotConfig(
        data_dir=args.data_dir,
        output_dir=args.output,
        formats=tuple(args.formats),
        bins=bins,
        num_pairs=args.num_pairs,
        seed=args.seed,
        no_plot=args.no_plot,
    )

    all_stats: list[dict[str, str | int | float]] = []
    all_inner_products: dict[str, dict[str, np.ndarray]] = {}
    written: list[Path] = []

    for dataset_name in args.datasets:
        stats_rows, ips_by_variant, figure_paths = process_dataset(dataset_name, config)
        all_stats.extend(stats_rows)
        all_inner_products[dataset_name] = ips_by_variant[dataset_name]
        written.extend(figure_paths)

    stats_path = config.output_dir / 'inner_product_distribution_stats.csv'
    write_stats_csv(all_stats, stats_path)
    written.append(stats_path)

    if not config.no_plot and len(all_inner_products) > 1:
        for variant in _VARIANTS:
            ips_for_variant = {
                dataset_name: all_inner_products[dataset_name][variant] for dataset_name in all_inner_products
            }
            written.extend(
                plot_inner_product_panels(
                    ips_for_variant,
                    variant=variant,
                    output_dir=config.output_dir,
                    formats=config.formats,
                    bins=config.bins,
                ),
            )

    print('Wrote:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    main()
