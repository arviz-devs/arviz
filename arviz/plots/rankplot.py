import numpy as np
import scipy.stats

from ..data import convert_to_dataset
from .plot_utils import (
    _scale_fig_size,
    xarray_var_iter,
    default_grid,
    _create_axes_grid,
    make_label,
)
from ..utils import _var_names


def _sturges_formula(dataset, mult=1):
    """Use Sturges' formula to determine number of bins.

    See https://en.wikipedia.org/wiki/Histogram#Sturges'_formula
    or https://doi.org/10.1080%2F01621459.1926.10502161

    Parameters
    ----------
    dataset: xarray.DataSet
        Must have the `draw` dimension

    mult: float
        Used to scale the number of bins up or down. Default is 1 for Sturges' formula.

    Returns
    -------
    int
        Number of bins to use
    """
    return int(np.ceil(mult * np.log2(dataset.draw.size)) + 1)


def plot_rank(data, var_names=None, coords=None, bins=None, ref_line=True, figsize=None, axes=None):

    posterior_data = convert_to_dataset(data, group="posterior")
    if coords is not None:
        posterior_data = posterior_data.sel(**coords)
    var_names = _var_names(var_names, posterior_data)
    plotters = list(xarray_var_iter(posterior_data, var_names=var_names, combined=True))

    if bins is None:
        # Use double sturges' formula
        bins = _sturges_formula(posterior_data, mult=2)

    if axes is None:
        rows, cols = default_grid(len(plotters))

        figsize, ax_labelsize, titlesize, _, _, _ = _scale_fig_size(
            figsize, None, rows=rows, cols=cols
        )
        _, axes = _create_axes_grid(len(plotters), rows, cols, figsize=figsize, squeeze=False)

    for ax, (var_name, selection, var_data) in zip(axes.ravel(), plotters):
        ranks = scipy.stats.rankdata(var_data).reshape(var_data.shape)
        all_counts = []
        for row in ranks:
            counts, bin_ary = np.histogram(row, bins=bins, range=(0, ranks.size))
            all_counts.append(counts)
        all_counts = np.array(all_counts)
        gap = all_counts.max() * 1.05
        width = bin_ary[1] - bin_ary[0]

        y_ticks = []

        # Center the bins
        bin_ary = (bin_ary[1:] + bin_ary[:-1]) / 2
        for idx, counts in enumerate(all_counts):
            y_ticks.append(idx * gap)
            if ref_line:
                # Line where data is uniform
                ax.axhline(y=y_ticks[-1] + counts.mean(), linestyle="--", color="C1")
            # fake an x-axis
            ax.axhline(y=y_ticks[-1], color="k")
            ax_color = ax.get_facecolor()
            ax.bar(
                bin_ary,
                counts,
                bottom=y_ticks[-1],
                width=width,
                align="center",
                color="C0",
                edgecolor=ax_color,
            )
        ax.set_xlabel("Rank (all chains)", fontsize=ax_labelsize)
        ax.set_ylabel("Chain", fontsize=ax_labelsize)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(np.arange(len(y_ticks)))
        ax.set_title(make_label(var_name, selection), fontsize=titlesize)

    return axes
