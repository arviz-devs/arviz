import numpy as np

from ..data import convert_to_dataset
from .plot_utils import (
    _scale_fig_size,
    xarray_var_iter,
    default_grid,
    _create_axes_grid,
    make_label,
)
from ..utils import _var_names


def plot_rank(data, var_names=None, coords=None, bins="auto", figsize=None, axes=None):

    posterior_data = convert_to_dataset(data, group="posterior")
    if coords is not None:
        posterior_data = posterior_data.sel(**coords)
    var_names = _var_names(var_names, posterior_data)
    plotters = list(xarray_var_iter(posterior_data, var_names=var_names, combined=True))

    if axes is None:
        rows, cols = default_grid(len(plotters))

        figsize, ax_labelsize, titlesize, _, _, _ = _scale_fig_size(
            figsize, None, rows=rows, cols=cols
        )
        _, axes = _create_axes_grid(len(plotters), rows, cols, figsize=figsize, squeeze=False)

    for ax, (var_name, selection, var_data) in zip(axes.ravel(), plotters):
        ranks = var_data.argsort(axis=None).reshape(*var_data.shape)
        all_counts = []
        for row in ranks:
            # If `bins` was auto, it gets overwritten here, and carried forward
            # using the `range` argument makes sure the bins cover all the data
            counts, bins = np.histogram(row, bins=bins, range=(0, ranks.size))
            all_counts.append(counts)
        all_counts = np.array(all_counts)
        gap = all_counts.max() * 1.05
        width = (bins[1] - bins[0]) * 0.95

        y_ticks = []
        for idx, counts in enumerate(all_counts):
            y_ticks.append(idx * gap)
            # Line where data is uniform
            ax.axhline(
                y=y_ticks[-1] + counts.sum() / bins.shape[0], linestyle="--", color="C1", zorder=-10
            )
            # fake an x-axis
            ax.axhline(y=y_ticks[-1], color="k")
            ax.bar(bins[:-1], counts, bottom=y_ticks[-1], width=width, align="edge", color="C0")
        ax.set_xlabel("Rank (all chains)")
        ax.set_ylabel("Chain", fontsize=ax_labelsize)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(np.arange(len(y_ticks)))
        ax.set_title(make_label(var_name, selection), fontsize=titlesize)

    return axes
