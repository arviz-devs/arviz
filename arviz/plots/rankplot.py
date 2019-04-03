"""Histograms of ranked posterior draws, plotted for each chain."""
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
    """Plot rank order statistics of chains.

    From the paper: Rank plots are histograms of the ranked posterior
    draws (ranked over all chains) plotted separately for each chain.
    If all of the chains are targeting the same posterior, we expect
    the ranks in each chain to be uniform, whereas if one chain has a
    different location or scale parameter, this will be reflected in
    the deviation from uniformity. If rank plots of all chains look
    similar, this indicates good mixing of the chains.

    This plot was introduced by Aki Vehtari, Andrew Gelman, Daniel
    Simpson, Bob Carpenter, Paul-Christian Burkner (2019):
    Rank-normalization, folding, and localization: An improved R-hat
    for assessing convergence of MCMC.
    arXiv preprint https://arxiv.org/abs/1903.08008


    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : string or list of variable names
        Variables to be plotted
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    bins : None or passed to np.histogram
        Binning strategy used for histogram. By default uses twice the
        result of Sturges' formula. See `np.histogram` documenation for
        other available arguments.
    ref_line : boolean
        Whether to include a dashed line showing where a uniform
        distribution would lie
    figsize : tuple
        Figure size. If None it will be defined automatically.
    ax : axes
        Matplotlib axes. Defaults to None.

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    Show a default rank plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_rank(data)

    Recreate Figure 13 from the arxiv preprint

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_rank(data, var_names='tau')
    """
    posterior_data = convert_to_dataset(data, group="posterior")
    if coords is not None:
        posterior_data = posterior_data.sel(**coords)
    var_names = _var_names(var_names, posterior_data)
    plotters = list(xarray_var_iter(posterior_data, var_names=var_names, combined=True))

    if bins is None:
        # Use double Sturges' formula
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

        # Center the bins
        bin_ary = (bin_ary[1:] + bin_ary[:-1]) / 2

        y_ticks = []
        for idx, counts in enumerate(all_counts):
            y_ticks.append(idx * gap)
            if ref_line:
                # Line where data is uniform
                ax.axhline(y=y_ticks[-1] + counts.mean(), linestyle="--", color="C1")
            # fake an x-axis
            ax.axhline(y=y_ticks[-1], color="k", lw=1)
            ax.bar(
                bin_ary,
                counts,
                bottom=y_ticks[-1],
                width=width,
                align="center",
                color="C0",
                edgecolor=ax.get_facecolor(),
            )
        ax.set_xlabel("Rank (all chains)", fontsize=ax_labelsize)
        ax.set_ylabel("Chain", fontsize=ax_labelsize)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(np.arange(len(y_ticks)))
        ax.set_title(make_label(var_name, selection), fontsize=titlesize)

    return axes
