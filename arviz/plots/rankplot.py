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
    filter_plotters_list,
    _sturges_formula,
)
from ..utils import _var_names
from ..stats.stats_utils import histogram


def plot_rank(
    data,
    var_names=None,
    coords=None,
    bins=None,
    kind="bars",
    colors="cycle",
    ref_line=True,
    labels=True,
    figsize=None,
    axes=None,
):
    """Plot rank order statistics of chains.

    From the paper: Rank plots are histograms of the ranked posterior draws (ranked over all
    chains) plotted separately for each chain.
    If all of the chains are targeting the same posterior, we expect the ranks in each chain to be
    uniform, whereas if one chain has a different location or scale parameter, this will be
    reflected in the deviation from uniformity. If rank plots of all chains look similar, this
    indicates good mixing of the chains.

    This plot was introduced by Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter,
    Paul-Christian Burkner (2019): Rank-normalization, folding, and localization: An improved R-hat
    for assessing convergence of MCMC. arXiv preprint https://arxiv.org/abs/1903.08008


    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object. Refer to documentation of
        az.convert_to_dataset for details
    var_names : string or list of variable names
        Variables to be plotted
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    bins : None or passed to np.histogram
        Binning strategy used for histogram. By default uses twice the result of Sturges' formula.
        See `np.histogram` documenation for, other available arguments.
    kind : string
        If bars (defaults), ranks are represented as stacked histograms (one per chain). If vlines
        ranks are represented as vertical lines above or below `ref_line`.
    colors : string or list of strings
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically choose a color per model from matplolib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to `cycle`.
    ref_line : boolean
        Whether to include a dashed line showing where a uniform distribution would lie
    labels : bool
        wheter to plot or not the x and y labels, defaults to True
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

    Use vlines to compare results for centered vs noncentered models

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered_data = az.load_arviz_data('centered_eight')
        >>> noncentered_data = az.load_arviz_data('non_centered_eight')
        >>> _, ax = plt.subplots(1, 2, figsize=(12, 3))
        >>> az.plot_rank(centered_data, var_names="mu", kind='vlines', axes=ax[0])
        >>> az.plot_rank(noncentered_data, var_names="mu", kind='vlines', axes=ax[1])

    """
    posterior_data = convert_to_dataset(data, group="posterior")
    if coords is not None:
        posterior_data = posterior_data.sel(**coords)
    var_names = _var_names(var_names, posterior_data)
    plotters = filter_plotters_list(
        list(xarray_var_iter(posterior_data, var_names=var_names, combined=True)), "plot_rank"
    )
    length_plotters = len(plotters)

    if bins is None:
        bins = _sturges_formula(posterior_data, mult=2)

    if axes is None:
        rows, cols = default_grid(length_plotters)

        figsize, ax_labelsize, titlesize, _, _, _ = _scale_fig_size(
            figsize, None, rows=rows, cols=cols
        )
        _, axes = _create_axes_grid(length_plotters, rows, cols, figsize=figsize, squeeze=False)
    else:
        figsize, ax_labelsize, titlesize, _, _, _ = _scale_fig_size(figsize, None)

    chains = len(posterior_data.chain)
    if colors == "cycle":
        colors = ["C{}".format(idx % 10) for idx in range(chains)]
    elif isinstance(colors, str):
        colors = [colors] * chains

    for ax, (var_name, selection, var_data) in zip(np.ravel(axes), plotters):
        ranks = scipy.stats.rankdata(var_data).reshape(var_data.shape)
        bin_ary = np.histogram_bin_edges(ranks, bins=bins, range=(0, ranks.size))
        all_counts = np.empty((len(ranks), len(bin_ary) - 1))
        for idx, row in enumerate(ranks):
            _, all_counts[idx], _ = histogram(row, bins=bin_ary)
        gap = all_counts.max() * 1.05
        width = bin_ary[1] - bin_ary[0]

        # Center the bins
        bin_ary = (bin_ary[1:] + bin_ary[:-1]) / 2

        y_ticks = []
        if kind == "bars":
            for idx, counts in enumerate(all_counts):
                y_ticks.append(idx * gap)
                ax.bar(
                    bin_ary,
                    counts,
                    bottom=y_ticks[-1],
                    width=width,
                    align="center",
                    color=colors[idx],
                    edgecolor=ax.get_facecolor(),
                )
                if ref_line:
                    ax.axhline(y=y_ticks[-1] + counts.mean(), linestyle="--", color="k")
            if labels:
                ax.set_ylabel("Chain", fontsize=ax_labelsize)
        elif kind == "vlines":
            ymin = np.full(len(all_counts), all_counts.mean())
            for idx, counts in enumerate(all_counts):
                ax.plot(bin_ary, counts, "o", color=colors[idx])
                ax.vlines(bin_ary, ymin, counts, lw=2, color=colors[idx])
            ax.set_ylim(0, all_counts.mean() * 2)
            if ref_line:
                ax.axhline(y=all_counts.mean(), linestyle="--", color="k")

        if labels:
            ax.set_xlabel("Rank (all chains)", fontsize=ax_labelsize)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(np.arange(len(y_ticks)))
            ax.set_title(make_label(var_name, selection), fontsize=titlesize)
        else:
            ax.set_yticks([])

    return axes
