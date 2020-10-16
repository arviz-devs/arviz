import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arviz.plots.plot_utils import xarray_var_iter, default_grid, _scale_fig_size, make_label
from arviz.data import convert_to_dataset
from arviz.plots.backends.matplotlib import create_axes_grid
from arviz.utils import _var_names


def geweke_like(data, var_names=None, splits=10, round_to=2):
    r"""Compute z-scores for convergence diagnostics.

    Concatenates all chains and split them in equal size portions. Them compare them pairwise by computing the
    difference of the mean divided by their pooled variances. This is esentially a Welch's t statistic.
    The computed z_scores are expected to be distributed as a standard normal distribution.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list
        Names of variables to include. Prefix the variables by `~` when you
        want to exclude them from the analysis: `["~beta"]` instead of `["beta"]` (see
        examples below).
    splits : int:
        Number of portions to split the concatenated chains. It must lead to portions of the same size.
    round_to: int
        Number of decimals used to round results. Defaults to 2. Use "none" to return raw numbers.

    Returns
    -------
    pandas.DataFrame
        Return value will contain summary statistics for each variable. The summaries are the mean of the z_scores,
        the standard deviation, and the proportion of z_scores in absolute value larger than 2.
    """

    posterior_data = convert_to_dataset(data, group="posterior")
    iterator = list(xarray_var_iter(
        posterior_data, var_names=var_names, combined=True))
    #var_names = _var_names(var_names, posterior_data, filter_vars)
    summary = {}
    for var_name, selection, var_data in iterator:
        z_scores = _geweke_like(var_data, splits=splits)
        v_name = make_label(var_name, selection, position="beside")
        summary[v_name] = (z_scores.mean(),  z_scores.std(),
                           np.mean(np.abs(z_scores) > 2))
    df_summary = pd.DataFrame.from_dict(
        summary, orient="index", columns=["mean", "std", ">|2|"])
    return df_summary.round(round_to)


def _geweke_like(ary, splits=10):
    ary_flat = np.ravel(ary)
    ary_split = np.array(np.split(ary_flat, splits))
    n_ary_s = len(ary_flat) / splits
    ary_means = ary_split.mean(1)
    ary_vars = ary_split.var(axis=1, ddof=1) / n_ary_s

    z_scores = np.zeros((splits**2-splits)//2)
    idx = 0
    for i in range(splits):
        for j in range(i+1, splits):
            z_scores[idx] = (ary_means[i] - ary_means[j]) / (ary_vars[i] + ary_vars[j])**0.5
            idx += 1
    return z_scores


def plot_geweke_like(data, var_names=None, filter_vars=None, splits=10, kind="scatter",
                     figsize=None, axes=None, backend_kwargs=None):
    """Compute and plot z-scores for convergence diagnostics.

    Concatenates all chains and split them in equal size portions. Then compare them pairwise by computing the
    difference of the mean divided by their pooled variances. This is esentially a Welch's t statistic.
    The computed z_scores are expected to be distributed as a standard normal distribution.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list
        Names of variables to include. Prefix the variables by `~` when you
        want to exclude them from the analysis: `["~beta"]` instead of `["beta"]` (see
        examples below).
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    splits : int:
        Number of portions to split the concatenated chains. It must lead to portions of the same size.
    kind : str:
        Available options are `scatter` or `forest`.
    figsize: tuple
        Figure size. If None it will be defined automatically.
    ax: numpy array-like of matplotlib axes or bokeh figures, optional
         If not supplied, Arviz will create its own array of plot areas (and return it).
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    """
    posterior_data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, posterior_data, filter_vars)

    if backend_kwargs is None:
        backend_kwargs = {}

    plotters = list(xarray_var_iter(
        posterior_data, var_names=var_names, combined=True))
    length_plotters = len(plotters)

    if kind == "scatter":

        rows, cols = default_grid(length_plotters)
        figsize, ax_labelsize, titlesize, xt_labelsize, _, _ = _scale_fig_size(
            figsize, None, rows=rows, cols=cols)

        backend_kwargs.setdefault("figsize", figsize)
        backend_kwargs.setdefault("sharex", True)
        backend_kwargs.setdefault("sharey", True)

        if axes is None:
            _, axes = create_axes_grid(
                length_plotters,
                rows,
                cols,
                backend_kwargs=backend_kwargs,
            )

        for ax, (var_name, selection, var_data) in zip(np.ravel(axes), plotters):
            z_scores = _geweke_like(var_data, splits=splits)
            ax.plot(z_scores, 'o')
            ax.axhline(-2, color='k', ls='--')
            ax.axhline(2, color='k', ls='--')

            ax.set_title(make_label(var_name, selection),
                         fontsize=titlesize*1.5, wrap=True)
            ax.tick_params(labelsize=xt_labelsize*1.5)

    elif kind == "forest":

        figsize, ax_labelsize, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, None,
                                                                                       rows=length_plotters,
                                                                                       cols=1)
        figsize = (figsize[0], figsize[1]/9)
        if axes is None:
            fig, axes = plt.subplots(
                length_plotters, 1, figsize=figsize, sharex=True)
            axes = np.ravel(axes)
        backend_kwargs.setdefault("squeeze", True)

        fig.set_constrained_layout(False)
        fig.subplots_adjust(hspace=0)
        for ax, (var_name, selection, var_data) in zip(axes, plotters):
            z_scores = _geweke_like(var_data, splits=splits)
            quant = np.quantile(z_scores, (.05, .16, .5, .84, .95))
            ax.plot((quant[1], quant[3]), [0, 0],  lw=linewidth *
                    4, color="C0", solid_capstyle="round")
            ax.plot((quant[0], quant[4]), [0, 0],  lw=linewidth *
                    1, color="C0", solid_capstyle="round")
            ax.plot(quant[2], 0, 'ko')
            for i in [-2, -1, 1, 2]:
                ax.axvline(i, ls="--", color="k")
            x_label = make_label(var_name, selection, "beside")
            ax.set_ylabel(x_label, rotation=0, labelpad=40 +
                          len(x_label)*2, fontsize=ax_labelsize*0.9)
            ax.set_yticks([])
            if ax != axes[0]:
                ax.spines['top'].set_visible(False)

    return axes