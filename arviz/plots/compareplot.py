"""Summary plot for model comparison."""
import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import _scale_text


def plot_compare(comp_df, insample_dev=True, plot_standard_error=True, plot_ic_diff=True,
                 figsize=None, textsize=None, plot_kwargs=None, ax=None):
    """Summary plot for model comparison.

    This plot is in the style of the one used in the book Statistical Rethinking
    by Richard McElreath.

    Parameters
    ----------
    comp_df: DataFrame
        the result of the `compare()` function
    insample_dev : bool, optional
        plot the in-sample deviance, that is the value of the IC without the penalization given by
        the effective number of parameters (pIC). Defaults to True
    plot_standard_error : bool, optional
        plot the standard error of the IC estimate. Defaults to True
    plot_ic_diff : bool, optional
        plot standard error of the difference in IC between each model and the top-ranked model.
        Defaults to True
    figsize : tuple, optional
        If None, size is (6, num of models) inches
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    plot_kwargs : dict, optional
        Optional arguments for plot elements. Currently accepts 'color_ic',
        'marker_ic', 'color_insample_dev', 'marker_insample_dev', 'color_dse',
        'marker_dse', 'ls_min_ic' 'color_ls_min_ic',  'fontsize'
    ax : axes, optional
        Matplotlib axes

    Returns
    -------
    ax : matplotlib axes
    """
    if figsize is None:
        figsize = (6, len(comp_df))

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if plot_kwargs is None:
        plot_kwargs = {}

    yticks_pos, step = np.linspace(0, -1, (comp_df.shape[0] * 2) - 1, retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] + step / 2

    yticks_labels = [''] * len(yticks_pos)

    information_criterion = 'waic'
    if information_criterion not in comp_df.columns:
        information_criterion = 'loo'

    if plot_ic_diff:
        yticks_labels[0] = comp_df.index[0]
        yticks_labels[2::2] = comp_df.index[1:]
        ax.set_yticks(yticks_pos)
        ax.errorbar(x=comp_df[information_criterion].iloc[1:],
                    y=yticks_pos[1::2],
                    xerr=comp_df.dse[1:],
                    color=plot_kwargs.get('color_dse', 'grey'),
                    fmt=plot_kwargs.get('marker_dse', '^'),
                    mew=linewidth,
                    elinewidth=linewidth)

    else:
        yticks_labels = comp_df.index
        ax.set_yticks(yticks_pos[::2])

    if plot_standard_error:
        ax.errorbar(x=comp_df[information_criterion],
                    y=yticks_pos[::2],
                    xerr=comp_df.se,
                    color=plot_kwargs.get('color_ic', 'k'),
                    fmt=plot_kwargs.get('marker_ic', 'o'),
                    mfc='None',
                    mew=linewidth,
                    lw=linewidth)
    else:
        ax.plot(comp_df[information_criterion],
                yticks_pos[::2],
                color=plot_kwargs.get('color_ic', 'k'),
                marker=plot_kwargs.get('marker_ic', 'o'),
                mfc='None',
                mew=linewidth,
                lw=0)

    if insample_dev:
        ax.plot(comp_df[information_criterion] - (2 * comp_df['p'+information_criterion]),
                yticks_pos[::2],
                color=plot_kwargs.get('color_insample_dev', 'k'),
                marker=plot_kwargs.get('marker_insample_dev', 'o'),
                mew=linewidth,
                lw=0)

    ax.axvline(comp_df[information_criterion].iloc[0],
               ls=plot_kwargs.get('ls_min_ic', '--'),
               color=plot_kwargs.get('color_ls_min_ic', 'grey'), lw=linewidth)

    ax.set_xlabel('Deviance', fontsize=textsize)
    ax.set_yticklabels(yticks_labels)
    ax.set_ylim(-1 + step, 0 - step)
    ax.tick_params(labelsize=textsize)

    return ax
