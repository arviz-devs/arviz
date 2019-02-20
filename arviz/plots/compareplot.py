"""Summary plot for model comparison."""
import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import _scale_fig_size


def plot_compare(
    comp_df,
    insample_dev=True,
    plot_standard_error=True,
    plot_ic_diff=True,
    figsize=None,
    textsize=None,
    plot_kwargs=None,
    ax=None,
):
    """
    Summary plot for model comparison.

    This plot is in the style of the one used in the book Statistical Rethinking (Chapter 6)
    by Richard McElreath.

    Notes
    -----
    Defaults to comparing Widely Accepted Information Criterion (WAIC) if present in comp_df column,
    otherwise compares Leave-one-out (loo)


    Parameters
    ----------
    comp_df: pd.DataFrame
        Result of the `az.compare()` method
    insample_dev : bool, optional
        Plot in-sample deviance, that is the value of the information criteria without the
        penalization given by the effective number of parameters (pIC). Defaults to True
    plot_standard_error : bool, optional
        Plot the standard error of the information criteria estimate. Defaults to True
    plot_ic_diff : bool, optional
        Plot standard error of the difference in information criteria between each model
         and the top-ranked model. Defaults to True
    figsize : tuple, optional
        If None, size is (6, num of models) inches
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    plot_kwargs : dict, optional
        Optional arguments for plot elements. Currently accepts 'color_ic',
        'marker_ic', 'color_insample_dev', 'marker_insample_dev', 'color_dse',
        'marker_dse', 'ls_min_ic' 'color_ls_min_ic',  'fontsize'
    ax : axes, optional
        Matplotlib axes

    Returns
    -------
    ax : matplotlib axes


    Examples
    --------
    Show default compare plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> model_compare = az.compare({'Centered 8 schools': az.load_arviz_data('centered_eight'),
        >>>                  'Non-centered 8 schools': az.load_arviz_data('non_centered_eight')})
        >>> az.plot_compare(model_compare)

    Plot standard error and information criteria difference only

    .. plot::
        :context: close-figs

        >>> az.plot_compare(model_compare, insample_dev=False)

    """
    if figsize is None:
        figsize = (6, len(comp_df))

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize, 1, 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if plot_kwargs is None:
        plot_kwargs = {}

    yticks_pos, step = np.linspace(0, -1, (comp_df.shape[0] * 2) - 1, retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] + step / 2

    yticks_labels = [""] * len(yticks_pos)

    _information_criterion = ["waic", "loo"]
    for information_criterion in _information_criterion:
        if information_criterion in comp_df.columns:
            break
    else:
        raise ValueError(
            "comp_df must contain one of the following"
            " information criterion: {}".format(_information_criterion)
        )

    if plot_ic_diff:
        yticks_labels[0] = comp_df.index[0]
        yticks_labels[2::2] = comp_df.index[1:]
        ax.set_yticks(yticks_pos)
        ax.errorbar(
            x=comp_df[information_criterion].iloc[1:],
            y=yticks_pos[1::2],
            xerr=comp_df.dse[1:],
            color=plot_kwargs.get("color_dse", "grey"),
            fmt=plot_kwargs.get("marker_dse", "^"),
            mew=linewidth,
            elinewidth=linewidth,
        )

    else:
        yticks_labels = comp_df.index
        ax.set_yticks(yticks_pos[::2])

    if plot_standard_error:
        ax.errorbar(
            x=comp_df[information_criterion],
            y=yticks_pos[::2],
            xerr=comp_df.se,
            color=plot_kwargs.get("color_ic", "k"),
            fmt=plot_kwargs.get("marker_ic", "o"),
            mfc="None",
            mew=linewidth,
            lw=linewidth,
        )
    else:
        ax.plot(
            comp_df[information_criterion],
            yticks_pos[::2],
            color=plot_kwargs.get("color_ic", "k"),
            marker=plot_kwargs.get("marker_ic", "o"),
            mfc="None",
            mew=linewidth,
            lw=0,
        )

    if insample_dev:
        ax.plot(
            comp_df[information_criterion] - (2 * comp_df["p_" + information_criterion]),
            yticks_pos[::2],
            color=plot_kwargs.get("color_insample_dev", "k"),
            marker=plot_kwargs.get("marker_insample_dev", "o"),
            mew=linewidth,
            lw=0,
        )

    ax.axvline(
        comp_df[information_criterion].iloc[0],
        ls=plot_kwargs.get("ls_min_ic", "--"),
        color=plot_kwargs.get("color_ls_min_ic", "grey"),
        lw=linewidth,
    )

    scale_col = information_criterion + "_scale"
    if scale_col in comp_df:
        scale = comp_df[scale_col].iloc[0].capitalize()
    else:
        scale = "Deviance"
    ax.set_xlabel(scale, fontsize=ax_labelsize)
    ax.set_yticklabels(yticks_labels)
    ax.set_ylim(-1 + step, 0 - step)
    ax.tick_params(labelsize=xt_labelsize)

    return ax
