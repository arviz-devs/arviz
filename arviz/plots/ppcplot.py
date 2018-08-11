import numpy as np
from .kdeplot import kdeplot
from .plot_utils import _scale_text, _create_axes_grid, default_grid


def ppcplot(data, ppc_sample, kind='kde', mean=True, figsize=None, textsize=None, ax=None):
    """
    Plot for Posterior Predictive Checks

    Parameters
    ----------
    data : Array-like
        Observed values
    ppc_samples : dict
        Posterior predictive check samples
    kind : str
        Type of plot to display (kde or cumulative)
    mean : bool
        Whether or not to plot the mean ppc distribution. Defaults to True
    figsize : figure size tuple
        If None, size is (6, 5)
    textsize: int
        Text size for labels. If None it will be auto-scaled based on figsize.
    ax: axes
        Matplotlib axes

    Returns
    -------
    ax : matplotlib axes
    """

    rows, cols = default_grid(len(ppc_sample))

    if figsize is None:
        figsize = (7, 5)

    _, ax = _create_axes_grid(len(ppc_sample), rows, cols, figsize=figsize)

    textsize, linewidth, _ = _scale_text(figsize, textsize, 2)

    for ax_, (var, ppss) in zip(np.atleast_1d(ax), ppc_sample.items()):
        if kind == 'kde':
            kdeplot(data, label='{}'.format(var),
                    plot_kwargs={'color': 'k', 'linewidth': linewidth, 'zorder': 3},
                    fill_kwargs={'alpha': 0},
                    ax=ax_)
            for pps in ppss:
                kdeplot(pps,
                        plot_kwargs={'color': 'C5', 'linewidth': 0.5 * linewidth},
                        fill_kwargs={'alpha': 0},
                        ax=ax_)
            ax_.plot([], color='C5', label='{}_pps'.format(var))
            if mean:
                kdeplot(ppss,
                        plot_kwargs={'color': 'C0',
                                     'linestyle': '--',
                                     'linewidth': linewidth,
                                     'zorder': 2},
                        label='mean {}_pps'.format(var),
                        ax=ax_)
            ax_.set_xlabel(var, fontsize=textsize)
            ax_.set_yticks([])

        elif kind == 'cumulative':
            ax_.plot(*_ecdf(data), color='k', lw=linewidth, label='{}'.format(var), zorder=3)
            for pps in ppss:
                ax_.plot(*_ecdf(pps), alpha=0.2, color='C5', lw=linewidth)
            ax_.plot([], color='C5', label='{}_pps'.format(var))
            if mean:
                ax_.plot(*_ecdf(ppss.flatten()), color='C0', ls='--', lw=linewidth,
                         label='mean {}_pps'.format(var))
            ax_.set_xlabel(var, fontsize=textsize)
            ax_.set_yticks([0, 0.5, 1])
        ax_.legend(fontsize=textsize)
        ax_.tick_params(labelsize=textsize)

    return ax


def _ecdf(data):
    len_data = len(data)
    data_s = np.sort(data)
    cdf = np.arange(1, len_data + 1) / len_data
    return data_s, cdf
