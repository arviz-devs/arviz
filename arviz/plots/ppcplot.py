"""Posterior predictive plot."""
import numpy as np
from .kdeplot import kdeplot
from .plot_utils import _scale_text, _create_axes_grid, default_grid


def ppcplot(data, kind='kde', alpha=0.2, mean=True, figsize=None, textsize=None):
    """
    Plot for Posterior Predictive checks.

    Note that this plot will flatten out any dimensions in the posterior predictive variables.

    Parameters
    ----------
    data : Array-like
        Observed values
    kind : str
        Type of plot to display (kde or cumulative)
    alpha : float
        Opacity of posterior predictive density curves
    mean : bool
        Whether or not to plot the mean posterior predictive distribution. Defaults to True
    figsize : figure size tuple
        If None, size is (6, 5)
    textsize: int
        Text size for labels. If None it will be auto-scaled based on figsize.

    Returns
    -------
    axes : matplotlib axes
    """
    for group in ('posterior_predictive', 'observed_data'):
        if not hasattr(data, group):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=group))

    observed = data.observed_data
    posterior_predictive = data.posterior_predictive

    rows, cols = default_grid(len(observed.data_vars))
    if figsize is None:
        figsize = (7 * cols, 5 * rows)
    _, axes = _create_axes_grid(len(observed.data_vars), rows, cols, figsize=figsize)

    textsize, linewidth, _ = _scale_text(figsize, textsize)
    for ax, var_name in zip(np.atleast_1d(axes), observed.data_vars):
        if kind == 'kde':
            kdeplot(observed[var_name].values.flatten(), label='Observed {}'.format(var_name),
                    plot_kwargs={'color': 'k', 'linewidth': linewidth, 'zorder': 3},
                    fill_kwargs={'alpha': 0},
                    ax=ax)
            for _, chain_vals in posterior_predictive[var_name].groupby('chain'):
                for _, vals in chain_vals.groupby('draw'):
                    kdeplot(vals,
                            plot_kwargs={'color': 'C4',
                                         'alpha': alpha,
                                         'linewidth': 0.5 * linewidth},
                            fill_kwargs={'alpha': 0},
                            ax=ax)
            ax.plot([], color='C4', label='Posterior predictive {}'.format(var_name))
            if mean:
                kdeplot(posterior_predictive[var_name].values.flatten(),
                        plot_kwargs={'color': 'C0',
                                     'linestyle': '--',
                                     'linewidth': linewidth,
                                     'zorder': 2},
                        label='Posterior predictive mean {}'.format(var_name),
                        ax=ax)
            ax.set_xlabel(var_name, fontsize=textsize)
            ax.set_yticks([])

        elif kind == 'cumulative':
            ax.plot(*_empirical_cdf(observed[var_name].values.flatten()),
                    color='k',
                    linewidth=linewidth,
                    label='Observed {}'.format(var_name),
                    zorder=3)
            for _, chain_vals in posterior_predictive[var_name].groupby('chain'):
                for _, vals in chain_vals.groupby('draw'):
                    ax.plot(*_empirical_cdf(vals), alpha=alpha, color='C4', linewidth=linewidth)
            ax.plot([], color='C4', label='Posterior predictive {}'.format(var_name))
            if mean:
                ax.plot(*_empirical_cdf(posterior_predictive[var_name].values.flatten()),
                        color='C0',
                        linestyle='--',
                        linewidth=linewidth,
                        label='Posterior predictive mean {}'.format(var_name))
            ax.set_xlabel(var_name, fontsize=textsize)
            ax.set_yticks([0, 0.5, 1])
        ax.legend(fontsize=textsize)
    return axes


def _empirical_cdf(data):
    """Compute empirical cdf of a numpy array.

    Parameters
    ----------
    data : np.array
        1d array

    Returns
    -------
    np.array, np.array
        x and y coordinates for the empirical cdf of the data
    """
    return np.sort(data), np.linspace(0, 1, len(data))
