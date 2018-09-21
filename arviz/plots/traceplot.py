"""Plot histogram and values from MCMC samples."""
import matplotlib.pyplot as plt
import numpy as np

from ..data import convert_to_dataset
from .kdeplot import plot_kde
from .plot_utils import _scale_text, get_bins, xarray_var_iter, make_label, get_coords


def plot_trace(data, var_names=None, coords=None, figsize=None, textsize=None, lines=None,
               combined=False, kde_kwargs=None, hist_kwargs=None, trace_kwargs=None):
    """Plot samples histograms and values.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, two variables are required.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : figure size tuple
        If None, size is (8, 8)
    textsize: int
        Text size for labels
    lines : tuple
        Tuple of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    combined : bool
        Flag for combining multiple chains into a single line. If False (default), chains will be
        plotted separately.
    kde_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_kde`. Only affects continuous variables.
    hist_kwargs : dict
        Extra keyword arguments passed to `plt.hist`. Only affects discrete variables.
    trace_kwargs : dict
        Extra keyword arguments passed to `plt.plot`
    Returns
    -------
    axes : matplotlib axes


    Examples
    --------
    Plot a subset variables

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('non_centered_eight')
        >>> coords = {'theta_t_dim_0': [0, 1], 'school':['Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords)

    Combine all chains into one distribution and trace

    .. plot::
        :context: close-figs

        >>> coords = {'theta_t_dim_0': [0, 1], 'school':['Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, combined=True)


    Plot reference lines against distribution and trace

    .. plot::
        :context: close-figs

        >>> lines = (('theta_t',{'theta_t_dim_0':0}, [-1]),)
        >>> coords = {'theta_t_dim_0': [0, 1], 'school':['Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, lines=lines)
    """
    data = convert_to_dataset(data, group='posterior')

    if coords is None:
        coords = {}

    if lines is None:
        lines = ()

    plotters = list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))

    if figsize is None:
        figsize = (12, len(plotters) * 2)

    if trace_kwargs is None:
        trace_kwargs = {}

    trace_kwargs.setdefault('alpha', 0.35)

    if kde_kwargs is None:
        kde_kwargs = {}

    if hist_kwargs is None:
        hist_kwargs = {}

    hist_kwargs.setdefault('alpha', 0.35)

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize, scale_ratio=1)
    trace_kwargs.setdefault('linewidth', linewidth)

    _, axes = plt.subplots(len(plotters), 2, squeeze=False, figsize=figsize)

    for i, (var_name, selection, value) in enumerate(plotters):
        if combined:
            value = value.flatten()
        value = np.atleast_2d(value)
        colors = []
        for row in value:
            axes[i, 1].plot(np.arange(len(row)), row, **trace_kwargs)
            colors.append(axes[i, 1].get_lines()[-1].get_color())
            kde_kwargs.setdefault('plot_kwargs', {})
            kde_kwargs['plot_kwargs']['color'] = colors[-1]
            if row.dtype.kind == 'i':
                _histplot_op(axes[i, 0], row, **hist_kwargs)
            else:
                plot_kde(row, textsize=textsize, ax=axes[i, 0], **kde_kwargs)

        axes[i, 0].set_yticks([])
        for idx in (0, 1):
            axes[i, idx].set_title(make_label(var_name, selection), fontsize=textsize)
            axes[i, idx].tick_params(labelsize=textsize)

        for _, _, vlines in (j for j in lines if j[0] == var_name and j[1] == selection):
            if isinstance(vlines, (float, int)):
                line_values = [vlines]
            else:
                line_values = np.atleast_1d(vlines).ravel()
            axes[i, 0].vlines(line_values, *axes[i, 0].get_ylim(), colors=colors,
                              linewidth=1.5, alpha=0.75)
            axes[i, 1].hlines(line_values, *axes[i, 1].get_xlim(), colors=colors,
                              linewidth=1.5, alpha=trace_kwargs['alpha'])
        axes[i, 0].set_ylim(ymin=0)
    plt.tight_layout()
    return axes


def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align='left', density=True, **kwargs)
    xticks = get_bins(data, max_bins=10, fenceposts=1)
    ax.set_xticks(xticks)
    return ax
