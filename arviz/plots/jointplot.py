import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from .kdeplot import kdeplot
from ..utils import convert_to_xarray
from .plot_utils import _scale_text, get_bins, xarray_var_iter, make_label


def jointplot(data, var_names=None, coords=None, figsize=None, textsize=None, kind='scatter',
              gridsize='auto', joint_kwargs=None, marginal_kwargs=None):
    """
    Plot a scatter or hexbin of two variables with their respective marginals distributions.

    Parameters
    ----------

    data : xarray, or object that can be converted (pystan or pymc3 draws)
        Posterior samples
    var_names : list of variable names
        Variables to be plotted, two variables are required.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : figure size tuple
        If None, size is (8, 8)
    textsize: int
        Text size for labels
    kind : str
        Type of plot to display (scatter of hexbin)
    gridsize : int or (int, int), optional.
        The number of hexagons in the x-direction. Ignored when hexbin is False. See `plt.hexbin`
        for details
    joint_shade : dicts, optional
        Additional keywords modifying the join distribution (central subplot)
    marginal_shade : dicts, optional
        Additional keywords modifying the marginals distributions (top and right subplot)
        (to control the shade)
    Returns
    -------
    axjoin : matplotlib axes, join (central) distribution
    ax_hist_x : matplotlib axes, x (top) distribution
    ax_hist_y : matplotlib axes, y (right) distribution
    """

    data = convert_to_xarray(data)
    if coords is None:
        coords = {}

    plotters = list(xarray_var_iter(data.sel(**coords), var_names=var_names, combined=True))

    if len(plotters) != 2:
        raise Exception(f'Number of variables to be plotted must 2 (you supplied {len(plotters)})')

    if figsize is None:
        figsize = (6, 6)

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize)

    if joint_kwargs is None:
        joint_kwargs = {}

    if marginal_kwargs is None:
        marginal_kwargs = {}

    plt.figure(figsize=figsize)

    axjoin, ax_hist_x, ax_hist_y = _define_axes()

    x_var_name = make_label(*plotters[0][:2])
    y_var_name = make_label(*plotters[1][:2])

    x = plotters[0][2].flatten()
    y = plotters[1][2].flatten()

    axjoin.set_xlabel(x_var_name, fontsize=textsize)
    axjoin.set_ylabel(y_var_name, fontsize=textsize)
    axjoin.tick_params(labelsize=textsize)

    if kind == 'scatter':
        axjoin.scatter(x, y, **joint_kwargs)
    elif kind == 'hexbin':
        if gridsize == 'auto':
            gridsize = int(len(x)**0.35)
        axjoin.hexbin(x, y, mincnt=1, gridsize=gridsize, **joint_kwargs)
        axjoin.grid(False)
    else:
        raise ValueError(f'Plot type {kind} not recognized.')

    for val, ax, orient, rotate in ((x, ax_hist_x, 'vertical', False),
                                    (y, ax_hist_y, 'horizontal', True)):
        if val.dtype.kind == 'i':
            bins = get_bins(val)
            ax.hist(val, bins=bins, align='left', density=True,
                    orientation=orient, **marginal_kwargs)
        else:
            marginal_kwargs.setdefault('plot_kwargs', {})
            marginal_kwargs['plot_kwargs']['linewidth'] = linewidth
            kdeplot(val, rotated=rotate, ax=ax, **marginal_kwargs)

    ax_hist_x.set_xlim(axjoin.get_xlim())
    ax_hist_y.set_ylim(axjoin.get_ylim())

    return axjoin, ax_hist_x, ax_hist_y


def _define_axes():
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_join = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axjoin = plt.axes(rect_join)
    ax_hist_x = plt.axes(rect_histx)
    ax_hist_y = plt.axes(rect_histy)

    ax_hist_x.xaxis.set_major_formatter(NullFormatter())
    ax_hist_y.yaxis.set_major_formatter(NullFormatter())
    ax_hist_x.set_yticks([])
    ax_hist_y.set_xticks([])

    return axjoin, ax_hist_x, ax_hist_y
