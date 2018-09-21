"""Joint scatter plot of two variables."""
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from ..data import convert_to_dataset
from .kdeplot import plot_kde
from .plot_utils import _scale_text, get_bins, xarray_var_iter, make_label, get_coords


def plot_joint(data, var_names=None, coords=None, figsize=None, textsize=None, kind='scatter',
               gridsize='auto', contour=True, fill_last=True, joint_kwargs=None,
               marginal_kwargs=None):
    """
    Plot a scatter or hexbin of two variables with their respective marginals distributions.

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
    kind : str
        Type of plot to display (scatter, kde or hexbin)
    gridsize : int or (int, int), optional.
        The number of hexagons in the x-direction. Ignored when hexbin is False. See `plt.hexbin`
        for details
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    joint_kwargs : dicts, optional
        Additional keywords modifying the join distribution (central subplot)
    marginal_kwargs : dicts, optional
        Additional keywords modifying the marginals distributions (top and right subplot)

    Returns
    -------
    axjoin : matplotlib axes, join (central) distribution
    ax_hist_x : matplotlib axes, x (top) distribution
    ax_hist_y : matplotlib axes, y (right) distribution
    """
    valid_kinds = ['scatter', 'kde', 'hexbin']
    if kind not in valid_kinds:
        raise ValueError(('Plot type {} not recognized.'
                          'Plot type must be in {}').format(kind, valid_kinds))

    data = convert_to_dataset(data, group='posterior')
    if coords is None:
        coords = {}

    plotters = list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))

    if len(plotters) != 2:
        raise Exception(
            'Number of variables to be plotted must 2 (you supplied {})'.format(
                len(plotters)
            )
        )

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
    elif kind == 'kde':
        plot_kde(x, y, contour=contour, fill_last=fill_last, ax=axjoin, **joint_kwargs)
    else:
        if gridsize == 'auto':
            gridsize = int(len(x)**0.35)
        axjoin.hexbin(x, y, mincnt=1, gridsize=gridsize, **joint_kwargs)
        axjoin.grid(False)

    for val, ax, orient, rotate in ((x, ax_hist_x, 'vertical', False),
                                    (y, ax_hist_y, 'horizontal', True)):
        if val.dtype.kind == 'i':
            bins = get_bins(val)
            ax.hist(val, bins=bins, align='left', density=True,
                    orientation=orient, **marginal_kwargs)
        else:
            marginal_kwargs.setdefault('plot_kwargs', {})
            marginal_kwargs['plot_kwargs']['linewidth'] = linewidth
            plot_kde(val, rotated=rotate, ax=ax, **marginal_kwargs)

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
