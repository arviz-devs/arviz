import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from .kdeplot import kdeplot
from ..utils import trace_to_dataframe
from .plot_utils import _scale_text


def jointplot(trace, varnames=None, figsize=None, text_size=None, hexbin=False, gridsize='auto', skip_first=0,
              joint_kwargs=None, marginal_kwargs=None):
    """
    Plot a scatter or hexbin of two variables with their respective marginals distributions.

    Parameters
    ----------

    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : figure size tuple
        If None, size is (8, 8)
    text_size: int
        Text size for labels
    hexbin : Boolean
        If True draws an hexbin plot
    gridsize : int or (int, int), optional.
        Only works when hexbin is True.
        The number of hexagons in the x-direction. The corresponding number of hexagons in the
        y-direction is chosen such that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    skip_first : int
        Number of first samples not shown in plots (burn-in)
    """
    trace = trace_to_dataframe(trace, combined=True)[skip_first:]

    if figsize is None:
        figsize = (8, 8)

    if text_size is None:
        text_size = _scale_text(figsize, text_size=text_size)

    if len(varnames) > 2:
        raise Exception('Number of variables to be plotted must 2')

    if joint_kwargs is None:
        joint_kwargs = {}

    if marginal_kwargs is None:
        marginal_kwargs = {}

    plt.figure(figsize=figsize)

    axjoin, axHistx, axHisty = _define_axes()

    x_var_name = varnames[0]
    y_var_name = varnames[1]

    x = trace[x_var_name].values
    y = trace[y_var_name].values

    axjoin.set_xlabel(x_var_name)
    axjoin.set_ylabel(y_var_name)

    if hexbin:
        if gridsize == 'auto':
            gridsize = int(len(trace)**0.35)
            print(gridsize)
        axjoin.hexbin(x, y, mincnt=1, gridsize=gridsize, **joint_kwargs)
        axjoin.grid(False)
    else:
        axjoin.scatter(x, y, **joint_kwargs)

    if x.dtype.kind == 'i':
        bins = range(x.min(), x.max() + 2)
        axHistx.hist(x, bins=bins, **marginal_kwargs)
    else:
        kdeplot(x, ax=axHistx, **marginal_kwargs)
    if y.dtype.kind == 'i':
        bins = range(y.min(), y.max() + 2)
        axHisty.hist(y, bins=bins, orientation='horizontal', **marginal_kwargs)
    else:
        kdeplot(y, ax=axHisty, rotated=True, **marginal_kwargs)

    axHistx.set_xlim(axjoin.get_xlim())
    axHisty.set_ylim(axjoin.get_ylim())


def _define_axes():
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_join = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axjoin = plt.axes(rect_join)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(NullFormatter())
    axHisty.yaxis.set_major_formatter(NullFormatter())
    axHistx.set_yticks([])
    axHisty.set_xticks([])

    return axjoin, axHistx, axHisty
