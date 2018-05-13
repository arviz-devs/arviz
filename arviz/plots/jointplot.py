import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from .kdeplot import kdeplot
from ..utils import trace_to_dataframe
from .plot_utils import _scale_text, get_bins


def jointplot(trace, varnames=None, figsize=None, textsize=None, kind='scatter', gridsize='auto',
              skip_first=0, joint_kwargs=None, marginal_kwargs=None):
    """
    Plot a scatter or hexbin of two variables with their respective marginals distributions.

    Parameters
    ----------

    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, two variables are required.
    figsize : figure size tuple
        If None, size is (8, 8)
    textsize: int
        Text size for labels
    kind : str
        Type of plot to display (scatter of hexbin)
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
    joint_shade : dicts, optional
        Additional keywords modifying the join distribution (central subplot)
    marginal_shade : dicts, optional
        Additional keywords modifying the marginals distributions (top and right subplot)
        (to control the shade)
    Returns
    -------
    axjoin : matplotlib axes, join (central) distribution
    axHistx : matplotlib axes, x (top) distribution
    axHisty : matplotlib axes, y (right) distribution
    """
    trace = trace_to_dataframe(trace[skip_first:] , combined=True)

    if figsize is None:
        figsize = (6, 6)

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize)

    if len(varnames) != 2:
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

    axjoin.set_xlabel(x_var_name, fontsize=textsize)
    axjoin.set_ylabel(y_var_name, fontsize=textsize)
    axjoin.tick_params(labelsize=textsize)

    if kind == 'scatter':
        axjoin.scatter(x, y, **joint_kwargs)
    elif kind == 'hexbin':
        if gridsize == 'auto':
            gridsize = int(len(trace)**0.35)
        axjoin.hexbin(x, y, mincnt=1, gridsize=gridsize, **joint_kwargs)
        axjoin.grid(False)
    else:
        raise ValueError('Plot type {} not recognized.'.format(kind))

    if x.dtype.kind == 'i':
        bins = get_bins(x)
        axHistx.hist(x, bins=bins, align='left', density=True,
                     **marginal_kwargs)
    else:
        kdeplot(x, ax=axHistx, **marginal_kwargs)
    if y.dtype.kind == 'i':
        bins = get_bins(y)
        axHisty.hist(y, bins=bins, align='left', density=True, orientation='horizontal',
                     **marginal_kwargs)
    else:
        kdeplot(y, ax=axHisty, rotated=True, lw=linewidth, **marginal_kwargs)

    axHistx.set_xlim(axjoin.get_xlim())
    axHisty.set_ylim(axjoin.get_ylim())

    return axjoin, axHistx, axHisty

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
