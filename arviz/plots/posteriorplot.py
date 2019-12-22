"""Plot posterior densities."""
from typing import Optional

from ..data import convert_to_dataset
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    get_coords,
    filter_plotters_list,
    get_plotting_function,
)
from ..utils import _var_names


def plot_posterior(
    data,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    credible_interval=0.94,
    multimodal=False,
    round_to: Optional[int] = None,
    point_estimate="mean",
    group="posterior",
    rope=None,
    ref_val=None,
    kind="kde",
    bw=4.5,
    bins=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs
):
    """Plot Posterior densities in the style of John K. Kruschke's book.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, two variables are required.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    credible_interval : float, optional
        Credible intervals. Defaults to 0.94. Use None to hide the credible interval
    multimodal : bool
        If true (default) it may compute more than one credible interval if the distribution is
        multimodal and the modes are well separated.
    round_to : int, optional
        Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
    point_estimate: str
        Must be in ('mode', 'mean', 'median', None)
    group : str, optional
        Specifies which InferenceData group should be plotted. Defaults to ‘posterior’.
    rope: tuple or dictionary of tuples
        Lower and upper values of the Region Of Practical Equivalence. If a list is provided, its
        length should match the number of variables.
    ref_val: float or dictionary of floats
        display the percentage below and above the values in ref_val. Must be None (default),
        a constant, a list or a dictionary like see an example below. If a list is provided, its
        length should match the number of variables.
    kind: str
        Type of plot to display (kde or hist) For discrete variables this argument is ignored and
        a histogram is always used.
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kind == kde`.
    bins : integer or sequence or 'auto', optional
        Controls the number of bins, accepts the same keywords `matplotlib.hist()` does. Only works
        if `kind == hist`. If None (default) it will use `auto` for continuous variables and
        `range(xmin, xmax + 1)` for discrete variables.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    Examples
    --------
    Show a default kernel density plot following style of John Kruschke

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_posterior(data)

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'])

    Plot Region of Practical Equivalence (rope) for all distributions

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], rope=(-1, 1))

    Plot Region of Practical Equivalence for selected distributions

    .. plot::
        :context: close-figs

        >>> rope = {'mu': [{'rope': (-2, 2)}], 'theta': [{'school': 'Choate', 'rope': (2, 4)}]}
        >>> az.plot_posterior(data, var_names=['mu', 'theta'], rope=rope)


    Add reference lines

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], ref_val=0)

    Show point estimate of distribution

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], point_estimate='mode')

    Show reference values using variable names and coordinates

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, ref_val= {"theta": [{"school": "Deerfield", "ref_val": 4},
        ...                                             {"school": "Choate", "ref_val": 3}]})

    Show reference values using a list

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, ref_val=[1] + [5] * 8 + [1])


    Plot posterior as a histogram

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'], kind='hist')

    Change size of credible interval

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'], credible_interval=.75)
    """
    data = convert_to_dataset(data, group=group)
    var_names = _var_names(var_names, data)

    if coords is None:
        coords = {}

    plotters = filter_plotters_list(
        list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True)),
        "plot_posterior",
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    kwargs.setdefault("linewidth", _linewidth)

    posteriorplot_kwargs = dict(
        ax=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        plotters=plotters,
        bw=bw,
        bins=bins,
        kind=kind,
        point_estimate=point_estimate,
        round_to=round_to,
        credible_interval=credible_interval,
        multimodal=multimodal,
        ref_val=ref_val,
        rope=rope,
        ax_labelsize=ax_labelsize,
        xt_labelsize=xt_labelsize,
        kwargs=kwargs,
        titlesize=titlesize,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":

        posteriorplot_kwargs.pop("xt_labelsize")
        posteriorplot_kwargs.pop("titlesize")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_posterior", "posteriorplot", backend)
    ax = plot(**posteriorplot_kwargs)
    return ax
