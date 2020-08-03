"""Histograms of ranked posterior draws, plotted for each chain."""
from itertools import cycle
import matplotlib.pyplot as plt

from ..data import convert_to_dataset
from .plot_utils import (
    xarray_var_iter,
    default_grid,
    filter_plotters_list,
    get_plotting_function,
)
from ..rcparams import rcParams
from ..utils import _var_names
from ..numeric_utils import _sturges_formula


def plot_rank(
    data,
    var_names=None,
    filter_vars=None,
    transform=None,
    coords=None,
    bins=None,
    kind="bars",
    colors="cycle",
    ref_line=True,
    labels=True,
    figsize=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """Plot rank order statistics of chains.

    From the paper: Rank plots are histograms of the ranked posterior draws (ranked over all
    chains) plotted separately for each chain.
    If all of the chains are targeting the same posterior, we expect the ranks in each chain to be
    uniform, whereas if one chain has a different location or scale parameter, this will be
    reflected in the deviation from uniformity. If rank plots of all chains look similar, this
    indicates good mixing of the chains.

    This plot was introduced by Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter,
    Paul-Christian Burkner (2019): Rank-normalization, folding, and localization: An improved R-hat
    for assessing convergence of MCMC. arXiv preprint https://arxiv.org/abs/1903.08008


    Parameters
    ----------
    data: obj
        Any object that can be converted to an az.InferenceData object. Refer to documentation of
        az.convert_to_dataset for details
    var_names: string or list of variable names
        Variables to be plotted. Prefix the variables by `~` when you want to exclude
        them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    transform: callable
        Function to transform data (defaults to None i.e.the identity function)
    coords: mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    bins: None or passed to np.histogram
        Binning strategy used for histogram. By default uses twice the result of Sturges' formula.
        See `np.histogram` documenation for, other available arguments.
    kind: string
        If bars (defaults), ranks are represented as stacked histograms (one per chain). If vlines
        ranks are represented as vertical lines above or below `ref_line`.
    colors: string or list of strings
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically choose a color per model from matplotlib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to `cycle`.
    ref_line: boolean
        Whether to include a dashed line showing where a uniform distribution would lie
    labels: bool
        wheter to plot or not the x and y labels, defaults to True
    figsize: tuple
        Figure size. If None it will be defined automatically.
    ax: numpy array-like of matplotlib axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show: bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

    Examples
    --------
    Show a default rank plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_rank(data)

    Recreate Figure 13 from the arxiv preprint

    .. plot::
        :context: close-figs

        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_rank(data, var_names='tau')

    Use vlines to compare results for centered vs noncentered models

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> centered_data = az.load_arviz_data('centered_eight')
        >>> noncentered_data = az.load_arviz_data('non_centered_eight')
        >>> _, ax = plt.subplots(1, 2, figsize=(12, 3))
        >>> az.plot_rank(centered_data, var_names="mu", kind='vlines', ax=ax[0])
        >>> az.plot_rank(noncentered_data, var_names="mu", kind='vlines', ax=ax[1])

    """
    if transform is not None:
        data = transform(data)
    posterior_data = convert_to_dataset(data, group="posterior")
    if coords is not None:
        posterior_data = posterior_data.sel(**coords)
    var_names = _var_names(var_names, posterior_data, filter_vars)
    plotters = filter_plotters_list(
        list(xarray_var_iter(posterior_data, var_names=var_names, combined=True)), "plot_rank"
    )
    length_plotters = len(plotters)

    if bins is None:
        bins = _sturges_formula(posterior_data, mult=2)

    rows, cols = default_grid(length_plotters)

    chains = len(posterior_data.chain)
    if colors == "cycle":
        colors = [
            prop
            for _, prop in zip(
                range(chains), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            )
        ]
    elif isinstance(colors, str):
        colors = [colors] * chains

    rankplot_kwargs = dict(
        axes=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        plotters=plotters,
        bins=bins,
        kind=kind,
        colors=colors,
        ref_line=ref_line,
        labels=labels,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_rank", "rankplot", backend)
    axes = plot(**rankplot_kwargs)
    return axes
