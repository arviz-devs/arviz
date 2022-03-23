"""Parallel coordinates plot showing posterior points with and without divergences marked."""
import numpy as np
from scipy.stats import rankdata

from ..data import convert_to_dataset
from ..labels import BaseLabeller
from ..sel_utils import xarray_to_ndarray
from ..rcparams import rcParams
from ..stats.stats_utils import stats_variance_2d as svar
from ..utils import _numba_var, _var_names, get_coords
from .plot_utils import get_plotting_function


def plot_parallel(
    data,
    var_names=None,
    filter_vars=None,
    coords=None,
    figsize=None,
    textsize=None,
    legend=True,
    colornd="k",
    colord="C1",
    shadend=0.025,
    labeller=None,
    ax=None,
    norm_method=None,
    backend=None,
    backend_config=None,
    backend_kwargs=None,
    show=None,
):
    """
    Plot parallel coordinates plot showing posterior points with and without divergences.

    Described by https://arxiv.org/abs/1709.01449

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object
        refer to documentation of :func:`arviz.convert_to_dataset` for details
    var_names: list of variable names
        Variables to be plotted, if `None` all variables are plotted. Can be used to change the
        order of the plotted variables. Prefix the variables by ``~`` when you want to exclude
        them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        ``pandas.filter``.
    coords: mapping, optional
        Coordinates of ``var_names`` to be plotted.
        Passed to :meth:`xarray.Dataset.sel`.
    figsize: tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on ``figsize``.
    legend: bool
        Flag for plotting legend (defaults to True)
    colornd: valid matplotlib color
        color for non-divergent points. Defaults to 'k'
    colord: valid matplotlib color
        color for divergent points. Defaults to 'C1'
    shadend: float
        Alpha blending value for non-divergent points, between 0 (invisible) and 1 (opaque).
        Defaults to .025
    labeller : labeller instance, optional
        Class providing the method ``make_label_vert`` to generate the labels in the plot.
        Read the :ref:`label_guide` for more details and usage examples.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    norm_method: str
        Method for normalizing the data. Methods include normal, minmax and rank.
        Defaults to none.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_config: dict, optional
        Currently specifies the bounds to use for bokeh axes.
        Defaults to value set in ``rcParams``.
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or
        :func:`bokeh.plotting.figure`.
    show: bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

    See Also
    --------
    plot_pair : Plot a scatter, kde and/or hexbin matrix with (optional) marginals on the diagonal.
    plot_trace : Plot distribution (histogram or kernel density estimates) and sampled values
                 or rank plot

    Examples
    --------
    Plot default parallel plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_parallel(data, var_names=["mu", "tau"])


    Plot parallel plot with normalization

    .. plot::
        :context: close-figs

        >>> az.plot_parallel(data, var_names=["mu", "tau"], norm_method='normal')

    """
    if coords is None:
        coords = {}

    if labeller is None:
        labeller = BaseLabeller()

    # Get diverging draws and combine chains
    divergent_data = convert_to_dataset(data, group="sample_stats")
    _, diverging_mask = xarray_to_ndarray(
        divergent_data,
        var_names=("diverging",),
        combined=True,
    )
    diverging_mask = np.squeeze(diverging_mask)

    # Get posterior draws and combine chains
    posterior_data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, posterior_data, filter_vars)
    var_names, _posterior = xarray_to_ndarray(
        get_coords(posterior_data, coords),
        var_names=var_names,
        combined=True,
        label_fun=labeller.make_label_vert,
    )
    if len(var_names) < 2:
        raise ValueError("Number of variables to be plotted must be 2 or greater.")
    if norm_method is not None:
        if norm_method == "normal":
            mean = np.mean(_posterior, axis=1)
            if _posterior.ndim <= 2:
                standard_deviation = np.sqrt(_numba_var(svar, np.var, _posterior, axis=1))
            else:
                standard_deviation = np.std(_posterior, axis=1)
            for i in range(0, np.shape(mean)[0]):
                _posterior[i, :] = (_posterior[i, :] - mean[i]) / standard_deviation[i]
        elif norm_method == "minmax":
            min_elem = np.min(_posterior, axis=1)
            max_elem = np.max(_posterior, axis=1)
            for i in range(0, np.shape(min_elem)[0]):
                _posterior[i, :] = ((_posterior[i, :]) - min_elem[i]) / (max_elem[i] - min_elem[i])
        elif norm_method == "rank":
            _posterior = rankdata(_posterior, axis=1, method="average")
        else:
            raise ValueError(f"{norm_method} is not supported. Use normal, minmax or rank.")

    parallel_kwargs = dict(
        ax=ax,
        colornd=colornd,
        colord=colord,
        shadend=shadend,
        diverging_mask=diverging_mask,
        posterior=_posterior,
        textsize=textsize,
        var_names=var_names,
        legend=legend,
        figsize=figsize,
        backend_kwargs=backend_kwargs,
        backend_config=backend_config,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_parallel", "parallelplot", backend)
    ax = plot(**parallel_kwargs)

    return ax
