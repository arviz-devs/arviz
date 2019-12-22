"""Parallel coordinates plot showing posterior points with and without divergences marked."""
import numpy as np
from scipy.stats.mstats import rankdata

from ..data import convert_to_dataset
from .plot_utils import _scale_fig_size, xarray_to_ndarray, get_coords, get_plotting_function
from ..utils import _var_names, _numba_var
from ..stats.stats_utils import stats_variance_2d as svar


def plot_parallel(
    data,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    legend=True,
    colornd="k",
    colord="C1",
    shadend=0.025,
    ax=None,
    norm_method=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Plot parallel coordinates plot showing posterior points with and without divergences.

    Described by https://arxiv.org/abs/1709.01449, suggested by Ari Hartikainen

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, if None all variable are plotted. Can be used to change the order
        of the plotted variables
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    legend : bool
        Flag for plotting legend (defaults to True)
    colornd : valid matplotlib color
        color for non-divergent points. Defaults to 'k'
    colord : valid matplotlib color
        color for divergent points. Defaults to 'C1'
    shadend : float
        Alpha blending value for non-divergent points, between 0 (invisible) and 1 (opaque).
        Defaults to .025
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    norm_method : str
        Method for normalizing the data. Methods include normal, minmax and rank.
        Defaults to none.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

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

    # Get diverging draws and combine chains
    divergent_data = convert_to_dataset(data, group="sample_stats")
    _, diverging_mask = xarray_to_ndarray(divergent_data, var_names=("diverging",), combined=True)
    diverging_mask = np.squeeze(diverging_mask)

    # Get posterior draws and combine chains
    posterior_data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, posterior_data)
    var_names, _posterior = xarray_to_ndarray(
        get_coords(posterior_data, coords), var_names=var_names, combined=True
    )
    if len(var_names) < 2:
        raise ValueError("This plot needs at least two variables")
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
            _posterior = rankdata(_posterior, axis=1)
        else:
            raise ValueError("{} is not supported. Use normal, minmax or rank.".format(norm_method))

    figsize, _, _, xt_labelsize, _, _ = _scale_fig_size(figsize, textsize, 1, 1)

    parallel_kwargs = dict(
        ax=ax,
        colornd=colornd,
        colord=colord,
        shadend=shadend,
        diverging_mask=diverging_mask,
        _posterior=_posterior,
        textsize=textsize,
        var_names=var_names,
        xt_labelsize=xt_labelsize,
        legend=legend,
        figsize=figsize,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":

        parallel_kwargs.pop("textsize")
        parallel_kwargs.pop("xt_labelsize")
        parallel_kwargs.pop("legend")
        parallel_kwargs.pop("colord")
        parallel_kwargs.pop("colornd")
        parallel_kwargs.pop("shadend")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_parallel", "parallelplot", backend)
    ax = plot(**parallel_kwargs)

    return ax
