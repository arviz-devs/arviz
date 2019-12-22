"""Plot a scatter or hexbin of sampled parameters."""
import warnings
import numpy as np

from ..data import convert_to_dataset, convert_to_inference_data
from .plot_utils import xarray_to_ndarray, get_coords, get_plotting_function
from ..utils import _var_names


def plot_pair(
    data,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    kind="scatter",
    gridsize="auto",
    contour=True,
    fill_last=True,
    divergences=False,
    colorbar=False,
    ax=None,
    divergences_kwargs=None,
    plot_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Plot a scatter or hexbin matrix of the sampled parameters.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, if None all variable are plotted
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    kind : str
        Type of plot to display (scatter, kde or hexbin)
    gridsize : int or (int, int), optional
        Only works for kind=hexbin.
        The number of hexagons in the x-direction. The corresponding number of hexagons in the
        y-direction is chosen such that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    divergences : Boolean
        If True divergences will be plotted in a different color
    colorbar : bool
        If True a colorbar will be included as part of the plot (Defaults to False).
        Only works when kind=hexbin
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    divergences_kwargs : dicts, optional
        Additional keywords passed to ax.scatter for divergences
    plot_kwargs : dicts, optional
        Additional keywords passed to ax.plot, az.plot_kde or ax.hexbin
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
    KDE Pair Plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered = az.load_arviz_data('centered_eight')
        >>> coords = {'school': ['Choate', 'Deerfield']}
        >>> az.plot_pair(centered,
        >>>             var_names=['theta', 'mu', 'tau'],
        >>>             kind='kde',
        >>>             coords=coords,
        >>>             divergences=True,
        >>>             textsize=18)

    Hexbin pair plot

    .. plot::
        :context: close-figs

        >>> az.plot_pair(centered,
        >>>             var_names=['theta', 'mu'],
        >>>             coords=coords,
        >>>             textsize=18,
        >>>             kind='hexbin')

    Pair plot showing divergences

    .. plot::
        :context: close-figs

        >>> az.plot_pair(centered,
        ...             var_names=['theta', 'mu', 'tau'],
        ...             coords=coords,
        ...             divergences=True,
        ...             textsize=18)
    """
    valid_kinds = ["scatter", "kde", "hexbin"]
    if kind not in valid_kinds:
        raise ValueError(
            ("Plot type {} not recognized." "Plot type must be in {}").format(kind, valid_kinds)
        )

    if coords is None:
        coords = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    if kind == "scatter":
        plot_kwargs.setdefault("marker", ".")
        plot_kwargs.setdefault("lw", 0)

    if divergences_kwargs is None:
        divergences_kwargs = {}

    divergences_kwargs.setdefault("marker", "o")
    divergences_kwargs.setdefault("markeredgecolor", "k")
    divergences_kwargs.setdefault("color", "C1")
    divergences_kwargs.setdefault("lw", 0)

    # Get posterior draws and combine chains
    data = convert_to_inference_data(data)
    posterior_data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, posterior_data)
    flat_var_names, _posterior = xarray_to_ndarray(
        get_coords(posterior_data, coords), var_names=var_names, combined=True
    )

    divergent_data = None
    diverging_mask = None
    # Get diverging draws and combine chains
    if divergences:
        if hasattr(data, "sample_stats") and hasattr(data.sample_stats, "diverging"):
            divergent_data = convert_to_dataset(data, group="sample_stats")
            _, diverging_mask = xarray_to_ndarray(
                divergent_data, var_names=("diverging",), combined=True
            )
            diverging_mask = np.squeeze(diverging_mask)
        else:
            divergences = False
            warnings.warn(
                "Divergences data not found, plotting without divergences. "
                "Make sure the sample method provides divergences data and "
                "that it is present in the `diverging` field of `sample_stats` "
                "or set divergences=False",
                SyntaxWarning,
            )

    if gridsize == "auto":
        gridsize = int(len(_posterior[0]) ** 0.35)

    numvars = len(flat_var_names)

    if numvars < 2:
        raise Exception("Number of variables to be plotted must be 2 or greater.")

    pairplot_kwargs = dict(
        ax=ax,
        _posterior=_posterior,
        numvars=numvars,
        figsize=figsize,
        textsize=textsize,
        kind=kind,
        plot_kwargs=plot_kwargs,
        contour=contour,
        fill_last=fill_last,
        gridsize=gridsize,
        colorbar=colorbar,
        divergences=divergences,
        diverging_mask=diverging_mask,
        divergences_kwargs=divergences_kwargs,
        flat_var_names=flat_var_names,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":
        pairplot_kwargs.pop("gridsize", None)
        pairplot_kwargs.pop("colorbar", None)
        pairplot_kwargs.pop("divergences_kwargs", None)
        pairplot_kwargs.pop("hexbin_values", None)

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_pair", "pairplot", backend)
    ax = plot(**pairplot_kwargs)
    return ax
