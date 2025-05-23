"""Plot a scatter, kde and/or hexbin of sampled parameters."""

import warnings
from typing import List, Optional, Union

import numpy as np

from ..data import convert_to_dataset
from ..labels import BaseLabeller
from ..sel_utils import xarray_to_ndarray, xarray_var_iter
from .plot_utils import get_plotting_function
from ..rcparams import rcParams
from ..utils import _var_names, get_coords


def plot_pair(
    data,
    group="posterior",
    var_names: Optional[List[str]] = None,
    filter_vars: Optional[str] = None,
    combine_dims=None,
    coords=None,
    marginals=False,
    figsize=None,
    textsize=None,
    kind: Union[str, List[str]] = "scatter",
    gridsize="auto",
    divergences=False,
    colorbar=False,
    labeller=None,
    ax=None,
    divergences_kwargs=None,
    scatter_kwargs=None,
    kde_kwargs=None,
    hexbin_kwargs=None,
    backend=None,
    backend_kwargs=None,
    marginal_kwargs=None,
    point_estimate=None,
    point_estimate_kwargs=None,
    point_estimate_marker_kwargs=None,
    reference_values=None,
    reference_values_kwargs=None,
    show=None,
):
    """
    Plot a scatter, kde and/or hexbin matrix with (optional) marginals on the diagonal.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details
    group: str, optional
        Specifies which InferenceData group should be plotted.  Defaults to 'posterior'.
    var_names: list of variable names, optional
        Variables to be plotted, if None all variable are plotted. Prefix the
        variables by ``~`` when you want to exclude them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        ``pandas.filter``.
    combine_dims : set_like of str, optional
        List of dimensions to reduce. Defaults to reducing only the "chain" and "draw" dimensions.
        See the :ref:`this section <common_combine_dims>` for usage examples.
    coords: mapping, optional
        Coordinates of var_names to be plotted. Passed to :meth:`xarray.Dataset.sel`.
    marginals: bool, optional
        If True pairplot will include marginal distributions for every variable
    figsize: figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int
        Text size for labels. If None it will be autoscaled based on ``figsize``.
    kind : str or List[str]
        Type of plot to display (scatter, kde and/or hexbin)
    gridsize: int or (int, int), optional
        Only works for ``kind=hexbin``. The number of hexagons in the x-direction.
        The corresponding number of hexagons in the y-direction is chosen
        such that the hexagons are approximately regular. Alternatively, gridsize
        can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    divergences: Boolean
        If True divergences will be plotted in a different color, only if group is either 'prior'
        or 'posterior'.
    colorbar: bool
        If True a colorbar will be included as part of the plot (Defaults to False).
        Only works when ``kind=hexbin``
    labeller : labeller instance, optional
        Class providing the method ``make_label_vert`` to generate the labels in the plot.
        Read the :ref:`label_guide` for more details and usage examples.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    divergences_kwargs: dicts, optional
        Additional keywords passed to :meth:`matplotlib.axes.Axes.scatter` for divergences
    scatter_kwargs:
        Additional keywords passed to :meth:`matplotlib.axes.Axes.scatter` when using scatter kind
    kde_kwargs: dict, optional
        Additional keywords passed to :func:`arviz.plot_kde` when using kde kind
    hexbin_kwargs: dict, optional
        Additional keywords passed to :meth:`matplotlib.axes.Axes.hexbin` when
        using hexbin kind
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or
        :func:`bokeh.plotting.figure`.
    marginal_kwargs: dict, optional
        Additional keywords passed to :func:`arviz.plot_dist`, modifying the
        marginal distributions plotted in the diagonal.
    point_estimate: str, optional
        Select point estimate from 'mean', 'mode' or 'median'. The point estimate will be
        plotted using a scatter marker and vertical/horizontal lines.
    point_estimate_kwargs: dict, optional
        Additional keywords passed to :meth:`matplotlib.axes.Axes.axvline`,
        :meth:`matplotlib.axes.Axes.axhline` (matplotlib) or
        :class:`bokeh:bokeh.models.Span` (bokeh)
    point_estimate_marker_kwargs: dict, optional
        Additional keywords passed to :meth:`matplotlib.axes.Axes.scatter`
        or :meth:`bokeh:bokeh.plotting.Figure.square` in point
        estimate plot. Not available in bokeh
    reference_values: dict, optional
        Reference values for the plotted variables. The Reference values will be plotted
        using a scatter marker
    reference_values_kwargs: dict, optional
        Additional keywords passed to :meth:`matplotlib.axes.Axes.plot` or
        :meth:`bokeh:bokeh.plotting.Figure.circle` in reference values plot
    show: bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

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

    Pair plot showing divergences and select variables with regular expressions

    .. plot::
        :context: close-figs

        >>> az.plot_pair(centered,
        ...             var_names=['^t', 'mu'],
        ...             filter_vars="regex",
        ...             coords=coords,
        ...             divergences=True,
        ...             textsize=18)
    """
    valid_kinds = ["scatter", "kde", "hexbin"]
    kind_boolean: Union[bool, List[bool]]
    if isinstance(kind, str):
        kind_boolean = kind in valid_kinds
    else:
        kind_boolean = [kind[i] in valid_kinds for i in range(len(kind))]
    if not np.all(kind_boolean):
        raise ValueError(f"Plot type {kind} not recognized. Plot type must be in {valid_kinds}")

    if coords is None:
        coords = {}

    if labeller is None:
        labeller = BaseLabeller()

    # Get posterior draws and combine chains
    dataset = convert_to_dataset(data, group=group)
    var_names = _var_names(var_names, dataset, filter_vars)
    plotters = list(
        xarray_var_iter(
            get_coords(dataset, coords), var_names=var_names, skip_dims=combine_dims, combined=True
        )
    )
    flat_var_names = []
    flat_ref_slices = []
    flat_var_labels = []
    for var_name, sel, isel, _ in plotters:
        dims = [dim for dim in dataset[var_name].dims if dim not in ["chain", "draw"]]
        flat_var_names.append(var_name)
        flat_ref_slices.append(tuple(isel[dim] if dim in isel else slice(None) for dim in dims))
        flat_var_labels.append(labeller.make_label_vert(var_name, sel, isel))

    divergent_data = None
    diverging_mask = None

    # Assigning divergence group based on group param
    if group == "posterior":
        divergent_group = "sample_stats"
    elif group == "prior":
        divergent_group = "sample_stats_prior"
    else:
        divergences = False

    # Get diverging draws and combine chains
    if divergences:
        if hasattr(data, divergent_group) and hasattr(getattr(data, divergent_group), "diverging"):
            divergent_data = convert_to_dataset(data, group=divergent_group)
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
                "or `sample_stats_prior` or set divergences=False",
                UserWarning,
            )

    if gridsize == "auto":
        gridsize = int(dataset.sizes["draw"] ** 0.35)

    numvars = len(flat_var_names)

    if numvars < 2:
        raise ValueError("Number of variables to be plotted must be 2 or greater.")

    pairplot_kwargs = dict(
        ax=ax,
        plotters=plotters,
        numvars=numvars,
        figsize=figsize,
        textsize=textsize,
        kind=kind,
        scatter_kwargs=scatter_kwargs,
        kde_kwargs=kde_kwargs,
        hexbin_kwargs=hexbin_kwargs,
        gridsize=gridsize,
        colorbar=colorbar,
        divergences=divergences,
        diverging_mask=diverging_mask,
        divergences_kwargs=divergences_kwargs,
        flat_var_names=flat_var_names,
        flat_ref_slices=flat_ref_slices,
        flat_var_labels=flat_var_labels,
        backend_kwargs=backend_kwargs,
        marginal_kwargs=marginal_kwargs,
        show=show,
        marginals=marginals,
        point_estimate=point_estimate,
        point_estimate_kwargs=point_estimate_kwargs,
        point_estimate_marker_kwargs=point_estimate_marker_kwargs,
        reference_values=reference_values,
        reference_values_kwargs=reference_values_kwargs,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_pair", "pairplot", backend)
    ax = plot(**pairplot_kwargs)
    return ax
