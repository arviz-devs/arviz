"""Pareto tail indices plot."""

import logging
import warnings

import numpy as np
from xarray import DataArray

from ..rcparams import rcParams
from ..stats import ELPDData
from ..utils import get_coords
from .plot_utils import format_coords_as_labels, get_plotting_function

_log = logging.getLogger(__name__)


def plot_khat(
    khats,
    color="C0",
    xlabels=False,
    show_hlines=False,
    show_bins=False,
    bin_format="{1:.1f}%",
    annotate=False,
    threshold=None,
    hover_label=False,
    hover_format="{1}",
    figsize=None,
    textsize=None,
    coords=None,
    legend=False,
    markersize=None,
    ax=None,
    hlines_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs
):
    r"""Plot Pareto tail indices :math:`\hat{k}` for diagnosing convergence in PSIS-LOO.

    Parameters
    ----------
    khats : ELPDData
        The input Pareto tail indices to be plotted.
    color : str or array_like, default "C0"
        Colors of the scatter plot, if color is a str all dots will have the same color,
        if it is the size of the observations, each dot will have the specified color,
        otherwise, it will be interpreted as a list of the dims to be used for the color
        code. If Matplotlib c argument is passed, it will override the color argument.
    xlabels : bool, default False
        Use coords as xticklabels.
    show_hlines : bool, default False
        Show the horizontal lines, by default at the values [0, 0.5, 0.7, 1].
    show_bins : bool, default False
        Show the percentage of khats falling in each bin, as delimited by hlines.
    bin_format : str, optional
        The string is used as formatting guide calling ``bin_format.format(count, pct)``.
    threshold : float, optional
        Show the labels of k values larger than `threshold`. If ``None`` (default), no
        observations will be highlighted.
    hover_label : bool, default False
        Show the datapoint label when hovering over it with the mouse. Requires an interactive
        backend.
    hover_format : str, default "{1}"
        String used to format the hover label via ``hover_format.format(idx, coord_label)``
    figsize : (float, float), optional
        Figure size. If ``None`` it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If ``None`` it will be autoscaled
        based on `figsize`.
    coords : mapping, optional
        Coordinates of points to plot. **All** values are used for computation, but only a
        a subset can be plotted for convenience. See :ref:`this section <common_coords>` for
        usage examples.
    legend : bool, default False
        Include a legend to the plot. Only taken into account when color argument is a dim name.
    markersize : int, optional
        markersize for scatter plot. Defaults to ``None`` in which case it will
        be chosen based on autoscaling for figsize.
    ax : axes, optional
        Matplotlib axes or bokeh figures.
    hlines_kwargs : dict, optional
        Additional keywords passed to
        :meth:`matplotlib.axes.Axes.hlines`.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.
    kwargs :
        Additional keywords passed to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    axes : matplotlib_axes or bokeh_figures

    See Also
    --------
    psislw : Pareto smoothed importance sampling (PSIS).

    Examples
    --------
    Plot estimated pareto shape parameters showing how many fall in each category.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> radon = az.load_arviz_data("radon")
        >>> loo_radon = az.loo(radon, pointwise=True)
        >>> az.plot_khat(loo_radon, show_bins=True)

    Show xlabels

    .. plot::
        :context: close-figs

        >>> centered_eight = az.load_arviz_data("centered_eight")
        >>> khats = az.loo(centered_eight, pointwise=True).pareto_k
        >>> az.plot_khat(khats, xlabels=True, threshold=1)

    Use custom color scheme

    .. plot::
        :context: close-figs

        >>> counties = radon.posterior.County[radon.constant_data.county_idx].values
        >>> colors = [
        ...     "blue" if county[-1] in ("A", "N") else "green" for county in counties
        ... ]
        >>> az.plot_khat(loo_radon, color=colors)

    Notes
    -----
    The Generalized Pareto distribution (GPD) diagnoses convergence rates for importance
    sampling. GPD has parameters offset, scale, and shape. The shape parameter (:math:`k`)
    tells the distribution's number of finite moments. The pre-asymptotic convergence rate
    of importance sampling can be estimated based on the fractional number of finite moments
    of the importance ratio distribution. GPD is fitted to the largest importance ratios and
    interprets the estimated shape parameter :math:`k`, i.e., :math:`\hat{k}` can then be
    used as a diagnostic (most importantly if :math:`\hat{k} > 0.7`, then the convergence
    rate is impractically low). See [1]_.

    References
    ----------
    .. [1] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., Gabry, J. (2024).
        Pareto Smoothed Importance Sampling. Journal of Machine Learning
        Research, 25(72):1-58.

    """
    if annotate:
        _log.warning("annotate will be deprecated, please use threshold instead")
        threshold = annotate

    if coords is None:
        coords = {}

    if color is None:
        color = "C0"

    if isinstance(khats, np.ndarray):
        warnings.warn(
            "support for arrays will be deprecated, please use ELPDData."
            "The reason for this, is that we need to know the numbers of draws"
            "sampled from the posterior",
            FutureWarning,
        )
        khats = khats.flatten()
        xlabels = False
        legend = False
        dims = []
        good_k = None
    else:
        if isinstance(khats, ELPDData):
            good_k = khats.good_k
            khats = khats.pareto_k
        else:
            good_k = None
            warnings.warn(
                "support for DataArrays will be deprecated, please use ELPDData."
                "The reason for this, is that we need to know the numbers of draws"
                "sampled from the posterior",
                FutureWarning,
            )
        if not isinstance(khats, DataArray):
            raise ValueError("Incorrect khat data input. Check the documentation")

        khats = get_coords(khats, coords)
        dims = khats.dims

    n_data_points = khats.size
    xdata = np.arange(n_data_points)
    if isinstance(khats, DataArray):
        coord_labels = format_coords_as_labels(khats)
    else:
        coord_labels = xdata.astype(str)

    plot_khat_kwargs = dict(
        hover_label=hover_label,
        hover_format=hover_format,
        ax=ax,
        figsize=figsize,
        xdata=xdata,
        khats=khats,
        good_k=good_k,
        kwargs=kwargs,
        threshold=threshold,
        coord_labels=coord_labels,
        show_hlines=show_hlines,
        show_bins=show_bins,
        hlines_kwargs=hlines_kwargs,
        xlabels=xlabels,
        legend=legend,
        color=color,
        dims=dims,
        textsize=textsize,
        markersize=markersize,
        n_data_points=n_data_points,
        bin_format=bin_format,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_khat", "khatplot", backend)
    axes = plot(**plot_khat_kwargs)
    return axes
