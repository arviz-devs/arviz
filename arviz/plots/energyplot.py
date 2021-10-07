"""Plot energy transition distribution in HMC inference."""
import warnings

from ..data import convert_to_dataset
from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_energy(
    data,
    kind=None,
    bfmi=True,
    figsize=None,
    legend=True,
    fill_alpha=(1, 0.75),
    fill_color=("C0", "C5"),
    bw="experimental",
    textsize=None,
    fill_kwargs=None,
    plot_kwargs=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """Plot energy transition distribution and marginal energy distribution in HMC algorithms.

    This may help to diagnose poor exploration by gradient-based algorithms like HMC or NUTS.

    Parameters
    ----------
    data : obj
        :class:`xarray.Dataset`, or any object that can be converted (must represent
        ``sample_stats`` and have an ``energy`` variable).
    kind : str
        Type of plot to display ("kde", "hist").
    bfmi : bool
        If True add to the plot the value of the estimated Bayesian fraction of missing information
    figsize : tuple
        Figure size. If None it will be defined automatically.
    legend : bool
        Flag for plotting legend. Defaults to True.
    fill_alpha : tuple of floats
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to (1, .75).
    fill_color : tuple of valid matplotlib color
        Color for Marginal energy distribution and Energy transition distribution.
        Defaults to ('C0', 'C5').
    bw: float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental". Defaults to "experimental".
        Only works if ``kind='kde'``.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    fill_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_kde` (to control the shade).
    plot_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_kde` or :func:`matplotlib.pyplot.hist`
        (if ``type='hist'``).
    ax: axes, optional
        :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.Figure`.
    backend: str, optional
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or
        :func:`bokeh.plotting.figure`.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    See Also
    --------
    bfmi : Calculate the estimated Bayesian fraction of missing information (BFMI).

    Examples
    --------
    Plot a default energy plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_energy(data)

    Represent energy plot via histograms

    .. plot::
        :context: close-figs

        >>> az.plot_energy(data, kind='hist')

    """
    energy = convert_to_dataset(data, group="sample_stats").energy.values

    if kind == "histogram":
        warnings.warn(
            "kind histogram will be deprecated in a future release. Use `hist` "
            "or set rcParam `plot.density_kind` to `hist`",
            FutureWarning,
        )
        kind = "hist"

    if kind is None:
        kind = rcParams["plot.density_kind"]

    plot_energy_kwargs = dict(
        ax=ax,
        energy=energy,
        kind=kind,
        bfmi=bfmi,
        figsize=figsize,
        textsize=textsize,
        fill_alpha=fill_alpha,
        fill_color=fill_color,
        fill_kwargs=fill_kwargs,
        plot_kwargs=plot_kwargs,
        bw=bw,
        legend=legend,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_energy", "energyplot", backend)
    ax = plot(**plot_energy_kwargs)
    return ax
