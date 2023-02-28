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
    r"""Plot energy transition distribution and marginal energy distribution in HMC algorithms.

    This may help to diagnose poor exploration by gradient-based algorithms like HMC or NUTS.
    The energy function in HMC can identify posteriors with heavy tailed distributions, that
    in practice are challenging for sampling.

    This plot is in the style of the one used in [1]_.

    Parameters
    ----------
    data : obj
        :class:`xarray.Dataset`, or any object that can be converted (must represent
        ``sample_stats`` and have an ``energy`` variable).
    kind : str, optional
        Type of plot to display ("kde", "hist").
    bfmi : bool, default True
        If True add to the plot the value of the estimated Bayesian fraction of missing
        information.
    figsize : (float, float), optional
        Figure size. If `None` it will be defined automatically.
    legend : bool, default True
        Flag for plotting legend.
    fill_alpha : tuple, default (1, 0.75)
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque).
    fill_color : tuple of valid matplotlib color, default ('C0', 'C5')
        Color for Marginal energy distribution and Energy transition distribution.
    bw : float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental". Defaults to "experimental".
        Only works if ``kind='kde'``.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If `None` it will be autoscaled
        based on `figsize`.
    fill_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_kde` (to control the shade).
    plot_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_kde` or :func:`matplotlib.pyplot.hist`
        (if ``type='hist'``).
    ax : axes, optional
        :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.Figure`.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    See Also
    --------
    bfmi : Calculate the estimated Bayesian fraction of missing information (BFMI).

    References
    ----------
    .. [1] Betancourt (2016). Diagnosing Suboptimal Cotangent Disintegrations in
    Hamiltonian Monte Carlo https://arxiv.org/abs/1604.00695

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
    energy = convert_to_dataset(data, group="sample_stats").energy.transpose("chain", "draw").values

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
