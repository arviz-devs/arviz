"""Plot ecdf and ecdf-difference plot with confidence bands."""
import numpy as np

from arviz.rcparams import rcParams
from arviz.plots.plot_utils import get_plotting_function


def plot_ecdf(
    values,
    values2=None,
    distribution=None,
    difference=False,
    pit=False,
    confidence_bands=False,
    granularity=100,
    num_trials=500,
    alpha=0.95,
    figsize=None,
    ecdf_fill=True,
    ax=None,
    show=None,
    backend=None,
    backend_kwargs=None,
    **kwargs
):
    """Plot ECDF and ECDF-Difference Plot with Confidence bands.

    Parameters
    ----------
    values : array-like
        Values to plot from an unknown continuous distribution
    values2 : array-like, optional
        Values to compare to the original sample
    distribution : function, optional
        Cumulative distribution function of the distribution we will compare our sample to
    pit : bool, optional
        If True plots the ECDF or ECDF-diff of PIT of sample
    confidence_bands : bool, optional
        If True plots the simultaneous confidence bands with 1 - alpha confidence level
    granularity : int, optional
        This denotes the granularity size of out plot
    num_trails : int, optional
        The number of trials to carry while finding gamma
    alpha : float, optional
        The type I error rate which s.t 1 - alpha denotes the confidence level of bands
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    ecdf_fill : bool, optional
        Use fill_between to mark the area inside the credible interval.
        Otherwise, plot the border lines.
    ax : axes, optional
        Matplotlib axes or bokeh figures.
    show : bool, optional
        Call backend show function.
    backend : str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`mpl:matplotlib.pyplot.subplots` or
        :meth:`bokeh:bokeh.plotting.figure`.

    Returns
    -------
    axes : matplotlib axes or bokeh figures
    """
    if values2 is None and distribution is None and confidence_bands is True:
        raise ValueError("For confidence bands you need to specify values2 or the distribution")

    if distribution is not None and values2 is not None:
        raise ValueError("To compare sample we need either distribution or values2 and not both")

    if values2 is None and distribution is None and pit is True:
        raise ValueError("For PIT specify either distribution or values2")

    if values2 is not None:
        values2 = np.ravel(values2)
        values2.sort()

    values = np.ravel(values)
    values.sort()

    ecdf_plot_args = dict(
        values=values,
        values2=values2,
        distribution=distribution,
        difference=difference,
        pit=pit,
        confidence_bands=confidence_bands,
        granularity=granularity,
        num_trials=num_trials,
        alpha=alpha,
        figsize=figsize,
        ecdf_fill=ecdf_fill,
        ax=ax,
        show=show,
        backend_kwargs=backend_kwargs,
        **kwargs
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_ecdf", "ecdfplot", backend)
    ax = plot(**ecdf_plot_args)

    return ax
