"""Plot separation plot for discrete outcome models."""
from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_separation(
    idata=None,
    y=None,
    y_hat=None,
    y_hat_line=False,
    expected_events=False,
    figsize=None,
    textsize=None,
    color=None,
    cmap=None,
    legend=True,
    ax=None,
    plot_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):

    """Plot separation plot for discrete outcome models.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object.
    y : array, DataArray or str
        Observed data. If str, idata must be present and contain the observed data group
    y_hat : array, DataArray or str
        Posterior predictive samples for ``y``. It must have the same shape as y plus an
        extra dimension at the end of size n_samples (chains and draws stacked). If str or
        None, idata must contain the posterior predictive group. If None, y_hat is taken
        equal to y, thus, y must be str too.
    y_hat_line : bool, optional
        Plot the sorted `y_hat` predictions.
    expected_events : bool, optional
        Plot the total number of expected events.
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    color : list or array_like, optional
        The first color will be used to plot the negative class while the second color will
        be assigned to the positive class.
    cmap : str, optional
        Colors for the separation plot will be taken from both ends of the color map
        respectively.
    legend : bool, optional
        Show the legend of the figure.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    plot_kwargs : dict, optional
        Additional keywords passed to ax.plot for `y_hat` line.
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

    References
    ----------
    * Greenhill, B. et al (2011) see https://doi.org/10.1111/j.1540-5907.2011.00525.x

    """

    separation_kwargs = dict(
        idata=idata,
        y=y,
        y_hat=y_hat,
        y_hat_line=y_hat_line,
        expected_events=expected_events,
        figsize=figsize,
        textsize=textsize,
        color=color,
        cmap=cmap,
        legend=legend,
        ax=ax,
        plot_kwargs=plot_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_separation", "separationplot", backend)
    axes = plot(**separation_kwargs)

    return axes
