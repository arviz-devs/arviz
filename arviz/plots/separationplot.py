"""Separation plot for discrete outcome models."""
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
    legend=True,
    ax=None,
    plot_kwargs=None,
    y_hat_line_kwargs=None,
    exp_events_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """Separation plot for binary outcome models.

    Model predictions are sorted and plotted using a color code according to
    the observed data.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object.
    y : array, DataArray or str
        Observed data. If str, idata must be present and contain the observed data group
    y_hat : array, DataArray or str
        Posterior predictive samples for ``y``. It must have the same shape as y. If str or
        None, idata must contain the posterior predictive group.
    y_hat_line : bool, optional
        Plot the sorted `y_hat` predictions.
    expected_events : bool, optional
        Plot the total number of expected events.
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    color : str, optional
        Color to assign to the postive class. The negative class will be plotted using the
        same color and an `alpha=0.3` transparency.
    legend : bool, optional
        Show the legend of the figure.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    plot_kwargs : dict, optional
        Additional keywords passed to :meth:`mpl:matplotlib.axes.Axes.bar` or :meth:`bokeh:bokeh.plotting.Figure.vbar` for separation plot.
    y_hat_line_kwargs : dict, optional
        Additional keywords passed to ax.plot for `y_hat` line.
    exp_events_kwargs : dict, optional
        Additional keywords passed to ax.scatter for expected_events marker.
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

    Examples
    --------
    Separation plot for a logistic regression model.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data('classification10d')
        >>> az.plot_separation(idata=idata, y='outcome', y_hat='outcome', figsize=(8, 1))

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
        legend=legend,
        ax=ax,
        plot_kwargs=plot_kwargs,
        y_hat_line_kwargs=y_hat_line_kwargs,
        exp_events_kwargs=exp_events_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_separation", "separationplot", backend)
    axes = plot(**separation_kwargs)

    return axes
