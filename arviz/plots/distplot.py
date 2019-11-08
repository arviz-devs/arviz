"""Plot distribution as histogram or kernel density estimates."""
import matplotlib.pyplot as plt

from .kdeplot import plot_kde
from .plot_utils import get_bins


def plot_dist(
    values,
    values2=None,
    color=None,
    kind="auto",
    cumulative=False,
    label=None,
    rotated=False,
    rug=False,
    bw=4.5,
    quantiles=None,
    contour=True,
    fill_last=True,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    hist_kwargs=None,
    ax=None,
    backend=None,
):
    """Plot distribution as histogram or kernel density estimates.

    By default continuous variables are plotted using KDEs and discrete ones using histograms

    Parameters
    ----------
    values : array-like
        Values to plot
    values2 : array-like, optional
        Values to plot. If present, a 2D KDE or a hexbin will be estimated
    color : string
        valid matplotlib color
    kind : string
        By default ("auto") continuous variables are plotted using KDEs and discrete ones using
        histograms. To override this use "hist" to plot histograms and "kde" for KDEs
    cumulative : bool
        If true plot the estimated cumulative distribution function. Defaults to False.
        Ignored for 2D KDE
    label : string
        Text to include as part of the legend
    rotated : bool
        Whether to rotate the 1D KDE plot 90 degrees.
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE
    bw : float
        Bandwidth scaling factor for 1D KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's
        rule of thumb (the default rule used by SciPy).
    quantiles : list
        Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
        Defaults to None.
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    plot_kwargs : dict
        Keywords passed to the pdf line of a 1D KDE.
    fill_kwargs : dict
        Keywords passed to the fill under the line (use fill_kwargs={'alpha': 0} to disable fill).
        Ignored for 2D KDE
    rug_kwargs : dict
        Keywords passed to the rug plot. Ignored if rug=False or for 2D KDE
        Use `space` keyword (float) to control the position of the rugplot. The larger this number
        the lower the rugplot.
    contour_kwargs : dict
        Keywords passed to the contourplot. Ignored for 1D KDE.
    hist_kwargs : dict
        Keywords passed to the histogram.
    ax : matplotlib axes

    Returns
    -------
    ax : matplotlib axes
    """
    dist_plot_args = dict(
        # Internal API
        # User Facing API that can be simplified
        values=values,
        values2=values2,
        color=color,
        kind=kind,
        cumulative=cumulative,
        label=label,
        rotated=rotated,
        rug=rug,
        bw=bw,
        quantiles=quantiles,
        contour=contour,
        fill_last=fill_last,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        contour_kwargs=contour_kwargs,
        hist_kwargs=hist_kwargs,
        ax=ax,
    )

    if backend is None or backend.lower() in ("mpl", "matplotlib"):
        from .backends.matplotlib.mpl_distplot import _plot_dist_mpl

        ax = _plot_dist_mpl(**dist_plot_args)
    elif backend == "bokeh":
        from .backends.bokeh.bokeh_distplot import _plot_dist_bokeh

        ax = _plot_dist_bokeh(**dist_plot_args)
    else:
        raise NotImplementedError(
            'Backend {} not implemented. Use {{"matplotlib", "bokeh"}}'.format(backend)
        )
    return ax
