"""Bokeh Autocorrplot."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import DataRange1d
from bokeh.models.annotations import Title

from . import backend_kwarg_defaults, backend_show
from ...plot_utils import _create_axes_grid, make_label
from ....stats import autocorr
from ....rcparams import rcParams


def plot_autocorr(
    axes, plotters, max_lag, figsize, rows, cols, line_width, combined, backend_kwargs, show,
):
    """Bokeh autocorrelation plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if axes is None:
        _, axes = _create_axes_grid(
            len(plotters),
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            sharex=True,
            sharey=True,
            backend="bokeh",
            backend_kwargs=backend_kwargs,
        )
    else:
        axes = np.atleast_2d(axes)

    for (var_name, selection, x), ax in zip(
        plotters, (item for item in axes.flatten() if item is not None)
    ):
        x_prime = x
        if combined:
            x_prime = x.flatten()
        y = autocorr(x_prime)

        ax.segment(
            x0=np.arange(len(y)),
            y0=0,
            x1=np.arange(len(y)),
            y1=y,
            line_width=line_width,
            line_color="black",
        )
        ax.line([0, 0], [0, max_lag], line_color="steelblue")

        title = Title()
        title.text = make_label(var_name, selection)
        ax.title = title
        ax.x_range = DataRange1d(
            start=0, end=max_lag, bounds=(0, len(np.arange(len(y)))), min_interval=5
        )
        ax.y_range = DataRange1d(start=-1, end=1, bounds=rcParams["plot.bokeh.bounds"])

    for _, ax in zip(
        plotters, (item for item in axes.flatten() if item is not None)
    ):
        ax.x_range = axes[0, 0].x_range
        ax.y_range = axes[0, 0].y_range

    if backend_show(show):
        bkp.show(gridplot(axes.tolist(), toolbar_location="above"))
    return axes
