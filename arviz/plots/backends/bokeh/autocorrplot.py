"""Bokeh Autocorrplot."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models.annotations import Title

from . import backend_kwarg_defaults
from ...plot_utils import make_label
from ....stats import autocorr


def plot_autocorr(
    axes, plotters, max_lag, line_width, combined, backend_kwargs,
):
    """Bokeh autocorrelation plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    for (var_name, selection, x), ax_ in zip(plotters, axes.flatten()):
        x_prime = x
        if combined:
            x_prime = x.flatten()
        y = autocorr(x_prime)

        ax_.segment(
            x0=np.arange(len(y)),
            y0=0,
            x1=np.arange(len(y)),
            y1=y,
            line_width=line_width,
            line_color="black",
        )
        ax_.line([0, 0], [0, max_lag], line_color="steelblue")

        title = Title()
        title.text = make_label(var_name, selection)
        ax_.title = title

    if axes.size > 0:
        axes[0, 0].x_range._property_values["start"] = 0  # pylint: disable=protected-access
        axes[0, 0].x_range._property_values["end"] = max_lag  # pylint: disable=protected-access
        axes[0, 0].y_range._property_values["start"] = -1  # pylint: disable=protected-access
        axes[0, 0].y_range._property_values["end"] = 1  # pylint: disable=protected-access

    if backend_kwargs["show"]:
        bkp.show(gridplot([list(item) for item in axes], toolbar_location="above"))
    return axes
