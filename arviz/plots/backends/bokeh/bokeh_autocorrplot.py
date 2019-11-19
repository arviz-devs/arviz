import numpy as np
import bokeh.plotting as bkp
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot

from ....stats import autocorr
from ...plot_utils import make_label

def _plot_autocorr(
    axes, data, var_names, max_lag, plotters, line_width, combined=False, show=True,
):
    for (var_name, selection, x), ax_ in zip(plotters, axes.flatten()):
        x_prime = x
        if combined:
            x_prime = x.flatten()
        y = autocorr(x_prime)

        for x_, y_ in enumerate(y[:max_lag]):
            ax_.line([x_, x_], [0, y_], line_width=line_width, line_color="black")

        ax_.line([0,0], [0,max_lag], line_color="steelblue")

        title = Title()
        title.text = make_label(var_name, selection)
        ax_title = title

    if axes.size > 0:
        axes[0, 0].x_range._property_values["end"] = max_lag
        axes[0, 0].x_range._property_values["start"] = -1
        axes[0, 0].x_range._property_values["end"] = 1


    if show:
        bkp.show(gridplot([list(item) for item in axes], toolbar_location="above"))
    return axes
