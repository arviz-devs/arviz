"""Bokeh KDE Plot."""
import bokeh.plotting as bkp
import matplotlib.pyplot as plt
import numpy as np


def _plot_kde_bokeh(
    density,
    lower,
    upper,
    density_q,
    xmin,
    xmax,
    ymin,
    ymax,
    gridsize,
    values,
    values2=None,
    rug=False,
    label=None,
    bw=4.5,
    quantiles=None,
    rotated=False,
    contour=True,
    fill_last=True,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    ax=None,
    legend=True,
    show=True,
):
    if ax is None:
        ax = bkp.figure(sizing_mode="stretch_both")

    if legend and label is not None:
        plot_kwargs["legend_label"] = label

    if isinstance(values, xr.Dataset):
        raise ValueError(
            "Xarray dataset object detected.Use plot_posterior, plot_density, plot_joint"
            "or plot_pair instead of plot_kde"
        )
    if isinstance(values, InferenceData):
        raise ValueError(" Inference Data object detected. Use plot_posterior instead of plot_kde")

    if values2 is None:
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault("line_color", plt.rcParams["axes.prop_cycle"].by_key()["color"][0])

        default_color = plot_kwargs.get("color")

        if fill_kwargs is None:
            fill_kwargs = {}

        fill_kwargs.setdefault("color", default_color)

        if rug_kwargs is None:
            rug_kwargs = {}
        else:
            # todo: add warning
            pass

        x = np.linspace(lower, upper, len(density))
        ax.line(x, density, **plot_kwargs)
    else:
        # todo
        raise NotImplementedError("Use matplotlib backend")

    if show:
        bkp.show(ax)
    return ax
