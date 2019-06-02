"""Plot pointwise elpd estimations of inference data."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D

from ..data import convert_to_inference_data
from .plot_utils import _scale_fig_size
from ..stats import waic, loo


def plot_elpd(
    idata_dict,
    ic="waic",
    color=None,
    xlabels=False,
    figsize=None,
    textsize=None,
    coords=None,
    legend=False,
    ax=None,
    scale="deviance",
    plot_kwargs=None,
):
    """
    Plot a scatter or hexbin matrix of the sampled parameters.

    Parameters
    ----------
    idata_dict : mapping, str -> InferenceData
        A dictionary mapping the model name to the object containing its inference data.
        Refer to az.convert_to_inference_data for details on possible dict items
    ic : str, optional
        Information Criterion (WAIC or LOO) used to compare models. Default WAIC.
    color : str or array_like, optional
        Colors of the scatter plot, if color is a str all dots will have the same color,
        if it is the size of the observations, each dot will have the specified color,
        otherwise, it will be interpreted as a list of the dims to be used for the color code
    xlabels : bool, optional
        Use coords as xticklabels
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    coords : mapping, optional
        Coordinates of points to plot. **All** values are used for computation, but only a
        a subset can be plotted for convenience.
    legend : bool, optional
        Include a legend to the plot. Only taken into account when color argument is a dim name.
    ax: axes, optional
        Matplotlib axes
    scale : str, optional
        scale argument passed to az.waic or az.loo, see their docs for details
    plot_kwargs : dicts, optional
        Additional keywords passed to ax.plot

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    Compare pointwise WAIC for centered and non centered models of the 8school problem

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata1 = az.load_arviz_data("centered_eight")
        >>> idata2 = az.load_arviz_data("non_centered_eight")
        >>> az.plot_elpd(
        >>>     {"centered model": idata1, "non centered model": idata2},
        >>>     xlabels=True
        >>> )

    """
    valid_ics = ["waic", "loo"]
    ic = ic.lower()
    if ic not in valid_ics:
        raise ValueError(
            ("Information Criteria type {} not recognized." "IC must be in {}").format(
                ic, valid_ics
            )
        )

    # Make sure all objects in idata_dict are InferenceData
    idata_dict = {key: convert_to_inference_data(idata) for key, idata in idata_dict.items()}
    numvars = len(idata_dict)
    models = list(idata_dict.keys())

    if coords is None:
        coords = {}

    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("marker", "+")

    def get_pointwise_as_dataarray(idata, coords=coords, ic=ic, scale=scale):
        if ic == "waic":
            pointwise_elpd = waic(idata, pointwise=True, scale=scale).waic_i
        else:
            pointwise_elpd = loo(idata, pointwise=True, scale=scale).loo_i
        like_dataarray = idata.sample_stats.log_likelihood
        _, _, *shape = like_dataarray.shape
        dims = [dim for dim in like_dataarray.dims if dim not in ["chain", "draw"]]
        present_coords = {dim: like_dataarray.coords.indexes[dim] for dim in dims}
        elpd_dataarray = xr.DataArray(
            pointwise_elpd.reshape(shape), dims=dims, coords=present_coords
        )
        elpd_dataarray = elpd_dataarray.sel(**coords)
        return elpd_dataarray

    pointwise_data = [get_pointwise_as_dataarray(idata_dict[model]) for model in models]
    xdata = np.arange(pointwise_data[0].size)

    if isinstance(color, str):
        present_dims = pointwise_data[0].dims
        if color in present_dims:
            coord_values = pointwise_data[0][color].values
            unique_coords = set(coord_values)
            color_mapping = {
                coord: num / len(unique_coords) for num, coord in enumerate(unique_coords)
            }
            if len(present_dims) > 1:
                multi_coords = pointwise_data[0].coords.to_index()
                coord_idx = present_dims.index(color)
                colors = [color_mapping[coord[coord_idx]] for coord in multi_coords]
            else:
                colors = [color_mapping[coord] for coord in coord_values]
            if legend:
                cmap_name = plot_kwargs.pop("cmap", plt.rcParams["image.cmap"])
                markersize = plot_kwargs.pop("s", plt.rcParams["lines.markersize"])
                cmap = getattr(cm, cmap_name)
                handles = [
                    Line2D(
                        [],
                        [],
                        color=cmap(color_mapping[coord]),
                        label=coord,
                        ms=markersize,
                        lw=0,
                        **plot_kwargs
                    )
                    for coord in unique_coords
                ]
                plot_kwargs.setdefault("cmap", cmap_name)
                plot_kwargs.setdefault("s", markersize ** 2)
            plot_kwargs.setdefault("c", colors)
        else:
            plot_kwargs.setdefault("c", color)
            legend = False
    else:
        legend = False
        plot_kwargs.setdefault("c", color)

    if xlabels:
        coord_labels = pointwise_data[0].coords.to_index().values
        if isinstance(coord_labels[0], tuple):
            fmt = ", ".join(["{}" for _ in coord_labels[0]])
            coord_labels[:] = [fmt.format(*x) for x in coord_labels]
        else:
            coord_labels[:] = ["{}".format(s) for s in coord_labels]

    if numvars < 2:
        raise Exception("Number of models to compare must be 2 or greater.")

    if numvars == 2:
        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )
        plot_kwargs.setdefault("s", markersize ** 2)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=(not xlabels and not legend))

        ax.scatter(xdata, pointwise_data[0] - pointwise_data[1], **plot_kwargs)

        ax.set_title("{} - {}".format(*models), fontsize=titlesize, wrap=True)
        ax.set_ylabel("ELPD difference", fontsize=ax_labelsize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)
        if xlabels:
            _set_xticklabels(ax, coord_labels)
            fig.autofmt_xdate()
        if legend:
            ncols = len(handles) // 6 + 1
            ax.legend(handles=handles, ncol=ncols, title=color)

    else:
        (figsize, ax_labelsize, titlesize, xt_labelsize, _, markersize) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )
        plot_kwargs.setdefault("s", markersize ** 2)

        if ax is None:
            fig, ax = plt.subplots(
                numvars - 1,
                numvars - 1,
                figsize=figsize,
                constrained_layout=(not xlabels and not legend),
            )

        for i in range(0, numvars - 1):
            var1 = pointwise_data[i]

            for j in range(0, numvars - 1):
                if j < i:
                    ax[j, i].axis("off")
                    continue

                var2 = pointwise_data[j + 1]

                ax[j, i].scatter(xdata, var1 - var2, **plot_kwargs)

                if j + 1 != numvars - 1:
                    ax[j, i].axes.get_xaxis().set_major_formatter(NullFormatter())
                    ax[j, i].set_xticks([])
                elif xlabels:
                    _set_xticklabels(ax[j, i], coord_labels)

                if i != 0:
                    ax[j, i].axes.get_yaxis().set_major_formatter(NullFormatter())
                    ax[j, i].set_yticks([])
                else:
                    ax[j, i].set_ylabel("ELPD difference", fontsize=ax_labelsize, wrap=True)

                ax[j, i].tick_params(labelsize=xt_labelsize)
                ax[j, i].set_title(
                    "{} - {}".format(models[i], models[j + 1]), fontsize=titlesize, wrap=True
                )
        if xlabels:
            fig.autofmt_xdate()
        if legend:
            ncols = len(handles) // 6 + 1
            ax[0, 1].legend(
                handles=handles, ncol=ncols, title=color, bbox_to_anchor=(0, 1), loc="upper left"
            )
    return ax


def _set_xticklabels(ax, coord_labels):
    """Set xticklabels to coordinate labels using Matplotlib default formatter."""
    xlim = ax.get_xlim()
    ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
    xticks = ax.get_xticks().astype(np.int64)
    xticks = xticks[(xticks > xlim[0]) & (xticks < xlim[1])]
    if len(xticks) > len(coord_labels):
        ax.set_xticks(np.arange(len(coord_labels)))
        ax.set_xticklabels(coord_labels)
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(coord_labels[xticks])
