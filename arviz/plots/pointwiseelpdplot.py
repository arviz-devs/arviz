"""Plot pointwise elpd estimations of inference data."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from ..data import convert_to_inference_data
from .plot_utils import _scale_fig_size
from ..stats import waic, loo


def plot_pointwise_elpd(
    idata_dict,
    ic="waic",
    color=None,
    figsize=None,
    textsize=None,
    coords=None,
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
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    coords : mapping, optional
        Coordinates of points to plot. **All** values are used for computation, but only a
        a subset can be plotted for convenience.
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
    """
    valid_ics = ["waic", "loo"]
    ic = ic.lower()
    if ic not in valid_ics:
        raise ValueError(
            ("Information Criteria type {} not recognized."
             "IC must be in {}").format(ic, valid_ics)
        )

    if coords is None:
        coords = {}

    if plot_kwargs is None:
        plot_kwargs = {}
        plot_kwargs.setdefault("marker", ".")
        plot_kwargs.setdefault("lw", 0)


    def get_pointwise_as_dataarray(idata, coords=coords, ic=ic, scale=scale):
        if ic == "waic":
            pointwise_elpd = waic(idata, pointwise=True, scale=scale).waic_i
        else:
            pointwise_elpd = loo(idata, pointwise=True, scale=scale).loo_i
        like_dataarray = idata.sample_stats.log_likelihood
        dims = [dim for dim in like_dataarray.dims if dim not in ["chain", "draw"]]
        present_coords = {dim: like_dataarray.coords.indexes[dim] for dim in dims}
        elpd_dataarray = xr.DataArray(pointwise_elpd, dims=dims, coords=present_coords)
        print(elpd_dataarray)
        print(coords)
        elpd_dataarray = elpd_dataarray.sel(**coords)
        print(elpd_dataarray)
        return elpd_dataarray


    # Make sure all objects in idata_dict are InferenceData
    idata_dict = {key: convert_to_inference_data(idata) for key, idata in idata_dict.items()}
    numvars = len(idata_dict)
    models = list(idata_dict.keys())
    pointwise_data = [get_pointwise_as_dataarray(idata_dict[model]) for model in models]


    if numvars < 2:
        raise Exception("Number of models to compare must be 2 or greater.")

    if numvars == 2:
        (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        ax.plot(pointwise_data[0]-pointwise_data[1], **plot_kwargs)

        ax.set_xlabel("{}".format(models[0]), fontsize=ax_labelsize, wrap=True)
        ax.set_ylabel("{}".format(models[1]), fontsize=ax_labelsize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)

    else:
        (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )

        if ax is None:
            _, ax = plt.subplots(
                numvars - 1, numvars - 1, figsize=figsize, constrained_layout=True
            )

        for i in range(0, numvars - 1):
            var1 = pointwise_data[i]

            for j in range(0, numvars - 1):
                if j < i:
                    ax[j, i].axis("off")
                    continue

                var2 = pointwise_data[j + 1]

                ax[j, i].plot(var1-var2, **plot_kwargs)

                if j + 1 != numvars - 1:
                    ax[j, i].axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_xlabel(
                        "{}".format(models[i]), fontsize=ax_labelsize, wrap=True
                    )
                if i != 0:
                    ax[j, i].axes.get_yaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_ylabel(
                        "{}".format(models[j + 1]), fontsize=ax_labelsize, wrap=True
                    )

                ax[j, i].tick_params(labelsize=xt_labelsize)

    return ax
