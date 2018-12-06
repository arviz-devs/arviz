"""KDE and histogram plots for multiple variables."""
import numpy as np

from ..data import convert_to_dataset
from ..stats import hpd
from .kdeplot import _fast_kde
from .plot_utils import (
    _scale_fig_size,
    make_label,
    xarray_var_iter,
    default_grid,
    _create_axes_grid,
)
from ..utils import _var_names


# pylint:disable-msg=too-many-function-args
def plot_density(
    data,
    group="posterior",
    data_labels=None,
    var_names=None,
    credible_interval=0.94,
    point_estimate="mean",
    colors="cycle",
    outline=True,
    hpd_markers="",
    shade=0.0,
    bw=4.5,
    figsize=None,
    textsize=None,
):
    """Generate KDE plots for continuous variables and histograms for discrete ones.

    Plots are truncated at their 100*(1-alpha)% credible intervals. Plots are grouped per variable
    and colors assigned to models.

    Parameters
    ----------
    data : Union[Object, Iterator[Object]]
        Any object that can be converted to an az.InferenceData object, or an Iterator returning
        a sequence of such objects.
        Refer to documentation of az.convert_to_dataset for details about such objects.
    group: Optional[str]
        Specifies which InferenceData group should be plotted.  Defaults to 'posterior'.
        Alternative values include 'prior' and any other strings used as dataset keys in the
        InferenceData.
    data_labels : Optional[List[str]]
        List with names for the datasets passed as "data." Useful when plotting more than one
        dataset.  Must be the same shape as the data parameter.  Defaults to None.
    var_names: Optional[List[str]]
        List of variables to plot.  If multiple datasets are supplied and var_names is not None,
        will print the same set of variables for each dataset.  Defaults to None, which results in
        all the variables being plotted.
    credible_interval : float
        Credible intervals. Should be in the interval (0, 1]. Defaults to 0.94.
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be 'mean', 'median' or None.
        Defaults to 'mean'.
    colors : Optional[Union[List[str],str]]
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically choose a color per model from matplolib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to `cycle`.
    outline : bool
        Use a line to draw KDEs and histograms. Default to True
    hpd_markers : str
        A valid `matplotlib.markers` like 'v', used to indicate the limits of the hpd interval.
        Defaults to empty string (no marker).
    shade : Optional[float]
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    bw : Optional[float]
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    figsize : Optional[Tuple[int, int]]
        Figure size. If None it will be defined automatically.
    textsize: Optional[float]
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.

    Returns
    -------
    ax : Matplotlib axes


    Examples
    --------
    Plot default density plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered = az.load_arviz_data('centered_eight')
        >>> non_centered = az.load_arviz_data('non_centered_eight')
        >>> az.plot_density([centered, non_centered])

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"])

    Plot a specific `az.InferenceData` group

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], group="prior")

    Specify credible interval

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], credible_interval=.5)

    Shade plots and/or remove outlines

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], outline=False, shade=.8)

    Specify binwidth for kernel density estimation

    .. plot::
        :context: close-figs

        >>> az.plot_density([centered, non_centered], var_names=["mu"], bw=.9)
    """
    var_names = _var_names(var_names)

    if not isinstance(data, (list, tuple)):
        datasets = [convert_to_dataset(data, group=group)]
    else:
        datasets = [convert_to_dataset(d, group=group) for d in data]

    if point_estimate not in ("mean", "median", None):
        raise ValueError(
            "Point estimate should be 'mean'," "median' or None, not {}".format(point_estimate)
        )

    n_data = len(datasets)

    if data_labels is None:
        if n_data > 1:
            data_labels = ["{}".format(idx) for idx in range(n_data)]
        else:
            data_labels = [""]
    elif len(data_labels) != n_data:
        raise ValueError(
            "The number of names for the models ({}) "
            "does not match the number of models ({})".format(len(data_labels), n_data)
        )

    if colors == "cycle":
        colors = ["C{}".format(idx % 10) for idx in range(n_data)]
    elif isinstance(colors, str):
        colors = [colors for _ in range(n_data)]

    if not 1 >= credible_interval > 0:
        raise ValueError("The value of credible_interval should be in the interval (0, 1]")

    to_plot = [list(xarray_var_iter(data, var_names, combined=True)) for data in datasets]
    all_labels = []
    length_plotters = []
    for plotters in to_plot:
        length_plotters.append(len(plotters))
        for var_name, selection, _ in plotters:
            label = make_label(var_name, selection)
            if label not in all_labels:
                all_labels.append(label)

    length_plotters = max(length_plotters)
    rows, cols = default_grid(length_plotters, max_cols=3)

    (figsize, _, titlesize, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    fig, ax = _create_axes_grid(length_plotters, rows, cols, figsize=figsize, squeeze=False)

    axis_map = {label: ax_ for label, ax_ in zip(all_labels, ax.flatten())}

    for m_idx, plotters in enumerate(to_plot):
        for var_name, selection, values in plotters:
            label = make_label(var_name, selection)
            _d_helper(
                values.flatten(),
                label,
                colors[m_idx],
                bw,
                titlesize,
                xt_labelsize,
                linewidth,
                markersize,
                credible_interval,
                point_estimate,
                hpd_markers,
                outline,
                shade,
                axis_map[label],
            )

    if n_data > 1:
        for m_idx, label in enumerate(data_labels):
            ax[0].plot([], label=label, c=colors[m_idx], markersize=markersize)
        ax[0].legend(fontsize=xt_labelsize)

    fig.tight_layout()

    return ax


def _d_helper(
    vec,
    vname,
    color,
    bw,
    titlesize,
    xt_labelsize,
    linewidth,
    markersize,
    credible_interval,
    point_estimate,
    hpd_markers,
    outline,
    shade,
    ax,
):
    """Plot an individual dimension.

    Parameters
    ----------
    vec : array
        1D array from trace
    vname : str
        variable name
    color : str
        matplotlib color
    bw : float
        Bandwidth scaling factor. Should be larger than 0. The higher this number the smoother the
        KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule of thumb
        (the default used rule by SciPy).
    titlesize : float
        font size for title
    xt_labelsize : float
       fontsize for xticks
    linewidth : float
        Thickness of lines
    markersize : float
        Size of markers
    credible_interval : float
        Credible intervals. Defaults to 0.94
    point_estimate : str or None
        'mean' or 'median'
    shade : float
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    ax : matplotlib axes
    """
    if vec.dtype.kind == "f":
        if credible_interval != 1:
            hpd_ = hpd(vec, credible_interval)
            new_vec = vec[(vec >= hpd_[0]) & (vec <= hpd_[1])]
        else:
            new_vec = vec

        density, xmin, xmax = _fast_kde(new_vec, bw=bw)
        density *= credible_interval
        x = np.linspace(xmin, xmax, len(density))
        ymin = density[0]
        ymax = density[-1]

        if outline:
            ax.plot(x, density, color=color, lw=linewidth)
            ax.plot([xmin, xmin], [-ymin / 100, ymin], color=color, ls="-", lw=linewidth)
            ax.plot([xmax, xmax], [-ymax / 100, ymax], color=color, ls="-", lw=linewidth)

        if shade:
            ax.fill_between(x, density, color=color, alpha=shade)

    else:
        xmin, xmax = hpd(vec, credible_interval)
        bins = range(xmin, xmax + 2)
        if outline:
            ax.hist(vec, bins=bins, color=color, histtype="step", align="left")
        if shade:
            ax.hist(vec, bins=bins, color=color, alpha=shade)

    if hpd_markers:
        ax.plot(xmin, 0, hpd_markers, color=color, markeredgecolor="k", markersize=markersize)
        ax.plot(xmax, 0, hpd_markers, color=color, markeredgecolor="k", markersize=markersize)

    if point_estimate is not None:
        if point_estimate == "mean":
            est = np.mean(vec)
        elif point_estimate == "median":
            est = np.median(vec)
        ax.plot(est, 0, "o", color=color, markeredgecolor="k", markersize=markersize)

    ax.set_yticks([])
    ax.set_title(vname, fontsize=titlesize, wrap=True)
    for pos in ["left", "right", "top"]:
        ax.spines[pos].set_visible(0)
    ax.tick_params(labelsize=xt_labelsize)
