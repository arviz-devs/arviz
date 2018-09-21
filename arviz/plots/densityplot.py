"""KDE and histogram plots for multiple variables."""
import numpy as np
import matplotlib.pyplot as plt

from ..data import convert_to_dataset
from ..stats import hpd
from .kdeplot import _fast_kde
from .plot_utils import _scale_text, make_label, xarray_var_iter


# pylint:disable-msg=too-many-function-args
def plot_density(data, data_labels=None, var_names=None, credible_interval=0.94,
                 point_estimate='mean', colors='cycle', outline=True, hpd_markers='', shade=0.,
                 bw=4.5, figsize=None, textsize=None):
    """Generate KDE plots for continuous variables and histograms for discrete ones.

    Plots are truncated at their 100*(1-alpha)% credible intervals. Plots are grouped per variable
    and colors assigned to models.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    data_labels : list[str]
        List with names for the samples in the list of datasets. Useful when
        plotting more than one trace.
    var_names: list, optional
        List of variables to plot (defaults to None, which results in all
        variables plotted).
    credible_interval : float
        Credible intervals. Defaults to 0.94.
    point_estimate : str or None
        Plot point estimate per variable. Values should be 'mean', 'median' or None.
        Defaults to 'mean'.
    colors : list or string, optional
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically choose a color per model from matplolib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to `cycle`.
    outline : boolean
        Use a line to draw KDEs and histograms. Default to True
    hpd_markers : str
        A valid `matplotlib.markers` like 'v', used to indicate the limits of the hpd interval.
        Defaults to empty string (no marker).
    shade : float
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    figsize : tuple
        Figure size. If None, size is (6, number of variables * 2)
    textsize: int
        Text size for labels and legend. If None it will be autoscaled based on figsize.

    Returns
    -------
    ax : Matplotlib axes
    """
    if not isinstance(data, (list, tuple)):
        datasets = [convert_to_dataset(data, group='posterior')]
    else:
        datasets = [convert_to_dataset(d, group='posterior') for d in data]

    if point_estimate not in ('mean', 'median', None):
        raise ValueError(
            "Point estimate should be 'mean', 'median' or None, not {}".format(
                point_estimate
            )
        )

    n_data = len(datasets)

    if data_labels is None:
        if n_data > 1:
            data_labels = ['{}'.format(idx) for idx in range(n_data)]
        else:
            data_labels = ['']
    elif len(data_labels) != n_data:
        raise ValueError(
            'The number of names for the models ({}) '
            'does not match the number of models ({})'.format(
                len(data_labels), n_data
            )
        )

    if colors == 'cycle':
        colors = ['C{}'.format(idx % 10) for idx in range(n_data)]
    elif isinstance(colors, str):
        colors = [colors for _ in range(n_data)]

    to_plot = [list(xarray_var_iter(data, var_names, combined=True)) for data in datasets]
    all_labels = set()
    for plotters in to_plot:
        for var_name, selection, _ in plotters:
            all_labels.add(make_label(var_name, selection))

    if figsize is None:
        figsize = (6, len(all_labels) * 2)

    textsize, linewidth, markersize = _scale_text(figsize, textsize=textsize)

    fig, axes = plt.subplots(len(all_labels), 1, squeeze=False, figsize=figsize)
    axis_map = {label: ax for label, ax in zip(all_labels, axes.flatten())}

    for m_idx, plotters in enumerate(to_plot):
        for var_name, selection, values in plotters:
            label = make_label(var_name, selection)
            _d_helper(values.flatten(), label, colors[m_idx], bw, textsize, linewidth, markersize,
                      credible_interval, point_estimate, hpd_markers, outline, shade,
                      axis_map[label])

    if n_data > 1:
        ax = axes.flatten()[0]
        for m_idx, label in enumerate(data_labels):
            ax.plot([], label=label, c=colors[m_idx], markersize=markersize)
        ax.legend(fontsize=textsize)

    fig.tight_layout()

    return axes


def _d_helper(vec, vname, color, bw, textsize, linewidth, markersize, credible_interval,
              point_estimate, hpd_markers, outline, shade, ax):
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
    textsize : float
        Fontsize of text
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
    if vec.dtype.kind == 'f':
        density, lower, upper = _fast_kde(vec, bw=bw)
        x = np.linspace(lower, upper, len(density))
        hpd_ = hpd(vec, credible_interval)
        cut = (x >= hpd_[0]) & (x <= hpd_[1])

        xmin = x[cut][0]
        xmax = x[cut][-1]
        ymin = density[cut][0]
        ymax = density[cut][-1]

        if outline:
            ax.plot(x[cut], density[cut], color=color, lw=linewidth)
            ax.plot([xmin, xmin], [-ymin/100, ymin], color=color, ls='-', lw=linewidth)
            ax.plot([xmax, xmax], [-ymax/100, ymax], color=color, ls='-', lw=linewidth)

        if shade:
            ax.fill_between(x, density, where=cut, color=color, alpha=shade)

    else:
        xmin, xmax = hpd(vec, credible_interval)
        bins = range(xmin, xmax + 2)
        if outline:
            ax.hist(vec, bins=bins, color=color, histtype='step', align='left')
        if shade:
            ax.hist(vec, bins=bins, color=color, alpha=shade)

    if hpd_markers:
        ax.plot(xmin, 0, hpd_markers, color=color, markeredgecolor='k', markersize=markersize)
        ax.plot(xmax, 0, hpd_markers, color=color, markeredgecolor='k', markersize=markersize)

    if point_estimate is not None:
        if point_estimate == 'mean':
            est = np.mean(vec)
        elif point_estimate == 'median':
            est = np.median(vec)
        ax.plot(est, -0.001, 'o', color=color, markeredgecolor='k', markersize=markersize)

    ax.set_yticks([])
    ax.set_title(vname)
    for pos in ['left', 'right', 'top']:
        ax.spines[pos].set_visible(0)
    ax.tick_params(labelsize=textsize)
