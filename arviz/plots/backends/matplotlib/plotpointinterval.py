import numpy as np

from ...plot_utils import calculate_point_estimate
from ....stats import hdi


def plot_point_interval(
    ax, values, point_estimate, hdi_prob, linewidth, markersize, rotated, interval_kwargs=None
):

    """ Plots point intervals
    
    Translates the data and represents them as point and interval summaries
    
    Parameters
    ----------
    ax : axes
        Matplotlib axes
    values : array-like
        Values to plot
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be ‘mean’, ‘median’, ‘mode’ or None. 
        Defaults to ‘auto’ i.e. it falls back to default set in rcParams.
    linewidth : int
        Line width throughout. If None it will be autoscaled based on figsize.
    markersize : int
        Markersize throughout. If None it will be autoscaled based on figsize.
    hdi_prob : float
        Valid only when point_interval is True. Plots HDI for chosen percentage of density. 
        Defaults to 0.94.
    rotated : bool
        Whether to rotate the dot plot by 90 degrees.
    interval_kwargs : dict
        Keyword passed to the point interval
    """

    endpoint = 100 * (1 - hdi_prob) / 2
    qlist_interval = [endpoint, 25, 75, 100 - endpoint]
    quantiles_interval = np.percentile(values, qlist_interval)
    quantiles_interval[0], quantiles_interval[-1] = hdi(
        values.flatten(), hdi_prob, multimodal=False
    )
    mid = len(quantiles_interval) // 2
    param_iter = zip(np.linspace(2 * linewidth, linewidth, mid, endpoint=True)[-1::-1], range(mid))

    for width, j in param_iter:
        if rotated:
            ax.vlines(
                0,
                quantiles_interval[j],
                quantiles_interval[-(j + 1)],
                linewidth=width,
                **interval_kwargs
            )
        else:
            ax.hlines(
                0,
                quantiles_interval[j],
                quantiles_interval[-(j + 1)],
                linewidth=width,
                **interval_kwargs
            )

    if point_estimate:
        point_value = calculate_point_estimate(point_estimate, values)
        if rotated:
            ax.plot(
                0, point_value, "o", markersize=markersize, color="black",
            )
        else:
            ax.plot(
                point_value, 0, "o", markersize=markersize, color="black",
            )

        return ax
