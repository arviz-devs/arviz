def plot_dist(
    values,
    values2=None,
    color="C0",
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
    hist_kwargs=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    ax=None,
):

    if ax is None:
        ax = plt.gca()

    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs.setdefault("cumulative", cumulative)

    if plot_kwargs is None:
        plot_kwargs = {}

    if rotated:
        hist_kwargs.setdefault("orientation", "horizontal")
    else:
        hist_kwargs.setdefault("orientation", "vertical")

    if kind == "auto":
        kind = "hist" if values.dtype.kind == "i" else "density"

    if kind == "hist":
        _histplot_op(
            values=values,
            values2=values2,
            color=color,
            label=label,
            rotated=rotated,
            ax=ax,
            hist_kwargs=hist_kwargs,
        )
    elif kind == "density":
        plot_kwargs.setdefault("color", color)
        legend = True if label is not None else False

        plot_kde(
            values,
            values2,
            cumulative=cumulative,
            rug=rug,
            label=label,
            bw=bw,
            quantiles=quantiles,
            rotated=rotated,
            contour=contour,
            legend=legend,
            fill_last=fill_last,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            contour_kwargs=contour_kwargs,
            ax=ax,
        )


def _histplot_op(values, values2, color, label, rotated, ax, hist_kwargs):
    """Add a histogram for the data to the axes."""

    if values2 is not None:
        raise NotImplementedError("Insert hexbin plot here")
    else:
        bins = _get_bins(values)
        ax.hist(
            values,
            bins=bins,
            color=color,
            label=label,
            rwidth=0.9,
            align="left",
            density=True,
            **hist_kwargs
        )
        if rotated:
            ax.set_yticks(bins[:-1])
        else:
            ax.set_xticks(bins[:-1])
        if label is not None:
            ax.legend()
    return ax
