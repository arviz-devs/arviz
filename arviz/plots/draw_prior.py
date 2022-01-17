"""Prior elicitation using roulette mode."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.patches as patches
from matplotlib.widgets import RadioButtons, TextBox
from math import ceil, floor

from ..rcparams import rcParams


def create_grid(x_min=0, x_max=1, nrows=10, ncols=10, ax=None):
    xx = np.arange(0, ncols)
    yy = np.arange(0, nrows)

    if ncols < 10:
        num = ncols
    else:
        num = 10

    ax[0, 0].set(
        xticks=np.linspace(0, ncols, num=num),
        xticklabels=[f"{i:.1f}" for i in np.linspace(x_min, x_max, num=num)]
        ,
    )

    coll = np.zeros((nrows, ncols), dtype=object)
    for idx, xi in enumerate(xx):
        for idy, yi in enumerate(yy):
            sq = patches.Rectangle((xi, yi), 1, 1, fill=True, facecolor="0.8", edgecolor="w")
            ax[0, 0].add_patch(sq)
            coll[idy, idx] = sq
    return coll


class Rectangles:
    def __init__(self, fig, coll, nrows, ncols, ax):
        self.fig = fig
        self.coll = coll
        self.nrows = nrows
        self.ncols = ncols
        self.ax = ax
        self.weights = {k: 0 for k in range(0, ncols)}
        fig.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        if event.inaxes == self.ax[0, 0]:
            x = event.xdata
            y = event.ydata
            idx = floor(x)
            idy = ceil(y)
            if self.weights[idx] >= idy:
                idy -= 1
                for row in range(self.nrows):
                    self.coll[row, idx].set_facecolor("0.8")
            self.weights[idx] = idy
            for idy in range(idy):
                self.coll[idy, idx].set_facecolor("C1")
            self.fig.canvas.draw()


def on_leave_fig(event, grid, radio, lower_q, upper_q, min_range, max_range, ncols, ax):
    x_min = float(min_range.text)
    x_max = float(max_range.text)
    x_range = x_max - x_min
    samples = []
    ok = 0

    if any(grid.weights.values()):
        for k, v in grid.weights.items():
            if v != 0:
                ok += 1
            la = np.repeat((k / ncols * x_range) + x_min + ((x_range / ncols) / 2), v*100+1)
            samples.extend(la)

        x = np.linspace(x_min, x_max, 500)
        dist_name = radio.value_selected
        if dist_name == "automatic":
            sum_pdf_old = -np.inf
            for dist in [stats.beta, stats.norm, stats.gamma, stats.lognorm]:
                if dist.name == "beta":
                    a, b, _, _ = dist.fit(samples, floc=x_min, fscale=x_range)
                    can_dist = dist(a, b, loc=x_min, scale=x_max)
                elif dist.name in "norm":
                    a, b = dist.fit(samples)
                    can_dist = dist(a, b)
                elif dist.name == "gamma" and x_min >= 0:
                    a, _, b = dist.fit(samples)
                    can_dist = dist(a, scale=b)
                    b = 1 / b
                elif dist.name == "lognorm" and x_min >= 0:
                    b, _, a = dist.fit(samples)
                    can_dist = dist(b, scale=a)
                    a = np.log(a)
                logpdf = can_dist.logpdf(x)
                sum_pdf = np.sum(logpdf[np.isfinite(logpdf)])
                pdf = np.exp(logpdf)
                if sum_pdf > sum_pdf_old:
                    sum_pdf_old = sum_pdf
                    ref_pdf = pdf
                    dist_name = dist.name
                    if dist_name == "beta" and (x_min < 0 or x_max > 1):
                        dist_name = "scaled beta"
                    params = a, b
                    fitted_dist = can_dist
        else:
            if dist_name == "beta":
                a, b, _, _ = stats.beta.fit(samples, floc=x_min, fscale=x_range)
                fitted_dist = stats.beta(a, b, loc=x_min, scale=x_range)
                ref_pdf = fitted_dist.pdf(x)
                if x_min < 0 or x_max > 1:
                    dist_name = "scaled beta"
            elif dist_name == "normal":
                a, b = stats.norm.fit(samples)
                fitted_dist = stats.norm(a, b)
                ref_pdf = fitted_dist.pdf(x)
            elif dist_name == "gamma":
                a, _, b = stats.gamma.fit(samples)
                fitted_dist = stats.gamma(a, scale=b)
                b = 1 / b
                ref_pdf = fitted_dist.pdf(x)
            elif dist_name == "lognorm":
                b, _, a = stats.lognorm.fit(samples)
                fitted_dist = stats.lognorm(b, scale=a)
                a = np.log(a)
                ref_pdf = fitted_dist.pdf(x)
            params = a, b

        title = f"{dist_name}({params[0]:.1f}, {params[1]:.1f})"

        reset_dist_panel(x_min, x_max, ax)
        if ok > 1:
            ax[1, 0].plot(x, ref_pdf)
            q0 = fitted_dist.ppf(float(lower_q.text))
            q1 = fitted_dist.ppf(float(upper_q.text))
            ax[1, 0].axvline(q0, 0, 1, ls="--", color="0.5")
            ax[1, 0].axvline(q1, 0, 1, ls="--", color="0.5")
            support = fitted_dist.support()
            for bound in support:
                if np.isfinite(bound):
                    ax[1, 0].plot(bound, 0, "ko")
            ax[1, 0].set_title(title)
    event.canvas.draw()


def update_grid(text, min_range, max_range, n_rows, n_cols, grid, ax):
    x_min = float(min_range.text)
    x_max = float(max_range.text)
    ncols = int(n_cols.text)
    nrows = int(n_rows.text)
    ax[0, 0].cla()
    coll = create_grid(x_min=x_min, x_max=x_max, nrows=nrows, ncols=ncols, ax=ax)
    grid.coll = coll
    grid.ncols = ncols
    grid.nrows = nrows
    grid.weights = {k: 0 for k in range(0, ncols)}
    reset_dist_panel(x_min, x_max, ax)
    ax[0, 0].set_yticks([])
    ax[0, 0].relim()
    ax[0, 0].autoscale_view()


def reset_dist_panel(x_min, x_max, ax):
    ax[1, 0].cla()
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim(x_min, x_max)


def draw_prior(x_min=0, x_max=1, nrows=10, ncols=10, backend=None):

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()
    if backend == "bokeh":
        raise TypeError("Draw prior is only supported with matplotlib backend.")

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell" and get_backend() != "nbAgg":
            raise Warning(
                "To run animations inside a notebook you have to use the nbAgg backend. "
                "Try with `%matplotlib notebook` or  `%matplotlib  nbAgg`. You can switch "
                "back to the default backend with `%matplotlib  inline` or "
                "`%matplotlib  auto`."
            )
    except NameError:
        pass

    fig, ax = plt.subplots(2, 2, gridspec_kw={"width_ratios": [5, 1]}, constrained_layout=True)

    coll = create_grid(x_min, x_max, nrows, ncols, ax=ax)

    ################################################################
    # Make checkbuttons
    labels = ["automatic", "beta", "normal", "gamma", "lognorm"]
    radio = RadioButtons(ax[0, 1], labels)

    # Set text boxes for x-range
    min_range_ax = plt.axes([0.915, 0.4, 0.07, 0.03], facecolor="0.2")
    [s.set_visible(False) for s in min_range_ax.spines.values()]
    max_range_ax = plt.axes([0.915, 0.35, 0.07, 0.03], facecolor="0.2")
    [s.set_visible(False) for s in max_range_ax.spines.values()]
    min_range = TextBox(min_range_ax, "x_min", x_min, hovercolor="0.8", label_pad=0.1)
    max_range = TextBox(max_range_ax, "x_max", x_max, hovercolor="0.8", label_pad=0.1)

    # Set text boxes for grid rows and columns
    n_cols_ax = plt.axes([0.915, 0.29, 0.07, 0.03], facecolor="0.2")
    [s.set_visible(False) for s in n_cols_ax.spines.values()]
    n_cols = TextBox(n_cols_ax, "x_bins", ncols, hovercolor="0.8", label_pad=0.1)
    n_rows_ax = plt.axes([0.915, 0.24, 0.07, 0.03], facecolor="0.2")
    [s.set_visible(False) for s in n_rows_ax.spines.values()]
    n_rows = TextBox(n_rows_ax, "y_bins", nrows, hovercolor="0.8", label_pad=0.1)

    # Set text boxes for quantiles
    lower_q_ax = plt.axes([0.915, 0.15, 0.07, 0.03], facecolor="0.2")
    [s.set_visible(False) for s in lower_q_ax.spines.values()]
    upper_q_ax = plt.axes([0.915, 0.10, 0.07, 0.03], facecolor="0.2")
    [s.set_visible(False) for s in upper_q_ax.spines.values()]
    lower_q = TextBox(lower_q_ax, "lower", 0.33, hovercolor="0.8", label_pad=0.1)
    upper_q = TextBox(upper_q_ax, "upper", 0.66, hovercolor="0.8", label_pad=0.1)

    ax[0, 0].set_title("draw your distribution")
    ax[0, 0].set_yticks([])
    ax[1, 0].set_yticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xticks([])
    ax[0, 0].relim()
    ax[0, 0].autoscale_view()
    ################################################################

    grid = Rectangles(fig, coll, nrows, ncols, ax)
    min_range.on_submit(
        lambda event: update_grid(event, min_range, max_range, n_rows, n_cols, grid, ax)
    )
    max_range.on_submit(
        lambda event: update_grid(event, min_range, max_range, n_rows, n_cols, grid, ax)
    )
    n_cols.on_submit(
        lambda event: update_grid(event, min_range, max_range, n_rows, n_cols, grid, ax)
    )
    n_rows.on_submit(
        lambda event: update_grid(event, min_range, max_range, n_rows, n_cols, grid, ax)
    )

    fig.canvas.mpl_connect(
        "figure_leave_event",
        lambda event: on_leave_fig(
            event, grid, radio, lower_q, upper_q, min_range, max_range, ncols, ax
        ),
    )
