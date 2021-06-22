
from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import create_axes_grid
from ...plot_utils import plot_point_interval



def plot_dot(
    values,
    binwidth,
    dotsize,
    stackratio,
    hdi_prob,
    quartiles,
    rotated,
    dotcolor,
    intervalcolor,
    markersize,
    figsize,
    linewidth,
    point_estimate,
    quantiles,
    point_interval,
    backend_kwargs,
    plot_kwargs,
    interval_kwargs,
):

    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs.setdefault('match_aspect', True)

    (figsize, _, _, _, auto_linewidth, auto_markersize) = _scale_fig_size(figsize, None)
    dotcolor = vectorized_to_hex(dotcolor)
    intervalcolor = vectorized_to_hex(intervalcolor)


    if plot_kwargs is None:
        plot_kwargs = {}
        plot_kwargs.setdefault("color", dotcolor)

    if linewidth is None:
        linewidth = auto_linewidth

    if markersize is None:
        markersize = auto_markersize

    if interval_kwargs is None:
        interval_kwargs = {}
        interval_kwargs.setdefault("color", intervalcolor)


    ## Wilkinson's Algorithm
    x, y = wilkinson_algorithm(values, quantiles, binwidth, stackratio, rotated)
    
    ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )
    
    ax.circle(x, y, radius=dotsize*(binwidth/2), **plot_kwargs, radius_dimension = 'y')
    if rotated:
        ax.xaxis.major_tick_line_color = None
        ax.xaxis.minor_tick_line_color = None
        ax.xaxis.major_label_text_font_size = '0pt'
    else:
        ax.yaxis.major_tick_line_color = None
        ax.yaxis.minor_tick_line_color = None
        ax.yaxis.major_label_text_font_size = '0pt'

    if point_interval:
        ax = plot_point_interval(
            ax, values, point_estimate, hdi_prob, quartiles, linewidth, markersize, rotated, interval_kwargs, 'bokeh'
        )
        
    show_layout(ax)
    
    return ax


def wilkinson_algorithm(
    values, 
    quantiles, 
    binwidth, 
    stackratio,  
    rotated):

    count = 0
    x, y = [], []
    
    while count < quantiles:
        stack_first_dot = values[count]
        num_dots_stack = 0
        while(values[count] < (binwidth + stack_first_dot)):
            num_dots_stack += 1
            count += 1
            if(count == quantiles):
                break
        x_coord = (stack_first_dot + values[count-1])/2
        y_coord = binwidth/2
        if(rotated):
            x_coord, y_coord = y_coord, x_coord
        x.append(x_coord)
        y.append(y_coord)
        for _ in range(num_dots_stack):
            x.append(x_coord)
            y.append(y_coord)
            if(rotated):
                x_coord += binwidth + (stackratio-1)*(binwidth)
            else:
                y_coord += binwidth + (stackratio-1)*(binwidth)

    return x, y