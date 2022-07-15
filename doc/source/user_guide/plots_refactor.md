# Plots refactor overview
The main idea behind this refactor is to separate ArviZ plotting tasks
into 3 separated and independent parts. Hopefully this will make it
easier to:

* Reuse ArviZ components for other goals such as generating
  grids and processing
* Simplify the maintenance and addition of multiple plotting backends.
* Support multiple execution/processing backends.
  - It looks like most PPLs still duplicate ess/rhat...

:::{note}
All names so far are mostly random and 100% open for dicsussion.
The focus right now should is on the API related proposals
and making sure they support everything we want to be possible.
:::

## Data organization
By "data organization" I mean taking the original InferenceData and
generating the plotting grids and selectors to generate the plots.

I propose this depends on a `ArviZGrid` class that can be generated
with `facet_wrap` and `facet_grid` methods.

```
ArviZGrid.facet_wrap(idata, col, overlay, **kwargs)
ArviZGrid.facet_wrap(idata, col, row, overlay, **kwargs)
```

Arguments `row, col, overlay` would take strings or lists of strings: dimension names
over which to facet or overlay. We could then also add more keyword arguments like
`backend_kwargs` or `col_wrap`... (would be nice to try and use similar arguments
to seaborn and xarray internal plotting). Then the `**kwargs` would
be keys of `property` and values of string or lists of strings, like row or col
for properties over which to iterate over and update that aesthetic with every plot.
This is the "standard" overlaying, but I think in our case it is also important to
have this extra `overlay` kwarg to overlay plots without changing any aesthetic
for spaghetti-like plots.

### Challenges
We have multiple complex plots that have multiple elements and we might want
to facet and define different colors for the different elements. I can't really
think about an API for this. An option could be to do this refactor
but keep some of the `plot_posterior`, `plot_ppc` with a similar API to
the current one, which creates the grid under the hood and uses the refactored
plots.

Facetting over model/multiple InferenceData. My current idea is
to have this class take both idata or an iterable of idata.
If the first case, then everything works as is, otherwise
an extra "fake" dimension `model` or `idata` could be used
in the `row/col/overlay/color/...`

We should probably also add some extra "aesthetics" which aren't
really aesthetics such as `shift` or `x/y` to generate a forest_plot
like figure from the `plot_interval` elements.

### Initial ideas on implementation
On implementation, my current idea is for the `__init__` method
of that class to generate the iterable/generator
using a similar process (or even exactly the same as our current
`xarray_sel_iter`) as well as multiple xarray DataArrays/Datasets
with the right shapes: an mpl-axes/bokeh-figure one, a `color`
one...

Then, when processing or plotting we loop over the generator
and get the right axes, color, linestyle... by subsetting over these
xarray objects (using only the present dimensions).
If we then have also methods to update these objects,
we could initialize the element, plot, then update the overlay/color
property and plot another element on the same axes,
or update the axes and make a different graphic with the same
overlay properties but on a different axes.

## Data processing
Still need to see how possible this is, but as a first approximation,
I think the ideal approach would be for processing functions
to take both the original inferencedata and the plotting generator.
It is therefore the processing function that can decide to loop over
the generator or to run the function on the inferencedata/dataset
directly (which will easily support dask, numba... and easy parallelization).

I am not discussing much API yet here because I think this step should generally
be hidden from users. If they ask for `plot_density` we generate the
kde, if they ask for plot_dot we run the wilkinson algorithm...
but it is still nice to abstract it I think.

It should also be easy to define multiple "data processing backends"
with minimal duplication.

### Challenges
There are some plots that "generate" a new dimension. For example `plot_ess`
"generates" the quantile/evolution dimension which is that is used
on the x-axis. I am still thinking about how to make this interact
with the plotting/facetting.

## Plotting
### User facing plotting
As I commented before, it might make sense to try and keep a similar API
to the one we have, using all the elements defined here but
"bundling" multiple elements that are often combined together: plot_ppc, plot_posterior...

As updated/a bit more lower level API, I am not sure if it would be preferable to
have plotting as methods of the data organization class or as functions
that take the data organization class. I am leaning towards the function
option because then it can also be an optional argument and generate
a default facet wrap over all but chain and draw dims like we currently do.
I think it will also be easier to replace the data organization steps but
not the processing/plotting

#### Challenges
Even if we try to abstract processing as much as possible, it still needs
to happen "inside" plotting. The processing depends on the plots
that we want to do. This relates again to the previous section.
The option I am leaning on right now is to have plotting
functions take two classes as arguments, the data organization and
the data processing one.

### Base plotting
ArviZ will have classes or functions (TODO: choose which) for the base plots we use:
lines, scatter, fill_between, bars... These objects would first have the x/y(/z) inputs
followed by _a subset of aesthetics_ and ending with `**kwargs` passed as is
to the actual plotting backend. The initial proposal is to have:

```
line(x, y, color, linestyle, linewidth, alpha, **kwargs)
scatter(x, y, color, marker, size, alpha, **kwargs)
fill_x(x, y1, y2, ...)
fill_y(y, x1, x2, ...)
bars_y(x, y_top, y_bottom=0, ...)
hexbin()
contour()
contourf()
errorbar()
circles() # for dotplot?
```

:::{important}
ArviZ plots would then **only** use these plotting intermediaries.
If at some point we need a new type of plot not covered we'd need
to add its intermediary first, then the actual plot.
:::

Doing this, we'll reduce all duplication between backends, and adding
a new backend to ArviZ (i.e. a json one to save plot info) would
reduce only to adding the intermediaries for that backend.

#### Base Plotting Example
Note: ignore issues with defaults

```python
class ArviZPlot:
    def __init__(self, bakend=None, target=None):
        self.backend = rcParams...
        self.target = self.init_target(target)  # for now, mpl axes or bokeh figure
        for meth in method_list:
            setattr(self, meth, getattr(self, f"{backend}_{meth}"))

    def matplotlib_line(self, x, y, color, linestyle, linewidth, alpha, **kwargs):
        self.target.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, **kwargs)

    def bokeh_line(self, x, y, color, linestyle, alpha, **kwargs):
        self.target.line(x, y, color=color, line_dash=linestyle, line_width=linewidth, alpha=alpha, **kwargs)
```
