---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(plotting_with_bokeh)=

# Plotting with Bokeh

Arviz has the option to use {doc}`Bokeh <bokeh:index>` as backend which offers interactive plots. Although most of the functions in the {ref}`plot <plot_api>` module work seamlessly with any backend, some advanced plots may require the use of backend specific features. In this guide, advanced plotting with Bokeh will be covered.

This page can be downloaded as a {jupyter-download:script}`Python script <plotting_with_bokeh>`
or as a {jupyter-download:nb}`Jupyter notebook <plotting_with_bokeh>`.

```{code-cell} ipython3
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

az.style.use("arviz-darkgrid")

# Confgure Bokeh as backend
az.rcParams["plot.backend"] = "bokeh"
az.output_notebook()
```

:::{note}
To display the plots as output in a Jupyter notebook, it's required to call the function  {func}`arviz.output_notebook` beforehand.
:::


## Customizing plots

(bokeh_backend_kwargs)=
### Using `backend_kwargs`

The `backend_kwargs` argument can be very useful for some specific configuration. That is parameters available in {class}`bokeh:bokeh.plotting.figure.Figure`. As the options available depends on the backend, this parameter is not as flexible as creating custom axes.

As an example, the following code changes the color background and the plot width.

```{code-cell} ipython3
# load data
data = az.load_arviz_data('radon')
```

```{code-cell} ipython3
az.plot_posterior(
    data,
    var_names=["g"],
    backend_kwargs={"width": 350, 
                    "background_fill_color": "#d3d0e3"});
```

(bokeh_show)=
### The parameter `show`

The parameter `show` is used to control whether the plot is displayed or not. This behavior can be useful when a plot will be displayed in a grid as the example in the following section.


(bokeh_ax)=
### Defining custom axes

The `ax` argument of any `plot` function allows to use created axes manually. In the example, this parameter allows to arrange 2 different plots in a grid, set limit to the x axis and share the axes' ranges between the two plots.


```{code-cell} ipython3
from bokeh.io import show
from bokeh.layouts import row
from bokeh.plotting import figure

# load data
observed_data = data.observed_data.y.to_numpy()
# create axes
f1 = figure(x_range=(observed_data.min() - 1, observed_data.max() + 1))
f2 = figure(x_range=f1.x_range, y_range=f1.y_range)
# plot
az.plot_ppc(data, group="prior", num_pp_samples=100, show=False, ax=f1)
az.plot_ppc(data, group="posterior", num_pp_samples=100, show=False, ax=f2)

az.show_layout([[f1], [f2]])
```

:::{note}
{func}`arviz.show_layout` creates a bokeh layout and calls shows if `show=True`, which is the default value.
:::

### Extending ArviZ-Bokeh plots

Arviz plot returns a {class}`~bokeh.plotting.Figure` object, therefore different Bokeh plots can be added to a plot created using Arviz:

```{code-cell} ipython3
# load data
data = az.load_arviz_data('regression1d')
X = data.observed_data.y_dim_0.values
Y = data.observed_data.y.values
y_pp = data.posterior_predictive.y.values
# plot
f1 = figure(plot_width=600, plot_height=600, toolbar_location="below")
az.plot_hdi(X, y_pp, color="#b5a7b6", show=False, ax=f1)
f1.scatter(X, Y, marker="circle", fill_color="#0d7591")

show(f1)
```

Similarly, custom axes allow to display Arviz and Bokeh plots in the same grid. In this example, the plot in `f1` has an Arviz plot extended with a Bokeh plot and `f2` has a `scatter` created using Bokeh directly.

```{code-cell} ipython3
# load data
observed_data = data.observed_data.y.values
# create axes
f1 = figure(plot_width=400, plot_height=400, toolbar_location="below")
f2 = figure(plot_width=400, plot_height=400, toolbar_location="below")
# plot
az.plot_hdi(X, y_pp, color="#b5a7b6", show=False, ax=f1)
f1.line(X, y_pp.mean(axis=(0, 1)), color="black")
f2.scatter(X, Y, marker="circle", fill_color="#0d7591")

show(row(f1, f2))
```

## Bokeh with Jupyter

:::{note}
In Juptyer notebooks, Arviz offers {func}`arviz.output_notebook` which is a wrapper over {func}`bokeh.io.output.output_notebook`.

To use Bokeh with JupyterLab, JupyterLab and widgets, please refer to this {ref}`user guide <bokeh:userguide_jupyter>`.
:::


## Server mode

:::{note}
In the examples above, bokeh is being used in `Standalone` mode. 
Bokeh can be also used to create interactive web applications. For more details see {ref}`Running a Bokeh server <bokeh:userguide_server>`.
:::
