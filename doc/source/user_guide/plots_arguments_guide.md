---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(plots_arguments_guide)=
# Plots' arguments guide

Arviz {ref}`plot <plot_api>` module is used for plotting data. It consists of lot of functions that serves different purposes.
Most of these plotting functions have common arguments. These common arguments are explained below with the examples.

<!--- TODO: use names like centered, non_centered, rugby, radon... -->

```{code-cell}
import arviz as az
data = az.load_arviz_data('centered_eight');
non_centered = az.load_arviz_data('non_centered_eight');
```

:::{warning} Page in construction
:::

(common_var_names)=
## `var_names`

Variables to be plotted, if None all variables are plotted. Prefix the variables by ~ when you want to exclude them from the plot. Let's see the examples.

Plot default autocorrelation

```{code-cell}
az.plot_posterior(data);
```

Plot one variable by setting `var_names=var1`

```{code-cell}
az.plot_posterior(data, var_names=['mu']);
```

Plot subset variables by specifying variable name exactly

```{code-cell}
az.plot_posterior(data, var_names=['mu', 'tau']);
```

Plot variables with regular expressions
```{code-cell}
az.plot_posterior(data, var_names=['mu', '^the'], filter_vars="regex");
```

(common_filter_vars)=
## `filter_vars`
If None (default), interpret `var_names` as the real variables names.

Plot using the default value of `filter_vars` which is `None`

```{code-cell}
az.plot_posterior(data, var_names=['mu']);
```

If “like”, interpret `var_names` as substrings of the real variables names.

Plot using `filter_vars="like"`

```{code-cell}
az.plot_posterior(data, var_names=['mu', '^the'], filter_vars="like");
```

If “regex”, interpret `var_names` as regular expressions on the real variables names. A la `pandas.filter`.

Plot using `filter_vars="regex"`

```{code-cell}
az.plot_posterior(data, var_names=['mu', '^the'], filter_vars="regex");
```

(common_coords)=
## `coords`
Dictionary mapping dimensions to selected coordinates to be plotted. Dimensions without a mapping specified will include all coordinates for that dimension. Defaults to including all coordinates for all dimensions if None.

Using coords argument to plot only a subset of data

```{code-cell}
coords = {"school": ["Choate","Phillips Exeter"]};
az.plot_posterior(data, var_names=["mu", "theta"], coords=coords);
```

(common_hdi_prob)=
## `hdi_prob`

(common_color)=
## `color`

(common_grid)=
## `grid`
Number of rows and columns. Defaults to None, the rows and columns are automatically inferred.

Plot variables in a 4x5 grid

```{code-cell}
az.plot_density([data, non_centered], grid=(4, 5));
```

(common_figsize)=
## `figsize`

`figsize` is short for figure size. If None it will be defined automatically.

(common_textsize)=
## `textsize`

(common_legend)=
## `legend`

(common_ax)=
## `ax`

(common_backend_kwargs)=
## `backend_kwargs`

(common_show)=
## `show`
