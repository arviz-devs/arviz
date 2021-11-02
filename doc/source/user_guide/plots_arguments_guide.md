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

Arviz {ref}`plot <plot_api>` module is used for plotting data. It consists many functions that each serve different purposes.
Most of these plotting functions have common arguments. These common arguments are explained in the following examples.

<!--- TODO: use names like centered, non_centered, rugby, radon... -->

```{code-cell}
import arviz as az
centered_eight = az.load_arviz_data('centered_eight');
non_centered_eight = az.load_arviz_data('non_centered_eight');
```

:::{warning} Page in construction
:::

(common_var_names)=
## `var_names`

Variables to be plotted, if None all variables are plotted. Prefix the variables by ~ when you want to exclude them from the plot. Let's see the examples.

Plot default autocorrelation

```{code-cell}
az.plot_posterior(centered_eight);
```

Plot one variable by setting `var_names=var1`

```{code-cell}
az.plot_posterior(centered_eight, var_names=['mu']);
```

Plot subset variables by specifying variable name exactly

```{code-cell}
az.plot_posterior(centered_eight, var_names=['mu', 'tau']);
```

Plot variables with regular expressions
```{code-cell}
az.plot_posterior(centered_eight, var_names=['mu', '^the'], filter_vars="regex");
```

(common_filter_vars)=
## `filter_vars`
If None (default), interpret `var_names` as the real variables names.

Plot using the default value of `filter_vars` which is `None`

```{code-cell}
az.plot_posterior(centered_eight, var_names=['mu']);
```

If “like”, interpret `var_names` as substrings of the real variables names.

Plot using `filter_vars="like"`

```{code-cell}
az.plot_posterior(centered_eight, var_names=['mu', 'the'], filter_vars="like");
```

If “regex”, interpret `var_names` as regular expressions on the real variables names. A la `pandas.filter`.

Plot using `filter_vars="regex"`

```{code-cell}
az.plot_posterior(centered_eight, var_names=['mu', '^the'], filter_vars="regex");
```

(common_coords)=
## `coords`
Dictionary mapping dimensions to selected coordinates to be plotted. Dimensions without a mapping specified will include all coordinates for that dimension. Defaults to including all coordinates for all dimensions if None.

Using coords argument to plot only a subset of data

```{code-cell}
coords = {"school": ["Choate","Phillips Exeter"]};
az.plot_posterior(centered_eight, var_names=["mu", "theta"], coords=coords);
```

(common_combine_dims)=
## `combine_dims`
Set like argument containing dimensions to reduce for plots that represent probability
distributions. When used, the dimensions on `combine_dims` are added to `chain` and
`draw` dimensions and reduced to generate a single KDE/histogram/dot plot from >2D
arrays.

`combine_dims` dims can be used to generate the KDE of the distribution of `theta`
when combining all schools together.

```{code-cell}
az.plot_posterior(centered_eight, var_names=["mu", "theta"], combine_dims={"school"})
```

:::{warning}
`plot_pair` also supports `combine_dims` argument, but it's the users responsibility
to ensure the variables to be plotted have compatible dimensions when reducing
them pairwise.
:::

Both `theta` and `theta_t` have the `school` dimension, so we can use `combine_dims`
in `plot_pair` to generate their global 2d KDE.

```{code-cell}
az.plot_pair(
    non_centered_eight, var_names=["theta", "theta_t"], combine_dims={"school"}, kind="kde"
)
```

`mu` however does not, so trying to plot their combined distribution errors out:

```{code-cell}
:tags: [raises-exception]

az.plot_pair(non_centered_eight, var_names=["theta", "mu"], combine_dims={"school"})
```

(common_hdi_prob)=
## `hdi_prob`

Probability for the highest density interval. Defaults to ``stats.hdi_prob`` rcParam.
    

(common_color)=
## `color`

(common_grid)=
## `grid`
Number of rows and columns. Defaults to None, the rows and columns are automatically inferred.

Plot variables in a 4x5 grid

```{code-cell}
az.plot_density([centered_eight, non_centered_eight], grid=(4, 5));
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
