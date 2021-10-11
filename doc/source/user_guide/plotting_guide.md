---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Plotting guide

ArviZ provides multiple plotting functions for different functionalities. You can check all the functions in the {ref}`plots <plot_api>` module.

(common_arguments)=
## Arguments
Most of the plotting functions have common arguments which are explained below with the examples.

(common_var_names)=
### `var_names`

Variables to be plotted, if None all variables are plotted. Prefix the variables by ~ when you want to exclude them from the plot. Vector-value stochastics are handled automatically.

(var_names_examples)=
#### Examples

Plot default autocorrelation

```{code-cell}
import arviz as az
data = az.load_arviz_data('centered_eight');
az.plot_posterior(data);
```

Plot all the variables by keeping `var_names=None`

```{code-cell}
az.plot_posterior(data, var_names=None);
```

Plot one variable by setting `var_names=var1`

```{code-cell}
az.plot_autocorr(data, var_names=['mu']);
```

Plot subset variables by specifying variable name exactly

```{code-cell}
az.plot_posterior(data, var_names=['mu', 'tau']);
```

Plot variables with regular expressions
```{code-cell}
az.plot_posterior(data, var_names=['mu', '^the'], filter_vars="regex");
```
