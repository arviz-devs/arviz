# Plotting guide

ArviZ provides multiple plotting functions for different functionalities. You can check all the functions in the {ref}`plots <plot_api>` module.

## Arguments
Most of the plotting functions have common arguments which are explained below with the examples.

### `var_names`

Variables to be plotted, if None all variables are plotted. Prefix the variables by ~ when you want to exclude them from the plot. Vector-value stochastics are handled automatically.

#### Examples

Plot default autocorrelation

``` python
import arviz as az
data = az.load_arviz_data('centered_eight')
az.plot_autocorr(data)
```

Plot subset variables by specifying variable name exactly

```
az.plot_autocorr(data, var_names=['mu', 'tau'] )
```

Combine chains by variable and select variables by excluding some with partial naming

```
az.plot_autocorr(data, var_names=['~thet'], filter_vars="like", combined=True)
```