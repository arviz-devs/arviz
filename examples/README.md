# Charts for Example Gallery

[See Example Gallery](https://python.arviz.org/en/latest/examples/index.html)

## How to add a new chart to the Example Gallery

`Matplotlib` file is required for the chart to appear in the example gallery. `Bokeh` uses the same metadata as long as the Plot Title and `{plot_name}` are the same.

## Use the templates below to create new charts.

| {Metadata} | Description |
| --- | --- |
| Plot Title | Title of the chart shown in example gallery and table of contents |
| Gallery Category | Single Category in example gallery and table of contents |

#### Gallery Categories
```
[
    "Mixed Plots",
    "Distributions",
    "Distribution Comparisons",
    "Inference Diagnostics",
    "Regression Timeseries",
    "Model Comparisons",
    "Model Validations",
    "Miscellaneous",
    "Styles",
]
```

#### Matplotlib (Required)

Create `mpl_plot_{plot_name}.py` under `matplotlib/`.

```
"""
{Plot Title}
=========
_gallery_category: {Gallery Category}
_overlay_desc: {Overlay Description}
"""
{Additional imports here}

import arviz as az

az.style.use("arviz-doc")

{Additional code here}
```

#### Bokeh (Recommended)

Create `bokeh_plot_{plot_name}.py` under `matplotlib/`.

```
"""
{Plot Title, must match the one in Matplotlib}
=========
"""
{Additional imports here}

import arviz as az

az.style.use("arviz-doc")

{Additional code here}
```