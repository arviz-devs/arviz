# How to add to the Example Gallery

`Matplotlib` file is required for the chart to appear in the {ref}`example_gallery`. `Bokeh` uses the same metadata as long as the Plot Title and `{plot_name}` are the same.

**Use the templates below to create new charts.**

| {Metadata} | Description |
| --- | --- |
| Plot Title | Title of the chart shown in example gallery and table of contents |
| Gallery Category | Single Category in example gallery and table of contents, defaults to Miscellaneous|
| Alt Text | Text for overlay and alternative text for img thumbnail |

**Gallery Categories**
```
[
    "Mixed Plots",
    "Distributions",
    "Distribution Comparison",
    "Inference Diagnostics",
    "Regression or Time Series",
    "Model Comparison",
    "Model Checking",
    "Miscellaneous",
    "Styles",
]
```

## Matplotlib Template (Required)

Create `mpl_plot_{plot_name}.py` under `matplotlib/`.

```
"""
{Plot Title}
=========
_gallery_category: {Gallery Category}
_alt_text: {Alt Text}
"""
{Additional imports here}

import arviz as az

az.style.use("arviz-doc")

{Additional code here}
```

## Bokeh Template

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