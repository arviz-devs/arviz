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

(getting_started)=
# Getting started

## Have you used ArviZ versions earlier than 1.0?

We recommend reading the {ref}`migration_guide` instead.

(installation)=
## Installation

```{code-block} bash
pip install "arviz[<I/O>, <plotting>]"
```

ArviZ brings together several smaller components into one library. Some features, like plotting or reading and writing files, may not be included by default, depending on what you have installed. To make sure you get the features you need, we recommend choosing them when you install ArviZ. You can select from the following options:

* I/O: `zarr`, `netcdf4` and `h5netcdf`
* Plotting: `matplotlib`, `bokeh` and `plotly`


For example, if you want to install Zarr and Matplotlib


```{code-block} bash
pip install "arviz[zarr, matplotlib]"
```

If instead you want to use NetCDF and interactive plotting with Plotly and Bokeh then you would run:

```{code-block} bash
pip install "arviz[h5netcdf, plotly, bokeh]"
```

Note that you can mix and match any of the available options. You’re not limited to choosing just one file format or one plotting library.

### Verifying the installation

```{code-block} python
import arviz as az
print(az.info)
```

This should print the version of ArviZ and the libraries that comprise it.
It should look like:

```{code-block} none
Status information for ArviZ 1.0.0

arviz_base 0.7.0 available, exposing its functions as part of the `arviz` namespace
arviz_stats 0.7.0 available, exposing its functions as part of the `arviz` namespace
arviz_plots 0.7.0 available, exposing its functions as part of the `arviz` namespace
```

If any of the 3 libraries is missing or can't be imported, the first step for troubleshooting
should be going over the error messages at import time:

```{code-block} python
import logging
logging.basicConfig(level=logging.INFO)
import arviz as az
```

## Learning about the concepts and algorithms powering ArviZ

We have an [online book](https://arviz-devs.github.io/EABM/) covering these concepts.

## ArviZ usage details

If you are already comfortable with the different tasks needed for Bayesian modeling
and want to know how to use ArviZ to simplify your workflows, we recommend starting
at the {ref}`arviz_plots:overview_plots` page or `arviz-plots`.

## Basic ArviZ workflow

A typical workflow in ArviZ consists of:

1. Loading inference results
2. Converting them into `InferenceData`
3. Inspecting summary statistics
4. Visualizing posterior samples

### Example

```python
import arviz as az

# Load example dataset
idata = az.load_arviz_data("centered_eight")

# Summary statistics
print(az.summary(idata))

# Trace plot
az.plot_trace(idata)

# Posterior distribution
az.plot_posterior(idata)
```
### Recommended installation

For most users, installing ArviZ with plotting support is recommended:

```bash
pip install "arviz[plotting]"
```

## What is InferenceData?

`InferenceData` is the core data structure used in ArviZ.

It stores:

- posterior samples
- prior samples
- observed data
- metadata

This standardized format allows ArviZ to provide consistent diagnostics
and visualizations across different probabilistic programming libraries.


