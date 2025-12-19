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

## Have you already used <1.0 ArviZ versions?

We recommend reading the {ref}`migration_guide` instead.

(installation)=
## Installation

```{code-block} bash
pip install "arviz[<I/O>, <plotting>]"
```

ArviZ is a meta library that pulls together smaller components.
Depending on the libraries you have installed the plotting or I/O functionality
might not be available so we recommend indicating which ones you want to use
whenever you install ArviZ. The options are:

* I/O: `zarr`, `netcdf4` and `h5netcdf`
* Plotting: `matplotlib`, `bokeh` and `plotly`

Some example install commands for ArviZ:

```{code-block} bash
pip install "arviz[zarr, matplotlib]"
pip install "arviz[h5netcdf, plotly, bokeh]"
```

Note you can use any combination of the available options, it is not restricted
to one I/O and one plotting library.

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
