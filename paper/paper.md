---
title: 'ArviZ: a modular and flexible library for exploratory analysis of Bayesian models'
tags:
  - Python
  - Bayesian statistics
  - Bayesian workflow
authors:
  - name: Osvaldo A Martin
    orcid: 0000-0001-7419-8978
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Oriol Abril-Pla
    orcid: 0000-0002-1847-9481
    equal-contrib: true
    corresponding: true
    affiliation: 2
  - name: Jordan Deklerk
    affiliation: 3
  - name: Colin Carroll
    orcid: 0000-0001-6977-0861
    affiliation: 2
  - name: Ari Hartikainen
    orcid: 0000-0002-4569-569X
    affiliation: 2
  - name: Aki Vehtari
    orcid: 0000-0003-2164-9469
    affiliation: 4,1
affiliations:
 - name:  Aalto University, Espoo, Finland
   index: 1
 - name: arviz-devs
   index: 2
 - name: DICK's Sporting Goods, Coraopolis, Pennsylvania
   index: 3
 - name: ELLIS Institute Finland
   index: 4
date: 22 December 2025
bibliography: references.bib
---

# Summary

`ArviZ` [@Kumar_2019] is a Python package for exploratory analysis of Bayesian models that has been widely used in academia and industry since its introduction in 2019. It's goal is to integrate seamlessly with established probabilistic programming languages and statistical interfaces, such as PyMC [@Abril-pla_2023], Stan (via the cmdstanpy interface) [@stan], Pyro, and NumPyro [@Phan_2019; @Bingham_2019], emcee [@emcee], and Bambi [@Capretto_2022], among others.

`ArviZ` is part of the broader ArviZ-project, which develops tools for Exploratory Analysis of Bayesian Models. The organization also maintains other initiatives, including arviz.jl (for Julia), PreliZ [@icazatti_2023], educational resources [@eabm_2025], and additional packages that are still in an experimental phase.

In this work, we present a redesigned version of `ArviZ` that emphasizes greater user control and modularity. This redesign delivers a more flexible and efficient toolkit for exploratory analysis of Bayesian models. With its renewed focus on modularity and usability, `ArviZ` is well-positioned to remain an essential tool for Bayesian modelers in both research and applied settings.

# Statement of need

Probabilistic programming has emerged as a powerful paradigm for statistical modeling, accompanied by a growing ecosystem of tools for model specification and inference. Effective modeling requires robust support for sampling diagnostics, model comparison, and model checking [@Gelman_2020; @Martin_2024; @Guo_2024]. `ArviZ` addresses this gap by providing a unified, backend-agnostic library to perform these tasks.

The methods implemented in `ArviZ` are grounded in well-established statistical principles and provide robust, interpretable diagnostics and visualizations [@Vehtari_2017; @Gelman_2019; @Paananen_2021; @Vehtari_2021; @Dimitriadis_2021; @Sailynoja_2022; @Kallioinen_2023; @Sailynoja_2025]. The redesigned version furthers these goals by introducing an easier-to-use interface for regular users and more powerful tooling for power users and developers of Bayesian tools. These updates align with recent developments in the probabilistic programming field. Additionally, the new design facilitates the use of components as modular building blocks for custom analyses. This frequent user request was difficult to accommodate under the old framework.

# Description

We present a redesigned version of `ArviZ` emphasizing greater user control and modularity. The new architecture enables users to customize the installation and use of specific components. The previous `ArviZ` design divided the package into three submodules, which are now available as three independent installable packages with improved design as described next.

General functionality, data processing, and data input/output have been streamlined and enhanced for greater versatility. Previously, `ArviZ` used the custom `InferenceData` class to organize and store the high-dimensional outputs of Bayesian inference in a structured, labeled format, enabling efficient analysis, metadata persistence, and serialization. These have been replaced with the `DataTree` class from xarray [@Hoyer_2017]. Additionally, converters allow more flexibility in dimensionality, naming, and indexing of their generated outputs.

Statistical functions are now accessible through two distinct interfaces:

* A low-level array interface with minimal dependencies, intended for advanced users
and developers of third-party libraries.
* A higher-level xarray interface designed for end users, which simplifies usage by automating common tasks and handling metadata.

Plotting functions have also been redesigned to support modularity at multiple levels:

* At a high level, `ArviZ` offers a collection of “batteries-included” plots. These are built-in plotting functions providing sensible defaults for common tasks like MCMC sampling diagnostics, predictive checks, and model comparison.
* At an intermediate level, the API enables easier customization of batteries-included plots and simplifies the creation of new plots. This is achieved through the `PlotCollection` class, which enables developers and advanced users to focus solely on the plotting logic, without needing to handle faceting or aesthetics.
* At a lower level, we have improved the separation between computational and plotting logic, reducing code duplication and enhancing modular design. These changes also facilitate support for multiple plotting backends, improving extensibility and maintainability. Currently, `ArviZ` supports three plotting backends: matplotlib [@Hunter_2007], Bokeh [@Bokeh_2018], and plotly [@plotly_2015].


## Examples

For the first example, we construct an array resembling data from MCMC sampling. We have 4 chains and 1000 draws for two posterior variables. We can compute the effective sample sizes for this array using the stats interface. For this, we need to specify which axes represent the chains and which the draws.

    import numpy as np
    from arviz import array_stats

    rng = np.random.default_rng()
    samples = rng.normal(size=(4, 1000, 2))
    array_stats.ess(samples, chain_axis=0, draw_axis=1)

We now contrast the array interface with the xarray interface, as we see there is no need to specify the chain and draw information, as this information is already encoded in the `DataTree` object.

    import arviz as az
    dt_samples = az.convert_to_datatree(samples)
    az.ess(dt_samples)

The only required argument for battery-included plots is the input data, typically a `DataTree` (`dt`), but in the following example we also apply optional customizations.

    az.style.use('arviz-variat')
    dt = az.load_arviz_data("centered_eight")
    pc = az.plot_dist(
        dt,
        kind="hist",
        visuals={"hist":{"alpha": 0.3}},
        aes={"color": ["school"]}
    );
    pc.add_legend("school", loc="outside right upper")

![plot_dist with color mapped to school dimension.](figures/figure_0.png "`plot_dist` is a built-in plot. Here we show an example of further customization. The color is mapped to the school dimension. A neutral color is automatically assigned to the variables without the school dimension (mu and tau). The histograms have been made translucent"){width=4.5in}

We have shown two small examples. For a more comprehensive overview, see the [`ArviZ` documentation](https://python.arviz.org/en/latest/) and the [EABM guide](https://arviz-devs.github.io/EABM/) [@eabm_2025]. These resources include a wide range of examples designed for all types of users, from casual users to advanced analysts and developers looking to use `ArviZ` in their projects or libraries.

## Acknowledgements

We thank our fiscal sponsor, NumFOCUS, a nonprofit 501(c)(3) public charity, for their operational and financial support. We also thank all the contributors to `arviz`, `arviz-base`, `arviz-stats` and `arviz-plots` repositories, including code contributors, documentation writers, issue reporters, and users who have provided feedback and suggestions.

This research was supported by:

* The Research Council of Finland Flagship Program "Finnish Center for Artificial Intelligence" (FCAI)
* Research Council of Finland grant 340721
* Essential Open Source Software Round 4 grant by the Chan Zuckerberg Initiative (CZI)

# References
