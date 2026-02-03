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
  - name: Seth D. Axen
    orcid: 0000-0003-3933-8247
    affiliation: 4
  - name: Colin Carroll
    orcid: 0000-0001-6977-0861
    affiliation: 2
  - name: Ari Hartikainen
    orcid: 0000-0002-4569-569X
    affiliation: 2
  - name: Aki Vehtari
    orcid: 0000-0003-2164-9469
    affiliation: "1, 5"
affiliations:
 - name:  Aalto University, Espoo, Finland
   index: 1
 - name: arviz-devs
   index: 2
 - name: DICK's Sporting Goods, Coraopolis, Pennsylvania
   index: 3
 - name: University of Tübingen
   index: 4
 - name: ELLIS Institute Finland
   index: 5
date: 17 January 2026
bibliography: references.bib
---

# Summary

When working with Bayesian models, a range of related tasks must be addressed beyond inference itself. These include diagnosing the quality of MCMC samples, model criticism, model comparison, etc. We collectively refer to these activities as exploratory analysis of Bayesian models.

In this work, we present a redesigned version of `ArviZ`, a Python package for exploratory analysis of Bayesian models. The redesign emphasizes greater user control and modularity. This redesign delivers a more flexible and efficient toolkit for exploratory analysis of Bayesian models. With its renewed focus on modularity and usability, `ArviZ` is well-positioned to remain an essential tool for Bayesian modelers in both research and applied settings.


# Statement of need

Probabilistic programming has emerged as a powerful paradigm for statistical modeling, accompanied by a growing ecosystem of tools for model specification and inference. Effective modeling requires robust support for uncertainty visualization, sampling diagnostics, model comparison, and model checking [@Gelman_2020; @Martin_2024; @Guo_2024]. `ArviZ` addresses this gap by providing a unified, backend-agnostic library to perform these tasks. The original `ArviZ` paper [@Kumar_2019] described the landscape of probabilistic programming tools at the time and the need for a unified, backend-agnostic library for exploratory analysis - a need that has only grown as the ecosystem has expanded.

The methods implemented in `ArviZ` are grounded in well-established statistical principles and provide robust, interpretable diagnostics and visualizations [@Vehtari_2017; @Gelman_2019; @Dimitriadis_2021; @Paananen_2021; @Padilla_2021; @Vehtari_2021; @Sailynoja_2022; @Kallioinen_2023; @Sailynoja_2025]. As modern Bayesian practice has increasingly emphasized iterative, simulation-based workflows for model building, checking, validation, and comparison, the original ArviZ design became harder to adapt to these emerging needs. The redesigned version furthers ArviZ’s goals by introducing an easier-to-use interface for routine analyses alongside more powerful, flexible tooling for advanced users and developers of Bayesian software. These updates align ArviZ with the current demands of Bayesian modeling and workflow development


# State of the field

In the Python Bayesian ecosystem, ArviZ occupies a niche comparable to tools in the R/Stan community such as posterior [@gelman_2013;@Vehtari_2021], loo [@Vehtari_2017;@loo], bayesplot [@bayesplot0;@bayesplot1], priorsense [@Kallioinen_2023], and ggdist [@kay_2024] sharing similar goals while reflecting different language ecosystems and workflows.

# Research Impact Statement

`ArviZ` [@Kumar_2019] is a Python package for exploratory analysis of Bayesian models that has been widely used in academia and industry since its introduction in 2019, with over 700 citations and 75 million downloads. Its goal is to integrate seamlessly with established probabilistic programming languages and statistical interfaces, such as PyMC [@Abril-pla_2023], Stan (via the cmdstanpy interface) [@stan], Pyro, NumPyro [@Phan_2019; @Bingham_2019], emcee [@emcee], and Bambi [@Capretto_2022], among others.

The maturity of `ArviZ` has also led to other initiatives, including ArviZ.jl [@arvizjl_2025] for Julia, PreliZ [@icazatti_2023] for prior elicitation and the development of educational resources [@eabm_2025].

# Software design

The previous `ArviZ` design divided the package into three submodules, which are now available as three independent installable packages. This redesign emphasizes greater user control and modularity. The new architecture enables users to customize the installation and use of specific components. Key design changes include: 

General functionality, data processing, and data input/output have been streamlined and enhanced for greater versatility. Previously, `ArviZ` used the custom `InferenceData` class to organize and store the high-dimensional outputs of Bayesian inference in a structured, labeled format, enabling efficient analysis, metadata persistence, and serialization. These have been replaced with the `DataTree` class from xarray [@Hoyer_2017], which, like the original `InferenceData`, supports grouping but is more flexible, enabling richer nesting and automatic support for all xarray I/O formats. Additionally, converters allow more flexibility in dimensionality, naming, and indexing of their generated outputs.

Statistical functions are now accessible through two distinct interfaces:

* A low-level array interface with only `numpy` [@harris_2020] and `scipy` [@virtanen_2020] as dependencies, intended for advanced users
and developers of third-party libraries.
* A higher-level xarray interface designed for end users, which simplifies usage by automating common tasks and handling metadata.

Plotting functions have also been redesigned to support modularity at multiple levels:

* At a high level, `ArviZ` offers a collection of “batteries-included” plots. These are built-in plotting functions providing sensible defaults for common tasks like MCMC sampling diagnostics, predictive checks, and model comparison.
* At an intermediate level, the API enables easier customization of batteries-included plots and simplifies the creation of new plots. This is achieved through the `PlotCollection` class, which enables developers and advanced users to focus solely on the plotting logic, delegating any faceting or aesthetic mappings to `PlotCollection`.
* At a lower level, we have improved the separation between computational and plotting logic, reducing code duplication and enhancing modular design. These changes also facilitate support for multiple plotting backends, improving extensibility and maintainability. Currently, `ArviZ` supports three plotting backends: matplotlib [@Hunter_2007], Bokeh [@Bokeh_2018], and plotly [@plotly_2015].

Thanks to this new design, the cost of adding "batteries-included" plots has reduced in more than half even though `ArviZ` now supports one extra backend. Consequently, redesigned `ArviZ` already has 37 "batteries-included", 10 more than the 0.x versions.

## Examples

For the first example, we use the low-level array interface to compute the effective sample sizes for some fake data. We construct an array resembling data from MCMC sampling with 4 chains and 1000 draws for two posterior variables. When using the array interface we need to specify which axes represent the chains and which the draws.

    import numpy as np
    from arviz_stats.base import array_stats

    rng = np.random.default_rng()
    samples = rng.normal(size=(4, 1000, 2))  # (chain, draw, variable)
    array_stats.ess(samples, chain_axis=0, draw_axis=1)

The array interface is lightweight and intended for advanced users and library developers. For most users, we instead recommend the xarray interface, as it is more user-friendly and automates many tasks. When converting the NumPy array to a `DataTree`, ArviZ assigns `chain` and `draw` as named dimensions based on the assumed dimension order, so this information is already encoded in the resulting object and does not need to be specified explicitly when calling other functions.

    import arviz as az
    dt_samples = az.convert_to_datatree(samples)
    az.ess(dt_samples)

The only required argument for battery-included plots is the input data, typically a `DataTree` (`dt`). In this example we also apply optional customizations.

    az.style.use('arviz-variat')
    dt = az.load_arviz_data("centered_eight")
    pc = az.plot_dist(
        dt,
        kind="dot",
        visuals={"dist":{"marker": "C6"},
                "point_estimate_text":False},
        aes={"color": ["school"]}
    );
    pc.add_legend("school", loc="outside right upper")

![plot_dist with color mapped to school dimension.](figures/figure_0.png "`plot_dist` is one of the batteries-included plots in `ArviZ`. In this example we demonstrate how it can be further customized. We change the default kind from "kde" to "dot" to produce quantile dot plots [@kay_2016], and map the school dimension to color so that each school is shown in a different hue. Variables that do not have a school dimension (such as mu and tau) are automatically assigned a neutral color. We also disable the point-estimate text and set a custom marker style for the dots, and finally add a legend for the school"){width=4.5in}

We have shown two small examples. For a more comprehensive overview, see the [`ArviZ` documentation](https://python.arviz.org/en/latest/) and the [EABM guide](https://arviz-devs.github.io/EABM/) [@eabm_2025]. These resources include a wide range of examples designed for all types of users, from casual users to advanced analysts and developers looking to use `ArviZ` in their projects or libraries.

## AI usage disclosure

Generative AI tools were used during software development and documentation in a limited capacity, primarily to assist with rewording and minor code suggestions. All AI-assisted contributions were reviewed and edited by the authors. Core design decisions, feature development, and scientific or technical judgment were carried out by the authors, and all code and claims were tested and manually verified to ensure correctness.

## Acknowledgements

We thank our fiscal sponsor, NumFOCUS, a nonprofit 501(c)(3) public charity, for their operational and financial support. We also thank all the contributors to `arviz`, `arviz-base`, `arviz-stats`, and `arviz-plots` repositories, including code contributors, documentation writers, issue reporters, and users who have provided feedback and suggestions.

This research was supported by:

* The Research Council of Finland Flagship Program "Finnish Center for Artificial Intelligence" (FCAI)
* Research Council of Finland grant 340721
* Essential Open Source Software Round 4 grant by the Chan Zuckerberg Initiative (CZI)
* Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645

# References
