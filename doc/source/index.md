---
html_theme.sidebar_secondary.remove:
sd_hide_title: true
---

# ArviZ: Exploratory analysis of Bayesian models

<div id="hero">

   <div id="hero-left">  <!-- Start Hero Left -->
      <h2 style="font-size: 60px; font-weight: bold; margin: 2rem auto 0;">ArviZ</h2>
      <h3 style="font-weight: bold; margin-top: 0;">Exploratory analysis of Bayesian models</h3>
      <p>ArviZ is a Python package for exploratory analysis of Bayesian models. It serves as a backend-agnostic tool for diagnosing and visualizing Bayesian inference.</p>

   <div class="homepage-button-container">
      <div class="homepage-button-container-row">
         <a href="./getting_started/index.html" class="homepage-button primary-button">Get Started</a>
         <a href="./examples/index.html" class="homepage-button secondary-button">See Gallery</a>
      </div>
      <div class="homepage-button-container-row">
         <a href="./api/index.html" class="homepage-button-link">See API Reference →</a>
      </div>
   </div>
   </div>  <!-- End Hero Left -->

   <div id="hero-right">  <!-- Start Hero Right -->

   ::::{grid} 1 2 2 2
   :gutter: 3

      :::{grid-item-card}
      :link: ./examples/plot_trace_bars.html
      :shadow: none
      :class-card: example-gallery

      ::::{div} example-img-plot-overlay
      Rank Bars Diagnostic with KDE using `plot_trace`

      :::{image} ./_images/mpl_plot_trace_bars.png
      :::

      :::{grid-item-card}
      :link: ./examples/plot_forest_mixed.html
      :shadow: none
      :class-card: example-gallery

      ::::{div} example-img-plot-overlay
      Forest Plot with ESS using `plot_forest`

      ::::{image} ./_images/mpl_plot_forest_mixed.png
      :::

      :::{grid-item-card}
      :link: ./examples/plot_dist.html
      :shadow: none
      :class-card: example-gallery

      ::::{div} example-img-plot-overlay
      Dist Plot using `plot_dist`

      ::::{image} ./_images/mpl_plot_dist.png
      :::

      :::{grid-item-card}
      :link: ./examples/plot_density.html
      :shadow: none
      :class-card: example-gallery

      ::::{div} example-img-plot-overlay
      Density Plot (Comparison) using `plot_density`

      ::::{image} ./_images/mpl_plot_density.png
      :::

      :::{grid-item-card}
      :link: ./examples/plot_pair.html
      :shadow: none
      :class-card: example-gallery

      ::::{div} example-img-plot-overlay

      Pair Plot using `plot_pair`

      ::::{image} ./_images/mpl_plot_pair.png
      :::

      :::{grid-item-card}
      :link: ./examples/plot_ppc.html
      :shadow: none
      :class-card: example-gallery

      ::::{div} example-img-plot-overlay

      Posterior Predictive Check Plot using `plot_ppc`

      ::::{image} ./_images/mpl_plot_ppc.png
      :::
   ::::

   </div>  <!-- End Hero Right -->
</div>

# Key Features

<!-- The goal is to provide backend-agnostic tools for diagnostics and visualizations of Bayesian inference in Python,
by first converting inference data into [`xarray`](https://xarray.pydata.org/en/stable/) objects.
See {ref}`here <xarray_for_arviz>` for more on xarray and ArviZ usage
and {ref}`here <schema>` for more on `InferenceData` structure
and specification.

A Julia wrapper, [ArviZ.jl](https://julia.arviz.org/) is also available. It provides built-in support for [Turing.jl](https://turing.ml/dev/), [CmdStan.jl](https://github.com/StanJulia/CmdStan.jl), [StanSample.jl](https://github.com/StanJulia/StanSample.jl) and [Stan.jl](https://github.com/StanJulia/Stan.jl).

ArviZ's functions work with NumPy arrays, dictionaries of arrays, xarray datasets, and has built-in support for [PyMC3](https://docs.pymc.io/), [PyStan](https://pystan.readthedocs.io/en/latest/), [CmdStanPy](https://github.com/stan-dev/cmdstanpy), [Pyro](http://pyro.ai/), [NumPyro](http://num.pyro.ai/), [emcee](https://emcee.readthedocs.io/en/stable/), and [TensorFlow Probability](https://www.tensorflow.org/probability) objects. Support for Edward2 is on the roadmap. -->

::::{grid} 2 2 3 4
:gutter: 3

   :::{grid-item-card} Interoperable
   :text-align: center
   Integrates with all major probabilistic programming libraries: PyMC, CmdStanPy, PyStan, Pyro, NumPyro, emcee...
   :::

   :::{grid-item-card} Large suite of visualizations
   :text-align: center
   Provides over 25 plotting functions for all parts of Bayesian workflow: visualizing distributions, diagnostics, model checking...See the gallery for examples.
   :::

   :::{grid-item-card} State of the art diagnostics
   :text-align: center
   Latest published diagnostics and statistics are implemented, tested and distributed with ArviZ.
   :::

   :::{grid-item-card} Flexible model comparison
   :text-align: center
   Includes functions for comparing models with information criteria, and cross validation (both approximate and brute force).
   :::

   :::{grid-item-card} Built for collaboration
   :text-align: center
   Designed for flexible cross-language serialization using netCDF or Zarr formats. ArviZ also has a [Julia version](https://julia.arviz.org/) that uses the same {ref}`data schema <schema>`.
   :::

   :::{grid-item-card} Labeled data
   :text-align: center
   Builds on top of xarray to work with labeled dimensions and coordinates
   :::

   :::{grid-item-card} Open source
   :text-align: center
   Distributed under the [Apache license](https://github.com/arviz-devs/arviz/blob/main/LICENSE), ArviZ is developed and maintained [publicly on GitHub](https://github.com/arviz-devs/arviz) by a vibrant, responsive, and diverse {ref}`community <community>`
   :::

::::


# Sponsors

<div class="sponsor-container">
   <a class="sponsor" href="https://www.helsinki.fi/en"><img src="_static/sponsor_university_helsinki.png" alt="University of Helsinki"></a>
   <a class="sponsor" href="https://fcai.fi/"><img src="_static/sponsor_fcai.png" alt="FCAI"></a>
   <a class="sponsor" href="https://chanzuckerberg.com/"><img src="_static/sponsor_czi.png" alt="Chan Zuckerberg Initiative"></a>
</div>


# Support ArviZ

   <div class="two-col">
      <!-- Contributions -->
      <div class="col" id="contributions">
         <h3>Contributions</h3>
         <p>Contributions and issue reports are very welcome at <a href="https://github.com/arviz-devs/arviz">the GitHub repository</a>. We have a <a href="https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md">contributing guide</a> to help you through the process. If you have any doubts, please do not hesitate to contact us on <a href="https://gitter.im/arviz-devs/community">gitter</a>.</p>
      </div>
      <!-- Citation -->
      <div class="col" id="citation">
         <h3>Citation</h3>
         <p>If you use ArviZ and want to <strong>cite</strong> it please use <a class="reference external" href="https://doi.org/10.21105/joss.01143"><img alt="JOSS" src="https://joss.theoj.org/papers/10.21105/joss.01143/status.svg"></a>.</p>
         <p>See our <a href="https://www.arviz.org/en/latest/support.html#cite">support page</a> for information on how to cite in BibTeX format.</p>
      </div>
   </div>
   <!-- Donate -->
   <div id="donate">
      <h3>Donate</h3>
      <div class="two-col">
         <div class="col">
            <a class="sponsor" href="https://numfocus.org/"><img src="_static/donate_numfocus.png" alt="NumFOCUS"></a>
         </div>
         <div class="col">
         <p>ArviZ is a non-profit project under the NumFOCUS umbrella. To support ArviZ financially, click the donate button below or visit the NumFOCUS website.</p>
         <a href="https://numfocus.org/donate-to-arviz" class="sponsor homepage-button primary-button">Donate to ArviZ</a>
         </div>
      </div>
   </div>


:::{toctree}
:maxdepth: 1
:hidden:
  Getting Started<getting_started/index>
  Example Gallery<examples/index>
  User Guide<user_guide/index>
  API Reference<api/index>
  Community<community>
  Contributing<contributing/index>
:::
