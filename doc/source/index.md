---
html_theme.sidebar_secondary.remove:
sd_hide_title: true
---

(homepage)=
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

::::::{grid} 1 2 2 2
:gutter: 3

:::::{grid-item-card}
:link: example_plot_trace_bars
:link-type: ref
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Rank Bars Diagnostic with KDE using `plot_trace`
:::

:::{image} ./_images/mpl_plot_trace_bars.png
:::
:::::

:::::{grid-item-card}
:link: example_plot_forest_mixed
:link-type: ref
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Forest Plot with ESS using `plot_forest`
:::

:::{image} ./_images/mpl_plot_forest_mixed.png
:::
:::::

:::::{grid-item-card}
:link: example_plot_dist
:link-type: ref
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Dist Plot using `plot_dist`
:::

:::{image} ./_images/mpl_plot_dist.png
:::
:::::

:::::{grid-item-card}
:link: example_plot_density
:link-type: ref
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Density Plot (Comparison) using `plot_density`
:::

:::{image} ./_images/mpl_plot_density.png
:::
:::::

:::::{grid-item-card}
:link: example_plot_pair
:link-type: ref
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Pair Plot using `plot_pair`
:::

:::{image} ./_images/mpl_plot_pair.png
:::
:::::

:::::{grid-item-card}
:link: example_plot_ppc
:link-type: ref
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Posterior Predictive Check Plot using `plot_ppc`
:::

:::{image} ./_images/mpl_plot_ppc.png
:::
:::::
::::::

<!-- grid ended above, do not put anything on the right of markdown closings -->

</div>  <!-- End Hero Right -->
</div>  <!-- End Hero -->

# Key Features

:::::{grid} 1 1 2 2
:gutter: 5

::::{grid-item-card} 
:shadow: none
:class-card: sd-border-0

:::{image} _static/key_feature_interoperability.svg
:::

:::{div} key-features-text
<strong>Interoperability</strong><br/>
Integrates with all major probabilistic programming libraries: PyMC, CmdStanPy, PyStan, Pyro, NumPyro, and emcee.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/key_feature_visualizations.svg
:::

:::{div} key-features-text
<strong>Large Suite of Visualizations</strong><br/>
Provides over 25 plotting functions for all parts of Bayesian workflow: visualizing distributions, diagnostics, and model checking. See the gallery for examples.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/key_feature_diagnostics.svg
:::

:::{div} key-features-text
<strong>State of the Art Diagnostics</strong><br/>
Latest published diagnostics and statistics are implemented, tested and distributed with ArviZ.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/key_feature_comparison.svg
:::

:::{div} key-features-text
<strong>Flexible Model Comparison</strong><br/>
Includes functions for comparing models with information criteria, and cross validation (both approximate and brute force).
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0 

:::{image} _static/key_feature_collaboration.svg
:::

:::{div} key-features-text
<strong>Built for Collaboration</strong><br/>
Designed for flexible cross-language serialization using netCDF or Zarr formats. ArviZ also has a [Julia version](https://julia.arviz.org/) that uses the same {ref}`data schema <schema>`.
:::
::::

::::{grid-item-card}
:shadow: none
:class-card: sd-border-0

:::{image} _static/key_feature_labeled_data.svg
:::

:::{div} key-features-text
<strong>Labeled Data</strong><br/>
Builds on top of xarray to work with labeled dimensions and coordinates.
:::
::::
:::::


# Using ArviZ

::::{grid} 1 1 2 2

:::{grid-item}

<h3>Contributions</h3>

Contributions and issue reports are very welcome at
[the GitHub repository](https://github.com/arviz-devs/arviz).
We have a {ref}`contributing guide <contributing_guide>` to help you through the process.
If you have any doubts, please do not hesitate to contact us on [gitter](https://gitter.im/arviz-devs/community).
:::
:::{grid-item}

<h3>Citation</h3>

If you use ArviZ, please it using <a class="reference external" href="https://doi.org/10.21105/joss.01143"><img alt="JOSS" src="https://joss.theoj.org/papers/10.21105/joss.01143/status.svg"></a>.

See our {ref}`support page <arviz_org:cite>` for information on how to cite in BibTeX format.
:::
::::


# Sponsors

::::{grid} 1 3 3 3
:::{grid-item}
[![helsinki_uni_logo](_static/sponsor_university_helsinki.png)](https://www.helsinki.fi/en)
:::
:::{grid-item}
[![fcai_logo](_static/sponsor_fcai.png)](https://fcai.fi/)
:::
:::{grid-item}
[![czi_logo](_static/sponsor_czi.png)](https://chanzuckerberg.com/)
:::
::::


# Supporting ArviZ

::::{grid} 1 1 3 3

:::{grid-item}
:child-align: justify
<h3>Become a Sponsor</h3>

See the Sponsors page for more details.

<a href="https://www.arviz.org/en/latest/sponsors_partners.html" class="homepage-button primary-button">See Details</a>
:::

:::{grid-item}
:child-align: justify
<h3>Shop ArviZ Merchandise</h3>

![logo_merch](_static/donate_merch.svg)

Merchandise is available for the ArviZ logo and favicon.

<div style="display: flex; flex-wrap: wrap; justify-content:center;">
<a href="https://numfocus.myspreadshop.com/arviz+logo?idea=629e289fc8ee26344a684241" class="homepage-button primary-button">Shop Logo</a>
<a href="https://numfocus.myspreadshop.com/arviz+favicon+design?idea=62a74f17ebe60a221692c6f2" class="homepage-button primary-button">Shop Favicon</a>
</div>
:::

:::{grid-item}
:child-align: justify
<h3>Donate</h3>

![numfocus_logo](_static/donate_numfocus.png)

ArviZ is a non-profit project under the NumFOCUS umbrella. To support ArviZ financially, consider donating through the NumFOCUS website.

<a href="https://numfocus.org/donate-to-arviz" class="homepage-button primary-button">Donate</a>
:::
::::


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
