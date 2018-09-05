<img src="https://arviz-devs.github.io/arviz/_static/logo.png" height=100></img>

[![Build Status](https://travis-ci.org/arviz-devs/arviz.svg?branch=master)](https://travis-ci.org/arviz-devs/arviz) [![Coverage Status](https://coveralls.io/repos/github/arviz-devs/arviz/badge.svg?branch=master)](https://coveralls.io/github/arviz-devs/arviz?branch=master)

# ArviZ

ArviZ (pronounced "AR-_vees_") is a Python package for exploratory analysis of Bayesian models.
Includes functions for posterior analysis, model checking, comparison and diagnostics.

## Documentation

The official Arviz documentation can be found here
https://arviz-devs.github.io/arviz/index.html

## Installation

The latest version can be installed from the master branch using pip:

```
pip install git+git://github.com/arviz-devs/arviz.git
```

Another option is to clone the repository and install using `python setup.py install`.

-------------------------------------------------------------------------------
## [Gallery](https://arviz-devs.github.io/arviz/examples/index.html)

<p>
<table>
<tr>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/ridgeplot.html">
  <img alt="Ridge plot"
  src="https://arviz-devs.github.io/arviz/_static/ridgeplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/parallelplot.html">
  <img alt="Parallel plot"
  src="https://arviz-devs.github.io/arviz/_static/parallelplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/traceplot.html">
  <img alt="Trace plot"
  src="https://arviz-devs.github.io/arviz/_static/traceplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/densityplot.html">
  <img alt="Density plot"
  src="https://arviz-devs.github.io/arviz/_static/densityplot_thumb.png" />
  </a>
  </td>

  </tr>
  <tr>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/posteriorplot.html">
  <img alt="Posterior plot"
  src="https://arviz-devs.github.io/arviz/_static/posteriorplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/jointplot.html">
  <img alt="Joint plot"
  src="https://arviz-devs.github.io/arviz/_static/jointplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/ppcplot.html">
  <img alt="Posterior predictive plot"
  src="https://arviz-devs.github.io/arviz/_static/ppcplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/pairplot.html">
  <img alt="Pairplot"
  src="https://arviz-devs.github.io/arviz/_static/pairplot_thumb.png" />
  </a>
  </td>

  </tr>
  <tr>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/energyplot.html">
  <img alt="Energy Plot"
  src="https://arviz-devs.github.io/arviz/_static/energyplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/violinplot.html">
  <img alt="Violin Plot"
  src="https://arviz-devs.github.io/arviz/_static/violinplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/forestplot.html">
  <img alt="Forest Plot"
  src="https://arviz-devs.github.io/arviz/_static/forestplot_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/autocorrplot.html">
  <img alt="Autocorrelation Plot"
  src="https://arviz-devs.github.io/arviz/_static/autocorrplot_thumb.png" />
  </a>
  </td>

</tr>
</table>

## Dependencies

Arviz is tested on Python 3.5 and 3.6, and depends on NumPy, SciPy, xarray, and Matplotlib.

## Developing

There is a Dockerfile which helps for isolating build problems and local development.
1. Install Docker for your operating system
2. Clone this repo,
3. Run `./scripts/start_container.sh`

 This should start a local docker container called arviz, as well as a Jupyter notebook server running on port 8888. The notebook should be opened in your browser automatically (you can disable this by passing --no-browser). The container will be running the code from your local copy of arviz, so you can test your changes.
