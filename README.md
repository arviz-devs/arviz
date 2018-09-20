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
  <a href="https://arviz-devs.github.io/arviz/examples/plot_forest_ridge.html">
  <img alt="Ridge plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_forest_ridge_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_parallel.html">
  <img alt="Parallel plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_parallel_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_trace.html">
  <img alt="Trace plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_trace_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_density.html">
  <img alt="Density plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_density_thumb.png" />
  </a>
  </td>

  </tr>
  <tr>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_posterior.html">
  <img alt="Posterior plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_posterior_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_joint.html">
  <img alt="Joint plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_joint_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_ppc.html">
  <img alt="Posterior predictive plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_ppc_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_pair.html">
  <img alt="Pair plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_pair_thumb.png" />
  </a>
  </td>

  </tr>
  <tr>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_energy.html">
  <img alt="Energy Plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_energy_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_violin.html">
  <img alt="Violin Plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_violin_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_forest.html">
  <img alt="Forest Plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_forest_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/plot_autocorr.html">
  <img alt="Autocorrelation Plot"
  src="https://arviz-devs.github.io/arviz/_static/plot_autocorr_thumb.png" />
  </a>
  </td>

</tr>
</table>

## Dependencies

Arviz is tested on Python 3.5 and 3.6, and depends on NumPy, SciPy, xarray, and Matplotlib.

## Developing

A typical development workflow is:

1. Install project requirements: `pip install requirements.txt`
2. Install additional testing requirements: `pip install requirements-dev.txt`
3. Write helpful code and tests.
4. Verify code style: `./scripts/lint.sh`
5. Run test suite: `pytest arviz/tests`
6. Make a pull request.

There is also a Dockerfile which helps for isolating build problems and local development.

1. Install Docker for your operating system
2. Clone this repo,
3. Run `./scripts/start_container.sh`

 This should start a local docker container called arviz, as well as a Jupyter notebook server running on port 8888. The notebook should be opened in your browser automatically (you can disable this by passing --no-browser). The container will be running the code from your local copy of arviz, so you can test your changes.
