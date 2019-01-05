<img src="https://arviz-devs.github.io/arviz/_static/logo.png" height=100></img>

[![Build Status](https://travis-ci.org/arviz-devs/arviz.svg?branch=master)](https://travis-ci.org/arviz-devs/arviz) [![Coverage Status](https://coveralls.io/repos/github/arviz-devs/arviz/badge.svg?branch=master)](https://coveralls.io/github/arviz-devs/arviz?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# ArviZ

ArviZ (pronounced "AR-_vees_") is a Python package for exploratory analysis of Bayesian models.
Includes functions for posterior analysis, model checking, comparison and diagnostics.

## Documentation

The ArviZ documentation can be found in the [official docs](https://arviz-devs.github.io/arviz/index.html).
First time users may find the [quickstart](https://arviz-devs.github.io/arviz/notebooks/Introduction.html)
to be helpful. Additional guidance can be found in the
[usage documentation](https://arviz-devs.github.io/arviz/usage.html).


## Installation

### Stable
ArviZ is available for installation from [PyPI](https://pypi.org/project/arviz/).
The latest stable version can be installed using pip:

```
pip install arviz
```

### Development
The latest development version can be installed from the master branch using pip:

```
pip install git+git://github.com/arviz-devs/arviz.git
```

Another option is to clone the repository and install using git and setuptools:

```
git clone https://github.com/arviz-devs/arviz.git
cd arviz
python setup.py install
```

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

ArviZ is tested on Python 3.5 and 3.6, and depends on NumPy, SciPy, xarray, and Matplotlib.

## Contributions
ArviZ is a community project and welcomes contributions. 
Additional information can be found in the [Contributing Readme](https://github.com/arviz-devs/arviz/blob/master/CONTRIBUTING.md)

### Developing

A typical development workflow is:

1. Install project requirements: `pip install -r requirements.txt`
2. Install additional testing requirements: `pip install -r requirements-dev.txt`
3. Write helpful code and tests.
4. Verify code style: `./scripts/lint.sh`
5. Run test suite: `pytest arviz/tests`
6. Make a pull request.

There is also a Dockerfile which helps for isolating build problems and local development.

1. Install Docker for your operating system
2. Clone this repo,
3. Run `./scripts/container.sh --build`

This will build a local image with the tag `arviz`. 
After building the image tests can be executing by running `docker run arviz`.
An interactive shell can be started by running `docker run -it arviz /bin/bash`. The correct conda environment will be activated automatically.


## Code of Conduct
ArviZ wishes to maintain a positive community. Additional details
can be found in the [Code of Conduct](https://github.com/arviz-devs/arviz/blob/master/Code_of_Conduct.MD)

