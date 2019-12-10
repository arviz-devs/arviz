<img src="https://arviz-devs.github.io/arviz/_static/logo.png" height=100></img>

[![Build Status](https://travis-ci.org/arviz-devs/arviz.svg?branch=master)](https://travis-ci.org/arviz-devs/arviz)
[![Azure Build Status](https://dev.azure.com/ArviZ/ArviZ/_apis/build/status/arviz-devs.arviz?branchName=master)](https://dev.azure.com/ArviZ/ArviZ/_build/latest?definitionId=1&branchName=master)
[![Coverage Status](https://coveralls.io/repos/github/arviz-devs/arviz/badge.svg?branch=master)](https://coveralls.io/github/arviz-devs/arviz?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/arviz-devs/community)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.01143/status.svg)](https://doi.org/10.21105/joss.01143) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2540945.svg)](https://doi.org/10.5281/zenodo.2540945)

# ArviZ

ArviZ (pronounced "AR-_vees_") is a Python package for exploratory analysis of Bayesian models.
Includes functions for posterior analysis, model checking, comparison and diagnostics.

### ArviZ in other languages
ArviZ also has a Julia wrapper available [ArviZ.jl](https://arviz-devs.github.io/ArviZ.jl/stable/).

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

ArviZ is also available through [conda-forge](https://anaconda.org/conda-forge/arviz).

```
conda install -c conda-forge arviz
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
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_forest_ridge.html">
  <img alt="Ridge plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_forest_ridge_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_parallel.html">
  <img alt="Parallel plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_parallel_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_trace.html">
  <img alt="Trace plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_trace_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_density.html">
  <img alt="Density plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_density_thumb.png" />
  </a>
  </td>

  </tr>
  <tr>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_posterior.html">
  <img alt="Posterior plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_posterior_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_joint.html">
  <img alt="Joint plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_joint_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_ppc.html">
  <img alt="Posterior predictive plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_ppc_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_pair.html">
  <img alt="Pair plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_pair_thumb.png" />
  </a>
  </td>

  </tr>
  <tr>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_energy.html">
  <img alt="Energy Plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_energy_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_violin.html">
  <img alt="Violin Plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_violin_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_forest.html">
  <img alt="Forest Plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_forest_thumb.png" />
  </a>
  </td>

  <td>
  <a href="https://arviz-devs.github.io/arviz/examples/matplotlib/mpl_plot_autocorr.html">
  <img alt="Autocorrelation Plot"
  src="https://arviz-devs.github.io/arviz/_static/mpl_plot_autocorr_thumb.png" />
  </a>
  </td>

</tr>
</table>

## Dependencies

ArviZ is tested on Python 3.5, 3.6 and 3.7, and depends on NumPy, SciPy, xarray, and Matplotlib.


## Citation


If you use ArviZ and want to cite it please use [![DOI](http://joss.theoj.org/papers/10.21105/joss.01143/status.svg)](https://doi.org/10.21105/joss.01143)

Here is the citation in BibTeX format

```
@article{arviz_2019,
	title = {{ArviZ} a unified library for exploratory analysis of {Bayesian} models in {Python}},
	author = {Kumar, Ravin and Colin, Carroll and Hartikainen, Ari and Martin, Osvaldo A.},
	journal = {The Journal of Open Source Software},
	year = {2019},
	doi = {10.21105/joss.01143},
	url = {http://joss.theoj.org/papers/10.21105/joss.01143},
}
```


## Contributions
ArviZ is a community project and welcomes contributions.
Additional information can be found in the [Contributing Readme](https://github.com/arviz-devs/arviz/blob/master/CONTRIBUTING.md)


## Code of Conduct
ArviZ wishes to maintain a positive community. Additional details
can be found in the [Code of Conduct](https://github.com/arviz-devs/arviz/blob/master/CODE_OF_CONDUCT.md)

## Sponsors
[![NumFOCUS](https://i0.wp.com/numfocus.org/wp-content/uploads/2019/06/AffiliatedProject.png)](https://numfocus.org)
