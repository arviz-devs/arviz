<img src="https://raw.githubusercontent.com/arviz-devs/arviz-project/main/arviz_logos/ArviZ.png#gh-light-mode-only" width=200></img>
<img src="https://raw.githubusercontent.com/arviz-devs/arviz-project/main/arviz_logos/ArviZ_white.png#gh-dark-mode-only" width=200></img>

[![PyPI version](https://badge.fury.io/py/arviz.svg)](https://badge.fury.io/py/arviz)
[![Azure Build Status](https://dev.azure.com/ArviZ/ArviZ/_apis/build/status/arviz-devs.arviz?branchName=main)](https://dev.azure.com/ArviZ/ArviZ/_build/latest?definitionId=1&branchName=main)
[![codecov](https://codecov.io/gh/arviz-devs/arviz/branch/main/graph/badge.svg)](https://codecov.io/gh/arviz-devs/arviz)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/arviz-devs/community)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.01143/status.svg)](https://doi.org/10.21105/joss.01143) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2540945.svg)](https://doi.org/10.5281/zenodo.2540945)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)

ArviZ (pronounced "AR-_vees_") is a Python package for exploratory analysis of Bayesian models.
Includes functions for posterior analysis, data storage, model checking, comparison and diagnostics.

### ArviZ in other languages
ArviZ also has a Julia wrapper available [ArviZ.jl](https://julia.arviz.org/).

## Documentation

The ArviZ documentation can be found in the [official docs](https://python.arviz.org/en/latest/index.html).
First time users may find the [quickstart](https://python.arviz.org/en/latest/getting_started/Introduction.html)
to be helpful. Additional guidance can be found in the
[user guide](https://python.arviz.org/en/latest/user_guide/index.html).


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
The latest development version can be installed from the main branch using pip:

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
## [Gallery](https://python.arviz.org/en/latest/examples/index.html)

<p>
<table>
<tr>

  <td>
  <a href= "https://python.arviz.org/en/latest/examples/plot_forest_ridge.html">
  <img alt="Ridge plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_forest_ridge.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_forest.html">
  <img alt="Forest Plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_forest.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_violin.html">
  <img alt="Violin Plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_violin.png" width="300" height="auto" />
  </a>
  </td>

</tr>
<tr>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_ppc.html">
  <img alt="Posterior predictive plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_ppc.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_dot.html">
  <img alt="Joint plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_dot.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_posterior.html">
  <img alt="Posterior plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_posterior.png" width="300" height="auto" />
  </a>
  </td>

</tr>
<tr>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_density.html">
  <img alt="Density plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_density.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_pair.html">
  <img alt="Pair plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_pair.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_pair_hex.html">
  <img alt="Hexbin Pair plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_pair_hex.png" width="300" height="auto" />
  </a>
  </td>

</tr>
<tr>
  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_trace.html">
  <img alt="Trace plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_trace.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_energy.html">
  <img alt="Energy Plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_energy.png" width="300" height="auto" />
  </a>
  </td>

  <td>
  <a href="https://python.arviz.org/en/latest/examples/plot_rank.html">
  <img alt="Rank Plot"
  src="https://python.arviz.org/en/latest/_images/mpl_plot_rank.png" width="300" height="auto" />
  </a>
  </td>

</tr>
</table>
<div>

  <a href="https://python.arviz.org/en/latest/examples/index.html">And more...</a>
</div>
## Dependencies

ArviZ is tested on Python 3.9, 3.10 and 3.11, and depends on NumPy, SciPy, xarray, and Matplotlib.


## Citation


If you use ArviZ and want to cite it please use [![DOI](http://joss.theoj.org/papers/10.21105/joss.01143/status.svg)](https://doi.org/10.21105/joss.01143)

Here is the citation in BibTeX format

```
@article{arviz_2019,
  doi = {10.21105/joss.01143},
  url = {https://doi.org/10.21105/joss.01143},
  year = {2019},
  publisher = {The Open Journal},
  volume = {4},
  number = {33},
  pages = {1143},
  author = {Ravin Kumar and Colin Carroll and Ari Hartikainen and Osvaldo Martin},
  title = {ArviZ a unified library for exploratory analysis of Bayesian models in Python},
  journal = {Journal of Open Source Software}
}
```


## Contributions
ArviZ is a community project and welcomes contributions.
Additional information can be found in the [Contributing Readme](https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md)


## Code of Conduct
ArviZ wishes to maintain a positive community. Additional details
can be found in the [Code of Conduct](https://github.com/arviz-devs/arviz/blob/main/CODE_OF_CONDUCT.md)

## Donations
ArviZ is a non-profit project under NumFOCUS umbrella. If you want to support ArviZ financially, you can donate [here](https://numfocus.org/donate-to-arviz).

## Sponsors
[![NumFOCUS](https://www.numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png)](https://numfocus.org)
