ArviZ: Exploratory analysis of Bayesian models
==============================================
|Build Status|
|Coverage Status|
|Zenodo|
|NumFocus|

.. |JOSS| image:: http://joss.theoj.org/papers/10.21105/joss.01143/status.svg
   :target: https://doi.org/10.21105/joss.01143

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2540944.svg
   :target: https://doi.org/10.5281/zenodo.2540944

.. |NumFocus| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: https://www.numfocus.org/

.. raw:: html


    <div style="clear: both"></div>
    <div class="container-fluid">
        <div class="row">
            <div class="col-md-6">


ArviZ is a Python package for exploratory analysis of Bayesian models. Includes functions for posterior analysis, sample diagnostics, model checking, and comparison.

The goal is to provide backend-agnostic tools for diagnostics and visualizations of Bayesian inference in Python, by first converting inference data into `xarray <https://xarray.pydata.org/en/stable/>`_ objects. See :doc:`here <notebooks/XarrayforArviZ>` for more on xarray and ArviZ.




Installation with pip is recommended

.. code:: bash

    pip install arviz

ArviZ will plot NumPy arrays, dictionaries of arrays, xarray datasets, and has built-in support for `PyMC3 <https://docs.pymc.io/>`_,
`PyStan <https://pystan.readthedocs.io/en/latest/>`_,
`Pyro <http://pyro.ai/>`_, and
`emcee <https://emcee.readthedocs.io/en/stable/>`_ objects. Support for PyMC4, TensorFlow Probability, Edward2, and Edward are on the roadmap.

Contributions and issue reports are very welcome at
`the github repository <https://github.com/arviz-devs/arviz>`_.


If you use ArviZ and want to cite it please use |JOSS|

Here is the citation in BibTeX format

.. code:: bash

    @article{arviz_2019,
	    title = {{ArviZ} a unified library for exploratory analysis of {Bayesian} models in {Python}},
	    author = {Kumar, Ravin and Carroll, Colin and Hartikainen, Ari and Martin, Osvaldo A.},
	    journal = {The Journal of Open Source Software},
	    year = {2019},
	    doi = {10.21105/joss.01143},
	    url = {http://joss.theoj.org/papers/10.21105/joss.01143},
    }

.. toctree::
    :maxdepth: 1

    Quickstart<notebooks/Introduction>
    Example Gallery<examples/index>
    Cookbook<notebooks/InferenceDataCookbook>
    api
    usage
    about


.. raw:: html

    </div>
    <div class="col-md-6">
        <div class="container-fluid hidden-xs hidden-sm">
            <a href="examples/plot_pair.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_pair_thumb.png">
            </div>
            </a>
            <a href="examples/plot_forest.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_forest_thumb.png">
            </div>
            </a>
            <a href="examples/plot_density.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_density_thumb.png">
            </div>
            </a>
            <a href="examples/plot_energy.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_energy_thumb.png">
            </div>
            </a>
            <a href="examples/plot_posterior.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_posterior_thumb.png">
            </div>
            </a>
            <a href="examples/plot_kde_2d.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_kde_2d_thumb.png">
            </div>
            </a>
            <a href="examples/plot_forest_ridge.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_forest_ridge_thumb.png">
            </div>
            </a>
            <a href="examples/plot_parallel.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_parallel_thumb.png">
            </div>
            </a>
            <a href="examples/plot_trace.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_trace_thumb.png">
            </div>
            </a>
            <a href="examples/plot_joint.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_joint_thumb.png">
            </div>
            </a>
            <a href="examples/plot_ppc.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_ppc_thumb.png">
            </div>
            </a>
            <a href="examples/plot_autocorr.html">
            <div class="col-md-3 thumbnail">
                <img src="_static/plot_autocorr_thumb.png">
            </div>
            </a>
        </div>
    </div>

    </div>
    </div>
    </div>


.. |Build Status| image:: https://travis-ci.org/arviz-devs/arviz.png?branch=master
   :target: https://travis-ci.org/arviz-devs/arviz
.. |Coverage Status| image:: https://coveralls.io/repos/github/arviz-devs/arviz/badge.svg?branch=master
   :target: https://coveralls.io/github/arviz-devs/arviz?branch=master
