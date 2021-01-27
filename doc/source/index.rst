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


    <div class="container-xl">
      <div class="row">
        <div class="col-md-6">


ArviZ is a Python package for exploratory analysis of Bayesian models. Includes functions for posterior analysis, data storage, sample diagnostics, model checking, and comparison.

The goal is to provide backend-agnostic tools for diagnostics and visualizations of Bayesian inference in Python,
by first converting inference data into `xarray <https://xarray.pydata.org/en/stable/>`_ objects.
See :ref:`here <xarray_for_arviz>` for more on xarray and ArviZ usage
and :ref:`here <schema>` for more on ``InferenceData`` structure
and specification.


**Installation** using pip

.. code:: bash

    pip install arviz

Alternatively you can use conda-forge

.. code:: bash

    conda install -c conda-forge arviz

For the latest (unstable) version

.. code:: bash

    pip install git+https://github.com/arviz-devs/arviz

**Contributions** and **issue reports** are very welcome at `the github repository <https://github.com/arviz-devs/arviz>`_. We have a `contributing guide <https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md>`_ to help you through the process. If you have any doubts, please do not hesitate to contact us on `gitter <https://gitter.im/arviz-devs/community>`_.

ArviZ's functions work with NumPy arrays, dictionaries of arrays, xarray datasets, and has built-in support for `PyMC3 <https://docs.pymc.io/>`_,
`PyStan <https://pystan.readthedocs.io/en/latest/>`_, `CmdStanPy <https://github.com/stan-dev/cmdstanpy>`_,
`Pyro <http://pyro.ai/>`_, `NumPyro <http://num.pyro.ai/>`_,
`emcee <https://emcee.readthedocs.io/en/stable/>`_, and
`TensorFlow Probability <https://www.tensorflow.org/probability>`_ objects. Support for PyMC4, Edward2, and Edward are on the roadmap.

A Julia wrapper, `ArviZ.jl <https://arviz-devs.github.io/ArviZ.jl/stable/>`_ is
also available. It provides built-in support for
`Turing.jl <https://turing.ml/dev/>`_, `CmdStan.jl
<https://github.com/StanJulia/CmdStan.jl>`_, `StanSample.jl
<https://github.com/StanJulia/StanSample.jl>`_ and `Stan.jl <https://github.com/StanJulia/Stan.jl>`_.


ArviZ is a non-profit project under NumFOCUS umbrella. If you want to **support ArviZ financially**, you can donate `here <https://numfocus.org/donate-to-arviz>`_.


If you use ArviZ and want to **cite** it please use |JOSS|. Here is the citation in BibTeX format

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
  :hidden:

  getting_started/index
  Example Gallery<examples/index>
  user_guide/index
  api/index
  usage
  contributing/index
  about


.. raw:: html

    </div>
    <div class="col-md-6">
      <div class="container">
        <div class="row">
            <div class="col">
            <a href="examples/plot_pair.html">
            <div class="img-thumbnail">
                <img src="_static/plot_pair_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_forest.html">
            <div class="img-thumbnail">
                <img src="_static/plot_forest_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_density.html">
            <div class="img-thumbnail">
                <img src="_static/plot_density_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_energy.html">
            <div class="img-thumbnail">
                <img src="_static/plot_energy_thumb.png">
            </div>
            </a>
            </div>
        </div>
        <div class="row">
            <div class="col">
            <a href="examples/plot_posterior.html">
            <div class="img-thumbnail">
                <img src="_static/plot_posterior_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_kde_2d.html">
            <div class="img-thumbnail">
                <img src="_static/plot_kde_2d_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_forest_ridge.html">
            <div class="img-thumbnail">
                <img src="_static/plot_forest_ridge_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_parallel.html">
            <div class="img-thumbnail">
                <img src="_static/plot_parallel_thumb.png">
            </div>
            </a>
            </div>
        </div>
        <div class="row">
            <div class="col">
            <a href="examples/plot_trace.html">
            <div class="img-thumbnail">
                <img src="_static/plot_trace_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_joint.html">
            <div class="img-thumbnail">
                <img src="_static/plot_joint_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_ppc.html">
            <div class="img-thumbnail">
                <img src="_static/plot_ppc_thumb.png">
            </div>
            </a>
            </div>
            <div class="col">
            <a href="examples/plot_autocorr.html">
            <div class="img-thumbnail">
                <img src="_static/plot_autocorr_thumb.png">
            </div>
            </a>
            </div>
        </div>
      </div>
    </div>

    </div>
    </div>


.. |Build Status| image:: https://dev.azure.com/ArviZ/ArviZ/_apis/build/status/arviz-devs.arviz?branchName=main
   :target: https://dev.azure.com/ArviZ/ArviZ/_build/latest?definitionId=1&branchName=main
.. |Coverage Status| image:: https://codecov.io/gh/arviz-devs/arviz/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/arviz-devs/arviz
