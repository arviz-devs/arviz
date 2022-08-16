:html_theme.sidebar_secondary.remove:

.. _homepage:

ArviZ: Exploratory analysis of Bayesian models
==============================================

ArviZ is a Python package for exploratory analysis of Bayesian models. Includes functions for posterior analysis, data storage, sample diagnostics, model checking, and comparison.

The goal is to provide backend-agnostic tools for diagnostics and visualizations of Bayesian inference in Python,
by first converting inference data into `xarray <https://xarray.pydata.org/en/stable/>`_ objects.
See :ref:`here <xarray_for_arviz>` for more on xarray and ArviZ usage
and :ref:`here <schema>` for more on ``InferenceData`` structure
and specification.

.. raw:: html

    <div class="home-flex-grid">
        <a href="examples/plot_pair.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_pair.png">
                <span class="home-img-plot-overlay">Pair Plot</span>
            </div>
        </a>

        <a href="examples/plot_forest.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_forest.png">
                <span class="home-img-plot-overlay">Forest Plot</span>
            </div>
        </a>

        <a href="examples/plot_density.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_density.png">
                <span class="home-img-plot-overlay">Density Plot</span>
            </div>
        </a>

        <a href="examples/plot_energy.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_energy.png">
                <span class="home-img-plot-overlay">Energy Plot</span>
            </div>
        </a>

        <a href="examples/plot_posterior.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_posterior.png">
                <span class="home-img-plot-overlay">Posterior Plot</span>
            </div>
        </a>

        <a href="examples/plot_kde_2d.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_kde_2d.png">
                <span class="home-img-plot-overlay">KDE 2D Plot</span>
            </div>
        </a>

        <a href="examples/plot_forest_ridge.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_forest_ridge.png">
                <span class="home-img-plot-overlay">Forest Ridge Plot</span>
            </div>
        </a>

        <a href="examples/plot_parallel.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_parallel.png">
                <span class="home-img-plot-overlay">Parallel Plot</span>
            </div>
        </a>

        <a href="examples/plot_trace.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_trace.png">
                <span class="home-img-plot-overlay">Trace Plot</span>
            </div>
        </a>

        <a href="examples/plot_dot.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_dot.png">
                <span class="home-img-plot-overlay">Dot Plot</span>
            </div>
        </a>

        <a href="examples/plot_ppc.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_ppc.png">
                <span class="home-img-plot-overlay">Posterior Predictive Checks</span>
            </div>
        </a>

        <a href="examples/plot_autocorr.html">
            <div class="home-img-plot img-thumbnail">
                <img src="./_images/mpl_plot_autocorr.png">
                <span class="home-img-plot-overlay">Autocorrelation Plot</span>
            </div>
        </a>
    </div>

Installation
------------

Using pip

.. code:: bash

    pip install arviz

Using conda-forge

.. code:: bash

    conda install -c conda-forge arviz

To install the latest development version of ArviZ, please check the :ref:`Installation guide <dev-version>` for details.

Contribution
------------

**Contributions** and **issue reports** are very welcome at `the github repository <https://github.com/arviz-devs/arviz>`_. We have a `contributing guide <https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md>`_ to help you through the process. If you have any doubts, please do not hesitate to contact us on `gitter <https://gitter.im/arviz-devs/community>`_.

ArviZ's functions work with NumPy arrays, dictionaries of arrays, xarray datasets, and has built-in support for `PyMC3 <https://docs.pymc.io/>`_,
`PyStan <https://pystan.readthedocs.io/en/latest/>`_, `CmdStanPy <https://github.com/stan-dev/cmdstanpy>`_,
`Pyro <http://pyro.ai/>`_, `NumPyro <http://num.pyro.ai/>`_,
`emcee <https://emcee.readthedocs.io/en/stable/>`_, and
`TensorFlow Probability <https://www.tensorflow.org/probability>`_ objects. Support for Edward2 is on the roadmap.

A Julia wrapper, `ArviZ.jl <https://julia.arviz.org/>`_ is
also available. It provides built-in support for
`Turing.jl <https://turing.ml/dev/>`_, `CmdStan.jl
<https://github.com/StanJulia/CmdStan.jl>`_, `StanSample.jl
<https://github.com/StanJulia/StanSample.jl>`_ and `Stan.jl <https://github.com/StanJulia/Stan.jl>`_.


ArviZ is a non-profit project under NumFOCUS umbrella. If you want to **support ArviZ financially**, you can donate `here <https://numfocus.org/donate-to-arviz>`_.

Citation
--------

If you use ArviZ and want to **cite** it please use |JOSS|. Here is the citation in BibTeX format

.. code:: bash

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


.. toctree::
  :maxdepth: 1
  :hidden:

  getting_started/index
  Example Gallery<examples/index>
  user_guide/index
  api/index
  community
  Contributing<contributing/index>


.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.01143/status.svg
   :target: https://doi.org/10.21105/joss.01143