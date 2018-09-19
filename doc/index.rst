ArviZ: Exploratory analysis of Bayesian models
==============================================
|Build Status|

.. raw:: html


    <div style="clear: both"></div>
    <div class="container-fluid">
        <div class="row">
            <div class="col-md-6">


ArviZ includes functions for posterior analysis, sample diagnostics, model checking, and comparison.

It is currently in heavy development.  The goal is to provide backend-agnostic tools
for diagnostics and visualizations of Bayesian inference in Python, by first
converting inference data into `xarray <https://xarray.pydata.org/en/stable/>`_ objects.

Currently `PyMC3 <http://docs.pymc.io/>`_ and
`PyStan <https://pystan.readthedocs.io/en/latest/>`_ are best supported, but
support for PyMC4, TensorFlow Probability, Pyro, Edward2, and Edward are on the
roadmap.

Contributions and issue reports are very welcome at
`the github repository <https://github.com/arviz-devs/arviz>`_.


.. toctree::
   :maxdepth: 1

   Quickstart<notebooks/Introduction>
   Example Gallery<examples/index>
   Current Roadmap<roadmap>
   Example Gallery<examples/index>
   Quickstart<notebooks/Introduction>
   api
   about


.. raw:: html

    </div>
    <div class="col-md-6">
        <div class="container-fluid hidden-xs hidden-sm">
            <a href="examples/ridgeplot.html">
            <div class="col-md-4 thumbnail">
                <img src="_static/ridgeplot_thumb.png">
            </div>
            </a>
            <a href="examples/parallelplot.html">
            <div class="col-md-4 thumbnail">
                <img src="_static/parallelplot_thumb.png">
            </div>
            </a>
            <a href="examples/traceplot.html">
            <div class="col-md-4 thumbnail">
                <img src="_static/traceplot_thumb.png">
            </div>
            </a>
            <a href="examples/jointplot.html">
            <div class="col-md-4 thumbnail">
                <img src="_static/jointplot_thumb.png">
            </div>
            </a>
            <a href="examples/ppcplot.html">
            <div class="col-md-4 thumbnail">
                <img src="_static/ppcplot_thumb.png">
            </div>
            </a>
            <a href="examples/autocorrplot.html">
            <div class="col-md-4 thumbnail">
                <img src="_static/autocorrplot_thumb.png">
            </div>
            </a>
        </div>
    </div>

    </div>
    </div>
    </div>


.. |Build Status| image:: https://travis-ci.org/arviz-devs/arviz.png?branch=master
   :target: https://travis-ci.org/arviz-devs/arviz
