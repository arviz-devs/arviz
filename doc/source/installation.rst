Installation
============

This section provides a detailed information about installing ArviZ. You
can also find information regarding the required and optional
dependencies for ArviZ.

ArviZ can be installed either using pip or conda-forge

**Installing with pip**
-----------------------

.. code:: bash

    pip install arviz

**Installing with conda-forge**
-------------------------------

.. code:: bash

    conda install -c conda-forge arviz

If you want to install the latest version (unstable) of ArviZ, then you
may use

.. code:: bash

    pip install git+https://github.com/arviz-devs/arviz

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

Required dependencies
=====================

The below required dependencies are automatically installed when you use
pip or conda-forge.

.. code:: bash

    setuptools>=38.4
    matplotlib>=3.0
    numpy>=1.12
    scipy>=0.19
    packaging
    pandas>=0.23
    xarray>=0.16.1
    netcdf4
    typing_extensions>=3.7.4.3,<4

Optional dependencies
=====================

There are certain packages required for optional features.

**Numba**
---------

Necessary to speed up the code computation. More details can be found
`here <https://arviz-devs.github.io/arviz/user_guide/Numba.html>`_.

**Bokeh**
---------

Necessary for creating advanced interactive visualisations. The mininum
required version is ``1.4.0``. More details about bokeh can be found `over here <http://docs.bokeh.org/en/dev/docs/first_steps/installation.html>`_.

**UltraJSON**
-------------

If available, ArviZ make use of faster ujson when ``arviz.from_json(filename)`` is
invoked.

**Dask**
--------

Necessary to scale the packages and the surrounding ecosystem.


