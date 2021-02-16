############
Installation
############

This section provides detailed information about installing ArviZ. Most ArviZ
functionality is available with the basic requirements, but ArviZ also has optional
dependencies to further enhance the library. This guide will cover both basic and fully-fledged ArviZ installs and several installation methods.

ArviZ can be installed either using pip or conda-forge

******
Stable
****** 

Using pip
=========

.. code:: bash

    pip install arviz

Use the below pip command to install ArviZ with all of it's :ref:`Optional-dependencies`.

.. code:: bash

    pip install arviz[all]
    
Using conda-forge
=================

.. code:: bash

    conda install -c conda-forge arviz

***********
Development
*********** 

If you want to install the latest development version of ArviZ, then you
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

************
Dependencies
************

Required dependencies
=====================

The required dependencies for installing ArviZ are:

.. literalinclude:: ../../../requirements.txt

.. _Optional-dependencies:

Optional dependencies
=====================

The list of optional dependencies to further enhance ArviZ are.

.. literalinclude:: ../../../requirements-optional.txt


- Numba

  Necessary to speed up the code computation. The installation details can be found
  `here <https://numba.pydata.org/numba-doc/latest/user/installing.html>`_. Further details on enhanced functionality provided in ArviZ by Numba can be 
  `found here <https://arviz-devs.github.io/arviz/user_guide/Numba.html>`_.

- Bokeh

  Necessary for creating advanced interactive visualisations. The Bokeh installation guide can be found `over here <http://docs.bokeh.org/en/dev/docs/first_steps/installation.html>`_.

- UltraJSON

  If available, ArviZ makes use of faster ujson when ``arviz.from_json(filename)`` is
  invoked. UltraJSON can be either installed via `pip <https://pypi.org/project/ujson/>`_ or `conda <https://anaconda.org/anaconda/ujson>`_.

- Dask

  Necessary to scale the packages and the surrounding ecosystem. The installation details can be found `at this link <https://docs.dask.org/en/latest/install.html>`_.
  



