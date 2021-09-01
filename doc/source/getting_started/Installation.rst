##################
Installation guide
##################

This section provides detailed information about installing ArviZ. Most ArviZ
functionality is available with the basic requirements, but ArviZ also has optional
dependencies to further enhance the library. This guide will cover both basic and fully-fledged ArviZ installs and several installation methods.


******
Stable
******

ArviZ can be installed either using pip or conda-forge.

Using pip
=========

.. code:: bash

    pip install arviz

Use the below pip command to install ArviZ with all of its :ref:`Optional-dependencies`.

.. code:: bash

    pip install "arviz[all]"

Using conda-forge
=================

.. code:: bash

    conda install -c conda-forge arviz

.. _dev-version:

***********
Development
***********

If you want to install the latest development version of ArviZ, use the following command:

.. code:: bash

    pip install git+https://github.com/arviz-devs/arviz

**Note**: It can take sometime to execute depending upon your internet connection.

.. _arviz-dependencies:

************
Dependencies
************

Required dependencies
=====================

The required dependencies for installing ArviZ are:

.. literalinclude:: ../../../requirements.txt

and

.. code:: bash

    python>=3.6

.. _Optional-dependencies:

Optional dependencies
=====================

The list of optional dependencies to further enhance ArviZ is given below.

.. literalinclude:: ../../../requirements-optional.txt


- Numba

  Necessary to speed up the code computation. The installation details can be found
  `here <https://numba.pydata.org/numba-doc/latest/user/installing.html>`_. Further details on enhanced functionality provided in ArviZ by Numba can be
  `found here <https://arviz-devs.github.io/arviz/user_guide/Numba.html>`_.

- Bokeh

  Necessary for creating advanced interactive visualisations. The Bokeh installation guide can be found `over here <http://docs.bokeh.org/en/dev/docs/first_steps/installation.html>`_.

- UltraJSON

  If available, ArviZ makes use of faster ujson when :func:`arviz.from_json` is
  invoked. UltraJSON can be either installed via `pip <https://pypi.org/project/ujson/>`_ or `conda <https://anaconda.org/anaconda/ujson>`_.

- Dask

  Necessary to scale the packages and the surrounding ecosystem. The installation details can be found `at this link <https://docs.dask.org/en/latest/install.html>`_.




