##################
Installation guide
##################

This section provides detailed information about installing ArviZ. Most ArviZ
functionality is available with the basic requirements, but ArviZ also has optional
dependencies to further enhance the library. This guide will cover both basic and fully-fledged ArviZ installs and several installation methods.


.. tip::

   If you are installing ArviZ for the first time, consider installing with

   .. code:: bash

      pip install "arviz[preview]"

   And using the library through the {mod}`arviz.preview` module.
   While some features in ``arviz.`` are not available
   in ``arviz.preview.``,  most have already been included.
   Additionally, ``arviz.preview.`` include many new ones that
   are not available in ``arviz.`` and never will be.


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

It is also possible to use ``pip install "arviz[preview]"`` to access
:ref:`upcoming refactored features <preview_api>`.

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

.. code::

    python>=3.10

ArviZ follows `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_
and `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_ to choose the minimum
supported versions.

.. _Optional-dependencies:

Optional dependencies
=====================

The list of optional dependencies to further enhance ArviZ is given below.

.. literalinclude:: ../../../requirements-optional.txt


- Numba

  Necessary to speed up the code computation. The installation details can be found
  `here <https://numba.readthedocs.io/en/stable/user/installing.html>`_. Further details on enhanced functionality provided in ArviZ by Numba can be
  :ref:`found here <numba_for_arviz>`.

- Bokeh

  Necessary for creating advanced interactive visualisations. The Bokeh installation guide can be found `over here <https://docs.bokeh.org/en/latest/docs/first_steps/installation.html>`_.

- UltraJSON

  If available, ArviZ makes use of faster ujson when :func:`arviz.from_json` is
  invoked. UltraJSON can be either installed via `pip <https://pypi.org/project/ujson/>`_ or `conda <https://anaconda.org/anaconda/ujson>`_.

- Dask

  Necessary to scale the packages and the surrounding ecosystem. The installation details can be found `at this link <https://docs.dask.org/en/latest/install.html>`_.




