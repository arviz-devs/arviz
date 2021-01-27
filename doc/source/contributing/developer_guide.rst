.. developer_guide:


===============
Developer Guide
===============


Backends
========
ArviZ now supports multiple backends. If adding another backend please ensure you meeting the
following design patterns.

Code Separation
---------------
Each backend should be placed in a different module per the backend.
See ``arviz.plots.backends`` for examples

The code in the root level of ``arviz.plots`` should not contain
any opinion on backend. The idea is that the root level plotting
function performs math and construct keywords, and the backends
code in ``arviz.plots.backends`` perform the backend specific
keyword argument defaulting and plot behavior

The convenience function ``get_plotting_function`` available in
``arviz.plots.get_plotting_function`` should be called to obtain
the correct plotting function from the associated backend. If
adding a new backend follow the pattern provided to programatically
call the correct backend

Test Separation
---------------
Tests for each backend should be split into their own module
See ``tests.test_plots_matplotlib`` for an example

Gallery Examples
----------------
Gallery examples are not required but encouraged. Examples are
compiled into the arviz documentation website. The ``examples`` directory
can be found in the root of the arviz git repository.


Documentation
=============

Docstring style
---------------
See the corresponding section in the `contributing guide <https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md#docstring-formatting-and-type-hints>`_

Hyperlinks
----------
Complementary functions such as ``compare`` and ``plot_compare`` should reference
each other in their docstrings using a hyperlink, not only by name. The same
should happen with external functions whose usage is assumed to be known; a
clear example of this situation are docstrings on ``kwargs`` passed to bokeh or
matplotlib methods. This section covers how to reference functions from any
part of the docstring or from the `See also` section.

Reference external libraries
""""""""""""""""""""""""""""

Sphinx is configured to ease referencing libraries ArviZ relies heavily on by
using `intersphinx <https://docs.readthedocs.io/en/stable/guides/intersphinx.html>`_.
See guidance on the reference about how to link to objects from external
libraries and the value of ``intersphinx_mapping`` in ``conf.py`` for the complete and up to
date list of libraries that can be referenced. Note that the ``:key:`` before
the reference must match the kind of object that is being referenced, it
generally will not be ``:ref:`` nor ``:doc:``. For
example, for functions ``:func:`` has to be used and for class methods
``:meth:``. The complete list of keys can be found `here <https://github.com/sphinx-doc/sphinx/blob/685e3fdb49c42b464e09ec955e1033e2a8729fff/sphinx/domains/python.py#L845-L881>`_.

The extension `sphobjinv <https://sphobjinv.readthedocs.io/en/latest/>`_ can
also be helpful in order to get the exact type and name of a reference. Below
is an example on getting a reference from matplotlib docs::

  $ sphobjinv suggest -t 90 -u https://matplotlib.org/objects.inv "axes.plot"

  Remote inventory found.

  :py:method:`matplotlib.axes.Axes.plot`
  :py:method:`matplotlib.axes.Axes.plot_date`
  :std:doc:`api/_as_gen/matplotlib.axes.Axes.plot`
  :std:doc:`api/_as_gen/matplotlib.axes.Axes.plot_date`

We can therefore link to matplotlib docs on ``Axes.plot`` from any docstring
using::

  :meth:`mpl:matplotlib.axes.Axes.plot`

The `intersphinx_mappings`
defined for ArviZ can be seen in `conf.py <https://github.com/arviz-devs/arviz/blob/main/doc/conf.py>`_.
Moreover, the intersphinx key is optional. Thus, the pattern to get sphinx to generate links is::

  :type_id:`(intersphinx_key:)object_id`

with the part between brackets being optional. See the docstring on
:meth:`~arviz.InferenceData.to_dataframe` and
`its source <https://arviz-devs.github.io/arviz/_modules/arviz/data/inference_data.html#InferenceData.to_dataframe>`_ for an example.

Referencing ArviZ objects
"""""""""""""""""""""""""

The same can be done to refer to ArviZ functions, in which case,
``:func:`arviz.loo``` is enough, there is no need to use ``intersphinx``.
Moreover, using ``:func:`~arviz.loo``` will only show ``loo`` as link text
due to the preceding ``~``.

In addition, the `See also` docstring section is also available. Sphinx will
automatically add links to other ArviZ objects listed in the `See also`
section.
