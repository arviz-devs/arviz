.. _label_guide:

===========
Label guide
===========

Basic labelling
---------------

All ArviZ plotting functions and some stats functions can take an optional ``labeller`` argument.
By default, labels show the variable name.
Multidimensional variables also show the coordinate value.

Example: Default labelling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython::

    In [1]: import arviz as az
       ...: import xarray as xr
       ...: schools = az.load_arviz_data("centered_eight")
       ...: az.summary(schools)

ArviZ supports label based indexing powered by `xarray <http://xarray.pydata.org/en/stable/getting-started-guide/why-xarray.html>`_.
Through label based indexing, you can use labels to plot a subset of selected variables.

Example: Label based indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a case where the coordinate values shown for the ``theta`` variable coordinate to the ``school`` dimension,
you can indicate ArviZ to plot ``tau`` by including it in the ``var_names`` argument to inspect its 1.03 :func:`~arviz.rhat` value.
To inspect the ``theta`` values for the ``Choate`` and ``St. Paul's`` coordinates, you can include ``theta`` in ``var_names`` and use the ``coords`` argument to select only these two coordinate values.
You can generate this plot with the following command:

.. ipython:: python

    @savefig label_guide_plot_trace_dist.png
    az.plot_trace_dist(
        schools,
        var_names=["tau", "theta"],
        coords={"school": ["Choate", "St. Paul's"]},
        compact=False
    );

Using the above command, you can now identify issues for low ``tau`` values.

Example: Using the labeller argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ``labeller`` argument to customize labels.
Unlike the default labels that show ``theta``, not :math:`\theta` (generated from ``$\theta$`` using :math:`\LaTeX`), the ``labeller`` argument presents the labels with proper math notation.

You can use :class:`~arviz_base.labels.MapLabeller` to rename the variable ``theta`` to ``$\theta$``, as shown in the following example:

.. ipython::

    In [1]: import arviz.labels as azl
       ...: labeller = azl.MapLabeller(var_name_map={"theta": r"$\theta$"})
       ...: coords = {"school": ["Deerfield", "Hotchkiss", "Lawrenceville"]}

    @savefig label_guide_plot_dist.png
    In [1]: az.plot_dist(
       ...:     schools,
       ...:     var_names="theta",
       ...:     coords=coords,
       ...:     labeller=labeller
       ...: );

.. seealso::

   For a list of labellers available in ArviZ, see the :ref:`the API reference page <arviz_base:labeller_api>`.

Sorting labels
--------------

ArviZ allows labels to be sorted in two ways:

1. Using the arguments passed to ArviZ plotting functions
2. Sorting the underlying :class:`xarray.Dataset`

The first option is more suitable for single time ordering whereas the second option is more suitable for sorting plots consistently.

.. note::

  Both ways are limited.
  Multidimensional variables can not be separated.
  For example, it is possible to sort ``theta, mu,`` or ``tau`` in any order, and within ``theta`` to sort the schools in any order, but it is not possible to sort half of the schools, then ``mu`` and ``tau`` and then the rest of the schools.


Sorting variable names
~~~~~~~~~~~~~~~~~~~~~~

.. ipython::

    In [1]: var_order = ["theta", "mu", "tau"]

.. tab-set::

    .. tab-item:: ArviZ args

        For variable names to appear sorted when calling ArviZ functions, pass a sorted list of the variable names.

        .. ipython::

            In [1]: az.summary(schools, var_names=var_order)

    .. tab-item:: xarray

        ArviZ's ``posterior`` group is a :class:`~xarray.DataTree`. To reorder
        its variables, subset the underlying :class:`~xarray.Dataset` (via
        ``.dataset``) with a sorted list of variable names.

        .. ipython::

            In [1]: az.summary(schools.posterior.dataset[var_order])

Sorting coordinate values
~~~~~~~~~~~~~~~~~~~~~~~~~

For sorting coordinate values, first, define the order, then store it, and use the result to sort the coordinate values.
You can define the order by creating a list manually or by using xarray objects as illustrated in the below example "Sorting out the schools by mean".

Example: Sorting the schools by mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Locate the means of each school by using the following command:

.. ipython::

    In [1]: school_means = schools.posterior["theta"].mean(("chain", "draw"))
       ...: school_means

* You can use the ``DataArray`` result to sort the coordinate values for ``theta``.

There are two ways of sorting:

#. Arviz args
#. xarray

.. tab-set::

    .. tab-item:: ArviZ args

        Sort the coordinate values to pass them as a ``coords`` argument and choose the order of the rows.

        .. ipython::

            In [1]: sorted_schools = schools.posterior["school"].sortby(school_means)
               ...: az.summary(schools, var_names="theta", coords={"school": sorted_schools})

    .. tab-item:: xarray

        ``DataTree`` does not implement :meth:`~xarray.Dataset.sortby` directly.
        Apply it to the underlying ``Dataset`` (via ``.dataset``).

        .. ipython::

            In [1]: az.summary(schools.posterior.dataset.sortby(school_means), var_names="theta")

Sorting dimensions
~~~~~~~~~~~~~~~~~~

In some cases, our multidimensional variables may not have only one more dimension (a length ``n`` dimension
in addition to the ``chain`` and ``draw`` ones)
but could have multiple more dimensions.
Let's imagine we have performed a set of fixed experiments on several days to multiple subjects,
three data dimensions overall.

We will create fake inference data with data mimicking this situation to show how to sort dimensions.
To keep things short and not clutter the guide too much with unnecessary output lines,
we will stick to a posterior of a single variable and the dimension sizes will be ``2, 3, 4``.

.. ipython::

    In [1]: from numpy.random import default_rng
       ...: import pandas as pd
       ...: rng = default_rng()
       ...: samples = rng.normal(size=(4, 500, 2, 3, 4))
       ...: coords = {
       ...:     "subject": ["ecoli", "pseudomonas", "clostridium"],
       ...:     "date": ["1-3-2020", "2-4-2020", "1-5-2020", "1-6-2020"],
       ...:     "experiment": [1, 2]
       ...: }
       ...: experiments = az.from_dict(
       ...:     {"posterior": {"b": samples}}, dims={"b": ["experiment", "subject", "date"]}, coords=coords
       ...: )
       ...: experiments.posterior

Given how we have constructed our dataset, the default order is ``experiment, subject, date``.

.. dropdown:: Click to see the default summary

  .. ipython::

      In [1]: az.summary(experiments)

However, the order we want is: ``subject, date, experiment``.
Now, to get the desired result, we need to modify the underlying xarray object.

.. ipython:: python

    dim_order = ("chain", "draw", "subject", "date", "experiment")
    az.summary(experiments.posterior.dataset.transpose(*dim_order))

.. note::

    ``DataTree`` does not implement :meth:`~xarray.Dataset.transpose` directly,
    so it must be applied to the underlying ``Dataset`` via ``.dataset``.
    ArviZ functions accept a ``Dataset`` directly, so no re-wrapping into
    a ``DataTree`` is needed.

Labeling with indexes
---------------------

As you may have seen, there are some labellers with ``Idx`` in their name:
:class:`~arviz_base.labels.IdxLabeller` and :class:`~arviz_base.labels.DimIdxLabeller`.
They show the positional index of the values instead of their corresponding coordinate value.

We have seen before that we can use the ``coords`` argument or
the :meth:`~arviz.InferenceData.sel` method to select data based on the coordinate values.
Similarly, we can use the :meth:`~arviz.InferenceData.isel` method to select data based on positional indexes.

.. ipython:: python

    az.plot_forest(schools, var_names="theta", labeller=azl.IdxLabeller())

After seeing the above plot, let's use ``isel`` method to generate the plot of a subset only.

.. ipython:: python

    az.plot_forest(schools.isel(school=[2, 5, 7]), var_names="theta", labeller=azl.IdxLabeller())

.. warning::

  Positional indexing is NOT label based indexing with numbers!

The positional indexes shown will correspond to the ordinal position in the *subsetted object*.
If you are not subsetting the object, you can use these indexes with ``isel`` without problem.
However, if you are subsetting the data (either directly or with the ``coords`` argument)
and want to use the positional indexes shown, you need to use them on the corresponding subset.

**Example**: If you use a dict named ``coords`` when calling a plotting function,
for ``isel`` to work it has to be called on
``original_idata.sel(**coords).isel(<desired positional idxs>)`` and
not on ``original_idata.isel(<desired positional idxs>)``.

Labeller mixtures
-----------------

.. TODO: Update this section to use ``plot_lm`` which benefits more directly
   from labeller mixtures and custom labellers.

Different labellers can be combined using :func:`~arviz_base.labels.mix_labellers`, so
that each labeller in the mixture contributes the formatting it specializes
in. This is generally more convenient than writing a labeller from scratch
when you only want to tweak one part of the default behaviour.

Custom labellers
----------------

.. TODO: Update this section to use ``plot_lm`` and override
   ``make_label_flat`` for ``plot_forest``/``summary`` use cases.

For full control over label formatting, subclass
:class:`~arviz_base.labels.BaseLabeller` and override the relevant method(s).
