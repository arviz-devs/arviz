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

    @savefig label_guide_plot_trace.png
    az.plot_trace(schools, var_names=["tau", "theta"], coords={"school": ["Choate", "St. Paul's"]}, compact=False);

Using the above command, you can now identify issues for low ``tau`` values.

Example: Using the labeller argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ``labeller`` argument to customize labels.
Unlike the default labels that show ``theta``, not :math:`\theta` (generated from ``$\theta$`` using :math:`\LaTeX`), the ``labeller`` argument presents the labels with proper math notation.


You can use :class:`~arviz.labels.MapLabeller` to rename the variable ``theta`` to ``$\theta$``, as shown in the following example:

.. ipython::

    In [1]: import arviz.labels as azl
       ...: labeller = azl.MapLabeller(var_name_map={"theta": r"$\theta$"})
       ...: coords = {"school": ["Deerfield", "Hotchkiss", "Lawrenceville"]}

    @savefig label_guide_plot_posterior.png
    In [1]: az.plot_posterior(schools, var_names="theta", coords=coords, labeller=labeller, ref_val=5);

.. seealso::

   For a list of labellers available in ArviZ, see the :ref:`the API reference page <labeller_api>`.

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

        In xarray, subsetting the Dataset with a sorted list of variable names will order the Dataset.

        .. ipython::

            In [1]: schools.posterior = schools.posterior[var_order]
               ...: az.summary(schools)

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

        You can use the :meth:`~xarray.Dataset.sortby` method to order our coordinate values directly at the source.

        .. ipython::

            In [1]: schools.posterior = schools.posterior.sortby(school_means)
               ...: az.summary(schools, var_names="theta")

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
       ...:     posterior={"b": samples}, dims={"b": ["experiment", "subject", "date"]}, coords=coords
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
    experiments = experiments.posterior.transpose(*dim_order)
    az.summary(experiments)

.. note::

    However, we don't need to overwrite or store the modified xarray object.
    Doing ``az.summary(experiments.posterior.transpose(*dim_order))`` would work just the same
    if we only want to use this order once.

Labeling with indexes
---------------------

As you may have seen, there are some labellers with ``Idx`` in their name:
:class:`~arviz.labels.IdxLabeller` and  :class:`~arviz.labels.DimIdxLabeller`.
They show the positional index of the values instead of their corresponding coordinate value.

We have seen before that we can use the ``coords`` argument or
the :meth:`~arviz.InferenceData.sel` method to select data based on the coordinate values.
Similarly, we can use the :meth:`~arviz.InferenceData.isel` method to select data based on positional indexes.

.. ipython:: python

    az.summary(schools, labeller=azl.IdxLabeller())

After seeing the above summary, let's use ``isel`` method to generate the summary of a subset only.

.. ipython:: python

    az.summary(schools.isel(school=[2, 5, 7]), labeller=azl.IdxLabeller())

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

In some cases, none of the available labellers do the right job.
For example, one case where this is bound to happen is with :func:`~arviz.plot_forest`.
When setting ``legend=True`` it does not really make sense to add the model name to the tick labels.
``plot_forest`` knows that, and if no ``labeller`` is passed, it uses either
:class:`~arviz.labels.BaseLabeller` or :class:`~arviz.labels.NoModelLabeller` depending on the value of ``legend``.
However, if we do want to use the ``labeller`` argument, we have to enforce this default ourselves:

.. ipython:: python

    schools2 = az.load_arviz_data("non_centered_eight")

    @savefig default_plot_forest.png
    az.plot_forest(
        (schools, schools2),
        model_names=("centered", "non_centered"),
        coords={"school": ["Deerfield", "Lawrenceville", "Mt. Hermon"]},
        figsize=(10,7),
        labeller=azl.DimCoordLabeller(),
        legend=True
    );

There is a lot of repeated information now.
The variable names, ``dims`` and ``coords`` are shown for both models.
Moreover, the models are labeled both in the legend and in the labels of the y axis.
For such cases, ArviZ provides a convenience function :func:`~arviz.labels.mix_labellers`
that combines labeller classes for some extra customization.

**Labeller classes** aim to split labeling into atomic tasks and have a method per task to maximize extensibility.
Thus, many new labellers can be created with this mixer function alone without needing to write a new class from scratch.
There are more usage examples of :func:`~arviz.labels.mix_labellers` in its docstring page, click on
it to go there.

.. ipython:: python

    MixtureLabeller = azl.mix_labellers((azl.DimCoordLabeller, azl.NoModelLabeller))

    @savefig mixture_plot_forest.png
    az.plot_forest(
        (schools, schools2),
        model_names=("centered", "non_centered"),
        coords={"school": ["Deerfield", "Lawrenceville", "Mt. Hermon"]},
        figsize=(10,7),
        labeller=MixtureLabeller(),
        legend=True
    );

Custom labellers
----------------
So far we have managed to customize the labels in the plots without writing a new class
from scratch. However, there could be cases where we have to customize our labels
further than what these sample labellers allow. In such cases, we have to subclass one of the
labellers in :ref:`arviz.labels <labeller_api>` and override some of its methods.

One case where we might need to do use this approach is when non indexing coordinates are present.
This happens for example after doing pointwise selection on multiple dimensions,
but we can also add extra dimensions to our models manually, as shown in TBD.
For this example, let's use pointwise selection.
Let's say one of the variables in the posterior represents a `covariance matrix <https://en.wikipedia.org/wiki/Covariance_matrix>`_, and we want
to keep it as is for other post-processing tasks instead of extracting the sub diagonal `triangular
matrix <https://en.wikipedia.org/wiki/Triangular_matrix)>`_ with no repeated info as a flattened array. Or any other pointwise selection.

Here is our data:


.. ipython:: python

    from numpy.random import default_rng
    import numpy as np
    import xarray as xr
    rng = default_rng()
    cov = rng.normal(size=(4, 500, 3, 3))
    cov = np.einsum("...ij,...kj", cov, cov)
    cov[:, :, [0, 1, 2], [0, 1, 2]] = 1
    subjects = ["ecoli", "pseudomonas", "clostridium"]
    idata = az.from_dict(
        {"cov": cov},
        dims={"cov": ["subject", "subject bis"]},
        coords={"subject": subjects, "subject bis": subjects}
    )
    idata.posterior

To select a non rectangular slice with xarray and to get the result flattened and without NaNs, we can
use ``DataArray`` s indexed with a dimension that is not present in our current dataset:

.. ipython:: python

    coords = {
        'subject': xr.DataArray(
            ["ecoli", "ecoli", "pseudomonas"], dims=['pointwise_sel']
        ),
        'subject bis': xr.DataArray(
            ["pseudomonas", "clostridium", "clostridium"], dims=['pointwise_sel']
        )
    }
    idata.posterior.sel(coords)

We see now that ``subject`` and ``subject bis`` are no longer indexing coordinates, and
therefore won't be available to the ``labeller``:

.. ipython:: python

    @savefig default_plot_posterior.png
    az.plot_posterior(idata, coords=coords);

To get around this limitation, we will store the ``coords`` used for pointwise selection
as a Dataset. We will pass this Dataset to the ``labeller`` so it can use the info it has available
(``pointwise_sel`` and its position in this case) to subset this ``coords`` Dataset
and use that instead to label.
One option is to format these non-indexing coordinates as a dictionary whose
keys are dimension names and values are coordinate labels and pass that to the parent's
``sel_to_str`` method:

.. ipython:: python

    coords_ds = xr.Dataset(coords)

    class NonIdxCoordLabeller(azl.BaseLabeller):
        """Use non indexing coordinates as labels."""
        def __init__(self, coords_ds):
            self.coords_ds = coords_ds
        def sel_to_str(self, sel, isel):
            new_sel = {k: v.values for k, v in self.coords_ds.sel(sel).items()}
            return super().sel_to_str(new_sel, new_sel)

    labeller = NonIdxCoordLabeller(coords_ds)

    @savefig custom_plot_posterior1.png
    az.plot_posterior(idata, coords=coords, labeller=labeller);

This has the following advantages:

- It requires very little extra code.
- It allows to combine our newly created ``NonIdxCoordLabeller`` with other labellers as we did in the previous section.

Another option is to go for a much more customized look, and handle everything
on :meth:`~arviz.labels.BaseLabeller.make_label_vert` to get labels like "Correlation between subjects x and y".

.. ipython:: python

    class NonIdxCoordLabeller(azl.BaseLabeller):
        """Use non indexing coordinates as labels."""
        def __init__(self, coords_ds):
            self.coords_ds = coords_ds
        def make_label_vert(self, var_name, sel, isel):
            coords_ds_subset = self.coords_ds.sel(sel)
            subj = coords_ds_subset["subject"].values
            subj_bis = coords_ds_subset["subject bis"].values
            return f"Correlation between subjects\n{subj} & {subj_bis}"

    labeller = NonIdxCoordLabeller(coords_ds)

    @savefig custom_plot_posterior2.png
    az.plot_posterior(idata, coords=coords, labeller=labeller);

This won't combine properly with other labellers, but it serves its function and
achieves complete customization of the labels, so we probably won't want to combine
it with other labellers either. The main drawback is that we have only overridden
``make_label_vert``, so functions like ``plot_forest`` or ``summary`` who
use :meth:`~arviz.labels.BaseLabeller.make_label_flat` will still fall back to the methods defined by ``BaseLabeller``.
