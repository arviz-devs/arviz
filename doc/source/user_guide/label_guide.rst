.. _label_guide:

===========
Label guide
===========

Basic labeling
--------------

All ArviZ plotting functions and some stats functions take an optional ``labeller`` argument.
By default, labels show the variable name and the coordinate value
(for multidimensional variables only).
The first example below uses this default labeling.

.. ipython::

  In [1]: import arviz as az
     ...: schools = az.load_arviz_data("centered_eight")
     ...: az.summary(schools)

Thanks to being powered by xarray, ArviZ supports label based indexing.
We can therefore use the labels we have seen in the summary to plot only a subset of the variables,
the one we are interested in.
Provided we know that the coordinate values shown for theta correspond to the `school` dimension,
we can plot only ``tau`` to better inspect it's 1.03 :func:`~arviz.rhat` and
``theta`` for ``Choate`` and ``St. Paul's``, the ones with higher means:

.. ipython:: python

    @savefig label_guide_plot_trace.png
    az.plot_trace(schools, var_names=["tau", "theta"], coords={"school": ["Choate", "St. Paul's"]}, compact=False);

So far so good, we can identify some issues for low ``tau`` values which is great start.
But say we want to make a report on Deerfield, Hotchkiss and Lawrenceville schools to
see the probability of ``theta > 5`` and we have to present it somewhere with math notation.
Our default labels show ``theta``, not $\theta$ (generated from ``$\theta$`` using $\LaTeX$).

Fear not, we can use the labeller argument to customize the labels.
The ``arviz.labels`` module contains some classes that cover some common customization classes.

In this case, we can use :class:`~arviz.labels.MapLabeller` and
tell it to rename the variable name ``theta`` to ``$\theta$``, like so:

.. ipython::

    In [1]: import arviz.labels as azl
       ...: labeller = azl.MapLabeller(var_name_map={"theta": r"$\theta$"})
       ...: coords = {"school": ["Deerfield", "Hotchkiss", "Lawrenceville"]}

    @savefig label_guide_plot_posterior.png
    In [1]: az.plot_posterior(schools, var_names="theta", coords=coords, labeller=labeller, ref_val=5);

You can see the labellers available in ArviZ at :ref:`their API reference page <labeller_api>`.
Their names aim to be descriptive and they all have examples in their docstring.
For further customization continue reading this guide.

Sorting labels
--------------

Labels in ArviZ can generally be sorted in two ways,
using the arguments passed to ArviZ plotting functions or
sorting the underlying xarray Dataset.
The first one is more convenient for single time ordering
whereas the second is better if you want plots consistently sorted that way and
is also more flexible, using ArviZ args is more limited.

Both alternatives have an important limitation though.
Multidimensional variables are always together.
We can sort ``theta, mu, tau`` in any order, and within ``theta`` we can sort the schools in any order,
but it's not possible to show half the schools, then ``mu`` and ``tau`` and then the rest of the schools.

Sorting variable names
......................

.. ipython::

    In [1]: var_order = ["theta", "mu", "tau"]

.. tabbed:: ArviZ args

  We can pass a list with the variable names sorted to modify the order in which they appear
  when calling ArviZ functions

  .. ipython::

      In [1]: az.summary(schools, var_names=var_order)

.. tabbed:: xarray

  In xarray, subsetting the Datset with a sorted list of variable names will order the Dataset.

  .. ipython::

      In [1]: schools.posterior = schools.posterior[var_order]
         ...: az.summary(schools)

Sorting coordinate values
.........................

We may also want to sort the schools by their mean.
To do so we first have to get the means of each school:

.. ipython::

    In [1]: school_means = schools.posterior["theta"].mean(("chain", "draw"))
       ...: school_means

We can then use this DataArray result to sort the coordinate values for ``theta``.
Again we have two alternatives:

.. tabbed:: ArviZ args

  Here the first step is to sort the coordinate values so we can pass them as `coords` argument and
  choose the order of the rows.
  If we want to manually sort the schools, `sorted_schools` can be defined straight away as a list

  .. ipython::

      In [1]: sorted_schools = schools.posterior["school"].sortby(school_means)
         ...: az.summary(schools, var_names="theta", coords={"school": sorted_schools})

.. tabbed:: xarray

  We can use the :meth:`~xarray.Dataset.sortby` method to order our coordinate values straight at the source

  .. ipython::

      In [1]: schools.posterior = schools.posterior.sortby(school_means)
         ...: az.summary(schools, var_names="theta")

Sorting dimensions
..................

In some cases, our multidimensional variables may not have only a length ``n`` dimension
(in addition to the ``chain`` and ``draw`` ones)
but could also have multiple dimensions.
Let's imagine we have performed a set of fixed experiments on several days to multiple subjects,
three data dimensions overall.

We will create a fake inference data with data mimicking this situation to show how to sort dimensions.
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

Given how we have constructed our dataset, the default order is ``experiment, subject, date``

.. dropdown:: Click to see the default summary

  .. ipython::

      In [1]: az.summary(experiments)

However, we actually want to have the dimensions in this order: ``subject, date, experiment``.
And in this case, we need to modify the underlying xarray object in order to get the desired result:

.. ipython:: python

    dim_order = ("chain", "draw", "subject", "date", "experiment")
    experiments = experiments.posterior.transpose(*dim_order)
    az.summary(experiments)

Note however that we don't need to overwrite or store the modified xarray object.
Doing ``az.summary(experiments.posterior.transpose(*dim_order))`` would work just the same
if we only want to use this order once.

Labeling with indexes
---------------------

As you may have seen, there are labellers with ``Idx`` in their name:
:class:`~arviz.labels.IdxLabeller` and  :class:`~arviz.labels.DimIdxLabeller`,
which show the positional index of the values instead of their corresponding coordinate value.

We have seen before that we can use the ``coords`` argument or
the :meth:`~arviz.InferenceData.sel` method to select data based on the coordinate values.
Similarly, we can use the :meth:`~arviz.InferenceData.isel` method to select data based on positional indexes.

.. ipython:: python

    az.summary(schools, labeller=azl.IdxLabeller())

After seeing this summary, we use ``isel`` to generate the summary of a subset only.

.. ipython:: python

    az.summary(schools.isel(school=[2, 5, 7]), labeller=azl.IdxLabeller())

.. warning::

  Positional indexing is NOT label based indexing with numbers!

The positional indexes shown will correspond to the ordinal position *in the subsetted object*.
If you are not subsetting the object, you can use these indexes with ``isel`` without problem.
However, if you are subsetting the data (either directly or with the ``coords`` argument)
and want to use the positional indexes shown, you need to use them on the corresponding subset.

An example. If you use a dict named ``coords`` when calling a plotting function,
for ``isel`` to work it has to be called on
``original_idata.sel(**coords).isel(<desired positional idxs>)`` and
not on ``original_idata.isel(<desired positional idxs>)``

Labeller mixtures
-----------------

In some cases, none of the available labellers will do the right job.
One case where this is bound to happen is with ``plot_forest``.
When setting ``legend=True`` it does not really make sense to add the model name to the tick labels.
``plot_forest`` knows that, and if no ``labeller`` is passed, it uses either
:class:`~arviz.labels.BaseLabeller` or :class:`~arviz.labels.NoModelLabeller` depending on the value of ``legend``.
If we do want to use the ``labeller`` argument however, we have to make sure to enforce this default ourselves:

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
The variable names, dims and coords are shown for both models and
the models are labeled both in the legend and in the labels of the y axis.
For cases like this, ArviZ provides a convenience function :func:`~arviz.labels.mix_labellers`
that combines labeller classes for some extra customization.
Labeller classes aim to split labeling into atomic tasks and have a method per task to maximize extensibility.
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
So far we have managed to customize the labels in the plots without having to write a new class
from scratch all by ourselves. However, there could be cases where we have to customize our labels
further than what these sample labellers allow. In such cases, we have to subclass one of the
labellers in ``arviz.labels`` and override some of its methods.

One case where we might need to do this is when non indexing coordinates are present.
This happens for example after doing pointwise selection on multiple dimensions,
but we can also add extra dimensions to our models manually, as shown in TBD.
For this example, let's use pointwise selection.
Say for example one of the variables in the posterior represents a covariance matrix, and we want
to keep it as is for other post-processing tasks instead of extracting the sub diagonal triangular
matrix with no repeated info as a flattened array. Or any other pointwise selection really.

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

To select a non rectangular slice with xarray and get the result flattened and without nans, we can
use DataArrays indexed with a dimension which is not present in our current dataset:

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
therefore won't be available to the labeller:

.. ipython:: python

    @savefig default_plot_posterior.png
    az.plot_posterior(idata, coords=coords);

To get around this limitation, we will store the coords used for pointwise selection
as a Dataset which we will pass to the labeller so it can use the info it has available
(``pointwise_sel`` and its position in this case) to subset this coords Dataset
and use that instead to label.
One option is to format these non-indexing coordinates as a dictionary whose
keys are dimension names and values coordinate labels and pass that to the parent's
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

This has the advantage of requiring very little extra code and to allow to combine
our newly created ``NonIdxCoordLabeller`` with other labellers like we did in
the previous section.

Another option is to go for a much more customized look, and handle everything
on ``make_label_vert`` to get labels like "Correlation between subjects x and y".

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

This won't combine properly with other labellers, but it serves its function, and
achieves complete customization of the labels, so we probably won't want to combine
it with other labellers either. The main drawback is that we have only overridden
``make_label_vert``, so functions like ``plot_forest`` or ``summary`` who
use ``make_label_flat`` will still fall back to the methods defined by ``BaseLabeller``.
