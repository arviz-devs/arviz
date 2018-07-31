Arviz Roadmap
=============

*August 1, 2018*

This is super preliminary and reflects thoughts on what needs to be done
to get to a stable release.

Road to v1.0
------------

I would guess that this all takes 2-6 months to do, though as an open source
project this can vary quite a bit (and *you* can speed this up). The initial PyPI
release may be quite unstable, but will allow faster iteration as PyStan
and PyMC3 are able to start developing with it. The initial goal here is
to support at least those two libraries, while remaining general enough
to support other probabilistic programming libraries in the future.

-  Convert plots to xarray

   -  ``violintraceplot``

   -  ``posteriorplot``

   -  ``parallelplot``

      -  Requires divergences to be useful

   -  ``pairplot``

      -  Looks a lot nicer with divergences

   -  ``energyplot``

      -  Requires sampler statistics

-  Convert stats to xarray

   -  Maybe implementing these directly in ``numpy`` instead, allowing
      PyMC3/PyStan to use them without going through ``xarray``?

   -  I haven’t looked closely enough at these to know which need more than
      a posterior

-  Integrate into PyMC3

   -  Will require a release of ``arviz`` on pypi

-  Initial strategy will probably be making ``arviz`` flexible enough
   that ``pymc3`` can edit the plots as desired (but try to keep code on
   ``pymc3`` side to handling styles, and not data)

-  Integrate with PyStan

   -  Similar to above, though will remain separate.

   -  Perhaps add ``stan`` and ``pymc3`` stylesheets
      directly into ArviZ?

Longer term
-----------

-  Support full netcdf data schema

   -  A standard, useful data format for probabilistic programming is
      *very* interesting

   -  We have three libraries (pystan, pymc3, pymc4) with three different
      inference backends (C++ templates, theano, tensorflow) that are used
      in industry and academia that are interested in using this, which is
      a great test suite

   -  In some of the plots we are not very careful about memory efficiency,
      but a benefit of xarray is being able to lazily read data from disk

-  My impression is that ``/divergences`` and ``/sampler_statistics``
   would be the next two datasets to support

-  Other backends? [MOSTLY REJECTED: Let's focus on matplotlib for now!]

   -  ``altair``

      -  Very modular, interactive, easy sharing via the web

      -  Supporting xarray `on their radar <https://github.com/altair-viz/altair/issues/891>`__

      -  Would require a pretty complete rewrite

      -  Right now only supports up to 5,000 points in an underlying
            dataset, which *most* traces come up against (there’s talk about
            using a netcdf backend which could help)

   -  ``bokeh``

      -  Also works on web

      -  *Not as familiar with pros/cons here*

-  Reach out to other libraries that may be interested

-  Should even automatically support other languages through ``netCDF``. Would be
   interesting to get an example from ``rstan`` into ``netCDF`` and then into ArviZ!
