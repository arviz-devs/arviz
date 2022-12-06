.. currentmodule:: arviz

.. _idata_api:

InferenceData
-------------

Constructor
...........

.. autosummary::
    :toctree: generated/

    InferenceData

Attributes
..........
``InferenceData`` objects store :class:`xarray:xarray.Dataset` as attributes.
The :ref:`schema` contains the guidance related to these attributes.

.. autosummary::
  :toctree: generated/

  InferenceData.get_index

IO / Conversion
...............

.. autosummary::
  :toctree: generated/

  InferenceData.to_dataframe
  InferenceData.to_json
  InferenceData.from_netcdf
  InferenceData.to_netcdf
  InferenceData.from_zarr
  InferenceData.to_zarr
  InferenceData.chunk
  InferenceData.compute
  InferenceData.load
  InferenceData.persist
  InferenceData.unify_chunks

Dictionary interface
....................

.. autosummary::
  :toctree: generated/

  InferenceData.groups
  InferenceData.items
  InferenceData.values

InferenceData contents
......................

.. autosummary::
  :toctree: generated/

  InferenceData.add_groups
  InferenceData.extend
  InferenceData.assign
  InferenceData.assign_coords
  InferenceData.rename
  InferenceData.rename_dims
  InferenceData.set_coords
  InferenceData.set_index

Indexing
........

.. autosummary::
    :toctree: generated/

    InferenceData.isel
    InferenceData.sel
    InferenceData.reset_index

Computation
...........

.. autosummary::
  :toctree: generated/

  InferenceData.map
  InferenceData.cumsum
  InferenceData.max
  InferenceData.mean
  InferenceData.median
  InferenceData.min
  InferenceData.quantile
  InferenceData.sum

Reshaping and reorganizing
..........................

.. autosummary::
  :toctree: generated/

  InferenceData.stack
  InferenceData.unstack
