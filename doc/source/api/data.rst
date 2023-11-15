.. currentmodule:: arviz

.. _data_api:

Data
----

Inference library converters
............................

.. autosummary::
   :toctree: generated/

   from_beanmachine
   from_cmdstan
   from_cmdstanpy
   from_emcee
   from_numpyro
   from_pyjags
   from_pyro
   from_pystan


IO / General conversion
.......................

.. autosummary::
  :toctree: generated/

  convert_to_inference_data
  convert_to_dataset
  dict_to_dataset
  from_datatree
  from_dict
  from_json
  from_netcdf
  to_datatree
  to_json
  to_netcdf
  from_zarr
  to_zarr


General functions
.................

.. autosummary::
   :toctree: generated/

   concat
   extract

Data examples
.............

.. autosummary::
   :toctree: generated/

   list_datasets
   load_arviz_data
