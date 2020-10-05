.. currentmodule:: arviz

.. _data_api:

Data
----

Inference library converters
............................

.. autosummary::
   :toctree: generated/

   from_cmdstan
   from_cmdstanpy
   from_emcee
   from_numpyro
   from_pyjags
   from_pymc3
   from_pymc3_predictions
   from_pyro
   from_pystan
   from_tfp


IO / General conversion
.......................

.. autosummary::
  :toctree: generated/

  convert_to_inference_data
  convert_to_dataset
  dict_to_dataset
  from_dict
  from_json
  from_netcdf
  to_netcdf


General functions
.................

.. autosummary::
   :toctree: generated/

   concat

Data examples
.............

.. autosummary::
   :toctree: generated/

   list_datasets
   load_arviz_data
