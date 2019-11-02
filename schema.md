# InferenceData schema specification
The `InferenceData` schema scheme defines a data structure compatible with [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) with 3 goals in mind, usefulness in the analysis of Bayesian inference results, reproducibility of Bayesian inference analysis and interoperability between different inference backends and programming languages.

Currently there are 2 implementations of this design:
* [ArviZ](https://arviz-devs.github.io/arviz/) in Python which integrates with:
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [PyMC3](https://docs.pymc.io)
  - [pyro](https://pyro.ai/)
      and [numpyro](https://pyro.ai/numpyro/)
  - [PyStan](https://pystan.readthedocs.io/en/latest/index.html),
      [CmdStan](https://mc-stan.org/users/interfaces/cmdstan)
      and [CmdStanPy](https://cmdstanpy.readthedocs.io/en/latest/index.html)
  - [tensorflow-probability](https://www.tensorflow.org/probability)
* [ArviZ.jl](https://github.com/sethaxen/ArviZ.jl) in julia which integrates with:
  - [Turing](https://turing.ml/dev/).

## Current design
`InferenceData` stores all quantities relevant in order to fulfill its goals in different groups. Each group, described below, stores a conceptually different quantity generally represented by several multidimensional labeled variables.

Each group should have one entry per variable, with the first two dimensions of each variable should be the sample identifier (`chain`, `draw`). Dimensions must be named and explicit their index values, called coordinates. Coordinates can have repeated identifiers and may not be numerical. Variable names must not share names with dimensions.

Moreover, each group contains the following attributes:
* `created_at`: the date of creation of the group.
* `inference_library`: the library used to run the inference.
* `inference_library_version`: version of the inference library used.

`InferenceData` data objects contain any combination the groups described below.

#### `posterior`
Samples from the posterior distribution p(theta|y).

#### `sample_stats`
Information and diagnostics about each `posterior` sample, provided by the inference backend. It may vary depending on the algorithm used by the backend (i.e. an affine invariant sampler has no energy associated). The name convention used for `sample_stats` variables is the following:
* `lp`: unnormalized log probability of the sample
* `step_size`
* `step_size_bar`
* `tune`: boolean variable indicating if the sampler is tuning or sampling
* `depth`:
* `tree_size`:
* `mean_tree_accept`:
* `diverging`: HMC-NUTS only, boolean variable indicating divergent transitions
* `energy`: HMC-NUTS only
* `energy_error`
* `max_energy_error`

#### `observed_data`
Observed data on which the `posterior` is conditional. It should only contain data which is modeled as a random variable. Each variable should have a counterpart in `posterior_predictive`. The `posterior_predictive` counterpart variable may have a different name.

#### `posterior_predictive`
Posterior predictive samples p(y|y) corresponding to the posterior predictive pdf evaluated at the `observed_data`. Samples should match with `posterior` ones and each variable should have a counterpart in `observed_data`. The `observed_data` counterpart variable may have a different name.

#### `constant_data`
Model constants, data included in the model which is not modeled as a random variable (i.e. the x in a linear regression). It should be the data used to generate the `posterior` and `posterior_predictive` samples.

#### `prior`
p(theta)

#### `sample_stats_prior`

#### `prior_predictive`

## Planned features

### Sampler parameters

### Out of sample posterior_predictive samples
#### `predictions`
Out of sample posterior predictive samples p(y'|y).

#### `constant_data_predictions`
Model constants used to get the `predictions` samples. Its variables should have a counterpart in `constant_data`.
