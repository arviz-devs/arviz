# InferenceData schema specification
The `InferenceData` schema scheme defines a data structure compatible with [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) with 3 goals in mind, usefulness in the analysis of Bayesian inference results, reproducibility of Bayesian inference analysis and interoperability between different inference backends and programming languages.

Currently there are 2 beta implementations of this design:
* [ArviZ](https://arviz-devs.github.io/arviz/) in Python which integrates with:
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [PyMC3](https://docs.pymc.io)
  - [pyro](https://pyro.ai/)
      and [numpyro](https://pyro.ai/numpyro/)
  - [PyStan](https://pystan.readthedocs.io/en/latest/index.html),
      [CmdStan](https://mc-stan.org/users/interfaces/cmdstan)
      and [CmdStanPy](https://cmdstanpy.readthedocs.io/en/latest/index.html)
  - [tensorflow-probability](https://www.tensorflow.org/probability)
* [ArviZ.jl](https://github.com/sethaxen/ArviZ.jl) in Julia which integrates with:
  - [Turing.jl](https://turing.ml/dev/) and indirectly any package using [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl) to store results
  - [CmdStan.jl](https://github.com/StanJulia/CmdStan.jl), [StanSample.jl](https://github.com/StanJulia/StanSample.jl) and [Stan.jl](https://github.com/StanJulia/Stan.jl)

## Contents
1. [Current design](#current-design)
   1. [`posterior`](#posterior)
   1. [`sample_stats`](#sample_stats)
   1. [`posterior_predictive`](#posterior_predictive)
   1. [`observed_data`](#observed_data)
   1. [`constant_data`](#constant_data)
   1. [`prior`](#prior)
   1. [`sample_stats_prior`](#sample_stats_prior)
   1. [`prior_predictive`](#prior_predictive)
1. [Planned features](#planned-features)
   1. [Sampler parameters](#sampler-parameters)
   1. [Out of sample posterior_predictive samples](#out-of-sample-posterior_predictive-samples)
1. [Examples](#examples)

## Current design
`InferenceData` stores all quantities relevant in order to fulfill its goals in different groups. Each group, described below, stores a conceptually different quantity generally represented by several multidimensional labeled variables.

Each group should have one entry per variable. When relevant, the first two dimensions of each variable should be the sample identifier (`chain`, `draw`). For groups like `observed_data` or `constant_data` these two initial dimensions are omitted. Dimensions must be named and specify their index values, called coordinates. Coordinates can have repeated identifiers and may not be numerical. Variable names must not share names with dimensions.

Moreover, each group contains the following attributes:
* `created_at`: the date of creation of the group.
* `inference_library`: the library used to run the inference.
* `inference_library_version`: version of the inference library used.

`InferenceData` data objects contain any combination the groups described below.

#### `posterior`
Samples from the posterior distribution p(theta|y).

#### `sample_stats`
Information and diagnostics for each `posterior` sample, provided by the inference backend. It may vary depending on the algorithm used by the backend (i.e. an affine invariant sampler has no energy associated). The name convention used for `sample_stats` variables is the following:
* `lp`: (unnormalized) log probability for sample
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

#### `posterior_predictive`
Posterior predictive samples p(y|y) corresponding to the posterior predictive distribution evaluated at the `observed_data`. Samples should match with `posterior` ones and each variable should have a counterpart in `observed_data`. The `observed_data` counterpart variable may have a different name.

#### `observed_data`
Observed data on which the `posterior` is conditional. It should only contain data which is modeled as a random variable. Each variable should have a counterpart in `posterior_predictive`, however, the `posterior_predictive` counterpart variable may have a different name.

#### `constant_data`
Model constants, data included in the model which is not modeled as a random variable. It should be the data used to generate the `posterior` and `posterior_predictive` samples.

#### `prior`
Samples from the prior distribution p(theta). Samples should not match `posterior` samples. However, this group will still follow the convention on `chain` and `draw` as first dimensions. Each variable should have a counterpart in `posterior`.

#### `sample_stats_prior`
Information and diagnostics for each `prior` sample, provided by the inference backend. It may vary depending on the algorithm used by the backend (i.e. an affine invariant sampler has no energy associated). Variable names follow the same convention defined in [`sample_stats`](#sample_stats).

#### `prior_predictive`
Samples from the prior predictive distribution. Samples should match `prior` samples and each variable should have a counterpart in `posterior_predictive`/`observed_data`.

## Planned features

### Sampler parameters

### Out of sample posterior_predictive samples
#### `predictions`
Out of sample posterior predictive samples p(y'|y). Sample should match `posterior` samples. Its variables should have a counterpart in `posterior_predictive`. However, variables in `predictions` and their counterpart in `posterior_predictive` may not share coordinated.

#### `predictions_constant_data`
Model constants used to get the `predictions` samples. Its variables should have a counterpart in `constant_data`. However, variables in `predictions_constant_data` and their counterpart in `constant_data` may not share coordinates.

## Examples
In order to clarify the definitions above, an example of `InferenceData` generation for a 1D linear regression is available in several programming languages and probabilistic programming frameworks. This particular inference task has been chosen because it is widely well known while still being very useful and it also allows to populate all the fields in the `InferenceData` object.
* Python
  - PyMC3
  - [PyStan](schema/PyStan_schema_example.ipynb)
