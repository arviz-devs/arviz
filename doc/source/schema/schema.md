(schema)=
# InferenceData schema specification
The `InferenceData` schema scheme defines a data structure compatible with [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) having 3 goals in mind: usefulness in the analysis of Bayesian inference results, reproducibility of Bayesian inference analysis and interoperability between different inference backends and programming languages.

Currently there are 2 beta implementations of this design:
* [ArviZ](https://arviz-devs.github.io/arviz/) in Python which integrates with:
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [PyMC3](https://docs.pymc.io)
  - [Pyro](https://pyro.ai/) and [NumPyro](https://pyro.ai/numpyro/)
  - [PyStan](https://pystan.readthedocs.io/en/latest/index.html), [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) and [CmdStanPy](https://cmdstanpy.readthedocs.io/en/latest/index.html)
  - [TensorFlow Probability](https://www.tensorflow.org/probability)
* [ArviZ.jl](https://github.com/arviz-devs/ArviZ.jl) in Julia which integrates with:
  - [CmdStan.jl](https://github.com/StanJulia/CmdStan.jl), [StanSample.jl](https://github.com/StanJulia/StanSample.jl) and [Stan.jl](https://github.com/StanJulia/Stan.jl)
  - [Turing.jl](https://turing.ml/dev/) and indirectly any package using [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl) to store results

## Terminology
The terminology used in this specification is based on [xarray's terminology](http://xarray.pydata.org/en/stable/terminology.html), however, no xarray knowledge is assumed in this description. There are also some extensions particular to  the InferenceData case.

* **Variable**: NetCDF-like variables are multidimensional labeled arrays representing a single quantity. Variables and their dimensions must be named. They can also have attributes describing it. Relevant terms related to InferenceData variables are: *variable_name*, *values* (its data), *dimensions*, *coordinates*, and *attributes*.
* **Dimension**: The dimensions of an object are its named axes. A variable containing 3D data can have dimensions `[chain, draw, dim0]`, thus, its `0th`-dimension is `chain`, its `1st`-dimension is `draw` and so on. Every dimension present in an InferenceData variable must share names with a *coordinate*. Given that dimensions must be named, dimension and dimension name are used equivalents.
* **Coordinate**: A named array that labels a dimension. A coordinate named `chain` with values `[0, 1, 2, 3]` would label the `chain` dimension. Coordinate names and values can be loosely though of as labels and tick labels along a dimension.
* **Attributes**: An ordered dictionary that can store arbitrary metadata.
* **Group**: Dataset containing one or several variables with a conceptual link between them. Variables inside a group will generally share some dimensions too. For example, the `posterior` group contains a representation of the posterior distribution conditioned on the observations in the `observed_data` group.
* **Matching samples**: Two variables (or groups) whose samples match are those that have been generated with the same set of samples. Therefore, they will share dimensions and coordinates corresponding to sampling process. Sample dimensions (generally `(chain, draw)`) are the ones introduced by the sampling process.
* **Matching variables**: Two groups with matching variables are groups that conceptually share variables, variable dimensions and coordinates of the variable dimensions but do not necessarily share variable names nor sample dimensions. Variable dimensions are the ones intrinsic to the data and model as opposed to sample dimensions which are the ones relative to the sampling process. When talking about specific variables, this same idea is expressed as one variable being the counterpart of the other.

## Current design
`InferenceData` stores all quantities relevant to fulfilling its goals in different groups. Different groups generally distinguish conceptually different quantities in Bayesian inference, however, convenience in creation and usage of InferenceData objects also plays a role. In general, each quantity (such as posterior distribution or observed data) will be represented by several multidimensional labeled variables.

Each group should have one entry per variable and each variable should be named. When relevant, the first two dimensions of each variable should be the sample identifier (`chain`, `draw`). For groups like `observed_data` or `constant_data` these two initial dimensions are omitted. Dimensions must be named and share name with a coordinate specifying the index values, called coordinate values. Coordinate values can be repeated and should not necessarily be numerical values. Variables must not share names with dimensions.

Moreover, each group contains the following attributes:
* `created_at`: the date of creation of the group.
* `inference_library`: the library used to run the inference.
* `inference_library_version`: version of the inference library used.

`InferenceData` data objects contain any combination the groups described below. There are some relations (detailed below) between the variables and dimensions of different groups. Hence, whenever related groups are present they should comply with this relations.

### `posterior`
Samples from the posterior distribution p(theta|y).

### `sample_stats`
Information and diagnostics for each `posterior` sample, provided by the inference
backend. It may vary depending on the algorithm used by the backend (i.e. an affine
invariant sampler has no energy associated). Therefore none of these parameters
should be assumed to be present in the `sample_stats` group. The convention
below serves to ensure that _if_ a variable is present with one of these names
it will correspond to the definition included here.

The name convention used for `sample_stats` variables is the following:

* `lp`: The joint log posterior density for the model (up to an additive constant).
* `acceptance_rate`: The average acceptance probabilities of all possible samples in the proposed tree.
* `step_size`: The current integration step size.
* `step_size_nom`: The nominal integration step size. The `step_size` may differ from this, for example if the step size is jittered. Should only be present if `step_size` is also present and it varies between samples (i.e. step size is jittered).
* `tree_depth`: The number of tree doublings in the balanced binary tree.
* `n_steps`: The number of leapfrog steps computed. It is related to `tree_depth` with `n_steps <=
  2^tree_dept`.
* `diverging`: (boolean) Indicates the presence of leapfrog transitions with large energy deviation
  from starting and subsequent termination of the trajectory. "large" is defined as `max_energy_error` going over a threshold.
* `energy`: The value of the Hamiltonian energy for the accepted proposal (up to an
additive constant).
* `energy_error`: The difference in the Hamiltonian energy between the initial point and
the accepted proposal.
* `max_energy_error`: The maximum absolute difference in Hamiltonian energy between the initial point and all possible samples in the proposed tree.
* `int_time`: The total integration time (static HMC sampler)


### `log_likelihood`
Pointwise log likelihood data. Samples should match with `posterior` ones and its variables
should match `observed_data` variables. The `observed_data` counterpart variable
may have a different name. Moreover, some cases such as a multivariate normal
may require some dimensions or coordinates to be different.

### `posterior_predictive`
Posterior predictive samples p(y|y) corresponding to the posterior predictive distribution evaluated at the `observed_data`. Samples should match with `posterior` ones and its variables should match `observed_data` variables. The `observed_data` counterpart variable may have a different name.

### `observed_data`
Observed data on which the `posterior` is conditional. It should only contain data which is modeled as a random variable. Each variable should have a counterpart in `posterior_predictive`, however, the `posterior_predictive` counterpart variable may have a different name.

### `constant_data`
Model constants, data included in the model which is not modeled as a random variable. It should be the data used to generate samples in all the groups except the `predictions` groups.

### `prior`
Samples from the prior distribution p(theta). Samples need not match `posterior` samples. However, this group will still follow the convention on `chain` and `draw` as first dimensions. It should have matching variables with the `posterior` group.

### `sample_stats_prior`
Information and diagnostics for the samples in the `prior` group, provided by the inference backend. It may vary depending on the algorithm used by the backend. Variable names follow the same convention defined in `sample_stats`.

### `prior_predictive`
Samples from the prior predictive distribution. Samples should match `prior` samples and each variable should have a counterpart in `posterior_predictive`/`observed_data`.

### `predictions`
Out of sample posterior predictive samples p(y'|y). Samples should match `posterior` samples. Its variables should have a counterpart in `posterior_predictive`. However, variables in `predictions` and their counterpart in `posterior_predictive` can have different coordinate values.

### `predictions_constant_data`
Model constants used to get the `predictions` samples. Its variables should have a counterpart in `constant_data`. However, variables in `predictions_constant_data` and their counterpart in `constant_data` can have different coordinate values.

## Planned features
The `InferenceData` structure is still evolving, with some feature being currently developed. This section aims to describe the roadmap of the specification.

### Sampler parameters
Parameters of the sampling algorithm and sampling backend to be used for analysis reproducibility.

## Examples
In order to clarify the definitions above, an example of `InferenceData` generation for a 1D linear regression is available in several programming languages and probabilistic programming frameworks. This particular inference task has been chosen because it is widely well known while still being useful and it also allows to populate all the fields in the `InferenceData` object.

### Python

```{toctree}
PyMC3 example <PyMC3_schema_example>
PyStan example <PyStan_schema_example>
```
