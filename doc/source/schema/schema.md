(schema)=
# InferenceData schema specification
The `InferenceData` schema approach defines a data structure compatible with [NetCDF](https://www.unidata.ucar.edu/software/netcdf/). Its purpose is to serve the following three goals:
1. Usefulness in the analysis of Bayesian inference results.
2. Reproducibility of Bayesian inference analysis.
3. Interoperability between different inference backends and programming languages.

Currently there are **two beta implementations** of this design:
* {ref}`ArviZ <homepage>` in **Python** which integrates with:
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [PyMC](https://www.pymc.io)
  - [Pyro](https://pyro.ai/) and [NumPyro](https://pyro.ai/numpyro/)
  - [PyStan](https://pystan.readthedocs.io/en/latest/index.html),
    [CmdStan](https://mc-stan.org/docs/cmdstan-guide/index.html) and
    [CmdStanPy](https://mc-stan.org/cmdstanpy/)
  - [TensorFlow Probability](https://www.tensorflow.org/probability)
* [InferenceObjects.jl](https://github.com/arviz-devs/InferenceObjects.jl) in **Julia** used in [ArviZ.jl](https://github.com/arviz-devs/ArviZ.jl), which integrates with:
  - [CmdStan.jl](https://github.com/StanJulia/CmdStan.jl), [Soss.jl](https://cscherrer.github.io/Soss.jl/stable/), [StanSample.jl](https://github.com/StanJulia/StanSample.jl) and [Stan.jl](https://github.com/StanJulia/Stan.jl)
  - [Turing.jl](https://turing.ml/dev/) and indirectly any package using [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl) to store results

## Terminology
The terminology used in this specification is based on {ref}`xarray's terminology <xarray:terminology>`, however, no xarray knowledge is assumed in this description, nor xarray is needed to use or interact with the schema.
There are also some extensions particular to  the {ref}`InferenceData <xarray_for_arviz>` case.

* **Variable**: NetCDF-like variables are multidimensional labeled arrays representing a single quantity. Variables and their dimensions must be named. They can also have attributes describing it. Relevant terms related to `InferenceData` variables are following:
  - *variable_name*
  - *values* (its data)
  - *dimensions*
  - *coordinates*
  - *attributes*
* **Dimension**: The dimensions of an object are its named axes. A variable containing 3D data can have dimensions `[chain, draw, dim0]`, i.e., its `0th`-dimension is `chain`, its `1st`-dimension is `draw`, and so on. Every dimension present in an `InferenceData` variable must share names with a *coordinate*. Given that dimensions must be named, dimension and dimension name are used equivalents.
* **Coordinate**: A named array that labels a dimension. A coordinate named `chain` with values `[0, 1, 2, 3]` would label the `chain` dimension. Coordinate names and values can be loosely thought of as labels and tick labels along a dimension, respectively.
* **Attribute**: An ordered dictionary that can store arbitrary metadata.
* **Group**: Dataset containing one or several variables with a conceptual link between them. Variables inside a group will generally share some dimensions too. For example, the `posterior` group contains a representation of the posterior distribution conditioned on the observations in the `observed_data` group.
* **Matching samples**: Two variables (or groups) will be called to have matching samples if they are generated with the same set of samples. Therefore, they will share dimensions and coordinates corresponding to the sampling process. Sample dimensions (generally `(chain, draw)`) are the ones introduced by the sampling process.
* **Matching variables**: Two groups with matching variables are groups that conceptually share variables, variable dimensions and coordinates of the variable dimensions but do not necessarily share variable names nor sample dimensions. Variable dimensions are the ones intrinsic to the data and model as opposed to sample dimensions which are the ones relative to the sampling process. When talking about specific variables, this same idea is expressed as one variable being the counterpart of the other.

## Current design
`InferenceData` stores all quantities that are relevant to fulfilling its goals in different groups. Different groups generally distinguish conceptually different quantities in Bayesian inference, however, convenience in {ref}`creation <creating_InferenceData>` and {ref}`usage <working_with_InferenceData>` of `InferenceData` objects also plays a role. In general, each quantity (such as posterior distribution or observed data) will be represented by several multidimensional labeled variables.

### Rules
Below are a few rules that should be followed:
* Each group should have one entry per variable and each variable should be named.
* Dimension names `chain`, `draw`, `sample` and `pred_id` are reserved for
  InferenceData use to indicate sample dimensions.
  - `chain` indicates the MCMC chain
  - `draw` indicates the iteration _within_ each MCMC chain.
    ArviZ assumes all chains have the same length for better interoperability with
    NumPy and xarray.
  - `sample` indicates a unique id per value combining chain and draw. i.e. we often don't
    care about `chain` and `draw` when plotting and only want all the samples of the distribution
    as a whole.
  - `pred_id` is interpreted as the dimension storing multiple independent and identically
    distributed values per sample.
* Dimensions in InferenceData (including sample dimensions) should be identified by name only. The
  dimension order does not matter, only their names.
* For groups like `observed_data` or `constant_data`, all sample dimensions can be
  omitted. For groups like `prior`, `posterior` or `posterior_predictive` either `sample` has to be
  present or both `chain` and `draw` dimensions need to be present. Any combinations that follow
  this are valid.
* Dimensions must be named and share name with a coordinate specifying the index values, called coordinate values.
* Coordinate values can be repeated and should not necessarily be numerical values.
* Variables must not share names with dimensions.
* Groups, variables or the InferenceData itself can have arbitrary metadata stored.

### Metadata
No metadata is _required_ to be present in order to be compliant with the InferenceData schema.
However, it is recommended to store the following fields when relevant:
* `name`: InferenceData objects represent multiple quantities related to Bayesian modelling,
  but they are all tied to a single model. The model identifier can be added as metadata
  to simplify the calls to model comparison functions.
* `created_at`: the date of creation of the group.
* `creation_library`: the library used to create the InferenceData (might not necessarly be ArviZ)
* `creation_library_version`: the version of `creation_library` that generated the InferenceData
* `creation_library_language`: the programming language from which `creation_library` was used to create the InferenceData
* `inference_library`: the library used to run the inference.
* `inference_library_version`: version of the inference library used.

Metadata can be stored at the whole `InferenceData` level but also at group level when needed.


### Relations between groups
`InferenceData` data objects contain any combination of the groups described below. There are also some relations (detailed below) between the variables and dimensions of different groups. Hence, whenever related groups are present they should comply with these relations. Neither the presence of groups not described below or the lack of some of the groups described below go against the schema.

#### `posterior`
Samples from the posterior distribution $p(\theta|y)$ in the parameter (also called constrained) space.

(schema/unconstrained_posterior)=
#### `unconstrained_posterior`
Samples from the posterior distribution p(theta_transformed|y) in the unconstrained (also called transformed) space.

Only variables that undergo a transformation for sampling should be present here.
Therefore, to get the samples for _all_ the variables in the unconstrained space,
variables should be taken from the `unconstrained_posterior` group if present,
and if not, then the values from the variable in the `posterior` group should be used.

Samples should match between the `posterior` and the `unconstrained_posterior` groups.
All variables in `unconstrained_posterior` should have a counterpart in `posterior`
with the same name. However, they don't need to have the same dimensions nor shape.

:::{note}
:class: dropdown

Both InferenceData groups and variables can have metadata, which in the `unconstrained_posterior`
case could be used to store the transformations each variable goes through to map between the
constrained and unconstrained spaces. The schema leaves this completely up to the user
and imposes no conventions or restrictions on such metadata.
:::

(schema/sample_stats)=
#### `sample_stats`
Information and diagnostics for each `posterior` sample, provided by the inference
backend. It may vary depending on the algorithm used by the backend (i.e. an affine
invariant sampler has no energy associated). Therefore none of these parameters
should be assumed to be present in the `sample_stats` group. The convention
below serves to ensure that if a variable is present with one of these names
it will correspond to the definition given in front of it.
Moreover, some `sample_stats` may be constant throughout the sampling
process; these variables don't need to have any sampling dimensions.


:::{dropdown} Naming convention used for `sample_stats` variables
:icon: list-unordered

* `lp`: The joint log posterior density for the model (up to an additive constant).
* `acceptance_rate`: The average acceptance probabilities of all possible samples in the proposed tree.
* `step_size`: The current integration step size.
* `step_size_nom`: The nominal integration step size. The `step_size` may differ from this for example, if the step size is jittered. It should only be present if `step_size` is also present and it varies between samples (i.e. step size is jittered).
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
* `inv_metric`: Inverse metric (also known as inverse _mass matrix_) used in HMC samplers for the computation of the Hamiltonian.
  When it is constant, the resulting implementation is known as Euclidean HMC;
  in that case, the variable wouldn't need to have any sampling dimensions
  even if part of the `sample_stats` group.
:::


#### `log_likelihood`
Pointwise log likelihood data. Samples should match with `posterior` ones and its variables
should match `observed_data` variables. The `observed_data` counterpart variable
may have a different name. Moreover, some cases such as a multivariate normal
may require some dimensions or coordinates to be different.

#### `posterior_predictive`
Posterior predictive samples p(y|y) corresponding to the posterior predictive distribution evaluated at the `observed_data`. Samples should match with `posterior` ones and its variables should match `observed_data` variables. The `observed_data` counterpart variable may have a different name.

#### `observed_data`
Observed data on which the `posterior` is conditional. It should only contain data which is modeled as a random variable. Each variable should have a counterpart in `posterior_predictive`, however, the `posterior_predictive` counterpart variable may have a different name.

#### `constant_data`
Model constants, data included in the model which is not modeled as a random variable. It should be the data used to generate samples in all the groups except the `predictions` groups.

#### `prior`
Samples from the prior distribution p(theta). Samples do not need to match `posterior` samples. However, this group will still follow the convention on `chain` and `draw` as first dimensions. It should have matching variables with the `posterior` group.

#### `prior_predictive`
Samples from the prior predictive distribution. Samples should match `prior` samples and each variable should have a counterpart in `posterior_predictive`/`observed_data`.

#### `predictions`
Out of sample posterior predictive samples p(y'|y). Samples should match `posterior` samples. Its variables should have a counterpart in `posterior_predictive`. However, variables in `predictions` and their counterpart in `posterior_predictive` can have different coordinate values.

#### `predictions_constant_data`
Model constants used to get the `predictions` samples. Its variables should have a counterpart in `constant_data`. However, variables in `predictions_constant_data` and their counterpart in `constant_data` can have different coordinate values.

:::{admonition} Note on sample stats, warmup and unconstrained groups
:class: note, dropdown

The schema does not define which warmup or unconstrained groups exist or can exist
by default. We recognize both the samplers and the models are continuously evolving.
Some models already require the use of sampling algorithms to get prior samples,
in which case we basically need to treat the prior and posterior groups in the same way.

We define the prefixes to allow libraries that use InferenceData to be aware
of the potential relations and hopefully support as many cases as possible.
Back to the case above, it might be necessary to generate a pair plot
for prior samples generated with NUTS _and_ its associated divergences,
which would then come from `sample_stats_prior`.
:::

#### Sample stats groups
Information and diagnostics for the samples in any InferenceData group
other than the posterior should be stored in a separate group with the
`sample_stats_` prefix. For example `sample_stats_prior`.

The same rules and conventions defined in {ref}`schema/sample_stats` apply to
any sample stats group.

#### Warmup groups
Samples generated during the adaptation/warmup phases of algorithms like HMC
can also be stored in InferenceData. In such cases, the data/samples
generated during the adaptation process should be stored in groups with
the same name with the `warmup_` prefix, e.g. `warmup_posterior`, `warmup_sample_stats_prior`.
The `warmup_` prefix goes before other prefixes.

#### Unconstrained groups
Samples on the unconstrained space in cases where the samples need to be generated with
the help of a sampling algorithm and the sampling algorithm requires transformations
to an unconstrained space.

It is described in more detail in {ref}`schema/unconstrained_posterior` section, which
is what we expect to be the most common section, but other groups could also have
an unconstrained linked group, e.g. `prior` and `unconstrained_prior`.

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
