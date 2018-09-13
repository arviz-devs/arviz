# Design for netcdf storage format for mcmc traces

/
All data relating to a mcmc run.

attrs:
backend: "stan" or "pymc3"
version: str
model_name?: str
comment?: str
model_version?: str
timestamp: int
author?: str

/coords
For each dimension, we store a corresponding variable here, that contains the labels for that dimension.

/model
Each backend can store a representation of the model in here. (I guess
source code for stan, no clue for pymc3 at this point.)

/sample_stats
Statistics computed while sampling, like step size (step_size), depth, diverging, energy (for HMC), log probability (lp), and log likelihood (log_likelihood).

/initial_point
    One point in parameter space for each chain, where that chain started.
    TODO We could also store that as an attribute to each var in /trace

/tuning_trace?
    Same as /trace, but optional. It contains the trace during tuning. Optional. (only if store_tune=True or so?)

/tuning_advi?
    Some data about an initialization advi run? History of convergence stats, final sd and mu params? Optional

/divergences
    Points in parameter space where the leapfrog starts that lead to a divergence (excluding tuning).
    Does stan have access to that info? We could also just store the accepted point of a divergent trajectory.

/observed_data
    All data that is used in observed variables (or the data (or transformed?) data sections in stan.)

/warnings
    A list of warnings during sampling. Eg low effective_n, divergences....
    TODO Not sure about the format. Can we somehow share at least part of that between stan/pymc?
    They mostly produce the same warnings I think.

/prior?
Samples from the prior distribution. Same shapes as in trace. (except for (sample, chain))

/prior_predictive?
Samples from the prior predictive distribution. Same vars as in /data

/posterior_predictive?
Samples from the posterior predicitve distribution. Same vars as in /data

/trace
TODO We could call this /posterior

    attrs:
            The final parameters for the sampler. ie the final mass matrix and step size.

    /trace//var1
            One entry for each variable. The first two dimensions should always be
            `(chain, sample)`. I guess the decision whether or not we want to expose a stacked version `draw=('chain', 'sample')`
            is up to arviz.

            Variable names must not share names with coordinate names.

            attrs:
                is_free: Whether or not the variable is a free variable for the sampler, or a transformed one
                domain: One of (“reals”, “pos-reals”, “integers”, “sym-pos-def”, "interval"...)
                TODO For stuff like sym-pos-def we need to know along which dims it is a matrix.
                TODO This data could also be stored in /model
                sym_pos_axis: [dim_idx1, dim_idx2]
                interval_lower:
                interval_upper
                TODO How would this deal with cases where the lower and upper bounds depend on the index?



TODO: In order to reproduce the run, it may make sense to also store some data on the random state (in numpy, this is a tuple of arrays), as
well as some version info.  Hopefully just `PyMC3, (3, 4, 1)` or similar works.
