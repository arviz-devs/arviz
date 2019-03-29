.. _api:

.. currentmodule:: arviz

API Reference
=============

.. _plot_api:

Plots
-----

.. autosummary::
    :toctree: generated/

    plot_autocorr
    plot_compare
    plot_density
    plot_energy
    plot_forest
    plot_hpd
    plot_joint
    plot_kde
    plot_khat
    plot_pair
    plot_parallel
    plot_posterior
    plot_ppc
    plot_trace

.. _stats_api:

Stats
-----

.. autosummary::
    :toctree: generated/

    compare
    hpd
    loo
    psislw
    r2_score
    summary
    waic

.. _diagnostics_api:

Diagnostics
-----------

.. autosummary::
    :toctree: generated/

    bfmi
    geweke
    effective_sample_size_mean
    effective_sample_size_sd
    effective_sample_size_bulk
    effective_sample_size_tail
    effective_sample_size_quantile
    rhat
    msce_mean
    msce_sd
    msce_quantile

.. _stats_utils_api:

Stats utils
-----------

.. autosummary::
    :toctree: generated/

    autocorr
    make_ufunc

Data
----

.. autosummary::
    :toctree: generated/


    convert_to_inference_data
    load_arviz_data
    to_netcdf
    from_netcdf
    from_cmdstan
    from_dict
    from_emcee
    from_pymc3
    from_pyro
    from_pystan
