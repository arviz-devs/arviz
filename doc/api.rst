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
    plot_rank
    plot_trace

.. _stats_api:

Stats
-----

.. autosummary::
    :toctree: generated/

    bfmi
    compare
    hpd
    loo
    r2_score
    summary
    waic
    psislw

.. _diagnostics_api:

Diagnostics
-----------

.. autosummary::
    :toctree: generated/

    effective_sample_size
    rhat
    geweke
    autocorr

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
