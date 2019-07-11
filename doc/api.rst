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
    plot_dist
    plot_elpd
    plot_energy
    plot_ess
    plot_forest
    plot_hpd
    plot_joint
    plot_kde
    plot_khat
    plot_loo_pit
    plot_mcse
    plot_pair
    plot_parallel
    plot_posterior
    plot_ppc
    plot_rank
    plot_trace
    plot_violin

.. _stats_api:

Stats
-----

.. autosummary::
    :toctree: generated/

    apply_test_function
    compare
    hpd
    loo
    loo_pit
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
    ess
    effective_sample_size
    rhat
    mcse

.. _stats_utils_api:

Stats utils
-----------

.. autosummary::
    :toctree: generated/

    autocov
    autocorr
    make_ufunc
    wrap_xarray_ufunc

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

Utils
-----

.. autosummary::
      :toctree: generated/
      :template: class.rst

      Numba
      interactive_backend
