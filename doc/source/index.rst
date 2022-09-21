:html_theme.sidebar_secondary.remove:

.. _homepage:

Overview
========

ArviZ
-----

Exploratory analysis of Bayesian models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ArviZ is a Python package for exploratory analysis of Bayesian models. It serves as a backend-agnostic tool for diagnosing and visualizing Bayesian inference.

.. raw:: html 

   <div class="homepage-button-container">
      <a href="./getting_started/index.html" class="homepage-button primary-button">Get Started</a>
      <a href="./examples/index.html" class="homepage-button secondary-button">See Gallery</a>
      <a href="./api/index.html" class="homepage-button-link">See API Reference →</a>
   </div>

Example Gallery
---------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: ./plot_trace_bars.html
      :shadow: none
      :class-card: example-gallery

      .. div:: example-img-plot-overlay

         Rank Bars Diagnostic with KDE using `plot_trace`

      .. image:: ./_images/mpl_plot_trace_bars.png
   
   .. grid-item-card::
      :link: ./examples/plot_forest_mixed.html
      :shadow: none
      :class-card: example-gallery

      .. div:: example-img-plot-overlay

         Forest Plot with ESS using `plot_forest`

      .. image:: ./_images/mpl_plot_forest_mixed.png

   .. grid-item-card::
      :link: ./plot_dist.html
      :shadow: none
      :class-card: example-gallery

      .. div:: example-img-plot-overlay

         Dist Plot using `plot_dist`

      .. image:: ./_images/mpl_plot_dist.png

   .. grid-item-card::
      :link: ./plot_density.html
      :shadow: none
      :class-card: example-gallery

      .. div:: example-img-plot-overlay

         Density Plot (Comparison) using `plot_density`

      .. image:: ./_images/mpl_plot_density.png
   
   .. grid-item-card::
      :link: ./plot_pair.html
      :shadow: none
      :class-card: example-gallery

      .. div:: example-img-plot-overlay

         Pair Plot using `plot_pair`

      .. image:: ./_images/mpl_plot_pair.png

   .. grid-item-card::
      :link: ./plot_ppc.html
      :shadow: none
      :class-card: example-gallery

      .. div:: example-img-plot-overlay

         Posterior Predictive Check Plot using `plot_ppc`

      .. image:: ./_images/mpl_plot_ppc.png
   

Key Features
============

The goal is to provide backend-agnostic tools for diagnostics and visualizations of Bayesian inference in Python,
by first converting inference data into `xarray <https://xarray.pydata.org/en/stable/>`_ objects.
See :ref:`here <xarray_for_arviz>` for more on xarray and ArviZ usage
and :ref:`here <schema>` for more on ``InferenceData`` structure
and specification.

A Julia wrapper, `ArviZ.jl <https://julia.arviz.org/>`_ is also available. It provides built-in support for `Turing.jl <https://turing.ml/dev/>`_, `CmdStan.jl <https://github.com/StanJulia/CmdStan.jl>`_, `StanSample.jl <https://github.com/StanJulia/StanSample.jl>`_ and `Stan.jl <https://github.com/StanJulia/Stan.jl>`_.

ArviZ's functions work with NumPy arrays, dictionaries of arrays, xarray datasets, and has built-in support for `PyMC3 <https://docs.pymc.io/>`_, `PyStan <https://pystan.readthedocs.io/en/latest/>`_, `CmdStanPy <https://github.com/stan-dev/cmdstanpy>`_, `Pyro <http://pyro.ai/>`_, `NumPyro <http://num.pyro.ai/>`_, `emcee <https://emcee.readthedocs.io/en/stable/>`_, and `TensorFlow Probability <https://www.tensorflow.org/probability>`_ objects. Support for Edward2 is on the roadmap.

.. grid:: 2 2 3 4
   :gutter: 3

   .. grid-item-card:: Key Feature 1
      :text-align: center
      
      Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 

   .. grid-item-card::
      :text-align: center

      Key Feature 2 (Variant)
      ^^^
      
      Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 

   .. grid-item-card:: Key Feature 3
      :text-align: center
      
      Description of Key Feature

   .. grid-item-card:: Key Feature 4
      :text-align: center
      
      Description of Key Feature

   .. grid-item-card:: Key Feature 5
      :text-align: center
      
      Description of Key Feature

   .. grid-item-card:: Key Feature 6
      :text-align: center
      
      Description of Key Feature

   .. grid-item-card:: Key Feature 7
      :text-align: center

      Description of Key Feature

   .. grid-item-card:: Key Feature 8
      :text-align: center
      
      Description of Key Feature


Support ArviZ
=============

.. raw:: html 

   <div class="two-col">
      <!-- Contributions -->
      <div class="col" id="contributions">
         <h3>Contributions</h3>
         <p>Contributions and issue reports are very welcome at <a href="https://github.com/arviz-devs/arviz">the GitHub repository</a>. We have a <a href="https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md">contributing guide</a> to help you through the process. If you have any doubts, please do not hesitate to contact us on <a href="https://gitter.im/arviz-devs/community">gitter</a>.</p>
      </div>
      <!-- Citation -->
      <div class="col" id="citation">
         <h3>Citation</h3>
         <p>If you use ArviZ and want to <strong>cite</strong> it please use <a class="reference external" href="https://doi.org/10.21105/joss.01143"><img alt="JOSS" src="https://joss.theoj.org/papers/10.21105/joss.01143/status.svg"></a>.</p>
         <p>See our <a href="https://www.arviz.org/en/latest/support.html#cite">support page</a> for information on how to cite in BibTeX format.</p>
      </div>
   </div>

   <div class="two-col">
      <!-- Sponsors -->
      <div class="col" id="sponsors">
         <h3>Sponsors</h3>
         <p>ArviZ receives support from Helsinki University, Finnish Center for Artificial Inteligence, and Chan Zuckerberg Initiative. <a href="https://www.arviz.org/en/latest/sponsors_partners.html">See here</a> for sponsorship details.</p>
         <div class="sponsor-container">
            <a class="sponsor" href="https://www.helsinki.fi/en"><img src="_static/sponsor_university_helsinki.png" alt="University of Helsinki"></a>
            <a class="sponsor" href="https://fcai.fi/"><img src="_static/sponsor_fcai.png" alt="FCAI"></a>
            <a class="sponsor" href="https://chanzuckerberg.com/"><img src="_static/sponsor_czi.png" alt="Chan Zuckerberg Initiative"></a>
         </div>
      </div>
      <!-- Donate -->
      <div class="col" id="donate">
         <h3>Donate</h3>
         <p>ArviZ is a non-profit project under the NumFOCUS umbrella. To support ArviZ financially, click the donate button below or visit the NumFOCUS website.</p>
         <div class="sponsor-container">
            <a class="sponsor" href="https://numfocus.org/"><img src="_static/donate_numfocus.png" alt="NumFOCUS"></a>
            <a href="https://numfocus.org/donate-to-arviz" class="sponsor homepage-button primary-button">Donate</a>
         </div>
      </div>
   </div>



.. toctree::
  :maxdepth: 1
  :hidden:

  getting_started/index
  Example Gallery<examples/index>
  user_guide/index
  api/index
  community
  Contributing<contributing/index>
