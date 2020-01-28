.. developer_guide:


===============
Developer Guide
===============


Backends
========
ArviZ now supports multiple backends. If adding another backend please ensure you meeting the
following design patterns.

Code Separation
---------------
Each backend should be placed in a different module per the backend.
See `arviz.plots.backends` for examples

The code in the root level of `arviz.plots` should not contain
any opinion on backend. The idea is that the root level plotting
function perform math and construct keywords, and the backends
code in `arviz.plots.backends` perform the backend specific
keyword argument defaulting and plot behavior

The convenience function `get_plotting_function` is available in
`arviz.plots.get_plotting_function` should be called to obtain
the correct plotting function from the associated backend. If
adding a new backend follow the pattern provided to programatically
call the correct backend

Test Separation
---------------
Tests for each backend should be split into their own module
See `tests.test_plots_matplotlib` for an example

Examples
--------
Examples are not required but encouraged. Examples are
compiled into a Gallery in the arviz documentation website
and serve as readily discoverable sources for users to find functionality
and the associated code
