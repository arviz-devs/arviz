# Plotting backends
ArviZ supports multiple backends. While adding another backend, please ensure you meet the
following design patterns.

## Code Separation
Each backend should be placed in a different module per the backend.
See `arviz.plots.backends` for examples.

The code in the root level of `arviz.plots` should not contain
any opinion on backend. The idea is that the root level plotting
function performs math and constructs keywords, and the backends
code in `arviz.plots.backends` perform the backend specific
keyword argument defaulting and plot behavior.

The convenience function `get_plotting_function` available in
`arviz.plots.get_plotting_function` should be called to obtain
the correct plotting function from the associated backend. If
adding a new backend follow the pattern provided to programmatically
call the correct backend.

## Test Separation
Tests for each backend should be split into their own module
See [tests.test_plots_matplotlib](https://github.com/arviz-devs/arviz/blob/main/arviz/tests/base_tests/test_plots_matplotlib.py) for an example.

## Gallery Examples
Gallery examples are not required but encouraged. Examples are
compiled into the ArviZ documentation website. The [examples](https://github.com/arviz-devs/arviz/tree/main/examples) directory
can be found in the root of the ArviZ git repository.
