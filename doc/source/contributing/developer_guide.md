(developer_guide)=

# Developer Guide
## Library architecture
ArviZ is organized in modules (the folders in [arviz directory](https://github.com/arviz-devs/arviz/tree/main/arviz)).
The main 3 modules are `data`, `plots` and `stats`.
Then we have 3 more folders. The [tests](https://github.com/arviz-devs/arviz/tree/main/arviz/tests) folder contains tests for all these 3 modules.

The [static](https://github.com/arviz-devs/arviz/tree/main/arviz/static) folder is only used to store style and CSS files to get HTML output for `InferenceData`. Finally we have the [wrappers](https://github.com/arviz-devs/arviz/tree/main/arviz/wrappers) folder that contains experimental (not tested yet either) features and interacts closely with both [data](https://github.com/arviz-devs/arviz/tree/main/arviz/data) and [stats](https://github.com/arviz-devs/arviz/tree/main/arviz/stats) modules.

In addition, there are some files on the higher level directory: `utils.py`, `sel_utils.py`,
`rcparams.py` and `labels.py`.

## Plots
ArviZ supports multiple backends. While adding another backend, please ensure you meet the
following design patterns.

### Code Separation
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

### Test Separation
Tests for each backend should be split into their own module
See [tests.test_plots_matplotlib](https://github.com/arviz-devs/arviz/blob/main/arviz/tests/base_tests/test_plots_matplotlib.py) for an example.

### Gallery Examples
Gallery examples are not required but encouraged. Examples are
compiled into the ArviZ documentation website. The [examples](https://github.com/arviz-devs/arviz/tree/main/examples) directory
can be found in the root of the ArviZ git repository.
