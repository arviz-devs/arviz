# Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are
inclined to do so to submit patches for new or existing issues via pull requests.
This is particularly the case for simple fixes, such as typos or tweaks to documentation,
which do not require a heavy investment of time and attention.

Contributors are also encouraged to contribute new code to enhance ArviZ's
functionality, also via pull requests.
Please consult the documentation (on this same {ref}`website <homepage>`)
to ensure that any new contribution does not strongly overlap with existing
functionality and open a "Feature Request" issue before starting to work
on the new feature.

:::{note}
Since version 1.0, ArviZ is organized as a metapackage that brings together functionality
from three packages: [`arviz-base`](https://github.com/arviz-devs/arviz-base),
[`arviz-stats`](https://github.com/arviz-devs/arviz-stats), and
[`arviz-plots`](https://github.com/arviz-devs/arviz-plots).

As a result, contributing to ArviZ typically means contributing to one of these packages rather
than to the main ArviZ repository itself.
When browsing issues or pull requests, keep in mind that most activity happens in the
corresponding package repositories.
:::

(steps_before_working)=
## Steps before starting work
Before starting work on a pull request, double-check that no one else
is working on the ticket in both issue tickets and pull requests.

ArviZ is a community-driven project and always has multiple people working
on it simultaneously. These guidelines define a set of rules to ensure
that we all make the most of our time and we don't have two contributors
working on the same changes. Let's see what to do when you encounter the following scenarios:

### If an issue ticket exists
If an issue exists, check the ticket to ensure no one else has started working on it.
If you are first to start work, comment on the ticket to make it evident to others.
If the comment looks old or abandoned, leave a comment asking if you may start work.

### If an issue ticket doesn't exist
Open an issue ticket for the issue and state that you'll be solving the issue with a pull request.
Optionally create a pull request and add `[WIP]` in the title to indicate Work in Progress.

### In the event of a conflict
In the event of two or more people working on the same issue,
the general precedence will go to the person who first commented on the issue.
If there are no comments, it will go to the first person to submit a PR for review.
Each situation will differ though, and the core contributors will make the best judgment call if needed.

### If the issue ticket has someone assigned to it
If the issue is assigned, then precedence goes to the assignee.
However, if there has been no activity for 2 weeks from the assignment date,
the ticket is open for all again and can be unassigned.

(dev_summary)=
## Development process

ArviZ follows a test-driven and quality-first development workflow across all repositories in the ArviZ-verse.
The typical development loop involves the following steps:

### 1. Set up an isolated environment
Always develop within a dedicated virtual environment (via `venv`, `conda`, or `mamba`) with Python 3.12 or newer.
Install the package in editable mode with development dependencies:

```bash
pip install -e ".[test,check]"
```

For repositories with optional dependencies (such as plotting backends in `arviz-plots` or I/O formats in `arviz-base`),
include the relevant extras you are working with (e.g., `pip install -e ".[test,check,matplotlib,bokeh,plotly]"`).

### 2. Configure linters and code formatting
ArviZ enforces code formatting and style checks using [pre-commit](https://pre-commit.com/) and [Ruff](https://docs.astral.sh/ruff/).
Configure the git hooks to run automatically before every commit:

```bash
pre-commit install
```

You can manually trigger style checks across all files at any time:

```bash
pre-commit run --all-files
```

Alternatively, run the automated checks via `tox`:

```bash
tox -e check
```

### 3. Write tests alongside your code
- **Bug fixes**: Always write a minimal reproducing test case first that fails before your fix and passes after.
- **New features / API changes**: Add unit tests verifying both standard usage and edge cases (such as improper inputs, dimension mismatches, or missing optional dependencies).
- Place tests in the `tests/` directory following the pytest convention, utilizing existing fixtures where applicable.

Execute tests using `pytest` directly or via `tox`:

```bash
pytest
```

### 4. Build and preview documentation
Whenever you modify user-facing functions, add new guides, or update docstrings, build the documentation locally to ensure there are no broken links, syntax errors, or unrendered cross-references:

```bash
tox -e docs
```

You can also use `tox -e livedocs` for a live-reloading preview in your browser while editing documentation files.

### 5. Verify the package build
Ensure the distribution build succeeds without errors:

```bash
python -m build
```


(docstring_formatting)=
## Docstring formatting

Docstrings should follow the [numpy docstring guide](https://numpydoc.readthedocs.io/).
Extra guidance can also be found in
[pandas docstring guide](https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html).
Please reasonably document any additions or changes to the codebase; when in doubt, add a docstring.

While ArviZ does not include type hints in function and method definitions,
it uses [Docstub](https://docstub.readthedocs.io) to generate `.pyi` files
from numpydoc-style docstrings.
For this reason, clear and complete docstrings are important,
as they are used to generate type information for third-party tools such as type checkers and IDEs.


## Documentation for user facing methods

If changes are made to a method documented in {ref}`ArviZ-base API <arviz_base:api_reference>`,
{ref}`ArviZ-stats API <arviz_stats:api_reference>`, or {ref}`ArviZ-plots API <arviz_plots:api_reference>`;
please consider adding inline examples, See Also links,
References when relevant (e.g., if it implements a published method),
and Notes with implementation details that might be relevant to users.
You can refer to {func}`~arviz_plots.plot_dist` and {func}`~arviz_plots.plot_forest` for good examples.

## Tests

Tests are run through dedicated `tox` tasks and `pytest`. Due to differences in architecture,
combinations of optional dependencies, and scope between libraries, each subpackage defines its own testing tasks:

* **ArviZ Metapackage (`arviz-devs/arviz`)**: Tests verify namespace consistency, version synchronization across installed ArviZ subpackages, backward-compatibility warnings (such as {class}`~arviz.MigrationWarning` when accessing legacy attributes like `InferenceData`), and alias bindings. Run with:
  ```bash
  tox -e test-namespace
  # or directly:
  pytest
  ```
* **ArviZ-base**: Tests data containers (based on `xarray.DataTree`), converters from PPL outputs (such as PyMC, CmdStanPy, NumPyro), and I/O backends (NetCDF, Zarr). See the [ArviZ-base documentation](https://python.arviz.org/projects/base/en/latest/).
* **ArviZ-stats**: Tests mathematical diagnostics, convergence metrics, and posterior summary statistics against analytical results and reference datasets. See {ref}`ArviZ-stats testing guide <arviz_stats:contributing_testing>`.
* **ArviZ-plots**: Tests plotting logic across Matplotlib, Bokeh, and Plotly backends. See {ref}`ArviZ-plots testing guide <arviz_plots:contributing_testing>`.
