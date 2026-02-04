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

TODO: Details and sections to be added...


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

If changes are made to a method documented in the {ref}`ArviZ API Guide <api>`
please consider adding inline documentation examples.
You can refer to {func}`~arviz_plots.plot_dist` and {func}`~arviz_plots.plot_forest` for a good examples.

## Tests

:::{caution} Section in construction
:::
