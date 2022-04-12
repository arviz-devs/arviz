# Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are
inclined to do so to submit patches for new or existing issues via pull
requests. This is particularly the case for simple fixes, such as typos
or tweaks to documentation, which do not require a heavy investment
of time and attention.

Contributors are also encouraged to contribute new code to enhance ArviZ's
functionality, also via pull requests.
Please consult the documentation (on this same {ref}`website <homepage>`)
to ensure that any new contribution does not strongly overlap with existing
functionality and open a "Feature Request" issue before starting to work
on the new feature.

## Steps before starting work
Before starting work on a pull request double-check that no one else
is working on the ticket in both issue tickets and pull requests.

ArviZ is a community-driven project and always has multiple people working
on it simultaneously. These guidelines define a set of rules to ensure
that we all make the most of our time and we don't have two contributors
working on the same changes. Let's see what to do when you encounter the following scenarios:

### If an issue ticket exists
If an issue exists check the ticket to ensure no one else has started working on it. If you are first to start work, comment on the ticket to make it evident to others. If the comment looks old or abandoned leave a comment asking if you may start work.

### If an issue ticket doesn't exist
Open an issue ticket for the issue and state that you'll be solving the issue with a pull request. Optionally create a pull request and add `[WIP]` in the title to indicate Work in Progress.

### In the event of a conflict
In the event of two or more people working on the same issue, the general precedence will go to the person who first commented on the issue. If no comments it will go to the first person to submit a PR for review. Each situation will differ though, and the core contributors will make the best judgment call if needed.

### If the issue ticket has someone assigned to it
If the issue is assigned then precedence goes to the assignee. However, if there has been no activity for 2 weeks from the assignment date, the ticket is open for all again and can be unassigned.

(dev_summary)=
## Development process - summary

## Code Formatting
For code generally follow the
[TensorFlow's style guide](https://www.tensorflow.org/community/contribute/code_style)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
Both more or less follow PEP 8.

Final formatting is done with [black](https://github.com/ambv/black).

(docstring_formatting)=
# Docstring formatting and type hints

Docstrings should follow the
[numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).
Extra guidance can also be found in
[pandas docstring guide](https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html).
Please reasonably document any additions or changes to the codebase,
when in doubt, add a docstring.

The different formatting and aim between numpydoc style type description and
[type hints](https://docs.python.org/3/library/typing.html)
should be noted. numpydoc style targets docstrings and aims to be human
readable whereas type hints target function definitions and `.pyi` files and
aim to help third party tools such as type checkers or IDEs. ArviZ does not
require functions to include type hints
however contributions including them are welcome.

## Documentation for user facing methods
If changes are made to a method documented in the {ref}`ArviZ API Guide <api>`
please consider adding inline documentation examples.
You can refer to {func}`az.plot_posterior <arviz.plot_posterior>` for a good example.

### Tests
Section in construction
