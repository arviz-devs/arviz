# Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are
inclined to do so to submit patches for new or existing issues via pull
requests. This is particularly the case for simple fixes, such as typos
or tweaks to documentation, which do not require a heavy investment
of time and attention.

Contributors are also encouraged to contribute new code to enhance ArviZ's
functionality, also via pull requests.
Please consult the [ArviZ documentation](https://arviz-devs.github.io/arviz/)
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
# Development process - summary
The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/),
clone it to your local machine, and develop on a feature branch. the details of this process are listed on
{ref}`pr_checklist`. For a detailed
description of the recommended development process, see {ref}`developer_guide`.

## Code Formatting
For code generally follow the
[TensorFlow's style guide](https://www.tensorflow.org/community/contribute/code_style)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
Both more or less follow PEP 8.

Final formatting is done with [black](https://github.com/ambv/black).


## Docstring formatting and type hints
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

## Tests
Section in construction

# Steps

1. Fork the [project repository](https://github.com/arviz-devs/arviz/) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the ArviZ repo from your GitHub account to your local disk.

   ::::{tab-set}

   :::{tab-item} SSH
   :sync: ssh

   ```
   $ git clone git@github.com:<your GitHub handle>/arviz.git
   ```
   :::

   :::{tab-item} HTTPS
   :sync: https

   ```
   $ git clone https://github.com/<your GitHub handle>/arviz.git
   ```
   :::

   ::::

3. Navigate to your arviz directory and add the base repository as a remote:

   ::::{tab-set}

   :::{tab-item} SSH
   :sync: ssh

   ```
   $ cd arviz
   $ git remote add upstream git@github.com:arviz-devs/arviz.git
   ```
   :::

   :::{tab-item} HTTPS
   :sync: https

   ```
   $ cd arviz
   $ git remote add upstream https://github.com/arviz-devs/arviz
   ```
   :::

   ::::

4. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   ```{warning}
   Always create a new ``feature`` branch before making any changes. Make your chnages
   in the ``feature`` branch. It's good practice to never routinely work on the ``main`` branch of any repository.
   ```

5. Project requirements are in ``requirements.txt``, and libraries used for development are in ``requirements-dev.txt``.  To set up a development environment, you may (probably in a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/)) run:

   ```bash
   $ pip install -r requirements.txt
   $ pip install -r requirements-dev.txt
   $ pip install -r requirements-docs.txt  # to generate docs locally
   ```

   Alternatively, for developing the project in [Docker](https://docs.docker.com/), there is a script to setup the Docker environment for development. See {ref}`developing_in_docker`.

6. Develop the feature on your feature branch. Add your changes using git commands, ``git add`` and then ``git commit``, like:

   ```bash
   $ git add modified_files
   $ git commit -m "commit message here"
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

7. Go to the GitHub web page of your fork of the ArviZ repo. Click the 'Pull request' button to send your changes to the project's maintainers for review. This will send an email to the committers.

(pr_checklist)=
# Pull request checklist

We recommend that your contribution complies with the following guidelines before you submit a pull request:

* If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

* All public methods must have informative docstrings with sample usage when appropriate.

* Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request a broad review of functionality or API, or (3) seek collaborators.

* All other tests pass when everything is rebuilt from scratch.
See {ref}`developing_in_docker` for information on running the test suite locally.

* When adding additional plotting functionality, provide at least one example script in the [arviz/examples/](https://github.com/arviz-devs/arviz/tree/main/examples) folder. Have a look at other examples for reference. Examples should demonstrate why the new functionality is useful in practice and, if possible, compare it to other methods available in ArviZ.

* Added tests follow the [pytest fixture pattern](https://docs.pytest.org/en/latest/fixture.html#fixture).

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

* Documentation follows Numpy style guide.

* Run any of the pre-existing examples in ``docs/source/notebooks`` that contain analyses that would be affected by your changes to ensure that nothing breaks. This is a useful opportunity to not only check your work for bugs that might not be revealed by unit test, but also to show how your contribution improves ArviZ for end users.

* If modifying a plot, render your plot to inspect for changes and copy image in the pull request message on Github.

You can also check for common programming errors with the following
tools:

* Save plots as part of tests. Plots will be saved to a directory named `test_images` by default.

  ```bash
  $ pytest arviz/tests/base_tests/<name of test>.py --save
  ```

* Optionally save plots to a user named directory. This is useful for comparing changes across branches.

  ```bash
  $ pytest arviz/tests/base_tests/<name of test>.py --save user_defined_directory
  ```


* Code coverage **cannot** decrease. Coverage can be checked with **pytest-cov** package:

  ```bash
  $ pip install pytest pytest-cov coverage
  $ pytest --cov=arviz --cov-report=html arviz/tests/
  ```

* Your code has been formatted with [black](https://github.com/ambv/black) with a line length of 100 characters.

  ```bash
  $ pip install black
  $ black arviz/ examples/ asv_benchmarks/
  ```

* Your code passes pylint

  ```bash
  $ pip install pylint
  $ pylint arviz/
  ```

* No code style warnings, check with:

  ```bash
  $ ./scripts/lint.sh
  ```

# Running the benchmark tests

To run the **benchmark tests** do the following:

    $ pip install asv
    $ cd arviz
    $ asv run

**This guide was derived from the [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md)**.
