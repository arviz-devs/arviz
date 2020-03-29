# Guidelines for Contributing

As a scientific community-driven software project, ArviZ welcomes contributions from interested individuals or groups. These guidelines are provided to give potential contributors information to make their contribution compliant with the conventions of the ArviZ project, and maximize the probability of such contributions to be merged as quickly and efficiently as possible.

There are 4 main ways of contributing to the ArviZ project (in descending order of difficulty or scope):

* Adding new or improved functionality to the existing codebase
* Fixing outstanding issues (bugs) with the existing codebase. They range from low-level software bugs to higher-level design problems.
* Contributing or improving the documentation (`docs`) or examples (`arviz/examples`)
* Submitting issues related to bugs or desired enhancements

# Opening issues

We appreciate being notified of problems with the existing ArviZ code. We prefer that issues be filed the on [Github Issue Tracker](https://github.com/arviz-devs/arviz/issues), rather than on social media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other issues or pull requests by using the GitHub search tool to look for key words in the project issue tracker.

# Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are
inclined to do so to submit patches for new or existing issues via pull
requests. This is particularly the case for simple fixes, such as typos
or tweaks to documentation, which do not require a heavy investment
of time and attention.

Contributors are also encouraged to contribute new code to enhance ArviZ's
functionality, also via pull requests.
Please consult the [ArviZ documentation](https://arviz-devs.github.io/arviz/)
to ensure that any new contribution does not strongly overlap with existing functionality.

## Steps before starting work
Before starting a work on a pull request double check that no one else is working on the ticket in both issue tickets and pull requests.

### If an issue ticket exists
If an issue exists check the ticket to ensure no one else has started work. If first to start work, comment on the ticket to make it evident to others. If the comment looks old or abandoned leave a comment asking if you may start work.

### If an issue ticket doesn't exist
Open an issue ticket for the issue and state that you'll be solving the issue with a pull request. Optionally create a pull request and add `[WIP]` in the title to indicate Work in Progress.

### In the event of a conflict
In the event of two or more people working on the same issue, the general precedence will go to the person who first commented in the issue. If no comments it will go to the first person to submit a PR for review. Each situation will differ though, and the core contributors will make the best judgement call if needed.

## Making the pull request
The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/), clone it to your local machine, and develop on a feature branch.

For more instructions see the
[Pull request checklist](#pull-request-checklist)


### Code Formatting
For code generally follow the
[TensorFlow's style guide](https://www.tensorflow.org/community/contribute/code_style)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
Both more or less follows PEP 8.

Final formatting is done with [black](https://github.com/ambv/black).
For more detailed steps on a typical development workflow see the
[Pull request checklist](#pull-request-checklist)


### Docstring formatting and type hints
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

#### Documentation for user facing methods
If changes are made to a method documented in the
[ArviZ API Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
please consider adding inline documentation examples.
`az.plot_posterior` is a particularly
[good example](https://arviz-devs.github.io/arviz/generated/arviz.plot_posterior.html#arviz.plot_posterior).


## Steps

1. Fork the [project repository](https://github.com/arviz-devs/arviz/) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the ArviZ repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/arviz.git
   $ cd arviz
   $ git remote add upstream git@github.com:arviz-devs/arviz.git
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never routinely work on the ``master`` branch of any repository.

4. Project requirements are in ``requirements.txt``, and libraries used for development are in ``requirements-dev.txt``.  To set up a development environment, you may (probably in a [virtual environment](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)) run:

   ```bash
   $ pip install -r requirements.txt
   $ pip install -r requirements-dev.txt
   ```

Alternatively, there is a script to create a docker environment for development.  See: [Developing in Docker](#Developing-in-Docker).

Note: Building the documentation locally requires saving Bokeh plots as
images. To do this, bokeh needs some [extra dependencies](https://docs.bokeh.org/en/latest/docs/user_guide/export.html#exporting-plots)
that can not be installed with pip and are therefore not in
``requirements-dev.txt``. To build documentation locally without having to
install these extra dependencies, Docker can be used. See: [Building documentation with Docker](#building-documentation-with-docker)

5. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit -m "commit message here"
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the ArviZ repo. Click the 'Pull request' button to send your changes to the project's maintainers for review. This will send an email to the committers.

## Pull request checklist

We recommend that your contribution complies with the following guidelines before you submit a pull request:

* If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

* All public methods must have informative docstrings with sample usage when appropriate.

* Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request broad review of functionality or API, or (3) seek collaborators.

* All other tests pass when everything is rebuilt from scratch.  See
[Developing in Docker](#Developing-in-Docker) for information on running the test suite locally.

* When adding additional functionality, provide at least one example script or Jupyter Notebook in the ``arviz/examples/`` folder. Have a look at other examples for reference. Examples should demonstrate why the new functionality is useful in practice and, if possible, compare it to other methods available in ArviZ.

* Added tests follow the [pytest fixture pattern](https://docs.pytest.org/en/latest/fixture.html#fixture)

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

* Documentation follows Numpy style guide

* Run any of the pre-existing examples in ``docs/source/notebooks`` that contain analyses that would be affected by your changes to ensure that nothing breaks. This is a useful opportunity to not only check your work for bugs that might not be revealed by unit test, but also to show how your contribution improves ArviZ for end users.

* If modifying a plot, render your plot to inspect for changes and copy image in the pull request message on Github

You can also check for common programming errors with the following
tools:

* Save plots as part of tests. Plots will save to a directory named test_images by default

  ```bash
  $ pytest arviz/tests/base_tests/<name of test>.py --save
  ```

* Optionally save plots to a user named directory. This is useful for comparing changes across branches

  ```bash
  $ pytest arviz/tests/base_tests/<name of test>.py --save user_defined_directory
  ```


* Code coverage **cannot** decrease. Coverage can be checked with **pytest-cov** package:

  ```bash
  $ pip install pytest pytest-cov coverage
  $ pytest --cov=arviz --cov-report=html arviz/tests/
  ```

* Your code has been formatted with [black](https://github.com/ambv/black) with a line length of 100 characters. Note that black only runs in Python 3.6+

  ```bash
  $ pip install black
  $ black arviz/ examples/
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

## Developing in Docker

We have provided a Dockerfile which helps for isolating build problems, and local development.
Install [Docker](https://www.docker.com/) for your operating system, clone this repo. Docker will generate an environment with your local copy of `arviz` with all the packages in Dockerfile.

### Testing in Docker
Testing the code using docker consists of executing the same file 3 times (you may need root privileges to run it).
First run `./scripts/container.sh --clear-cache`. Then run `./scripts/container.sh --build`. This starts a local docker image called `arviz`. Finally run the tests with `./scripts/container.sh --test`. This should be quite close to how the tests run on TravisCI.

NOTE: If you run into errors due to `__pycache__` files (i.e. while testing in
docker after testing locally or installing with pip after testing with
docker), try running `./scripts/container.sh --clear-cache` before the errored
command.

### Using the Docker image interactively
Once the Docker image is built with `./scripts/container.sh --build`, interactive containers can also be run. Therefore, code can be edited and executed using the docker container, but modifying directly the working directory of the host machine.

To start a bash shell inside Docker, run:

    $ docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ -it arviz bash

Alternatively, to start a jupyter notebook, there are two steps, first run:

    $ docker run --mount type=bind,source="$(pwd)",target=/opt/arviz/ --name jupyter-dock -it -d -p 8888:8888 arviz
    $ docker exec jupyter-dock bash -c "pip install jupyter"
    $ docker exec -it jupyter-dock bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"

This will output something similar to `http://(<docker container id> or <ip>):8888/?token=<token id>`, and can be accessed at `http://localhost:8888/?token=<token id>`.

### Building documentation with Docker
The documentation can be build with Docker by running `./scripts/container.sh
--sphinx-build`. The docker image contains by default all dependencies needed
for building the documentation. After having build the docs in the Docker
container, they can be checked at `doc/build`.

## Running the benchmark tests

To run the **benchmark tests** do the following:

    $ pip install asv
    $ cd arviz
    $ asv run

#### This guide was derived from the [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md)
