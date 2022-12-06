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
