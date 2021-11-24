# Pull request step-by-step

The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/),
clone it to your local machine, and develop on a feature branch.

(pr_steps)=
## Steps

1. Fork the [project repository](https://github.com/arviz-devs/arviz/) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

(fork_step_pr)=
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

(feature_branch_step_pr)=
4. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   ```{warning}
   Always create a new ``feature`` branch before making any changes. Make your changes
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

   :::{tip}
   Now that the PR is ready to submit, check the {ref}`pr_checklist`.
   :::
