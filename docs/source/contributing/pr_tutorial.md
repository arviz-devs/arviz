(pr_tutorial)=
# Pull request step-by-step

The preferred workflow for contributing to ArviZ is to fork
the [GitHub repository](https://github.com/arviz-devs/arviz/),
clone it to your local machine, and develop on a feature branch.

This tutorial applies to any of these repositories:

* [arviz-devs/arviz](https://github.com/arviz-devs/arviz)
* [arviz-devs/arviz-base](https://github.com/arviz-devs/arviz-base)
* [arviz-devs/arviz-stats](https://github.com/arviz-devs/arviz-stats)
* [arviz-devs/arviz-plots](https://github.com/arviz-devs/arviz-plots)


(pr_steps)=
## Steps

:::{admonition} Have you already gone through this tutorial?
:class: dropdown tip

If this is not your first time going over this tutorial and you want to
work on your 2nd+ PR you can skip steps 1-3, and instead do:

```bash
git checkout main
git fetch upstream
git rebase upstream/main
```

This will make sure you are on the `main` branch of your local repository and sync it
with the one on GitHub. After this you can continue with {ref}`step 4 <feature_branch_step_pr>`
to create your feature branch from the up-to-date `main`
:::

1. Fork the project repository by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

(fork_step_pr)=
2. Clone your fork of the target ArviZ repository from your GitHub account to your local disk.

   ::::{tab-set}

   :::{tab-item} SSH
   :sync: ssh

   ```
   git clone git@github.com:<your GitHub handle>/<arviz repo>.git
   ```
   :::

   :::{tab-item} HTTPS
   :sync: https

   ```
   git clone https://github.com/<your GitHub handle>/<arviz repo>.git
   ```
   :::

   ::::

3. Navigate to your arviz directory and add the base repository as a remote:

   ::::{tab-set}

   :::{tab-item} SSH
   :sync: ssh

   ```
   cd <arviz repo>
   git remote add upstream git@github.com:arviz-devs/<arviz repo>.git
   ```
   :::

   :::{tab-item} HTTPS
   :sync: https

   ```
   cd <arviz repo>
   git remote add upstream https://github.com/arviz-devs/<arviz repo>.git
   ```
   :::

   ::::

(feature_branch_step_pr)=
4. Create a ``feature`` branch to hold your development changes:

   ```bash
   git checkout -b my-feature
   ```

   ```{warning}
   Always create a new ``feature`` branch before making any changes. Make your changes
   in the ``feature`` branch. It's good practice to never routinely work on the ``main`` branch of any repository
   and we have a pre-commit check that prevents committing to `main`
   ```

5. We use [tox](https://tox.wiki/en/latest/index.html) to help with common development tasks.
   To get your development environment up and running we recommend installing `tox` and the
   arviz local package you are working on:

   ```bash
   pip install tox
   pip install -e .
   ```

6. Work on your feature, bugfix, documentation improvement...

7. Execute the relevant development related tasks. Each repository has slightly different needs
   for testing or doc building so we start by checking the available tox commands:

   ```bash
   tox list -m dev
   ```

   which will print something like:

   ```none
   check          -> Perform style checks
   docs           -> Build HTML docs
   test-namespace -> Run ArviZ metapackage tests
   cleandocs      -> Clean HTML doc build outputs
   viewdocs       -> View HTML docs with the default browser
   ```

   we then execute the relevant commands. If we have been working a bugfix and didn't
   change anything related to documentation we can skip those tasks and run only the
   `check` and `test-namespace` ones:

   ```bash
   tox -e test-namespace
   tox -e check
   ```

(commit_push_step_pr)=
6. Once you are happy with the changes, add your changes using git commands,
   ``git add`` and then ``git commit``, like:

   ```bash
   git add modified_files
   git commit -m "commit message here"
   ```

   to record your changes locally. After committing, it is a good idea to sync with the base
   repository in case there have been any changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Then push the changes to your GitHub account with:

   ```bash
   git push -u origin my-feature
   ```

7. Go to the GitHub web page of the respective ArviZ repository you were working on.
   Click the 'Pull request' button to send your changes to the project's maintainers for review.
   This will send an email to the committers.

   :::{tip}
   Now that the PR is ready to submit, check the {ref}`pr_checklist`.
   :::

8. Thanks for contributing! {octicon}`heart-fill` Now Wait for reviews to come it.
   Most reviewers work on ArviZ on a volunteer basis, depending on availability and
   workload it can take several days until you get a review.

9. Address review comments. Some PRs can be merged directly, but the most common scenario
   if for further commits to be needed. To update your PR you only need to commit to the
   same branch you were working on and push as shown in {ref}`step 6 <commit_push_step_pr>`.
