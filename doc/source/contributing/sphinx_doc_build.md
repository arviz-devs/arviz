(sphinx_doc_build)=
# Building the documentation
Want to help with ArviZ documentation? There are several ways you could assist. You could help by suggesting changes via a pull request (PR), improving our documentation, reporting, and fixing issues. ArviZ documentation is constantly being improved and expanded by users and contributors, if you see anything that’s missing or have ideas on how to improve the existing documentation, please consider getting involved.

We recommend getting familiar with our [GitHub repository](https://github.com/arviz-devs/arviz/) before contributing to the ArviZ documentation.
Our documentation is built using Sphinx and written in Markdown, but can also be written using reStructuredText or Jupyter notebook. The files are appended with the extensions .md, .rst, and .ipynb, respectively.

## Table of content:
- [Getting started with documentation](Getting_started_with_documentation)
- [Using Git on the command-line](Using_Git_on_the_command-line)
- [Editing documents using Git](Editing_documents_using_Git)
- [Building documents with Docker](Building_documents_with_Docker)
- [Pull request checks](Pull_request_checks)
- [Previewing doc changes](Previewing_doc_changes)
- [Resources](Resources)
   - Markdown guide
   - reStructuredText guide
   

## Getting started with documentation

Creating or improving the documentation requires that you build it locally. There are a few methods to choose from:

- Using Git on the command line
- Building documentation with Docker

## Using Git on the command-line

The preferred workflow for contributing to ArviZ documentation is to fork the [GitHub repository](https://github.com/arviz-devs/arviz/), clone it to your local machine, develop on a feature branch, commit the changes, push it to GitHub and open a new pull request (PR). See the step-by-step guide {ref}`here <pr_steps>`.

You'll need to install and configure Git and have a Github account to work with the document locally. Alternatively, you could work with the documents using Docker. See building document with Docker {ref}`building_doc_with_docker`.

## Editing documents using Git

1. On your local machine create a new directory (folder): This is where the repository would reside.

2. Launch the command-line tool on your local machine: This is where you will run the commands. Before using the command line, ensure that Git is installed and configured.

3. Change to the working directory: Change the path to your directory, use the command:

```bash
   $ cd project_folder_name
   ```

4. Fork the repository: At the upper right corner, you’ll see `fork`, click on it to create a copy of the same repository in your account.

5. Clone the repository into your machine: clone your fork of the ArviZ repository to have a local copy of the repo:

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

6. Create a new ``feature`` that is descriptively named to manage your changes. Use the following command:
```bash
   $ git checkout -b my-feature
   ```
Then, change to the working directory as follows:

```bash
   $ cd arviz
``` 

7. Install and activate a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments.html). Then, install the {ref}`requirements_package` to work locally with the documents.

8. Once you clone the repo to your local machine, the repository includes files and folders. To locate ArviZ ``doc`` folder, please follow these steps:
   - Open the local directory in your text editor
   - Open ``arviz`` repository folder
   - Navigate to the ``doc`` folder: It contains the source folder and files. 
   - Navigate to the ``source`` folder: This folder contains files and folders for each section of ArviZ documentation. The ``index.rst`` file serves as a home page for the documentation.

9. To see all the changes made, run:
```bash
   $ git status
```
This will list all the changes you made. You'll be able to see the files you’ve modified. 

10. Add changes to feature: To add new changes to the  feature branch after editing the documents, use the command: 
```bash 
   $ git add modified_files
```
This will add the modified file in the working directory to the staging area.


11. Commit changes: 
After you’ve edited the document, run:
```bash
   $ git commit -m "your message"
```
This will commit the changes to the local branch. Briefly, add a message describing the changes you made to enable maintainers track, review, and merge edits more easily.

12. To record changes locally: Synchronize the changes with the base repository after committing:
```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

13. Push the current branch to your GitHub account.
```bash
    $ git push -u origin my-feature
   ``` 

14. Once you’ve pushed your commits, visit the GitHub page for your fork of the ArviZ repository. You will see a notification detailing your push to your branch on the comparing changes page along with the button labeled **Compare and pull request**.
     - Click Compare & pull request
     - Describe the changes you made to the document on the Open a pull request page. Refer to any ticket issues the pull request fixes. As an example, if the pull request closes an existing issue #0001, describe it in the description as "closes #0001".

15. Click the ``pull request`` button to send your changes to the project’s maintainers for review. 

## Building documentation with Docker

You can also edit the AviZ documentation using Docker. See {ref}`building_doc_with_docker` to find instructions on how to set up dependencies, edit and preview the changes made to files, as well as other information that will guide you through the process.

## Pull request checks

Each PR has a list of checks to ensure that your changes adhere to the rules being followed by the ArviZ docs.
You can check why a specific test is failing by clicking the `Details` next to it. It will take you to errors and warning page. This page shows the details of errors, for example in case of docstrings:

`arviz/plots/pairplot.py:127:0: C0301: Line too long (142/100) (line-too-long)`.

It means line 127 of the file `pairplot.py` is too long.
For running tests locally, see the {ref}`pr_checklist`.


(preview_change)=
## Previewing doc changes

There is an easy way to check the preview of docs by opening a PR on GitHub. ArviZ uses `readthedocs` to automatically build the documentation.
For previewing documentation changes, take the following steps:

1. Go to the checks of your PR. Wait for the `docs/readthedocs.org:arviz` to complete.

   ```{note}
   The green tick indicates that the given check has been built successfully.
   ```

2. Click the `Details` button next to it.
3. It will take you to the preview of ArviZ docs of your PR.
4. Go to the webpage of the file you are working on.
5. Check the preview of your changes on that page.

```{note} Note
The preview version of ArviZ docs will have a warning box that says "This page was created from a pull request (#Your PR number)." It shows the PR number whose changes have been implemented.
```

For example, a warning box will look like this:

```{warning}
This page was created from a pull request (#PR Number).
```

## Resources

[Markdown guide](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html)

[reStructuredText guide](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)