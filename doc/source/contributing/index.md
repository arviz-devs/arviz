(contributing_guide)=
# Contributing to ArviZ
## Welcome
Welcome to the [ArviZ](https://www.arviz.org/en/latest/index.html) project!
If you’re reading this guide, that probably means you’d like to get involved
and start contributing to the project.
ArviZ is a  multi-language open-source project that provides tools for exploratory analysis of Bayesian models.
The package includes functions for posterior analysis, data storage,
sample diagnostics, model checking, and comparison.

As a scientific, community-driven, open-source software project,
we welcome contributions from interested individuals or groups.
The guidelines specified in this document can help new contributors
make their contributions compliant with the conventions of the ArviZ python library,
and maximize the probability of such contributions being merged as quickly and as efficiently as possible.

All kinds of contributions to ArviZ, either on its codebase, its documentation,
or its community are valuable and we appreciate your help.
We constantly encourage non-code contributors to become core contributors
and participate in [ArviZ governance](https://www.arviz.org/en/latest/governance/index.html#governance).

## Before you begin
Before contributing to ArviZ, please make sure to read and observe the [Code of Conduct](https://github.com/arviz-devs/arviz/blob/main/CODE_OF_CONDUCT.md).

## Communication
Contact us on [Gitter](https://gitter.im/arviz-devs/community)
if you want to contribute to the project but you are not sure where you can contribute or how to start.

## Contributing to ArviZ
Historically, the most common way of contributing to the project is
sending pull requests on the main [ArviZ Github repository](https://github.com/arviz-devs/arviz),
but there are many other ways to help the project.

Each subsection within this section gives an overview of common contribution types
to ArviZ. If you prefer video instead of written form, jump to the next section:
[Contributing to ArviZ Webinar](https://python.arviz.org/en/latest/contributing/index.html#contributing-webinar) for a recording of an ArviZ core contributor
on both technical and social aspects of contributing to the library.

You can contribute to ArviZ in the following ways:

### Create an issue
[Submit issues](https://github.com/arviz-devs/arviz/issues/new/choose) for existing bugs or desired enhancements. Check  [Issue reports](https://python.arviz.org/en/latest/contributing/issue_reports.html#issue-reports) for details on how to write an issue.

### Translate ArviZ website
You can [translate](https://python.arviz.org/en/latest/contributing/translate.html#translate) our website to languages other than English.

### Review PRs
[Review pull requests](https://python.arviz.org/en/latest/contributing/review_prs.html#review-prs) to ensure that the contributions are well tested and documented.
For example, you can check if the documentation renders correctly and has no typos or mistakes.

### Manage issues
Helping to manage or triage the open issues can be a great contribution and
a great opportunity to learn about the various areas of the project.
You can [triage existing issues](https://python.arviz.org/en/latest/contributing/issue_triaging.html#issue-triaging) to make them clear and/or provide temporary workarounds for them.

### Write documentation
You can contribute to ArviZ’s documentation by either creating new content or editing existing content.
For instance, you can add new ArviZ examples.
To get involved with the documentation:
1. Familiarize yourself with the [documentation content structure](https://python.arviz.org/en/latest/contributing/content_structure.html#content-structure) and the [tool chain](https://python.arviz.org/en/latest/contributing/doc_toolchain.html#doc-toolchain) involved in creating the website.
2. Understand the basic workflow for opening a [pull request](https://python.arviz.org/en/latest/contributing/pr_tutorial.html#pr-tutorial) and [reviewing changes](https://python.arviz.org/en/latest/contributing/review_prs.html#review-prs).
3. Work on your content.
4. [Build](https://python.arviz.org/en/latest/contributing/sphinx_doc_build.html#sphinx-doc-build) your documentation and preview the doc changes.

### Support outreach initiatives
Support ArviZ [outreach initiatives](https://python.arviz.org/en/latest/contributing/outreach.html#oureach-contrib) such as writing blog posts, case studies, getting people to use ArviZ at your company or university, etc.

### Make changes in the code
Fix outstanding issues (bugs) in the existing codebase.
The issue can range from low-level software bugs to high-level design problems.
You can also add new features to the codebase or improve the existing functionality.
To fix bugs or add new features, do the following:
1. Familiarize yourself with the [guidelines](https://python.arviz.org/en/latest/contributing/contributing_prs.html#steps-before-working) before starting your work.
2. Understand the [development process](https://python.arviz.org/en/latest/contributing/contributing_prs.html#dev-summary) and code conventions.
3. Understand the basic workflow for opening a [pull request](https://python.arviz.org/en/latest/contributing/pr_tutorial.html#pr-tutorial) and [reviewing](https://python.arviz.org/en/latest/contributing/review_prs.html#review-prs) changes.
4. Review and test your code.

(contributing_webinar)=
## Contributing to ArviZ webinar

:::{youtube} 457ZTes4xOI
:::

```{toctree}
:hidden:
:caption: Contribution types

issue_reports
translate
issue_triaging
outreach
review_prs
contributing_prs
```

:::{toctree}
:hidden:
:caption: Tutorials

pr_tutorial
:::

:::{toctree}
:hidden:
:caption: How-to guides

sphinx_doc_build
updating_example_data
running_benchmarks
how_to_release
how_to_add_to_example_gallery
:::

:::{toctree}
:hidden:
:caption: Reference

pr_checklist
architecture
content_structure
docstrings
syntax_guide
developing_in_docker
:::

:::{toctree}
:hidden:
:caption: In depth explanations

doc_toolchain
diataxis_for_arviz
plotting_backends
:::
