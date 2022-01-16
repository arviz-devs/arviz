# Issue reports

We appreciate being notified of problems with the existing ArviZ code.
We prefer that issues be filed on the
[Github Issue Tracker](https://github.com/arviz-devs/arviz/issues),
rather than on social media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other
issues or pull requests by using the GitHub search tool to look for keywords
in the project issue tracker.

When writing an issue, make sure to include any supporting information,
in particular, the version of ArviZ that you are using and how did you install it.
The issue tracker has several templates available to help in writing the issue
and including useful supporting information.

## Minimal reproducible example
If your issue reports a bug or tries to show a specific behaviour,
provide a way to reproduce the issue. Consider the advice on
[stackoverflow](https://stackoverflow.com/help/minimal-reproducible-example)
about writing a minimal reproducible code example.

If even a minimal version of the code is still too long to comfortably share
in the issue text, upload the code snippet as a
[GitHub Gist](https://gist.github.com/) or a standalone Jupyter notebook.

## Sharing error tracebacks
If relevant, also include the complete error messages and tracebacks.
To avoid cluttering the issue body, you can optionally use the `details`
html tag in GitHub to create a simple dropdown, like so:

````
<details><summary>Click to see full traceback</summary>

```
long and barely comprehensible error traceback
```

</details>
````