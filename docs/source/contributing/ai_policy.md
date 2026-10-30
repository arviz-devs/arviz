(ai_policy)=
# AI usage policy

AI tools such as chatbots, coding assistants and agents can be helpful when working on ArviZ, and you're welcome to use them. This page explains what we expect when you do, in any ArviZ repository.

An agent can write a large pull request in minutes, but a maintainer still has to read all of it. Review is also how contributors learn the codebase, so it only works when you understand what you submit.

## Owning your changes

Every change in your pull request is yours, whoever or whatever wrote it, so you should be able to explain it without going back to the agent. If the agent suggests something you don't understand, ask it to explain and check the answer. If you still don't fully understand a part but believe it's correct, you may submit it as a last resort. Say so in the pull request and explain how you checked it.

## Agent-assisted vs fully agentic PRs

We welcome agent-assisted pull requests, where you direct the work, review every change and run the tests yourself. We don't accept fully agentic ones, where an agent writes the code, the tests and the description and the result goes in without meaningful review. Agents may not open issues or pull requests, post comments or mention maintainers on their own.

We may close fully agentic pull requests, and those whose authors can't explain them, without a detailed review, and block people who keep opening them.

## Communication

Please write pull request descriptions, issues and review replies yourself. Agent-written descriptions tend to be verbose and uninformative. An agent can help with a first draft or with polishing yours, and translation tools are fine, but the final text should reflect your own understanding. Keep descriptions brief, covering what changed, why, and how you verified it.

Answer reviewers' questions yourself, not with the agent's replies. We may hide comments that are pasted agent output.

## Disclosure

Please note in the pull request description which AI tools you used and for which parts of the work, for example "AI use: Claude Code found the bug and drafted the test, which I then rewrote."

## Tests

Agent-written tests often pass whether or not the code is correct. Look out for tests that:

- only confirm the code's current output, because they were written after it
- compute the expected value the same way as the code under test
- only check shapes, dtypes or that the output is finite
- use inputs where the bug can't show up

Where you can, write the test first and check that it fails. For a bug fix, the test should fail once you revert the fix. Take expected values from an independent source, such as a hand calculation or an established implementation, and keep tests few and targeted. See also the testing guidelines for {ref}`ArviZ-stats <arviz_stats:contributing_testing>` and {ref}`ArviZ-plots <arviz_plots:contributing_testing>`.

## Scope of changes

Bug fixes should address problems that affect real use. If you can only trigger a problem with inputs built for the purpose, open an issue instead. Keep the diff to what the change needs, and please don't open many pull requests at once.

For new features, API changes and larger refactors, open an issue first so we can agree on the approach. If a pull request is too large to review properly, we may close it or ask you to split it up.

## Conventions and documentation

Code written with AI tools should match the code around it, without the extra comments and long docstrings agents tend to add. Run the style checks and tests locally before you push, as described in the {ref}`pr_checklist`. Documentation and notebooks need the same care, since AI tools often get statistical details wrong or invent references.

## Copyright

Contributions to ArviZ are released under the Apache 2.0 license, so only submit code you're allowed to license that way. That includes checking that nothing came from the tool's training data under an incompatible license, and that the tool's terms let you contribute its output.

## Other projects

Our approach to AI usage is inspired by other open source projects, including [xarray](https://docs.xarray.dev/en/stable/contribute/ai-policy.html), [NumPy](https://numpy.org/devdocs/dev/ai_policy.html), [SciPy](https://docs.scipy.org/doc/scipy/dev/conduct/ai_policy.html), [statsmodels](https://www.statsmodels.org/devel/dev/ai-policy.html), [scikit-learn](https://scikit-learn.org/dev/developers/contributing.html#automated-contributions-policy), [matplotlib](https://matplotlib.org/stable/devel/contribute.html#use-of-generative-ai), [LLVM](https://llvm.org/docs/AIToolPolicy.html), [Apache Arrow](https://arrow.apache.org/docs/dev/developers/overview.html#ai-generated-code) and [Apache Airflow](https://github.com/apache/airflow/blob/main/contributing-docs/05_pull_requests.rst#gen-ai-assisted-contributions), and by the Apache Software Foundation's [guidance on generative tooling](https://www.apache.org/legal/generative-tooling.html). We expect to revise this page as we gain experience with these tools.
