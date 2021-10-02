(issue_triaging)=
# Issue Triaging


As a community-driven, open source collective, we value all user contributions at ArviZ. While most users often tend to look for ways to contribute code via PRs, we welcome and encourage help with issue traiging.

We consider isssue triaging an integral part of the library development. It is the main communication
channel between contributors and users of ArviZ. Moreover, it is a task that gives contributors
more flexibility than contributing code.

This page aims to describe how to get started as an issue triager with ArviZ and provide some
guidelines related to the task.

:::{note}
Even if it is often ignored, issue triaging is generally regarded as an important task,
and it is getting more traction lately. A clear example of that is the
[Labelathon](https://ropensci.org/commcalls/apr2021-pkg-community/) recently hosted by
the rOpenSci community. The [Labelathon was recorded](https://ropensci.org/commcalls/apr2021-pkg-community/)
and has also had a couple of [follow-up blogpost](https://ropensci.org/tags/labelathon/) which are great resources related to
issue triaging.
:::

## How to subscribe to ArviZ issues

The first step for you to get started would be to make sure you subscribe and get notifications of any new issue published on ArviZ's GitHub repo. You can then chose to work on it at the moment or come back to it later. Staying subscribed will ensure you are notified whenever issues are created or updated and therefore, see where your help might be needed.

If you are not familiar on how to set up notifications on GitHub, please check the following - [Setting Up Notifications on GitHub](https://docs.github.com/en/github/managing-subscriptions-and-notifications-on-github/setting-up-notifications/configuring-notifications#configuring-your-watch-settings-for-an-individual-repository).
Once you are set up, you can [view your subscriptions](https://docs.github.com/en/github/managing-subscriptions-and-notifications-on-github/managing-subscriptions-for-activity-on-github/viewing-your-subscriptions) and [manage your subscritions](https://docs.github.com/en/github/managing-subscriptions-and-notifications-on-github/managing-subscriptions-for-activity-on-github/managing-your-subscriptions) to ensure you are not being inundated with the volume and are getting notifications only to issues of your interest.

## Triage guidelines and suggestions

Similar to contributing code via PRs, most issue triaging tasks don't require any
specific permissions. Anyone with a GitHub account can help with issue triaging.

:::{important}
The list below provides ideas and examples of issue triaging entails. However, it is not a comprehnsive compliation. Often users encounter issues not forseen or experienced by developers. We encourage users go ahead and take ownership and bring these to the attention of the person who have posted or the contributors working on it. 
:::

Make sure the issue contains a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example), if relevant.
: Sometimes, the issue doesn't contain an example, however, it can still be clear about the problem.
  In that scenario, someone other than the person who posted the issue can generate an example.
  Issues with a reproducible example allow contributors to focus on fixing the bug or testing the
  proposed enhancement directly instead of having to first understand and reproduce the issue. This
  therefore makes things easier for contributors, but at the same time will reduce the time it takes
  to answer or fix an issue, helping issue posters directly too.

Ensure the issue is clear and has references, if needed.
: If an issue is not completely clear, you can comment by asking for clarifications or
  adding extra references so the issue is clear enough for someone else to start working on.
  One example would be [#1694 (comment)](https://github.com/arviz-devs/arviz/issues/1694#issuecomment-840683745)

Suggest fixes or workarounds.
: In some cases, the issue might be a usage question more than an issue (for example
  [#1758](https://github.com/arviz-devs/arviz/issues/1758)), in which case it can be answered directly,
  in some other cases it might be a version mismatch (i.e. users expecting a fresh out of the oven
  feature to be present in the latest version when it's only available on GitHub development
  version like in [#1773](https://github.com/arviz-devs/arviz/issues/1773)).
  Or there may even be a not ideal yet acceptable workaround for users to avoid the issue
  before it's fixed (for example [#1467](https://github.com/arviz-devs/arviz/issues/1467)).

Guide newcomers
: It is also important to introduce people who comment on ArviZ issues for the first time to the
  community, as well as helping people find issues to work on that match their interest
  and abilities.

:::{note}
If you regularly help with issue triaging you'll probably be asked to join the team and be given
some triaging or write permissions on the ArviZ GitHub repo. This would allow you to label and
assign issues in the ArviZ repo.

You can read more about our team and how to join in the {ref}`about_us` page.
:::
