(diataxis_for_arviz)=
# Diátaxis for ArviZ

ArviZ documentation has the {doc}`diataxis:index` as a North Star.

There are however some specificities to ArviZ case and knowing Diátaxis alone
is not enough to be able to understand ArviZ doc structure and correctly place
new content pages within it.
At the same time, having a basic understanding of Diátaxis is needed to understand
this page and the content structure in ArviZ docs.

These are:

* Multiple target audiences and documentation goals
* Strong visual component
* Work in progress

## Multiple target audiences
Diátaxis is a framework based around **user needs** in their exercise of a
**practical craft** (more detail at {doc}`diataxis:introduction` page in Diatáxis website).

The scope of the ArviZ website however is not limited to ArviZ users nor it is limited to practical
crafts like using ArviZ or contributing to it.

After careful consideration, we decided to split the website into three components:
two practical crafts, using and contributing
(each of which follows Diátaxis independently of the other)
and general information about the ArviZ project,
it's people, community, marketing info...
which can nor should follow the Diátaxis framework.

## Strong visual component
ArviZ is a library with a strong visualization component.
This translates for example into some duplicity when it comes to reference content.

Both the {ref}`example_gallery` and the {ref}`api` pages are reference content.
The API reference indexes by name and contains the description of each object,
whereas the Example Gallery indexes by image and to avoid excessive duplication,
links to the respective API page.

This allows users who know the name of the function or who want to check the options
available for the function they are currently using to quickly find it in the
API reference while also allowing users who know how a plot looks but don't know
their name nor the right ArviZ function to find it in the Example Gallery.

## Work in progress
In addition, ArviZ documentation itself and the application of Diátaxis to it
is still an active work in progress. That means that some pages will not
match what is described in {ref}`content_structure` because their updating to
enforce Diátaxis is still pending.
