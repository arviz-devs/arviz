(content_structure)=
# Documentation content structure

ArviZ documentation has the {doc}`diataxis:index` as a North Star.
This page assumes basic familiarity with Diátaxis.

The ArviZ website is divided into three content blocks:

* ArviZ user documentation
* ArviZ contributor documentation
* ArviZ Project website

## ArviZ user documentation
It is the block formed by the navbar sections
`Getting Started`, `Example Gallery`, `User Guide` and `API Reference`.

It's audience are ArviZ users and it uses Diátaxis to achieve its goal of
teaching how to use ArviZ.

Getting Started
: The Getting Started section should contain mostly tutorials, however, some
  how-to content or overview explanations are also allowed and even encouraged

Example Gallery
: The Example Gallery is a visual reference section that should contain
  **no description** of the functions used. It should instead link to the
  API Reference section.

  Its goal is to serve as visual index for users who know how their desired plot
  looks like but not the name of the ArviZ function that generates it

User Guide
: The User Guide is targeted to people who already have basic ArviZ knowledge
  and should mostly contain how-to guides.
  It can also contain in-depth explanations and some non-python objects reference
  content.

API Reference
: It should index and describe all ArviZ objects that are part of the public API,
  that is, are exposed to ArviZ users. It should be generated from a list of
  public objects only via sphinx extensions, taking the actual content from
  the docstrings in each ArviZ object.

## ArviZ contributor documentation
It is all inside the `Contributing` navbar section.

It's audience are any and all ArviZ contributors, from new contributors to members
of the core team.
It follows Diátaxis quite closely by using multiple captioned toctrees.

Contribution types overview
: Diátaxis explanation content. Contributing to ArviZ can be loosely undestood
  as a single practical craft, but it is more of multiple practical crafts
  coming together.

  This section aims to provide high level overviews explaining different
  contribution types, what skills are needed for them, why are they important
  and guide the reader to their next steps in one of the following sections.

Tutorials
: Diátaxis tutorial content. Tutorials with clearly defined and small scope,
  guiding new contributors in a single "atomic" contribution or even step
  in the contribution process.

How-to guides
: Diátaxis how-to content. Guides that outline the steps for common processes
  in the contributing workflow. Documenting this ensures it is available
  for newcomers but also that all contributors use the same processes
  which can reduce misunderstanding and frictions.

Reference
: Diátaxis reference content.

In depth explanations
: Diátaxis explanation content.

## ArviZ Project website
It is the block formed by the navbar sections
`Community`, `About Us` and by the homepage.

It's audience is anyone interested in the ArviZ project.
This can well be users or contributors but it could also be someone
who is neither. It has also no relation to any practical craft
and does not follow Diátaxis at all.

It should contain all the information related to the ArviZ community,
meeting places (online of physical), shared resources, related
projects and communities; and to the project itself, its members,
its roadmap, its procedures...
