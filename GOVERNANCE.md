# Main Governance Document
*Note* This Governance has not been implemented yet. Please see Election PR or document for specifics of where we are in establishing governance

The Project
===========

The ArviZ Project (The Project) is an open source software project
affiliated with the 501c3 NumFocus Foundation. The goal of The Project is to
develop open source software and deploy open and public websites and services
for reproducible, exploratory and interactive computing. The Software developed
by The Project is released under the Apache 2 open source license,
developed openly and hosted in public GitHub repositories under the
[GitHub organization](https://github.com/arviz-devs). Examples of
Project Software include the ArviZ code and the Documentation, etc. The Services run by the
Project consist of public websites and web-services that are hosted
at [https://arviz-devs.github.io/arviz/](https://arviz-devs.github.io/arviz/)
The Project is developed by a team of distributed developers, called
Contributors. Contributors are individuals who have contributed code,
documentation, designs or other work to one or more Project repositories.
Anyone can be a Contributor. Contributors can be affiliated with any legal
entity or none. Contributors participate in the project by submitting,
reviewing and discussing GitHub Pull Requests and Issues and participating in
open and public Project discussions on GitHub, Slack, and Gitter chat rooms. The foundation of Project participation is openness
and transparency.

There have been many Contributors to the Project, whose contributions are listed in the
logs of any of the repositories under the ArviZ-devs organization.

The Project Community consists of all Contributors and Users of the Project.
Contributors work on behalf of and are responsible to the larger Project
Community and we strive to keep the barrier between Contributors and Users as
low as possible.

The Project is formally affiliated with the 501c3 NumFOCUS Foundation
([http://numfocus.org](http://numfocus.org)),  NumFOCUS is the
only legal entity that has a formal relationship with the project.

### Governance

This section describes the governance and leadership model of The Project.

The foundations of Project governance are:

-   Openness & Transparency
-   Active Contribution
-   Institutional Neutrality

Traditionally, project leadership was unstructured but primarily driven
by a subset of Core Contributors whose active and consistent
contributions have been recognized by their receiving “commit rights” to the
Project GitHub repositories. In general all Project decisions are made through
consensus among the Core Contributors with input from the Community.

While this approach has served us well, as the Project grows and faces more
legal and financial decisions and interacts with other institutions, we see a
need for a more formal governance model. Moving forward The Project leadership
will consist of a Random Variables Council. We view this governance model as
the formalization of what we are already doing, rather than a change in
direction.


Community Architecture
-----------------------

* N Person steering council (Must be odd, will be decided in first election)
* Core Contributors (of which Council members are also a part of)
* General Contributors

Anyone working with ArviZ has the responsibility to personally uphold
the Code of Conduct. Core Contributors have the additional responsibility
of _enforcing_ the Code of Conduct to maintain a safe community.


Random Variables Council
------------------------
The Project will have a Steering Council that consists of Core Contributors
who have produced contributions that are substantial in quality and quantity,
and sustained over at least one year. The overall role of the Council is to
ensure, taking input from the Community, the
long-term well-being of the project, both technically and as a community.

During the everyday project activities, council members participate in all
discussions, code review and other project activities as peers with all other
Contributors and the Community. In these everyday activities, Council Members
do not have any special power or privilege through their membership on the
Council. However, it is expected that because of the quality and quantity of
their contributions and their expert knowledge of the Project Software and
Services that Council Members will provide useful guidance, both technical and
in terms of project direction, to potentially less experienced contributors.

Council Members will have the responsibility of
* Removing members, including Council Members, if they are in violation of the Code of Conduct
* Make decisions when regular community discussion doesn’t produce consensus on an issue in a reasonable time frame.
* Make decisions about strategic collaborations with other organizations or individuals.
* Make decisions about the overall scope, vision and direction of the project.
* Developing funding sources
* Deciding how to disburse funds with consultation from Core Contributors

The council may choose to delegate these responsibilities to sub-committees. If so, Council members must update this document to make the delegation clear.

Note that individual council member does not have the power to unilaterally wield these responsibilities, but the council as a whole must jointly make these decisions. In other words, Council Members are first and foremost Core Contributors, but only when needed they can collectively make decisions for the health of the project.

ArviZ will be holding its first election to determine its initial council in the coming weeks and this document will be updated.

### Private communications of the Council

Unless specifically required, all Council discussions and activities will be
between public (Github, gitter), and partially public channels (Slack)
and done in collaboration and discussion with the Core Contributors
and Community. The Council will have a private channel that will be used
sparingly and only when a specific matter requires privacy. When private
communications and decisions are needed, the Council will do its best to
summarize those to the Community after eliding personal/private/sensitive
information that should not be posted to the public internet.

### Conflict of interest

It is expected that Council Members will be employed at a wide
range of companies, universities and non-profit organizations. Because of this,
it is possible that Members will have conflict of interests. Such conflict of
interests include, but are not limited to:

-   Financial interests, such as investments, employment or contracting work,
    outside of The Project that may influence their work on The Project.
-   Access to proprietary information of their employer that could potentially
    leak into their work with the Project.

All members of the Council shall disclose to the rest of the
Council any conflict of interest they may have. Members with a conflict of
interest in a particular issue may participate in Council discussions on that
issue, but must recuse themselves from voting on the issue.

### Council Selection Process

#### Eligibility
* Must be core contributor for at least one year

#### Nominations
* Nominations are taken over a public github issue ticket over the course of 2 weeks.
* Only Core Contributors may nominate folks
* Self Nominations are allowed
* At the conclusion of the 2 weeks, the list of nominations is posted on the ticket and this ticket is closed.


#### Election Process

* Voting occurs over a period of at least 1 week, at the conclusion of the nominations. Voting is blind and mediated by either an application or a third party like Numfocus.
Each voter can vote zero or more times, once per each candidate. As this is not about ranking but about capabilities, voters vote on a yes/no basis per candidate -- “would I trust this person to lead ArviZ?”.
* In the event of a tie there will be a runoff election for the tied candidates. To avoid further ties and discriminate more among the tied candidates, this vote will be held by [Majority Judgment](https://en.wikipedia.org/wiki/Majority_judgment): for each candidate, voters judge their suitability for office as either Excellent, Very Good, Good, Acceptable, Poor, or Reject. Multiple candidates may be given the same grade by a voter. The candidate with the highest median grade is the winner.
* If more than one candidate has the same highest median-grade, the MJ winner is discovered by removing (one-by-one) any grades equal in value to the shared median grade from each tied candidate's total. This is repeated until only one of the previously tied candidates is currently found to have the highest median-grade.
* If ties are still present after this second round, the winner will be chosen at random. Each person tied will pick a number between 0 and 100, and a random integer will be generated from random.org. The person with the closet circular distance will be selected.
* At the conclusion of voting, all the results will be posted.
* The decision about who can vote after the first election is deferred to the council. See below for details

#### Length of Tenure and Reverification
* Council members term limits are 4 years, after which point their seat will come up for reelection.
* Each year on April 7th council members will be asked to restate their commitment to being on the council
* Attempts should be made to reach every council member over at least 2 communication media. For example: email, slack, phone, or github.
* If a council member does not restate their commitment their seat will be vacated.
* Inactivity can be determined by lack of substantial contribution, including votes on council, code or discussion contributions, contributions in the community or otherwise.
* In the event of a vacancy in the council, an election will be held to fill the position.
* There is no limit on the number of terms a Council Member can serve

#### Vote of No Confidence
* In exceptional circumstances, council members as well as core contributors may remove a sitting council member via a vote of no confidence. Core contributors can also call for a vote to remove the entire council -- in which case, Council Members do not vote.
* A no-confidence vote is triggered when a core team member (i.e Council member or Core contributor) calls for one publicly on an appropriate project communication channel, and two other core team members second the proposal. The initial call for a no-confidence vote must specify which type is intended -- whether it is targeting a single member or the council as a whole.
* The vote lasts for two weeks, and the people taking part in it vary:
  * If this is a single-member vote called by Core contributors, both Council members and Core contributors vote, and the vote is deemed successful if at least two thirds of voters express a lack of confidence.
  * If this is a whole-council vote, then it was necessarily called by Core contributors (since Council members can’t remove the whole Council) and only Core contributors vote. The vote is deemed successful if at least two thirds of voters express a lack of confidence.
  * If this is a single-member vote called by Council Members, only Council Members vote, and the vote is deemed successful if at least half the voters express a lack of confidence. Council Members also have the possibility to call for the whole core team to vote (i.e Council members and Core contributors), although this is not the default option. The threshold for successful vote is also at 50% of voters for this option.
* If a single-member vote succeeds, then that member is removed from the council and the resulting vacancy can be handled in the usual way.
* If a whole-council vote succeeds, the council is dissolved and a new council election is triggered immediately.

#### Ejecting Core Contributors
* Core contributors can be ejected through a simple majority vote by the council
* Upon ejecting a core contributor the council must publish an issue ticket, or public document detailing the
  * Violations
  * Evidence if available
  * Remediation plan (if necessary)
  * Signatures majority of council members to validate correctness and accuracy
* Core contributors can also voluntarily leave the project by notifying the community through a public means or by notifying the entire council.

### One Time Decisions
* The first election will also include a vote for the number of Council Members, options being 5 and 7. A simple majority will indicate preference.
  * At the conclusion of the vote this section should be removed

#### Voting Criteria For Future Elections
Voting for first election is restricted to establish stable governance, and to defer major decision to elected leaders
* For the first election only the people registered following the guidelines in elections/ArviZ_2020.md can vote
* In the first year, the council must determine voting eligibility for future elections between two criteria:
  * Core contributors
  * The contributing community at large

### Core Contributors
Core Contributors are those who have provided consistent and meaningful contributions to ArviZ.
These can be, but are not limited to, code contributions, community contributions, tutorial
development etc.

#### Core Contributor Nominations and Confirmation Process
Current Core Contributors can nominate candidates for consideration by the council. The council
can make the determination for acceptance with a process of their choosing.

#### Current Core Contributors
* Oriol Abril-Pla (@OriolAbril) 
* Alex Andorra (@AlexAndorra)
* Seth Axen (@sethaxen)
* Colin Carroll (@ColCarroll)
* Robert P. Goldman (@rpgoldman)
* Ari Hartikainen (@ahartikainen)
* Ravin Kumar (@canyon289)
* Osvaldo Martin (@aloctavodia)
* Mitzi Morris (@mitzimorris)
* Du Phan (@fehiepsi)
* Aki Vehtari (@avehtari)

#### Core Contributor Responsibilities
* Enforce code of conduct
* Maintain a check against Council
