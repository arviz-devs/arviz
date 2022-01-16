# Main Governance Document

## The Project

The ArviZ Project (The Project) is an open source software project
affiliated with the 501c3 NumFOCUS Foundation. The goal of The Project is to
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
open and public Project discussions on GitHub, Slack, and Gitter chat rooms.
The foundation of Project participation is openness and transparency.

There have been many Contributors to the Project, whose contributions are listed in the
logs of any of the repositories under the ArviZ-devs organization.

The Project Community consists of all Contributors and Users of the Project.
Contributors work on behalf of and are responsible to the larger Project
Community and we strive to keep the barrier between Contributors and Users as
low as possible.

The Project is formally affiliated with the 501c3 NumFOCUS Foundation
([http://numfocus.org](http://numfocus.org)),  NumFOCUS is the
only legal entity that has a formal relationship with the project.

## Governance

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


## Community Architecture

* General Contributors
* Core Contributors (of which Council members are also a part of)
* 7 Person steering council (Random Variables Council)

Anyone working with ArviZ has the responsibility to personally uphold
the Code of Conduct. Core Contributors have the additional responsibility
of _enforcing_ the Code of Conduct to maintain a safe community.

## Core Contributors
Core Contributors are those who have provided consistent and meaningful contributions to ArviZ.
These can be, but are not limited to, code contributions, community contributions, tutorial
development etc. Core Contributors will be given the ability to manage the ArviZ GitHub
repository, including code merges to main. This does not necessarily mean Core Contributors
must submit code, but more so signifies trust with the project as a whole.

### Core Contributor Responsibilities
* Enforce code of conduct
* Maintain a check against Council

### Core Contributor Nominations and Confirmation Process
Current Core Contributors can nominate candidates to become Core Contributors by
requesting so in a GitHub issue. If nominated candidates accept their nomination
(explicit comment approving nomination on the issue or "thumbs-up" emoji on the same issue),
then they can be considered by the Council: on the first of the month following a
nomination, the Council will vote on each nominee using the process above.

Voting will be private with results published on the issue ticket. In the case of a
rejection, results must include the reasons behind the decision (e.g. the time since
starting to contribute is deemed too short for now). The candidate
would then have to wait 3 months to be considered again.

### Current Core Contributors
* Oriol Abril-Pla (@OriolAbril)
* Agustina Arroyuelo (@agustinaarroyuelo)
* Alex Andorra (@AlexAndorra)
* Seth Axen (@sethaxen)
* Colin Carroll (@ColCarroll)
* Piyush Gautam (@percygautam)
* Robert P. Goldman (@rpgoldman)
* Ari Hartikainen (@ahartikainen)
* Ravin Kumar (@canyon289)
* Osvaldo Martin (@aloctavodia)
* Mitzi Morris (@mitzimorris)
* Du Phan (@fehiepsi)
* Aki Vehtari (@avehtari)

## Random Variables Council
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
* Making decisions when regular community discussion does not produce consensus on an issue
  in a reasonable time frame.
* Making decisions about strategic collaborations with other organizations or individuals.
* Making decisions about the overall scope, vision and direction of the project.
* Developing funding sources
* Deciding how to disburse funds with consultation from Core Contributors

The council may choose to delegate these responsibilities to sub-committees. If so, Council members must update this document to make the delegation clear.

Note that an individual council member does not have the power to unilaterally wield these
responsibilities, but the council as a whole must jointly make these decisions.
In other words, Council Members are first and foremost Core Contributors, but only
when needed they can collectively make decisions for the health of the project.

### Current Random Variables Council members
The current RV Council members are:

* Oriol Abril-Pla (@OriolAbril)
* Alex Andorra (@AlexAndorra)
* Seth Axen (@sethaxen)
* Colin Carroll (@ColCarroll)
* Ari Hartikainen (@ahartikainen)
* Ravin Kumar (@canyon289)
* Osvaldo Martin (@aloctavodia)

The election record can be found at [elections/ArviZ_2020.md](https://github.com/arviz-devs/arviz/blob/main/elections/ArviZ_2020.md)

### Council Decision Making Process
By and large we expect the decisions in ArviZ to be made _ad hoc_ and require little formal
coordination and with the community at large. However, for controversial proposals and
new Core Contributors the council may need to intervene to make the final decision
in a group vote.

#### Call for a vote
Core Contributors can call for a vote to resolve a target issue they feel has
been stale for too long and for which informal consensus appears unlikely. For a vote
to be called, the target issue must be at least 2 months old.

To do so, they have to open a proposal issue ticket labeled "Council Vote".
The proposal issue should contain a link to the target issue and a proposal
on how to resolve it. Proposals should include a statement making clear what it
means to "agree" or to "disagree".

Before voting starts, at least 3 days will be left for Core
Contributors to raise doubts about the proposal's _phrasing_, no extra discussion will take
place in the proposal issue. Proposal issues should be locked from creation
to prevent attracting discussion from people not familiar with the decision
process.

#### Voting process

* Each Council Member will vote either "Yes", "No", or "Neutral".
* It is recommended that all Council Members expose their reasons when voting.
  "No" votes, however, _must_ list the reasons for disagreement. Any "No" vote
  with no reason listed will be considered a "Neutral" vote.
* An absence of vote is considered as "Neutral".
* Voting will remain open for at least 3 days.
* For the proposal to pass, at least 60% of the council must vote "Yes", and no more
  than 20% can vote "No".

For decisions about the project the Council will perform it directly on the
proposal issue.
For decisions about people, such as electing or ejecting Core Contributors, the Council
will vote privately. However the decision will be posted publicly in an issue ticket.

### Private communications of the Council

Unless specifically required, all Council discussions and activities will be
between public (GitHub, Gitter), and partially public channels (Slack)
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
* Nominations are taken over a public GitHub issue ticket over the course of 2 weeks.
* Only Core Contributors may nominate folks
* Self Nominations are allowed
* At the conclusion of the 2 weeks, the list of nominations is posted on the ticket and this ticket is closed.


#### Election Process

* Voting occurs over a period of at least 1 week, at the conclusion of the nominations.
  Voting is blind and mediated by either an application or a third party like NumFOCUS.
  Each voter can vote zero or more times, once per each candidate. As this is not about
  ranking but about capabilities, voters vote on a yes/neutral/no basis
  per candidate -- “would I trust this person to lead ArviZ?”.
* Candidates are evaluated independently, each candidate having 60% or more of
  yes votes _and_ less or equal than 20% of no votes is chosen. If the number
  of chosen candidates is >=4 and <=10 all candidates are confirmed and the
  election process stops here.
* In the event that either not enough or too many candidates were confirmed,
  candidates are ranked by interpreting yes=+1, neutral=0 and no=-1. If too
  many candidates were confirmed, the 10 candidates with higher rank are
  elected. If not enough candidates were chosen, the 4 candidates with higher
  rank are elected.
* In the event of a tie there will be a runoff election for the tied candidates. To avoid
  further ties and discriminate more among the tied candidates, this vote will be held
  by [Majority Judgment](https://en.wikipedia.org/wiki/Majority_judgment) (MJ): for each
  candidate, voters judge their suitability for office as either "Excellent", "Very Good",
  "Good", "Acceptable", "Poor", or "Reject". Multiple candidates may be given the same
  grade by a voter. The candidate with the highest median grade is the winner.
* If more than one candidate has the same highest median-grade, the MJ winner is
  discovered by removing (one-by-one) any grades equal in value to the shared median
  grade from each tied candidate's total. This is repeated until only one of the
  previously tied candidates is currently found to have the highest median-grade.
* If ties are still present after this second round, the winner will be chosen at random.
  Each person tied will pick an integer number in the `[1, 100]` interval and send it
  privately to the third party mediating the election. After receiving all the numbers,
  said third party will draw a random integer from random.org. The person with the
  closest circular distance, defined as `min(|a-b|, 100-|a-b|)`, will be selected.
  This process will be repeated as many times as necessary as there may be ties
  resulting from candidates choosing the same number.
* At the conclusion of voting, all the results will be posted. And at least 24
  hours will be left to challenge the election result in case there were
  suspicions of irregularities or the process had not been correctly carried
  out.

#### Length of Tenure and Reverification
* Council members term limits are 4 years, after which point their seat will come up for reelection.
* Each year on April 7th council members will be asked to restate their commitment to being on the council
* Attempts should be made to reach every council member over at least 2 communication media. For example: email, Slack, phone, or GitHub.
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
* Core contributors can be ejected through a simple majority vote by the council. Council members vote "Yes" or "No".
* Upon ejecting a core contributor the council must publish an issue ticket, or public document detailing the
  * Violations
  * Evidence if available
  * Remediation plan (if necessary)
  * Signatures majority of council members to validate correctness and accuracy
* Core contributors can also voluntarily leave the project by notifying the community through a public means or by notifying the entire council.

#### Voting Criteria For Future Elections
Voting for first election is restricted to establish stable governance, and to defer major decision to elected leaders
* For the first election only the people registered following the guidelines in `elections/ArviZ_2020.md` can vote
* In the first year, the council must determine voting eligibility for future elections between two criteria:
  * Core contributors
  * The contributing community at large
