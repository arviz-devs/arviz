# ArviZ threat model

Last reviewed: 2026-10-22

## Scope and assumptions

ArviZ is a scientific Python library commonly used locally to analyse numerical data. This initial assessment focuses on the main ArviZ repository, including source changes, dependencies, GitHub Actions, package building and publishing, and data-loading workflows. Related projects such as `arviz-base`, `arviz-stats`, and `arviz-plots` may introduce additional risks that require separate review.

Users commonly analyse their own datasets, but downloaded files and data from external sources should not automatically be considered safe. ArviZ also relies on upstream libraries for much of its file parsing and data processing. This document identifies plausible threats; it is not a comprehensive security audit.

## Source, build, and release integrity

**Compromised build dependencies or actions.** A compromised dependency, GitHub Action, or downloaded build tool could execute malicious code during continuous integration or package building. The publishing workflow pins its third-party actions to commit SHAs and sets permissions at the workflow and job levels. The build job also enables provenance attestations. These controls reduce some risks but do not eliminate the possibility of compromised code or tools.

**Unauthorized source or release changes.** A compromised maintainer account or insufficiently restricted repository settings could allow unauthorized source changes or package publication. The publishing workflow uses PyPI Trusted Publishing through OIDC and a `publish` environment intended to require maintainer sign-off. Repository branch protections, tag protections, account security, and environment settings should be reviewed to verify that these controls are enforced as intended.

**Downstream impact.** A malicious source change or compromised build could produce a release that executes unwanted code in downstream users' environments. Provenance can help establish where an artifact was built, but does not prove that its contents are benign. A clear process for assessing affected releases, publishing fixes, and notifying downstream users is important.

## Dependencies and data during normal use

**Compromised or vulnerable dependencies.** A malicious or vulnerable direct or transitive dependency could affect users during installation or execution. ArviZ relies on upstream projects for many low-level operations, while its dependency constraints influence which upstream fixes users can install. Dependency updates should balance compatibility with timely security fixes. A full audit of upstream implementations is outside this initial assessment.

**Untrusted or unexpected data.** Malformed or unexpectedly large datasets may cause errors, excessive resource consumption, or unsafe behavior in ArviZ or an underlying parser. The handling of downloaded example data and the guarantees provided by any integrity checks should be verified before drawing conclusions about the protection they provide. Data integrity checks alone would not prove that a file is safe to parse.

## Known gaps and incident response

ArviZ's [security policy](https://github.com/arviz-devs/arviz/security/policy) directs vulnerability reports to Tidelift. The related [Develop an Incident Response Plan issue](https://github.com/arviz-devs/arviz/issues/2580) tracks further incident-response work.

Remaining questions include how reports are triaged, who coordinates remediation, how affected releases are handled, and how downstream users are notified when necessary. Repository branch and tag protections, publishing environment settings, and data-loading integrity checks also warrant further verification.

This is an initial assessment of plausible threats, not a statement that the scenarios described are confirmed vulnerabilities. Please open an issue or propose a change if an important scenario is missing.