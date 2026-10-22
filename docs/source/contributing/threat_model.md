# ArviZ threat model

Last reviewed: 2026-10-22

## Scope and assumptions

ArviZ is a high-level scientific Python library generally used locally to analyse numerical data. This initial assessment focuses on the main ArviZ repository, including source changes, dependencies, GitHub Actions, package building and publishing, and data-loading workflows. Related projects such as `arviz-base`, `arviz-stats`, and `arviz-plots` are also in scope, but their implementations have not been audited here.

Inputs are usually users' own datasets, but downloaded files and external data should not automatically be considered safe. ArviZ relies on upstream libraries for much of its low-level parsing and data processing. This document describes plausible threats, not confirmed vulnerabilities.

## Source, build, and release integrity

**Compromised build dependencies or actions.** A compromised dependency, GitHub Action, or downloaded build tool could execute malicious code during CI or package building. The [publishing workflow](https://github.com/arviz-devs/arviz/blob/main/.github/workflows/publish.yml) pins third-party actions to commit SHAs, scopes job permissions, and enables build provenance attestations. However, the build action downloads and runs the current `uv` installer without verifying a pinned content hash. The October 2026 [Python packaging announcement about the emSign/eMudhra CA certificate incident](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576) illustrates why TLS alone may not be sufficient: affected tools could accept forged server certificates. Pinning and verifying the installer content would reduce this risk.

**Unauthorized source or release changes.** A compromised maintainer account or insufficiently restricted repository settings could allow unauthorized source changes or package publication. The [publishing workflow](https://github.com/arviz-devs/arviz/blob/main/.github/workflows/publish.yml) uses PyPI Trusted Publishing through OIDC and a `publish` environment intended to require maintainer sign-off. Branch and tag protections, account security, environment approvals, and publishing permissions still need verification.

**Downstream impact.** A malicious source change or compromised build could produce a release that executes unwanted code in downstream users' environments. Build provenance improves traceability but does not prove that an artifact is benign. Release verification and a process for assessing affected releases, publishing fixes, and notifying downstream users remain important.

## Dependencies and data during normal use

**Compromised or vulnerable runtime dependencies.** A malicious or vulnerable direct or transitive dependency could affect users during installation or execution. ArviZ relies on upstream projects for many low-level operations, while its own dependency constraints influence which upstream fixes users can install. Dependency updates should avoid unnecessarily blocking security fixes. Auditing upstream implementations is outside this first pass.

**Unexpected behavior from untrusted data.** A malformed or unexpectedly large dataset could trigger errors or resource exhaustion in ArviZ or an underlying parser. Inputs are generally user-owned numerical data, so this risk appears less central than supply-chain risks, but downloaded files should not be assumed safe. The example-data loading and integrity-checking behavior still needs targeted verification; no specific checksum protection is claimed here. Supported formats and relevant dependency advisories should be reviewed before prioritizing further safeguards.

## Known gaps and incident response

ArviZ's [security policy](https://github.com/arviz-devs/arviz/security/policy) directs vulnerability reports to Tidelift. The related [Develop an Incident Response Plan issue](https://github.com/arviz-devs/arviz/issues/2580) tracks further incident-response work.

Remaining questions include report triage, remediation ownership, handling affected releases, and notifying downstream users. Repository branch and tag protections, publishing environment settings, and example-data integrity checks also warrant further verification. This threat model should be updated as these controls and response procedures are clarified.

This is an initial assessment of plausible threats, not a statement that the scenarios described are confirmed vulnerabilities. Feedback on overlooked scenarios is welcome.
