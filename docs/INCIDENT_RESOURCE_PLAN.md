# ArviZ Incident Response Plan

**Version:** 1.0
**Last Updated:** 2024
**Owner:** ArviZ Maintainer Team

---

## Purpose

This document defines how the ArviZ maintainer team detects, responds to, and recovers from security incidents. It applies to the ArviZ library, its GitHub repositories, PyPI packages, and associated infrastructure.

This plan fulfills ArviZ's commitment to responsible open-source security practices, including participation in GitHub's Secure Open Source Fund.

---

## Scope

This plan covers:

- Security vulnerabilities reported in ArviZ source code.
- Compromise of the ArviZ PyPI package (`arviz`).
- Compromise of GitHub repository access or maintainer accounts.
- Supply chain incidents (malicious dependencies).
- Accidental exposure of secrets or credentials in the repository.

---

## Roles and Responsibilities

| Role | Responsibility |
|------|---------------|
| **Incident Lead** | The maintainer who first receives or identifies the incident. Coordinates the response, owns communication, and makes escalation decisions. |
| **Technical Lead** | Leads investigation and patch development. May be the same person as Incident Lead for small teams. |
| **Communications Lead** | Drafts user-facing advisories and coordinates disclosure timing. |
| **All Maintainers** | Support investigation, review patches, assist with release. |

For ArviZ, the **on-call rotation** follows the current maintainer list in `MAINTAINERS.md`. If the initial recipient cannot lead, they must immediately hand off to another active maintainer.

---

## Severity Classification

Use the following to classify incidents:

| Severity | Criteria | Example |
|----------|----------|---------|
| **Critical** | Arbitrary code execution, package takeover, supply chain compromise | Malicious code injected into PyPI release |
| **High** | Significant data exposure, severe dependency CVE with direct user impact | Unsafe deserialization of crafted input files |
| **Medium** | Limited impact, requires unusual user action, or affects a niche use case | Exposure of local file paths in error messages |
| **Low** | Minimal impact, no user action required | Outdated dependency with no known exploit path |

When in doubt, treat the incident as one severity level higher until investigation clarifies the impact.

---

## Phase 1: Detection and Intake

### Sources

Incidents may be detected via:

- A private vulnerability report via GitHub Security Advisories.
- An email to the security contact.
- A public disclosure (e.g., a tweet, CVE filing, or issue opened in error).
- Automated scanning (e.g., Dependabot, GitHub secret scanning, pip-audit in CI).
- Internal discovery by a maintainer.

### Intake Steps

1. **Acknowledge receipt** within 2 business days (see `SECURITY.md`).
2. **Create a private tracking issue** using GitHub's private security advisory draft, or a private channel accessible only to maintainers.
3. **Assign an Incident Lead** and notify other active maintainers.
4. **Do not discuss details publicly** until the incident is resolved and coordinated disclosure is ready.

> ⚠️ If a report arrives as a public GitHub issue, immediately convert it to a private security advisory or ask the reporter to resubmit, then close and lock the issue. Do not copy vulnerability details into any public discussion.

---

## Phase 2: Triage and Assessment

Complete within **5 business days** of receipt.

### Triage Checklist

- [ ] Can the vulnerability be reproduced? Document the reproduction steps.
- [ ] What versions of ArviZ are affected?
- [ ] What is the attack vector (local, network, requires user interaction)?
- [ ] What is the potential impact (confidentiality, integrity, availability)?
- [ ] Is there a known exploit in the wild?
- [ ] Does this affect any downstream packages or major users of ArviZ?
- [ ] Assign a severity level (Critical / High / Medium / Low).

### CVSS Scoring (Optional but Recommended)

For High or Critical issues, compute a [CVSS v3.1 base score](https://www.first.org/cvss/calculator/3.1) to support consistent severity classification and advisory publication.

---

## Phase 3: Containment

### Immediate Actions (Critical/High only)

- **Yank the affected PyPI release** if a malicious or critically flawed package was published:
  ```
  pip install twine
  twine yank arviz==<version> --reason "Security vulnerability"
  ```
- **Revoke compromised credentials** (PyPI tokens, GitHub tokens, SSH keys) immediately via the respective service dashboards.
- **Disable affected GitHub Actions workflows** if they are part of a supply chain compromise.
- **Notify PyPI or GitHub Security** if the platform itself needs to take action (e.g., package takeover).

### General Containment

- Limit knowledge of the vulnerability to active maintainers and, if needed, trusted security contacts.
- Avoid merging any PRs related to the vulnerability into the public repository until the fix is ready for coordinated release.

---

## Phase 4: Investigation and Fix Development

### Investigation

1. Identify the root cause — is this a code bug, a dependency issue, a configuration problem, or a process failure?
2. Determine the full scope of affected versions and configurations.
3. Search git history for when the vulnerability was introduced (`git log -S '<pattern>'`).
4. Check whether any forks or dependent packages might be separately affected.

### Patch Development

1. Develop the fix on a **private fork or branch** not visible in the public repository.
   - GitHub Security Advisories support private temporary forks for exactly this purpose.
2. Write or update tests that cover the vulnerability.
3. Have at least one other maintainer review the patch before release.
4. Ensure the fix does not introduce regressions (run the full test suite).

---

## Phase 5: Release and Disclosure

### Release Steps

1. Merge the fix into the main branch.
2. Prepare a release that includes only the security fix (for Critical/High) or bundle with a scheduled release (for Medium/Low).
3. Publish to PyPI.
4. Verify the published package installs correctly and the fix is present:
   ```bash
   pip install arviz==<new_version>
   python -c "import arviz; print(arviz.__version__)"
   ```

### Coordinated Disclosure

Publish the GitHub Security Advisory simultaneously with (or immediately after) the release:

1. Navigate to **Security > Advisories** in the ArviZ GitHub repository.
2. Complete the advisory with:
   - Affected versions
   - Fixed version
   - Description of the vulnerability (without unnecessarily enabling exploitation)
   - Credit to the reporter (with their permission)
   - CVE ID (request one via GitHub if not already assigned)
3. Publish the advisory.

### Notifying the Ecosystem

For Critical or High severity issues, additionally:

- Post an announcement to the [ArviZ Discourse](https://discourse.pymc.io/) or equivalent community forum.
- Consider notifying major downstream users (e.g., PyMC, Bambi) directly via private message before public disclosure, if time allows.
- Notify [PyPI](https://pypi.org/security/) if the incident involved a compromised release.

---

## Phase 6: Post-Incident Review

Complete within **2 weeks** of resolution for Critical/High issues; within the next maintainer meeting for Medium/Low.

### Review Checklist

- [ ] Write a brief internal post-mortem (timeline, root cause, impact, resolution).
- [ ] Identify what went well and what could be improved.
- [ ] Update this Incident Response Plan if any steps were unclear or missing.
- [ ] Add or update automated checks (CI, Dependabot, secret scanning) to catch similar issues earlier.
- [ ] Close out the private tracking issue.
- [ ] Thank the reporter.

---

## Communication Templates

### Initial Acknowledgement (to reporter)

> Thank you for reporting this to us. We have received your report and will investigate promptly. We aim to provide an initial assessment within 5 business days. We will keep you updated on our progress and coordinate the disclosure timeline with you.

### Status Update (to reporter)

> We have completed our initial triage of the vulnerability you reported. [Brief non-sensitive update on status]. We are currently [working on a fix / gathering more information / preparing a release]. Our current target for resolution is [date or timeframe]. Please let us know if you have additional information.

### Public Advisory Summary

> A [severity] security vulnerability was identified in ArviZ versions [X.Y.Z] and earlier. [One-sentence non-exploitable description]. Users are strongly encouraged to upgrade to version [A.B.C], which contains a fix. No workaround is available / A workaround is available [if applicable]. We thank [reporter name or "the reporter"] for responsibly disclosing this issue.

---

## Tooling and Resources

| Tool | Purpose | Link |
|------|---------|-------|
| GitHub Security Advisories | Private tracking and coordinated disclosure | Repository > Security > Advisories |
| GitHub Secret Scanning | Detect leaked credentials | Enabled by default on public repos |
| Dependabot | Automated dependency vulnerability alerts | Repository > Settings > Security |
| PyPI Yanking | Remove a compromised release | `twine yank` or PyPI web UI |
| CVSS Calculator | Severity scoring | https://www.first.org/cvss/calculator/3.1 |
| CVE Request (via GitHub) | Assign a CVE to a published advisory | Available in GitHub Security Advisory UI |
| pip-audit | Audit installed packages for known CVEs | `pip install pip-audit && pip-audit` |

---

## Contacts

| Contact | Details |
|---------|---------|
| Security email | `security@arviz-devs.org` *(update with actual address)* |
| PyPI account owners | *(list internal document or private maintainer wiki)* |
| GitHub organization owners | *(list internal document or private maintainer wiki)* |
| GitHub Security support | https://support.github.com |
| PyPI security team | https://pypi.org/security/ |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial version, created as part of GitHub Secure Open Source Fund participation |

---

*This plan is reviewed and updated at least annually, or after any significant security incident.*