# Security Policy

## Supported Versions

ArviZ supports security updates for the following versions:

| Version | Supported          |
|---------|--------------------|
| Latest stable release | ✅ Yes |
| Previous minor release | ✅ Yes (critical fixes only) |
| Older releases | ❌ No |

We strongly encourage all users to keep ArviZ up to date.

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in ArviZ, please report it responsibly via one of the following channels:

### Option 1: GitHub Private Security Advisory (Preferred)

Use GitHub's built-in private vulnerability reporting:

1. Navigate to the [ArviZ Security Advisories page](https://github.com/arviz-devs/arviz/security/advisories).
2. Click **"Report a vulnerability"**.
3. Fill in the details of the vulnerability.

This keeps the disclosure confidential until a fix is ready.

### Option 2: Email

Send a detailed report to the ArviZ maintainers at:

**`security@arviz-devs.org`** *(update with actual address)*

Please encrypt sensitive reports using our [PGP key](https://github.com/arviz-devs/arviz/security) *(update with actual key link if available)*.

---

## What to Include in Your Report

To help us triage and resolve the issue quickly, please include:

- A clear description of the vulnerability and its potential impact.
- Steps to reproduce the issue (proof-of-concept code is welcome).
- The ArviZ version(s) affected.
- The environment (OS, Python version, dependency versions).
- Any suggested mitigations or patches, if you have them.

---

## Response Timeline

| Milestone | Target Timeframe |
|-----------|-----------------|
| Acknowledgement of report | Within **2 business days** |
| Initial triage and severity assessment | Within **5 business days** |
| Status update to reporter | Within **10 business days** |
| Patch released (critical/high severity) | Within **30 days** of confirmed vulnerability |
| Patch released (medium/low severity) | Next scheduled release |

We will keep you informed at each stage. If you do not hear from us within 2 business days, please follow up.

---

## Disclosure Policy

ArviZ follows **coordinated disclosure**:

1. Reporter submits a private vulnerability report.
2. The maintainer team acknowledges, triages, and develops a fix.
3. A fix is prepared and tested privately.
4. A new release is published containing the fix.
5. A public GitHub Security Advisory (GHSA) is published simultaneously with the release.
6. The reporter is credited (unless they prefer to remain anonymous).

We ask that reporters allow us a reasonable time (typically 90 days) to fix and disclose before making any public disclosure.

---

## Scope

Security issues we are interested in include, but are not limited to:

- Arbitrary code execution via crafted input files (e.g., malicious NetCDF/HDF5/JSON files).
- Unsafe deserialization of data objects.
- Dependency vulnerabilities that affect ArviZ users.
- Information disclosure or data exfiltration.

**Out of scope:**

- Vulnerabilities in third-party dependencies that do not directly affect ArviZ (please report those upstream).
- Issues that require physical access to the user's machine.
- Social engineering attacks.

---

## Acknowledgements

We sincerely thank all researchers and users who responsibly disclose vulnerabilities. Contributors will be acknowledged in the release notes and security advisory, unless they prefer anonymity.

This security policy is informed by our participation in [GitHub's Secure Open Source Fund](https://resources.github.com/security/secure-open-source-fund/).