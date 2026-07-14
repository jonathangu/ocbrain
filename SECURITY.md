# Security policy

## Supported version

Security fixes target the latest release on the `main` branch. Older releases
may not receive backports.

## Report a vulnerability privately

Use GitHub's
[private vulnerability reporting form](https://github.com/jonathangu/ocbrain/security/advisories/new).
Do not open a public issue for a suspected vulnerability or data exposure.

Use synthetic data in a reproduction. Never upload a live OCBrain database,
transcript corpus, token, secret, personal identifier, or owner-specific
configuration.

OCBrain restricts a database file to its owner, but the SQLite contents are not
encrypted at rest. Use full-disk encryption and an appropriately protected
backup destination when local-device compromise is in scope.

Please include the affected version or commit, operating system, impact, steps
to reproduce, and any mitigation you have already tested. The maintainer will
acknowledge the report, assess severity and supported versions, and coordinate
disclosure and a fix through the private advisory.
