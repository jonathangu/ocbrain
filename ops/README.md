# Operations artifacts

OCBrain installs only the on-demand local MCP launcher. It does not install
recurring launchd work; an operator who wants a scheduled loop opts in
explicitly (see `docs/SCHEDULED_MAINTENANCE.md`).

`hooks/pre-push` is the tracked git hook that runs
`ocbrain public-safety-check` over the outgoing commit range and blocks a push
that would carry private paths, denylisted identifiers, or new secrets into the
public repo. Install it with `ocbrain install-hooks` or
`scripts/install-hooks.sh`.

The three retired `com.jonathangu.ocbrain.*.plist` placeholders are gone. They
named the autopilot and stall-diagnostic loops, both deleted in v2 along with
the code they pointed at; every table those loops wrote was empty. An operator
upgrading from a legacy install should still unload and delete any
`com.jonathangu.ocbrain.*` agent left in `~/Library/LaunchAgents`.

## The ops manifest: `ocbrain doctor --ops`

Nine of the eleven defects logged against this system in its first four
production days lived in state outside the repo — launchd plists, env blocks,
hand-copied hooks, untracked pointer files. The database checks stayed green
through all of them, because the database was fine; the *wiring* was not.

`~/.ocbrain/ops-manifest.json` records what this machine is supposed to have:
which launchd jobs, with which environment, which hooks copied from which repo
examples, which control files present. It is machine-local and untracked, like
everything else under `~/.ocbrain/`.

```bash
# Deployment day (or after any deliberate wiring change): snapshot intent.
ocbrain doctor --ops --write-manifest

# Any other day: report every drift.
ocbrain doctor --ops
```

Drift is bidirectional: a job that lost an env key is a finding, and so is an
env key added on the machine that the manifest never heard of — an
unmanifested flag is a decision nobody recorded. An absent manifest is a
warning with instructions, never a failure; a fresh install has nothing to
assert yet.
