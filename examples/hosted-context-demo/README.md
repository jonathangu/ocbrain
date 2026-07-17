# Hosted-context demonstration

This tracked manifest contains only the public OCBrain installation facts in
`source.md`. It exists so a new user can explicitly prove a non-empty hosted
`brain.context -> brain.source` round trip without importing another person's
history or promoting arbitrary agent input.

Review both files before applying them. Applying the manifest writes durable
evidence, proposal, and approval events to the selected database. The required
`--allow-hosted-egress` flag acknowledges that the four fact bodies may be
returned to Codex, Claude Code, OpenClaw, or another hosted-model client. It
does not authorize the database, full source file, or local path to leave the
machine.

From the repository root, after `ocbrain init`:

```bash
.venv/bin/ocbrain --db data/ocbrain.sqlite curated-apply \
  examples/hosted-context-demo/manifest.json \
  --allow-hosted-egress \
  --actor "human-curated:YOUR-NAME"
```

Then start a fresh configured client and use this acceptance request:

```text
Call brain.context with query="What are the current OCBrain installation
requirements and client constraints?", project="ocbrain", and the narrowest
known repo/task scope. Show the raw packet. Expand one issued source with
brain.source and report hash_verified. Record honest feedback and close out.
```

The project in the request must match the manifest's `ocbrain` project. To
keep the demonstration separate from a real brain, initialize a second ignored
database and point the launcher at it with `OCBRAIN_DB`.
