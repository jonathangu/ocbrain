# Curated-memory example

This synthetic example shows the smallest explicit path from reviewed source
text to source-backed serving beliefs in a fresh v1 core.

1. Copy this directory outside the repository.
2. Replace `source.md` with a small source you have reviewed.
3. Replace the manifest facts with statements that the source actually
   supports.
4. Compute the source SHA-256 and update `sources[].sha256`.
5. Set the narrowest true project, visibility, and egress policy.
6. Review the complete manifest before applying it.

From the repository root:

```bash
.venv/bin/ocbrain --db data/ocbrain.sqlite curated-apply \
  /absolute/path/to/your/manifest.json \
  --actor "human-curated:YOUR-NAME"
```

Relative source paths resolve from the manifest directory. Applying a manifest
verifies the source hashes and appends evidence, proposal, and approval events.
Reapplying unchanged facts is idempotent.

This example is intentionally `local_only`. If a reviewed manifest contains
`hosted_ok` facts, `curated-apply` refuses to write anything unless the operator
also passes `--allow-hosted-egress`. That flag acknowledges delivery of the
exact fact bodies, not the complete source file or database. See
`../hosted-context-demo/` for the public end-to-end acceptance example.

Do not commit a real personal source, absolute owner path, private project fact,
database, transcript, token, or generated runtime artifact to the public
repository.
