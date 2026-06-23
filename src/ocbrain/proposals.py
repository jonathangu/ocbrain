from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ocbrain.db import get_knowledge, knowledge_evidence, knowledge_summary


def write_proposal(
    conn,
    knowledge_id: str,
    output_dir: Path,
    *,
    allow_unapproved: bool = False,
    actor: str = "ocbrain",
) -> Path:
    row = get_knowledge(conn, knowledge_id)
    if row is None:
        raise ValueError(f"knowledge not found: {knowledge_id}")
    if row["status"] != "candidate" and not allow_unapproved:
        raise PermissionError(
            f"knowledge {knowledge_id} must be a candidate before proposal generation"
        )
    if row["gate"] != "human" and not allow_unapproved:
        raise PermissionError(
            f"knowledge {knowledge_id} must be human-gated before proposal generation"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"knowledge-{row['type']}-{knowledge_id}.md"
    path = output_dir / filename
    evidence = knowledge_evidence(conn, knowledge_id)
    summary = knowledge_summary(row, evidence)
    proposal_hash = stable_proposal_hash(summary)
    title = row["title"] or row["slug"] or row["subject"] or knowledge_id

    lines = [
        "---",
        f"id: {knowledge_id}",
        "object_kind: knowledge",
        f"type: {row['type']}",
        f"gate: {row['gate']}",
        f"risk: {row['risk']}",
        f"status: {row['status']}",
        f"scope: {row['privacy_scope']}",
        f"confidence: {row['confidence']}",
        f"knowledge_created_at: {row['created_at']}",
        f"proposal_hash: {proposal_hash}",
        "---",
        "",
        f"# {title}",
        "",
        "## Candidate",
        "",
    ]
    if row["type"] == "value":
        lines += [
            f"- Subject: `{row['subject']}`",
            f"- Predicate: `{row['predicate']}`",
            f"- Value: `{summary.get('value')}`",
        ]
    else:
        lines += [
            f"- Slug: `{row['slug']}`",
            f"- Body URI: `{row['body_uri']}`",
            f"- Doc kind: `{row['doc_kind']}`",
        ]
    lines += [
        "",
        "## Handling",
        "",
        "- Human-gated. Do not auto-apply.",
        f"- Proposed by `{actor}`.",
    ]
    if evidence:
        lines += ["", "## Evidence", ""]
        for item in evidence:
            source = item.get("source_uri") or item.get("artifact_uri") or item["id"]
            lines.append(f"- `{item['relation']}` [{item['id']}] {item['claim']} (`{source}`)")

    content = "\n".join(lines).rstrip() + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    return path


def stable_proposal_hash(summary: dict) -> str:
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
