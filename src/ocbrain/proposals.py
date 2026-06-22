from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ocbrain.db import get_candidate, now_iso, transition_candidate


def write_proposal(
    conn,
    candidate_id: str,
    output_dir: Path,
    *,
    allow_unapproved: bool = False,
    actor: str = "ocbrain",
) -> Path:
    row = get_candidate(conn, candidate_id)
    if row is None:
        raise ValueError(f"candidate not found: {candidate_id}")
    if row["status"] not in {"approved", "proposed", "applied"} and not allow_unapproved:
        raise PermissionError(
            f"candidate {candidate_id} must be approved before proposal generation"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{row['target']}-{candidate_id}.md"
    path = output_dir / filename
    hints = json.loads(row["hints_json"])
    evidence = json.loads(row["evidence_json"])
    proposal_hash = stable_proposal_hash(row, hints, evidence)

    lines = [
        "---",
        f"id: {candidate_id}",
        f"target: {row['target']}",
        f"scope: {row['scope']}",
        f"risk: {row['risk']}",
        f"confidence: {row['confidence']}",
        f"candidate_created_at: {row['created_at']}",
        f"proposal_hash: {proposal_hash}",
        "---",
        "",
        f"# {row['title']}",
        "",
        row["body"],
        "",
        "## Handling",
        "",
    ]
    if row["target"] in {"skill", "policy"}:
        lines.append("- Proposal-first. Do not auto-apply.")
    elif row["target"] == "memory":
        lines.append("- Stage as concise source-backed memory.")
    elif row["target"] == "wiki":
        lines.append("- Draft or update wiki page preserving provenance.")
    else:
        lines.append("- Keep ignored unless manually promoted.")

    if hints:
        lines += ["", "## Hints", ""]
        lines += [f"- {hint}" for hint in hints]

    if evidence:
        lines += ["", "## Evidence", ""]
        for item in evidence:
            loc = item.get("uri", "")
            if item.get("line_start"):
                loc += f":{item['line_start']}"
            lines.append(f"- `{loc}`: {item.get('excerpt', '')}")

    content = "\n".join(lines).rstrip() + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    if row["status"] != "proposed":
        transition_candidate(
            conn,
            candidate_id,
            action="propose",
            next_status="proposed",
            actor=actor,
            reason=f"proposal written to {path}",
        )
    conn.execute(
        """
        INSERT OR IGNORE INTO artifact_links (
          id, candidate_id, surface, uri, applied_at, applied_by
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            f"proposal_{candidate_id}",
            candidate_id,
            "proposal",
            str(path),
            now_iso(),
            actor,
        ),
    )
    conn.commit()
    return path


def stable_proposal_hash(row, hints: list[str], evidence: list[dict]) -> str:
    payload = {
        "id": row["id"],
        "target": row["target"],
        "title": row["title"],
        "body": row["body"],
        "confidence": row["confidence"],
        "scope": row["scope"],
        "risk": row["risk"],
        "hints": hints,
        "evidence": evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
