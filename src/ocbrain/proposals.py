from __future__ import annotations

import json
from pathlib import Path

from ocbrain.db import get_candidate, now_iso


def write_proposal(conn, candidate_id: str, output_dir: Path) -> Path:
    row = get_candidate(conn, candidate_id)
    if row is None:
        raise ValueError(f"candidate not found: {candidate_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{row['target']}-{candidate_id}.md"
    path = output_dir / filename
    hints = json.loads(row["hints_json"])
    evidence = json.loads(row["evidence_json"])

    lines = [
        "---",
        f"id: {candidate_id}",
        f"target: {row['target']}",
        f"scope: {row['scope']}",
        f"risk: {row['risk']}",
        f"confidence: {row['confidence']}",
        f"created_at: {now_iso()}",
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

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    conn.execute(
        "UPDATE candidates SET status = 'proposed', updated_at = ? WHERE id = ?",
        (now_iso(), candidate_id),
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
            "ocbrain",
        ),
    )
    conn.commit()
    return path
