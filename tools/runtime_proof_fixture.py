from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocbrain.db import connect, init_db, insert_candidate
from ocbrain.excerpt import write_excerpt
from ocbrain.mcp import handle_request
from ocbrain.schema import Candidate, Evidence, Risk, Scope, Target


def seed_fixture(db_path: Path) -> dict[str, str]:
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    init_db(conn)
    candidates = {
        "approved_workspace": Candidate(
            target=Target.WIKI,
            title="Runtime excerpt reads approved workspace context",
            body=(
                "Runtime excerpt proof: approved workspace candidates can appear in "
                "Codex, Claude, and OpenClaw native context blocks."
            ),
            confidence=0.91,
            scope=Scope.WORKSPACE,
            risk=Risk.LOW,
            evidence=[
                Evidence(
                    uri="fixture://runtime-proof/approved-workspace",
                    excerpt="Approved workspace candidates can appear in native context blocks.",
                )
            ],
            claim_key="wiki runtime excerpt approved workspace context",
        ),
        "proposed_project": Candidate(
            target=Target.MEMORY,
            title="MCP get serves proposed project memory",
            body="Runtime proof: proposed candidates are reviewed and safe for default MCP get.",
            confidence=0.88,
            scope=Scope.PROJECT,
            risk=Risk.MEDIUM,
            evidence=[
                Evidence(
                    uri="fixture://runtime-proof/proposed-project",
                    excerpt="Proposed candidates are reviewed default-read candidates.",
                )
            ],
            claim_key="memory mcp get serves proposed project memory",
        ),
        "draft_workspace": Candidate(
            target=Target.WIKI,
            title="Draft candidate requires opt-in",
            body="Runtime proof: draft candidates must not be returned by default MCP get.",
            confidence=0.73,
            scope=Scope.WORKSPACE,
            risk=Risk.LOW,
            evidence=[
                Evidence(
                    uri="fixture://runtime-proof/draft-workspace",
                    excerpt="Draft candidates require explicit include_draft.",
                )
            ],
            claim_key="wiki draft candidate requires opt in",
        ),
        "approved_private": Candidate(
            target=Target.MEMORY,
            title="Private candidate requires opt-in",
            body="Runtime proof: private candidates must not be returned without include_private.",
            confidence=0.86,
            scope=Scope.PRIVATE,
            risk=Risk.LOW,
            evidence=[
                Evidence(
                    uri="fixture://runtime-proof/private-approved",
                    excerpt="Private candidates require explicit include_private.",
                )
            ],
            claim_key="memory private candidate requires opt in",
        ),
    }
    ids: dict[str, str] = {}
    for name, candidate in candidates.items():
        candidate_id = insert_candidate(conn, candidate)
        if candidate_id is None:
            raise RuntimeError(f"failed to insert fixture candidate: {name}")
        ids[name] = candidate_id
    conn.execute(
        "UPDATE candidates SET status = 'approved' WHERE id = ?",
        (ids["approved_workspace"],),
    )
    conn.execute(
        "UPDATE candidates SET status = 'proposed' WHERE id = ?",
        (ids["proposed_project"],),
    )
    conn.execute(
        "UPDATE candidates SET status = 'approved' WHERE id = ?",
        (ids["approved_private"],),
    )
    conn.commit()
    return ids


def call_get(conn, candidate_id: str, **arguments: Any) -> dict[str, Any]:
    return handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": candidate_id,
            "method": "tools/call",
            "params": {
                "name": "brain.get",
                "arguments": {"id": candidate_id, **arguments},
            },
        },
    )


def build_proof(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "runtime-proof.sqlite"
    ids = seed_fixture(db_path)
    conn = connect(db_path)
    init_db(conn)

    excerpts = {}
    for runtime, filename in (
        ("codex", "AGENTS.md"),
        ("claude", "CLAUDE.md"),
        ("openclaw", "OPENCLAW.md"),
    ):
        path = output_dir / filename
        write_excerpt(conn, path, runtime=runtime, scope=None, limit=10)
        excerpts[runtime] = str(path)

    proof = {
        "fixture_db": str(db_path),
        "candidate_ids": ids,
        "excerpts": excerpts,
        "mcp": {
            "approved_workspace_default": call_get(conn, ids["approved_workspace"]),
            "proposed_project_default": call_get(conn, ids["proposed_project"]),
            "draft_workspace_default": call_get(conn, ids["draft_workspace"]),
            "draft_workspace_include_draft": call_get(
                conn, ids["draft_workspace"], include_draft=True
            ),
            "approved_private_default": call_get(conn, ids["approved_private"]),
            "approved_private_include_private": call_get(
                conn, ids["approved_private"], include_private=True
            ),
        },
    }
    (output_dir / "fixture-ids.json").write_text(
        json.dumps(ids, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "mcp-proof.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create controlled ocbrain runtime proof artifacts"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    proof = build_proof(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "candidate_ids": proof["candidate_ids"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
