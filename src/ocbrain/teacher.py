from __future__ import annotations

import sqlite3
from typing import Any

from ocbrain.egress import egress_preview
from ocbrain.events import REWARD_BANDS, canonical_json
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext

DEFAULT_TEACHER_MODEL = "hosted_teacher"


def hosted_teacher_request(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    query: str | None = None,
    objective: str = "compile_scoped_beliefs",
    model: str = DEFAULT_TEACHER_MODEL,
    limit: int = 20,
    record: bool = True,
) -> dict[str, Any]:
    preview = egress_preview(
        conn,
        context=context,
        target="hosted_teacher",
        query=query,
        record=record,
    )
    included = preview["included"][:limit]
    request_id = stable_id(
        "teacher_request",
        objective,
        model,
        canonical_json(context.to_dict()),
        query or "",
        preview.get("audit_id") or preview["payload_hash"],
    )
    response_schema = teacher_response_schema()
    request = {
        "model": model,
        "objective": objective,
        "system": (
            "You are a scoped OCBrain teacher. Return only JSON matching the schema. "
            "Compile claims only from supplied evidence ids, preserve scope exactly, "
            "and use discard when evidence is weak or contradictory."
        ),
        "input": {
            "context": context.to_dict(),
            "query": query,
            "eligible_evidence": included,
            "response_schema": response_schema,
        },
    }
    dispatch_state = "approval_required" if included else "no_eligible_evidence"
    return {
        "request_id": request_id,
        "dispatch_state": dispatch_state,
        "call_performed": False,
        "approval": {
            "required": bool(included),
            "reason": "hosted teacher calls and hosted egress require explicit approval",
            "required_approvals": ["hosted_teacher_calls", "hosted_egress"],
        },
        "egress": preview,
        "request": request,
        "summary": {
            "eligible_evidence": len(included),
            "rejected_evidence": preview.get("rejected_count", len(preview["rejected"])),
            "payload_hash": preview["payload_hash"],
            "audit_id": preview.get("audit_id"),
        },
    }


def teacher_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["compilations"],
        "properties": {
            "compilations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "belief_id",
                        "body",
                        "evidence_ids",
                        "scope",
                        "confidence",
                        "teacher_rationale",
                        "reward_band",
                    ],
                    "properties": {
                        "belief_id": {"type": "string"},
                        "body": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "scope": {"type": "object"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "teacher_rationale": {"type": "string"},
                        "reward_band": {
                            "type": "string",
                            "enum": sorted(REWARD_BANDS),
                        },
                    },
                },
            },
            "notes": {"type": "string"},
        },
    }
