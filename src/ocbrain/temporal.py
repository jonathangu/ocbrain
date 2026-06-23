from __future__ import annotations

import re
from typing import Any

from ocbrain.text import compact_whitespace

TEMPORAL_TERMS = ("latest", "current", "installed", "version", "checked", "today", "now")


def has_temporal_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in TEMPORAL_TERMS)


def temporal_subject_key(text: str, *, limit: int = 180) -> str:
    normalized = compact_whitespace(text.lower())
    normalized = re.sub(r"\b20\d{2}[-/]\d{2}[-/]\d{2}(?:t[0-9:.+-]+z?)?\b", " ", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+){1,}(?:[-+][a-z0-9.-]+)?\b", " ", normalized)
    normalized = re.sub(r"\b[a-f0-9]{12,40}\b", " ", normalized)
    normalized = re.sub(r"candidate|operational fact|stage|source|from", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return compact_whitespace(normalized)[:limit]


def temporal_rows(conn, *, target: str = "memory") -> list[dict[str, Any]]:
    rows = []
    for row in conn.execute(
        """
        SELECT
          candidates.id,
          candidates.target,
          candidates.scope,
          candidates.title,
          candidates.body,
          candidates.status,
          candidates.created_at,
          events.created_at AS event_created_at,
          events.source_uri
        FROM candidates
        LEFT JOIN events ON events.id = candidates.event_id
        WHERE candidates.status != 'stale'
          AND candidates.target = ?
        ORDER BY COALESCE(events.created_at, candidates.created_at), candidates.id
        """,
        (target,),
    ):
        text = f"{row['title']} {row['body']}"
        if not has_temporal_term(text):
            continue
        subject_key = temporal_subject_key(row["body"])
        if not subject_key:
            continue
        item = dict(row)
        item["subject_key"] = subject_key
        rows.append(item)
    return rows


def temporal_supersession_groups(conn, *, limit: int | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in temporal_rows(conn):
        grouped.setdefault((row["scope"], row["subject_key"]), []).append(row)

    groups = []
    for (scope, subject_key), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        ordered = sorted(
            rows,
            key=lambda row: (
                row["event_created_at"] or row["created_at"] or "",
                row["created_at"] or "",
                row["id"],
            ),
        )
        kept = ordered[-1]
        stale = ordered[:-1]
        groups.append(
            {
                "scope": scope,
                "subject_key": subject_key,
                "keep_candidate_id": kept["id"],
                "stale_candidate_ids": [row["id"] for row in stale],
                "count": len(ordered),
                "sample_titles": [row["title"] for row in ordered[:3]],
            }
        )
        if limit is not None and len(groups) >= limit:
            break
    return groups
