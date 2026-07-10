"""Import explicit, local-only persona curation without exposing its text."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ocbrain.config import OcbrainConfig, load_config
from ocbrain.dataset.batching import DatasetWriteBatch
from ocbrain.dataset.mine_persona import is_style_admissible
from ocbrain.dataset.quality import store_example
from ocbrain.ids import content_hash
from ocbrain.text import find_probable_secret_leaks


def import_persona_curation(
    conn: sqlite3.Connection,
    input_path: str | Path,
    *,
    cfg: OcbrainConfig | None = None,
) -> dict[str, Any]:
    """Import private JSONL ``prompt``/``response`` pairs as graded-ready rows.

    The command returns metadata only. It never prints curation text or embeds
    the operator's input path in a dataset row. Local grading remains a separate
    step, so curation cannot silently self-award a passing LLM grade.
    """
    cfg = cfg or load_config()
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"curation line {line_number} is not an object")
        rows.append(value)

    batch = DatasetWriteBatch(
        conn,
        max_operations=cfg.dataset.write_batch_size,
        max_seconds=cfg.dataset.write_batch_seconds,
    )
    imported = excluded = skipped = 0
    source_label = content_hash(path.read_bytes().hex())[:16]
    for index, row in enumerate(rows, 1):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"curation item {index} has no prompt")
        if not isinstance(response, str) or not is_style_admissible(response):
            skipped += 1
            continue
        if find_probable_secret_leaks(prompt) or find_probable_secret_leaks(response):
            raise ValueError(f"curation item {index} contains a probable secret")
        source_uri = f"curation://{source_label}/{index}"
        batch.ensure()
        from ocbrain.db import upsert_evidence

        evidence_id = upsert_evidence(
            conn,
            source_type="persona_curation",
            source_runtime="local",
            source_uri=source_uri,
            content_hash=content_hash(f"{prompt}\0{response}"),
            claim=f"curated persona example {index}",
            privacy_scope="workspace",
        )
        batch.operation()
        batch.ensure()
        result = store_example(
            conn,
            dataset="persona",
            # Reuse the schema's authored_doc provenance class; the explicit
            # curation distinction lives in metadata without a table rebuild.
            source_kind="authored_doc",
            source_uri=source_uri,
            evidence_ids=[evidence_id],
            privacy_scope="workspace",
            body={
                "messages": [
                    {"role": "system", "content": cfg.dataset.persona_system_prompt},
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": response.strip()},
                ]
            },
            metadata={"sender_verified": True, "curated": True},
            target_text=response,
            base_label="good",
            base_confidence=1.0,
            base_reasons=["explicit_persona_curation"],
            n_turns=3,
        )
        batch.operation()
        if result is None:
            skipped += 1
        elif result["quality_label"] == "excluded":
            excluded += 1
        else:
            imported += 1
    batch.flush()
    return {
        "action": "dataset-persona-curate",
        "changed": imported,
        "items": len(rows),
        "imported": imported,
        "excluded": excluded,
        "skipped": skipped,
        "source_hash": source_label,
        "local_only": True,
        "writer_lock": batch.metrics(),
    }
