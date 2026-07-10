"""ocbrain dataset factory (spec §7).

Public surface for the mining half of the factory. ``export_all`` and ``stats``
live in lane-5's ``export.py``/``stats.py`` and are intentionally NOT imported
here so this package has no dependency on them.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ocbrain.config import OcbrainConfig, load_config
from ocbrain.dataset.mine_dpo import DpoPair, find_event_pairs, find_transcript_pairs, mine_dpo
from ocbrain.dataset.mine_persona import commit_examples, mine_persona, telegram_examples
from ocbrain.dataset.mine_sft import Exchange, label_exchange, mine_sft, segment_exchanges
from ocbrain.dataset.quality import scrub_reasons, store_example
from ocbrain.dataset.transcripts import (
    Session,
    Turn,
    UserClass,
    classify_user_text,
    is_conversation_transcript,
    iter_unmined_transcripts,
    parse_transcript,
)
from ocbrain.fsutil import ParseCache, db_side_dir

__all__ = [
    "DpoPair",
    "Exchange",
    "Session",
    "Turn",
    "UserClass",
    "classify_user_text",
    "commit_examples",
    "find_event_pairs",
    "find_transcript_pairs",
    "is_conversation_transcript",
    "iter_unmined_transcripts",
    "label_exchange",
    "mine_all",
    "mine_dpo",
    "mine_persona",
    "mine_sft",
    "parse_transcript",
    "scrub_reasons",
    "segment_exchanges",
    "store_example",
    "telegram_examples",
]


def mine_all(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    roots: list[str] | None = None,
    repos: list[str] | None = None,
    verified_only: bool = False,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Run SFT, DPO, and persona mining (spec §4.1 stage 11). Idempotent.

    ``roots`` defaults to the review/session roots; ``repos`` defaults to the
    configured persona git repos (discovered when empty).
    """
    cfg = cfg or load_config()
    roots = roots if roots is not None else list(cfg.review.session_roots)
    budget = None if time_budget_seconds is None else time_budget_seconds / 3.0
    # One run-shared parse memo so the three miners don't independently read +
    # normalize the same new/changed transcript (up to 3x → 1x when they parse
    # with identical options, e.g. no founder configured). Anchored beside the
    # live SQLite DB so tmp-DB tests keep it out of the live data/ tree.
    cache = ParseCache(db_side_dir(conn, "parse_cache"))
    sft = mine_sft(conn, cfg=cfg, roots=roots, time_budget_seconds=budget, parse_cache=cache)
    # NOTE: mine_dpo (owned by the v3/dpo lane) does not yet accept parse_cache,
    # so its parse of each new/changed file is not shared. Reported as a
    # cross-lane follow-up: adding a ``parse_cache`` param to mine_dpo mirroring
    # this call closes the last third of the within-run re-parse.
    dpo = mine_dpo(conn, cfg=cfg, roots=roots, time_budget_seconds=budget)
    persona = mine_persona(
        conn,
        cfg=cfg,
        roots=roots,
        repos=repos,
        verified_only=verified_only,
        time_budget_seconds=budget,
        parse_cache=cache,
    )
    locks = [part["writer_lock"] for part in (sft, dpo, persona)]
    writer_lock = {
        "batch_max_operations": cfg.dataset.write_batch_size,
        "batch_max_seconds": cfg.dataset.write_batch_seconds,
        "operations": sum(item["operations"] for item in locks),
        "batches_committed": sum(item["batches_committed"] for item in locks),
        "lock_wait_seconds": round(sum(item["lock_wait_seconds"] for item in locks), 6),
        "max_lock_wait_seconds": max(item["max_lock_wait_seconds"] for item in locks),
        "writer_lock_seconds": round(sum(item["writer_lock_seconds"] for item in locks), 6),
        "max_writer_lock_seconds": max(item["max_writer_lock_seconds"] for item in locks),
    }
    return {
        "ok": True,
        "sft": sft,
        "dpo": dpo,
        "persona": persona,
        "writer_lock": writer_lock,
    }
