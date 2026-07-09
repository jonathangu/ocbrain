"""excerpt_render stage helpers — idempotent managed-block rendering (spec §4, v0.3).

Covers :func:`ocbrain.excerpt.render_excerpt_file` and the char-budget /
composition helpers it builds on: block creation, in-place update preserving
surrounding content, refusal of quarantined rows, byte-idempotency (no rewrite,
mtime preserved on an unchanged block), and the char budget.
"""

from __future__ import annotations

from pathlib import Path

from ocbrain.db import connect, init_db, upsert_knowledge
from ocbrain.excerpt import (
    BEGIN,
    END,
    build_excerpt,
    compose_managed_block,
    render_excerpt_file,
)


def _db(tmp_path: Path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    return conn


def _inject(conn, subject: str, value_text: str) -> str:
    kid = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        origin="loop",
        subject=subject,
        predicate="note",
        value_text=value_text,
        status="current",
        inject=True,
        confidence=0.9,
    )
    conn.commit()
    return kid


def test_block_created_appends_at_end_preserving_content(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _inject(conn, "cache-policy", "cache TTL is thirty seconds")
    target = tmp_path / "MEMORY.md"
    target.write_text("# Memory\n\nExisting doctrine line.\n", encoding="utf-8")

    res = render_excerpt_file(conn, target, runtime="autopilot", limit=20)
    assert res["changed"] == 1
    assert res["served"] == 1
    text = target.read_text(encoding="utf-8")
    # Managed block present at the END; original content preserved above it.
    assert BEGIN in text and END in text
    assert text.index("Existing doctrine line.") < text.index(BEGIN)
    assert kid in text
    assert "cache-policy" in text
    # A served retrieval was logged for the rendered row.
    served = conn.execute(
        "SELECT COUNT(*) FROM retrieval_uses WHERE outcome = 'served'"
    ).fetchone()[0]
    assert served == 1


def test_block_updated_in_place_when_knowledge_changes(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _inject(conn, "cache-policy", "cache TTL is thirty seconds")
    target = tmp_path / "MEMORY.md"
    target.write_text("# Memory\n\nExisting doctrine line.\n", encoding="utf-8")
    render_excerpt_file(conn, target, runtime="autopilot", limit=20)

    # A new injectable row must show up in the block on the next render, and the
    # surrounding (non-marker) content must survive the in-place replacement.
    _inject(conn, "retry-policy", "retry three times with backoff")
    res = render_excerpt_file(conn, target, runtime="autopilot", limit=20)
    assert res["changed"] == 1
    text = target.read_text(encoding="utf-8")
    assert "cache-policy" in text and "retry-policy" in text
    assert "Existing doctrine line." in text
    # Exactly one managed block (no duplication on update).
    assert text.count(BEGIN) == 1 and text.count(END) == 1


def test_refuses_quarantined_row(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _inject(conn, "cache-policy", "cache TTL is thirty seconds")
    poisoned = _inject(conn, "poison", "benign at insert time")
    # Quarantine the row out-of-band: current + inject=1 but a quarantine_reason
    # set, exactly what list_current_knowledge must exclude from the block.
    conn.execute(
        "UPDATE knowledge SET quarantine_reason = ? WHERE id = ?",
        ("manual_quarantine", poisoned),
    )
    conn.commit()
    target = tmp_path / "MEMORY.md"

    render_excerpt_file(conn, target, runtime="autopilot", limit=20)
    text = target.read_text(encoding="utf-8")
    assert poisoned not in text
    assert "cache-policy" in text


def test_byte_idempotent_no_rewrite_preserves_mtime(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _inject(conn, "cache-policy", "cache TTL is thirty seconds")
    target = tmp_path / "MEMORY.md"
    target.write_text("# Memory\n\nExisting doctrine line.\n", encoding="utf-8")

    first = render_excerpt_file(conn, target, runtime="autopilot", limit=20)
    assert first["changed"] == 1
    before = target.read_text(encoding="utf-8")
    mtime_before = target.stat().st_mtime_ns
    served_before = conn.execute(
        "SELECT COUNT(*) FROM retrieval_uses WHERE outcome = 'served'"
    ).fetchone()[0]

    # Unchanged knowledge → no write, no new served rows, mtime untouched.
    second = render_excerpt_file(conn, target, runtime="autopilot", limit=20)
    assert second["changed"] == 0 and second["skipped"] == "unchanged"
    assert target.read_text(encoding="utf-8") == before
    assert target.stat().st_mtime_ns == mtime_before
    served_after = conn.execute(
        "SELECT COUNT(*) FROM retrieval_uses WHERE outcome = 'served'"
    ).fetchone()[0]
    assert served_after == served_before


def test_compose_managed_block_is_stable_on_reapply(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _inject(conn, "cache-policy", "cache TTL is thirty seconds")
    block = build_excerpt(conn, runtime="autopilot", limit=20)
    base = "# Memory\n\nExisting doctrine line.\n"
    once = compose_managed_block(base, block)
    twice = compose_managed_block(once, block)
    assert once == twice  # re-applying the same block is a fixed point


def test_char_budget_caps_block_size(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    for i in range(40):
        _inject(conn, f"subject-{i:02d}", f"value number {i} with some padding text")
    unbounded = build_excerpt(conn, runtime="autopilot", limit=100)
    bounded = build_excerpt(conn, runtime="autopilot", limit=100, max_chars=1200)
    assert len(bounded) <= 1200
    assert len(bounded) < len(unbounded)
    # The budgeted block still carries the header contract and at least one row.
    assert BEGIN in bounded and END in bounded
    assert "## Shared brain" in bounded
    assert "[know_" in bounded
