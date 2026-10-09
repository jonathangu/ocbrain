from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ocbrain.core_v1 import init_core_v1
from ocbrain.db import connect

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "receipt-closeouts.py"
_spec = importlib.util.spec_from_file_location("receipt_closeouts", SCRIPT)
assert _spec is not None and _spec.loader is not None
receipt_closeouts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(receipt_closeouts)


@pytest.fixture
def harness(tmp_path: Path):
    db = tmp_path / "core.sqlite"
    conn = connect(db)
    init_core_v1(conn)
    conn.close()
    root = tmp_path / "lanes"
    lane = root / "Lane-One"
    lane.mkdir(parents=True)
    state = tmp_path / "state.json"

    def run(*extra: str, capsys: pytest.CaptureFixture[str]) -> dict[str, int]:
        rc = receipt_closeouts.main(
            [
                "--db", str(db), "--roots", str(root), "--project", "ocbrain",
                "--state", str(state), *extra,
            ]
        )
        assert rc == 0
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    def rows() -> dict[str, dict]:
        conn = connect(db)
        try:
            return {
                r["task_ref"]: dict(r)
                for r in conn.execute(
                    "SELECT task_ref, status, summary, parent_closeout_id, awaiting, unresolved "
                    "FROM task_closeouts"
                )
            }
        finally:
            conn.close()

    return lane, run, rows


def test_receipts_file_closeouts_idempotently_and_chain_on_edit(harness, capsys) -> None:
    lane, run, rows = harness
    (lane / "RECEIPT-done.md").write_text(
        "# Receipt: Shipped the thing\n\nPR https://github.com/o/r/pull/7\n\nSTATUS: DONE\n"
    )
    (lane / "RECEIPT-heading.md").write_text("# Heading variant\n\n## STATUS: DONE\n")
    (lane / "builder-blocked.md").write_text(
        "# Blocked leaf\n\nSTATUS: BLOCKED needs a key\nRun `ssh asa2 fix-it` first\n"
    )
    (lane / "RECEIPT-failed.md").write_text("# Failed leaf\n\n**STATUS: FAILED** tests red\n")
    (lane / "RECEIPT-running.md").write_text("# Still running\n\nno status yet\n")
    (lane / "notes.md").write_text("STATUS: DONE\n")

    counts = run(capsys=capsys)
    assert counts == {
        "scanned": 5,
        "filed": 4,
        "skipped_unchanged": 0,
        "skipped_no_status": 1,
        "errors": 0,
    }
    filed = rows()
    assert filed["lane-one/receipt-done"]["status"] == "completed"
    assert filed["lane-one/receipt-done"]["summary"] == "Shipped the thing"
    assert filed["lane-one/receipt-heading"]["status"] == "completed"
    blocked = filed["lane-one/builder-blocked"]
    assert blocked["status"] == "blocked"
    assert "needs a key" in blocked["unresolved"]
    assert "`ssh asa2 fix-it`" in blocked["awaiting"]
    assert filed["lane-one/receipt-failed"]["status"] == "failed"
    assert "tests red" in filed["lane-one/receipt-failed"]["unresolved"]

    again = run(capsys=capsys)
    assert again["filed"] == 0
    assert again["skipped_unchanged"] == 4
    assert len(rows()) == 4

    (lane / "RECEIPT-done.md").write_text(
        "# Receipt: Shipped the thing\n\nPR https://github.com/o/r/pull/7\n\nmore\n\nSTATUS: DONE\n"
    )
    edited = run(capsys=capsys)
    assert edited["filed"] == 1
    conn = connect(lane.parent.parent / "core.sqlite")
    try:
        chained = conn.execute(
            "SELECT parent_closeout_id FROM task_closeouts WHERE task_ref=? "
            "ORDER BY closed_at DESC, rowid DESC LIMIT 1",
            ("lane-one/receipt-done",),
        ).fetchone()
        assert chained["parent_closeout_id"]
    finally:
        conn.close()


def test_dry_run_prints_arguments_and_writes_nothing(harness, capsys) -> None:
    lane, run, rows = harness
    (lane / "RECEIPT-a.md").write_text("# A\n\nSTATUS: DONE\n")

    run("--dry-run", capsys=capsys)

    assert rows() == {}
