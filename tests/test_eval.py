import json
from pathlib import Path

from ocbrain import cli
from ocbrain.db import EventInput, connect, init_db, insert_candidate, upsert_event
from ocbrain.schema import Candidate, Evidence, Risk, Target


def seed_candidate(db_path: Path, candidate: Candidate, body: str = "Architecture note.") -> None:
    conn = connect(db_path)
    init_db(conn)
    event = EventInput(
        id="evt_test",
        source_type="doc",
        source_uri="/tmp/source.md",
        content_hash="hash",
        title="Source",
        summary=body,
        body=body,
    )
    assert upsert_event(conn, event)
    assert insert_candidate(conn, candidate, event.id)
    conn.commit()


def test_eval_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    seed_candidate(
        db_path,
        Candidate(
            target=Target.WIKI,
            title="Architecture memory bridge",
            body="The bridge stores reviewed source-backed facts.",
            confidence=0.8,
            evidence=[Evidence(uri="/tmp/source.md", excerpt="Architecture memory bridge")],
        ),
    )
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "eval",
                "--sample-size",
                "1",
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["summary"]["verdict"] in {"pass", "warn", "fail"}
    assert report["sampled_candidates"][0]["target"] == "wiki"
    assert "ocbrain Eval Report" in output_md.read_text(encoding="utf-8")


def test_eval_fail_on_leak_returns_nonzero(tmp_path: Path) -> None:
    db_path = tmp_path / "ocbrain.sqlite"
    seed_candidate(
        db_path,
        Candidate(
            target=Target.POLICY,
            title="Never leak secrets",
            body="Never store sk-abcdefghijklmnopqrstuvwxyz123456 in output.",
            confidence=0.9,
            risk=Risk.HIGH,
            evidence=[
                Evidence(
                    uri="/tmp/source.md",
                    excerpt="Never store sk-abcdefghijklmnopqrstuvwxyz123456 in output.",
                )
            ],
            hints=["patch-suggestion-only"],
        ),
    )

    assert cli.main(["--db", str(db_path), "eval", "--sample-size", "1", "--fail-on-leak"]) == 1

