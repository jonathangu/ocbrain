import json
from pathlib import Path

from ocbrain import cli
from ocbrain.db import connect, init_db
from ocbrain.loops import LoopIngestOptions, dry_run_loop_ingest


def write_result(
    root: Path,
    item_id: str,
    *,
    decision: str = "kept",
    family: str = "typecheck_narrowing",
    delta: int = -8,
    artifact_uri: str = "diff.patch",
) -> Path:
    item_dir = root / item_id
    item_dir.mkdir(parents=True)
    (item_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    result_path = item_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "ocbrain.loop_result.v1",
                "loop_id": "repo-quality-loop",
                "run_id": "2026-06-23-nightly",
                "item_id": item_id,
                "worker_session_uri": "~/.openclaw/agents/session.jsonl",
                "project": "ocbrain",
                "objective": "Improve repo quality with verifier evidence.",
                "hypothesis": "Narrowing fixes avoidable type errors.",
                "mechanism": "The branch narrows too late.",
                "experiment_family": family,
                "changed_files": ["src/parser.ts"],
                "artifact_uris": [artifact_uri],
                "eval": {
                    "command": "npm run typecheck",
                    "metric_name": "typecheck_errors",
                    "direction": "lower_is_better",
                    "baseline_value": 17,
                    "result_value": 9 if delta < 0 else 17,
                    "delta_value": delta,
                    "passed": True,
                },
                "guardrails": [{"name": "tests", "command": "npm test", "passed": True}],
                "verifier": {
                    "command": "python loops/scripts/verify_result.py --item exp_001",
                    "passed": True,
                    "evidence_uri": "verifier.json",
                },
                "decision": decision,
                "failure_reason": None,
                "lesson_candidates": [
                    {
                        "target": "memory",
                        "body": "typecheck_narrowing reduced typecheck errors while tests passed.",
                    }
                ],
                "next_candidates": ["Apply narrowing inspection to adjacent modules."],
                "safety": {
                    "tool_profile": "coding",
                    "approval_gates_crossed": [],
                    "blocked_actions_attempted": [],
                },
                "hashes": {"result_hash": "sha256:placeholder"},
                "created_at": "2026-06-23T06:30:00-07:00",
            }
        ),
        encoding="utf-8",
    )
    return result_path


def test_loop_ingest_reconstructs_counts_and_candidates(tmp_path: Path) -> None:
    artifacts = tmp_path / "loops" / "artifacts" / "repo-quality-loop" / "2026-06-23-nightly"
    write_result(artifacts, "exp_001", decision="kept", delta=-8)
    write_result(artifacts, "exp_002", decision="kept", delta=-4)
    write_result(artifacts, "exp_003", decision="reverted", delta=0)

    result = dry_run_loop_ingest(
        LoopIngestOptions(
            loop_id="repo-quality-loop",
            run_id="2026-06-23-nightly",
            artifacts_root=artifacts,
        )
    )

    assert result["run_status"] == "completed"
    assert result["summary"]["items"] == 3
    assert result["summary"]["kept"] == 2
    assert result["summary"]["reverted"] == 1
    assert result["metrics"]["primary"]["name"] == "typecheck_errors"
    assert result["metrics"]["primary"]["best"] == 9
    assert result["experiment_families"][0]["status"] == "promising"
    assert any(candidate["target"] == "skill" for candidate in result["candidates"]) is False
    assert any("typecheck_narrowing" in candidate["body"] for candidate in result["candidates"])


def test_loop_ingest_reports_missing_artifact_tripwire(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    write_result(artifacts, "exp_001", artifact_uri="missing-eval.json")

    result = dry_run_loop_ingest(
        LoopIngestOptions(
            loop_id="repo-quality-loop",
            run_id="2026-06-23-nightly",
            artifacts_root=artifacts,
        )
    )

    assert result["tripwires"]
    assert result["tripwires"][0]["kind"] == "artifact_missing"
    assert "missing-eval.json" in result["tripwires"][0]["message"]


def test_loop_ingest_proposes_skill_after_repeated_success(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    write_result(artifacts, "exp_001", decision="kept", delta=-8)
    write_result(artifacts, "exp_002", decision="kept", delta=-4)
    write_result(artifacts, "exp_003", decision="kept", delta=-2)

    result = dry_run_loop_ingest(
        LoopIngestOptions(
            loop_id="repo-quality-loop",
            run_id="2026-06-23-nightly",
            artifacts_root=artifacts,
        )
    )

    skill_candidates = [
        candidate for candidate in result["candidates"] if candidate["target"] == "skill"
    ]
    assert skill_candidates
    assert skill_candidates[0]["status"] == "proposal_only"
    assert "typecheck_narrowing" in skill_candidates[0]["body"]


def test_loop_ingest_rejects_invalid_envelope(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    item_dir = artifacts / "exp_001"
    item_dir.mkdir(parents=True)
    (item_dir / "result.json").write_text(
        json.dumps({"schema_version": "wrong", "loop_id": "other"}),
        encoding="utf-8",
    )

    result = dry_run_loop_ingest(
        LoopIngestOptions(
            loop_id="repo-quality-loop",
            run_id="2026-06-23-nightly",
            artifacts_root=artifacts,
        )
    )

    assert result["run_status"] == "needs_review"
    assert result["envelopes"]["valid"] == 0
    assert result["envelopes"]["invalid"] >= 1


def test_loop_ingest_cli_is_dry_run_and_does_not_create_db(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "artifacts"
    write_result(artifacts, "exp_001")
    db_path = tmp_path / "ocbrain.sqlite"

    assert (
        cli.main(
            [
                "--db",
                str(db_path),
                "loop-ingest",
                "--loop-id",
                "repo-quality-loop",
                "--run-id",
                "2026-06-23-nightly",
                "--artifacts",
                str(artifacts),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    assert payload["summary"]["kept"] == 1
    assert not db_path.exists()


def test_loop_ingest_apply_is_idempotent(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "artifacts"
    write_result(artifacts, "exp_001")
    write_result(artifacts, "exp_002", artifact_uri="missing-eval.json")
    db_path = tmp_path / "ocbrain.sqlite"
    command = [
        "--db",
        str(db_path),
        "loop-ingest",
        "--loop-id",
        "repo-quality-loop",
        "--run-id",
        "2026-06-23-nightly",
        "--artifacts",
        str(artifacts),
        "--apply",
        "--json",
    ]

    assert cli.main(command) == 0
    first_payload = json.loads(capsys.readouterr().out)
    assert cli.main(command) == 0
    second_payload = json.loads(capsys.readouterr().out)

    conn = connect(db_path)
    assert first_payload["applied"]["items"] == 2
    assert second_payload["applied"]["items"] == 2
    assert conn.execute("SELECT COUNT(*) FROM loop_runs").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM loop_items").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM loop_iterations").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM loop_metrics").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM loop_artifacts").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM loop_tripwires").fetchone()[0] == 1


def test_loop_tables_exist_after_init(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'loop_%'"
        )
    }

    assert {
        "loop_programs",
        "loop_runs",
        "loop_items",
        "loop_iterations",
        "loop_metrics",
        "loop_artifacts",
        "loop_tripwires",
        "loop_candidate_links",
    } <= tables
