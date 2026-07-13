"""v0.4.1 fail-closed hosted-inference and training defaults."""

from __future__ import annotations

import json

from ocbrain.config import load_config
from ocbrain_ops.cli import main as ops_main
from ocbrain_training.cli import main as training_main


def test_hosted_and_training_lanes_default_off(tmp_path) -> None:
    cfg = load_config(tmp_path / "missing.json")

    assert cfg.judge.enabled is False
    assert cfg.embed.enabled is False
    assert cfg.teacher.enabled is False
    assert cfg.dataset.training_enabled is False


def test_authority_boundaries_require_explicit_opt_in(tmp_path) -> None:
    cfg = load_config(
        tmp_path / "missing.json",
        env={
            "OCBRAIN_JUDGE_ENABLED": "true",
            "OCBRAIN_EMBED_ENABLED": "true",
            "OCBRAIN_TEACHER_ENABLED": "true",
            "OCBRAIN_DATASET_TRAINING_ENABLED": "true",
        },
    )

    assert cfg.judge.enabled is True
    assert cfg.embed.enabled is True
    assert cfg.teacher.enabled is True
    assert cfg.dataset.training_enabled is True


def test_teacher_cli_fails_closed_before_opening_db(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("OCBRAIN_CONFIG", str(tmp_path / "missing.json"))
    legacy_path = tmp_path / "legacy.sqlite"
    ops_path = tmp_path / "ops.sqlite"

    assert (
        ops_main(
            [
                "--ops-db",
                str(ops_path),
                "--legacy-db",
                str(legacy_path),
                "event-teacher-request",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "blocked"
    assert result["call_performed"] is False
    assert result["reason"] == "hosted_teacher_disabled_by_default"
    assert not legacy_path.exists()


def test_pilot_cli_fails_closed_before_opening_db(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("OCBRAIN_CONFIG", str(tmp_path / "missing.json"))
    training_path = tmp_path / "training.sqlite"

    assert (
        training_main(
            ["--training-db", str(training_path), "dataset-pilot-prepare"]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "blocked"
    assert result["reason"] == "dataset_training_disabled_by_default"
    assert not training_path.exists()
