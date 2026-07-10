from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "grade-pilot-blind.py"
    spec = importlib.util.spec_from_file_location("grade_pilot_blind", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cases(path: Path) -> dict[str, str]:
    winners = {f"cal-{index}": "a" if index % 2 else "b" for index in range(6)}
    rows = [
        {
            "eval_id": f"cal-{index}",
            # Embedded machine-authored expectations are deliberately ignored.
            "expected_winner": "b" if winners[f"cal-{index}"] == "a" else "a",
        }
        for index in range(6)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return winners


def _labels(path: Path, winners: dict[str, str]) -> None:
    rows = [
        {"eval_id": eval_id, "winner": winner, "labeled_by": "Human Operator"}
        for eval_id, winner in winners.items()
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_calibration_gate_records_aggregate_and_passes(tmp_path, monkeypatch):
    module = _module()
    cases = tmp_path / "cases.jsonl"
    labels = tmp_path / "labels.jsonl"
    report = tmp_path / "report.json"
    winners = _cases(cases)
    _labels(labels, winners)
    monkeypatch.setattr(
        module,
        "_rate",
        lambda endpoint, model, pair, timeout: {
            "winner": winners[pair["eval_id"]],
        },
    )

    result = module._calibrate(
        "http://127.0.0.1:11434/api/chat",
        "local",
        cases,
        labels,
        timeout=1,
        minimum_accuracy=0.8,
        report_path=report,
    )
    assert result["passed"] is True
    assert result["accuracy"] == 1.0
    assert result["human_labeled"] is True
    assert result["results"][0]["human_winner"] == winners["cal-0"]
    assert json.loads(report.read_text())["correct"] == 6


def test_calibration_gate_fails_before_blind_scoring(tmp_path, monkeypatch):
    module = _module()
    cases = tmp_path / "cases.jsonl"
    labels = tmp_path / "labels.jsonl"
    report = tmp_path / "report.json"
    winners = _cases(cases)
    _labels(labels, winners)
    monkeypatch.setattr(
        module,
        "_rate",
        lambda endpoint, model, pair, timeout: {"winner": "tie"},
    )

    with pytest.raises(RuntimeError, match="calibration failed"):
        module._calibrate(
            "http://127.0.0.1:11434/api/chat",
            "local",
            cases,
            labels,
            timeout=1,
            minimum_accuracy=0.8,
            report_path=report,
        )
    assert json.loads(report.read_text())["passed"] is False


def test_calibration_requires_complete_human_provenance(tmp_path):
    module = _module()
    cases = tmp_path / "cases.jsonl"
    labels = tmp_path / "labels.jsonl"
    winners = _cases(cases)
    rows = [{"eval_id": eval_id, "winner": winner} for eval_id, winner in winners.items()]
    labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="labeled_by provenance"):
        module._calibrate(
            "http://127.0.0.1:11434/api/chat",
            "local",
            cases,
            labels,
            timeout=1,
            minimum_accuracy=0.8,
            report_path=tmp_path / "report.json",
        )
