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


def _cases(path: Path) -> None:
    rows = [
        {
            "eval_id": f"cal-{index}",
            "expected_winner": "a" if index % 2 else "b",
        }
        for index in range(6)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_calibration_gate_records_aggregate_and_passes(tmp_path, monkeypatch):
    module = _module()
    cases = tmp_path / "cases.jsonl"
    report = tmp_path / "report.json"
    _cases(cases)
    monkeypatch.setattr(
        module,
        "_rate",
        lambda endpoint, model, pair, timeout: {
            "winner": pair["expected_winner"],
        },
    )

    result = module._calibrate(
        "http://127.0.0.1:11434/api/chat",
        "local",
        cases,
        timeout=1,
        minimum_accuracy=0.8,
        report_path=report,
    )
    assert result["passed"] is True
    assert result["accuracy"] == 1.0
    assert json.loads(report.read_text())["correct"] == 6


def test_calibration_gate_fails_before_blind_scoring(tmp_path, monkeypatch):
    module = _module()
    cases = tmp_path / "cases.jsonl"
    report = tmp_path / "report.json"
    _cases(cases)
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
            timeout=1,
            minimum_accuracy=0.8,
            report_path=report,
        )
    assert json.loads(report.read_text())["passed"] is False
