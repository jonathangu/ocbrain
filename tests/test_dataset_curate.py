from __future__ import annotations

import json

import pytest

from ocbrain.dataset.curate import import_persona_curation
from ocbrain.db import connect, init_db


def test_persona_curation_imports_without_exposing_text_in_result(tmp_path):
    conn = connect(tmp_path / "brain.sqlite")
    init_db(conn)
    source = tmp_path / "voice.jsonl"
    response = (
        "Start with the decision record because it makes the system visible. "
        "Once that loop is trustworthy, the policy can become more ambitious."
    )
    source.write_text(json.dumps({"prompt": "Where should we begin?", "response": response}))

    result = import_persona_curation(conn, source)
    assert result["imported"] == 1
    assert result["local_only"] is True
    assert response not in json.dumps(result)
    row = conn.execute(
        "SELECT source_kind, grade_score, example_json FROM dataset_examples"
    ).fetchone()
    assert row["source_kind"] == "authored_doc"
    assert row["grade_score"] is None
    assert response in row["example_json"]


def test_persona_curation_rejects_probable_secrets_before_storage(tmp_path):
    conn = connect(tmp_path / "brain.sqlite")
    init_db(conn)
    source = tmp_path / "bad.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "Explain the setup.",
                "response": (
                    "Use this credential sk-abcdefghijklmnopqrstuvwxyz1234567890 "
                    "to configure the system, then continue with the remaining work."
                ),
            }
        )
    )
    with pytest.raises(ValueError, match="probable secret"):
        import_persona_curation(conn, source)
    assert conn.execute("SELECT COUNT(*) FROM dataset_examples").fetchone()[0] == 0
