from __future__ import annotations

import json

from ocbrain.dataset.calibration import import_calibrations
from ocbrain.db import connect, init_db


def test_calibration_import_distinguishes_complete_human_ideal(tmp_path):
    conn = connect(tmp_path / "calibration.sqlite")
    init_db(conn)
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "eval_id": "voice_1",
                        "winner": "b",
                        "reason": "B is direct but needs the reason.",
                        "ideal_response": "Choose B because it preserves the useful distinction.",
                        "ideal_response_source": "human",
                        "labeled_by": "human:operator",
                    }
                ),
                json.dumps(
                    {
                        "eval_id": "voice_2",
                        "winner": "a",
                        "critique": "A is closer, but neither is ideal.",
                    }
                ),
            ]
        )
        + "\n"
    )

    result = import_calibrations(conn, path)
    assert result["complete"] == 1
    assert result["incomplete"] == 1
    assert result["contains_calibration_text"] is False
    statuses = {
        row["eval_id"]: row["status"]
        for row in conn.execute("SELECT eval_id, status FROM dataset_calibrations")
    }
    assert statuses == {"voice_1": "complete", "voice_2": "incomplete"}
