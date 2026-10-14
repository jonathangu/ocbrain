import pytest

from ocbrain.fsutil import history_runtime


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/Users/x/.claude-jgu/projects/p/s.jsonl", "claude"),
        ("/Users/x/.claude/projects/p/s.jsonl", "claude"),
        ("/Users/x/.codex/projects/s.jsonl", "codex"),
        ("/Users/x/.hermes/sessions/s.jsonl", "hermes"),
        ("/Users/x/.cursor/sessions/s.jsonl", "cursor"),
        ("/Users/x/.openclaw/agents/s.jsonl", "openclaw"),
        ("/Users/x/elsewhere/s.jsonl", "unknown"),
    ],
)
def test_history_runtime(path, expected):
    from pathlib import Path

    assert history_runtime(Path(path)) == expected
