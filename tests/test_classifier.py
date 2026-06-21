from pathlib import Path

from ocbrain.classifier import classify_artifact
from ocbrain.schema import Target


def test_design_artifact_routes_to_wiki(tmp_path: Path) -> None:
    artifact = tmp_path / "brief.md"
    artifact.write_text("# Brief\n\nArchitecture uses MCP, memory, wiki, and skills.\n", encoding="utf-8")

    candidates = classify_artifact(artifact)

    assert any(candidate.target == Target.WIKI for candidate in candidates)


def test_empty_noise_routes_to_ignore(tmp_path: Path) -> None:
    artifact = tmp_path / "note.md"
    artifact.write_text("# Note\n\nhello\n", encoding="utf-8")

    candidates = classify_artifact(artifact)

    assert [candidate.target for candidate in candidates] == [Target.IGNORE]
