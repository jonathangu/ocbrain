from pathlib import Path

from ocbrain.classifier import classify_artifact
from ocbrain.schema import Target


def test_design_artifact_routes_to_wiki(tmp_path: Path) -> None:
    artifact = tmp_path / "brief.md"
    artifact.write_text(
        "# Brief\n\nArchitecture uses MCP, memory, wiki, and skills.\n", encoding="utf-8"
    )

    candidates = classify_artifact(artifact)

    wiki = next(candidate for candidate in candidates if candidate.target == Target.WIKI)
    assert "Architecture uses MCP" in wiki.body
    assert wiki.claim_key


def test_empty_noise_routes_to_ignore(tmp_path: Path) -> None:
    artifact = tmp_path / "note.md"
    artifact.write_text("# Note\n\nhello\n", encoding="utf-8")

    candidates = classify_artifact(artifact)

    assert [candidate.target for candidate in candidates] == [Target.IGNORE]


def test_policy_language_is_high_risk(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.md"
    artifact.write_text("# Rule\n\nNever auto-apply policy or hooks.\n", encoding="utf-8")

    candidates = classify_artifact(artifact)

    policy = next(candidate for candidate in candidates if candidate.target == Target.POLICY)
    assert policy.risk == "high"
    assert "Never auto-apply policy or hooks" in policy.body


def test_frontmatter_is_not_used_as_claim(tmp_path: Path) -> None:
    artifact = tmp_path / "brief.md"
    artifact.write_text(
        "---\n"
        "status: ok\n"
        "pageType: source\n"
        "- openclaw home: `/Users/guclaw/.openclaw`\n"
        "---\n"
        "# Brief\n\n"
        "Architecture uses MCP as the shared access layer.\n",
        encoding="utf-8",
    )

    candidates = classify_artifact(artifact)

    wiki = next(candidate for candidate in candidates if candidate.target == Target.WIKI)
    assert "Architecture uses MCP" in wiki.body
    assert "source: ---" not in wiki.body
