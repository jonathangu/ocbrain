from pathlib import Path

from ocbrain.classifier import classify_artifact, classify_text
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


def test_classifier_ignores_structural_json_envelopes() -> None:
    candidates = classify_text(
        (
            '{"type":"session","version":3,"timestamp":"2026-05-02T11:49:37.489Z",'
            '"cwd":"/Users/guclaw/.openclaw/workspace"}\n'
            '{"role":"assistant","content":"OpenClawBrain runtime hook registered"}\n'
        ),
        evidence=[],
        title_hint="session.jsonl",
        source_type="session",
    )

    assert [candidate.target for candidate in candidates] == [Target.IGNORE]


def test_first_meaningful_evidence_skips_log_braces(tmp_path: Path) -> None:
    artifact = tmp_path / "proof.log"
    artifact.write_text(
        "{\n"
        '"status":"ok"\n'
        "Architecture uses MCP search for compact reviewed context.\n",
        encoding="utf-8",
    )

    candidates = classify_artifact(artifact)

    wiki = next(candidate for candidate in candidates if candidate.target == Target.WIKI)
    assert "Architecture uses MCP search" in wiki.body
    assert wiki.claim_key == "wiki architecture uses mcp search for compact reviewed context"


def test_first_meaningful_evidence_skips_json_install_envelope(tmp_path: Path) -> None:
    artifact = tmp_path / "install.log"
    artifact.write_text(
        "{\n"
        '"command": "install",\n'
        '"openclawHome": "/Users/guclaw/.openclaw",\n'
        '"detail": "Verified installed version 0.4.42 through the canonical converge path."\n',
        encoding="utf-8",
    )

    candidates = classify_artifact(artifact)

    memory = next(candidate for candidate in candidates if candidate.target == Target.MEMORY)
    assert "Verified installed version 0.4.42" in memory.body
    assert 'source: "command": "install"' not in memory.body


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


def test_status_log_boilerplate_is_not_used_as_claim(tmp_path: Path) -> None:
    artifact = tmp_path / "status.log"
    artifact.write_text(
        "STATUS ok\n"
        "lifecycle attach=attached learner=yes watch=watching\n"
        "surface daemonSource=managed_service hook=openclawbrain@0.4.44 version=0.4.44\n",
        encoding="utf-8",
    )

    candidates = classify_artifact(artifact)

    memory = next(candidate for candidate in candidates if candidate.target == Target.MEMORY)
    assert "lifecycle attach=attached" in memory.body
    assert memory.title == (
        "Operational fact candidate: lifecycle attach=attached learner=yes watch=watching"
    )
    assert "source: STATUS ok" not in memory.body


def test_plugin_inspect_boilerplate_is_not_used_as_claim(tmp_path: Path) -> None:
    artifact = tmp_path / "plugin.log"
    artifact.write_text(
        "OpenClawBrain\n"
        "id: openclawbrain\n"
        "Learned memory and context from OpenClawBrain\n"
        "Status: loaded\n"
        "Version: 0.4.40\n"
        "Typed hooks:\n"
        "before_prompt_build (priority 5)\n",
        encoding="utf-8",
    )

    candidates = classify_artifact(artifact)

    memory = next(candidate for candidate in candidates if candidate.target == Target.MEMORY)
    assert "Version: 0.4.40" in memory.body
    assert memory.title == "Operational fact candidate: Version: 0.4.40"
    assert "source: OpenClawBrain" not in memory.body
    assert "source: Format: openclaw" not in memory.body


def test_candidate_title_uses_evidence_claim_not_document_header(tmp_path: Path) -> None:
    artifact = tmp_path / "proof.md"
    artifact.write_text(
        "# Operator Proof\n\n"
        "STATUS ok\n"
        "Architecture uses MCP search for compact reviewed context.\n",
        encoding="utf-8",
    )

    candidates = classify_artifact(artifact)

    wiki = next(candidate for candidate in candidates if candidate.target == Target.WIKI)
    assert wiki.title == (
        "Wiki synthesis: Architecture uses MCP search for compact reviewed context."
    )
