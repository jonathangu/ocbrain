from __future__ import annotations

from pathlib import Path

from ocbrain.schema import Candidate, Evidence, Risk, Target
from ocbrain.text import title_from_text


def classify_artifact(path: Path) -> list[Candidate]:
    """Return deterministic dry-run candidates for a markdown artifact.

    This is intentionally conservative. The first real implementation can swap this
    heuristic pass for an LLM-backed classifier while keeping the output contract.
    """
    text = path.read_text(encoding="utf-8")
    evidence = _first_meaningful_evidence(path, text.splitlines())
    return classify_text(text, evidence=evidence, title_hint=path.name)


def classify_text(
    text: str,
    *,
    evidence: list[Evidence],
    title_hint: str = "artifact",
    source_type: str | None = None,
) -> list[Candidate]:
    title = title_from_text(text, title_hint)
    candidates: list[Candidate] = []

    lower = text.lower()
    if any(
        token in lower
        for token in (
            "architecture",
            "design",
            "mcp",
            "wiki",
            "shared brain",
            "openclawbrain",
            "consolidation",
        )
    ):
        candidates.append(
            Candidate(
                target=Target.WIKI,
                title=f"Wiki synthesis: {title}",
                body=(
                    "Artifact appears to contain stable architecture or design synthesis. "
                    "Route to wiki draft rather than long-form memory."
                ),
                confidence=0.72,
                evidence=evidence,
                hints=["draft-only", "preserve provenance"],
            )
        )

    if any(token in lower for token in ("version", "installed", "latest", "checked")):
        candidates.append(
            Candidate(
                target=Target.MEMORY,
                title=f"Operational fact candidate: {title}",
                body=(
                    "Artifact appears to include dated operational state or version facts. "
                    "Extract only concise source-backed facts."
                ),
                confidence=0.68,
                evidence=evidence,
            )
        )

    if any(
        token in lower
        for token in ("repeatable", "workflow", "procedure", "skill proposal", "runbook")
    ):
        candidates.append(
            Candidate(
                target=Target.SKILL,
                title=f"Skill proposal candidate: {title}",
                body=(
                    "Artifact appears to describe repeatable behavior. "
                    "Route through Skill Workshop as a pending proposal."
                ),
                confidence=0.62,
                risk=Risk.MEDIUM,
                evidence=evidence,
                hints=["proposal-first", "do-not-auto-apply"],
            )
        )

    if any(
        token in lower for token in ("must", "never", "policy", "constraint", "hook", "forbid")
    ):
        candidates.append(
            Candidate(
                target=Target.POLICY,
                title=f"Constraint candidate: {title}",
                body=(
                    "Artifact appears to contain constraints or enforcement language. "
                    "Create a patch suggestion only; do not auto-apply."
                ),
                confidence=0.55,
                risk=Risk.HIGH,
                evidence=evidence,
                hints=["patch-suggestion-only"],
            )
        )

    if not candidates:
        candidates.append(
            Candidate(
                target=Target.IGNORE,
                title=f"No durable candidate: {title}",
                body="No strong memory/wiki/skill/policy candidate was detected.",
                confidence=0.5,
                evidence=evidence,
            )
        )

    if source_type == "session":
        for candidate in candidates:
            candidate.hints.append("session-derived")

    return candidates


def _first_meaningful_evidence(path: Path, lines: list[str]) -> list[Evidence]:
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return [
                Evidence(uri=str(path), excerpt=stripped[:500], line_start=index, line_end=index)
            ]
    return [Evidence(uri=str(path), excerpt="", line_start=None, line_end=None)]
