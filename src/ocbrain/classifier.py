from __future__ import annotations

from pathlib import Path

from ocbrain.schema import Candidate, Evidence, Risk, Target


def classify_artifact(path: Path) -> list[Candidate]:
    """Return deterministic dry-run candidates for a markdown artifact.

    This is intentionally conservative. The first real implementation can swap this
    heuristic pass for an LLM-backed classifier while keeping the output contract.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    evidence = _first_meaningful_evidence(path, lines)
    candidates: list[Candidate] = []

    lower = text.lower()
    if any(token in lower for token in ("architecture", "design", "mcp", "wiki", "skill")):
        candidates.append(
            Candidate(
                target=Target.WIKI,
                title="Architecture synthesis candidate",
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
                title="Operational version/state facts",
                body=(
                    "Artifact appears to include dated operational state or version facts. "
                    "Extract only concise source-backed facts."
                ),
                confidence=0.68,
                evidence=evidence,
            )
        )

    if any(token in lower for token in ("repeatable", "workflow", "procedure", "skill proposal")):
        candidates.append(
            Candidate(
                target=Target.SKILL,
                title="Repeatable workflow candidate",
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

    if any(token in lower for token in ("must", "never", "policy", "constraint", "hook")):
        candidates.append(
            Candidate(
                target=Target.POLICY,
                title="Constraint candidate",
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
                title="No durable candidate found",
                body="No strong memory/wiki/skill/policy candidate was detected.",
                confidence=0.5,
                evidence=evidence,
            )
        )

    return candidates


def _first_meaningful_evidence(path: Path, lines: list[str]) -> list[Evidence]:
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return [Evidence(uri=str(path), excerpt=stripped[:500], line_start=index, line_end=index)]
    return [Evidence(uri=str(path), excerpt="", line_start=None, line_end=None)]
