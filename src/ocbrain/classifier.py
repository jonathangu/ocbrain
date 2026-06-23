from __future__ import annotations

import re
from pathlib import Path

from ocbrain.db import EventInput
from ocbrain.ingest import IngestOptions, event_from_file
from ocbrain.schema import Candidate, Evidence, Risk, Target
from ocbrain.text import claim_key, compact_whitespace, title_from_text


def classify_artifact(path: Path) -> list[Candidate]:
    """Return deterministic candidates from a redacted artifact envelope.

    This is intentionally conservative. The first real implementation can swap this
    heuristic pass for an LLM-backed classifier while keeping the output contract.
    """
    event = event_from_file(path, IngestOptions())
    if event is None:
        return []
    return classify_event(event)


def classify_event(event: EventInput) -> list[Candidate]:
    evidence = _first_meaningful_evidence(Path(event.source_uri), event.body.splitlines())
    return classify_text(
        event.body,
        evidence=evidence,
        title_hint=event.title,
        source_type=event.source_type,
    )


def classify_text(
    text: str,
    *,
    evidence: list[Evidence],
    title_hint: str = "artifact",
    source_type: str | None = None,
) -> list[Candidate]:
    source_title = title_from_text(text, title_hint)
    candidates: list[Candidate] = []
    claim = claim_from_evidence(evidence, text)
    if not claim or is_low_value_claim(claim):
        return [
            Candidate(
                target=Target.IGNORE,
                title=f"No durable candidate: {source_title}",
                body="No strong memory/wiki/skill/policy candidate was detected.",
                confidence=0.5,
                evidence=evidence,
                claim_key=claim_key(f"ignore {claim}"),
            )
        ]

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
                title=candidate_title("Wiki synthesis", claim, source_title),
                body=f"Draft wiki synthesis from source: {claim}",
                confidence=0.72,
                evidence=evidence,
                hints=["draft-only", "preserve provenance"],
                claim_key=claim_key(f"wiki {claim}"),
            )
        )

    if any(token in lower for token in ("version", "installed", "latest", "checked")):
        candidates.append(
            Candidate(
                target=Target.MEMORY,
                title=candidate_title("Operational fact candidate", claim, source_title),
                body=f"Stage operational fact from source: {claim}",
                confidence=0.68,
                evidence=evidence,
                claim_key=claim_key(f"memory {claim}"),
            )
        )

    if any(
        token in lower
        for token in ("repeatable", "workflow", "procedure", "skill proposal", "runbook")
    ):
        candidates.append(
            Candidate(
                target=Target.SKILL,
                title=candidate_title("Skill proposal candidate", claim, source_title),
                body=f"Draft repeatable workflow from source: {claim}",
                confidence=0.62,
                risk=Risk.MEDIUM,
                evidence=evidence,
                hints=["proposal-first", "do-not-auto-apply"],
                claim_key=claim_key(f"skill {claim}"),
            )
        )

    if any(
        token in lower for token in ("must", "never", "policy", "constraint", "hook", "forbid")
    ):
        candidates.append(
            Candidate(
                target=Target.POLICY,
                title=candidate_title("Constraint candidate", claim, source_title),
                body=f"Patch-suggestion constraint from source: {claim}",
                confidence=0.55,
                risk=Risk.HIGH,
                evidence=evidence,
                hints=["patch-suggestion-only"],
                claim_key=claim_key(f"policy {claim}"),
            )
        )

    if not candidates:
        candidates.append(
            Candidate(
                target=Target.IGNORE,
                title=f"No durable candidate: {source_title}",
                body="No strong memory/wiki/skill/policy candidate was detected.",
                confidence=0.5,
                evidence=evidence,
                claim_key=claim_key(f"ignore {claim}"),
            )
        )

    if source_type == "session":
        for candidate in candidates:
            candidate.hints.append("session-derived")

    return candidates


def candidate_title(prefix: str, claim: str, fallback: str) -> str:
    return f"{prefix}: {title_from_claim(claim, fallback)}"


def title_from_claim(claim: str, fallback: str) -> str:
    normalized = compact_whitespace(claim)
    if not normalized:
        return fallback[:160]

    normalized = normalized.lstrip("-* ").strip()
    metadata_match = re.fullmatch(r"(?i)(title|summary|task|objective)\s*:\s*(.+)", normalized)
    if metadata_match:
        normalized = metadata_match.group(2).strip()
    normalized = normalized.strip("\"'` ")

    sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    words = sentence.split()
    if len(words) > 14:
        sentence = " ".join(words[:14])
    return sentence[:120] or fallback[:160]


def claim_from_evidence(evidence: list[Evidence], text: str) -> str:
    for item in evidence:
        excerpt = compact_whitespace(item.excerpt)
        if excerpt:
            return excerpt[:320]
    return compact_whitespace(text)[:320]


def _first_meaningful_evidence(path: Path, lines: list[str]) -> list[Evidence]:
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not is_low_value_evidence_line(stripped):
            return [
                Evidence(uri=str(path), excerpt=stripped[:500], line_start=index, line_end=index)
            ]
    return [Evidence(uri=str(path), excerpt="", line_start=None, line_end=None)]


def is_low_value_evidence_line(stripped: str) -> bool:
    lowered = stripped.lower()
    if stripped in {"---", "```", "{", "}", "[", "]"}:
        return True
    if stripped.startswith(("{", "[", "}", "]")) and (
        '"type"' in stripped or '"version"' in stripped or '"timestamp"' in stripped
    ):
        return True
    if len(stripped) <= 2:
        return True
    if re.fullmatch(r"[-*]?\s*date:\s*\d{4}-\d{2}-\d{2}.*", lowered):
        return True
    if lowered.startswith(("- **session key**", "- session key", "session key")):
        return True
    if lowered.startswith(("- status:", "status:", "- source:", "source:")):
        return True
    if lowered.startswith('"status":') and "ok" in lowered:
        return True
    if lowered in {"status ok", "openclawbrain"}:
        return True
    if lowered.startswith(
        (
            "id:",
            "learned memory and context from openclawbrain",
            "format:",
            "origin:",
            "shape:",
            "capability mode:",
            "legacy before_agent_start:",
            "typed hooks:",
            "services:",
            "install:",
            "owner:",
            "- owner:",
        )
    ):
        return True
    if lowered.startswith(("- bundle verdict:", "bundle verdict:", "- severity:", "severity:")):
        return True
    if re.fullmatch(
        r'"(command|openclawhome|activationroot|plugin|attach|restart)"\s*:.*',
        lowered,
    ):
        return True
    if re.fullmatch(r'"(action|changed|changereasons|required|capture)"\s*:.*', lowered):
        return True
    if lowered.startswith(("pagetype:", "- openclaw home:")):
        return True
    if "brain loaded: runtime hook registered" in lowered:
        return True
    return lowered in {"- status: `ok`", "status: ok"}


def is_low_value_claim(claim: str) -> bool:
    lowered = compact_whitespace(claim).lower()
    if not lowered:
        return True
    if is_low_value_evidence_line(lowered):
        return True
    if lowered in {"{", "}", "[", "]", "```", "---"}:
        return True
    if lowered.startswith(("{", "[")) and (
        '"type"' in lowered or '"version"' in lowered or '"timestamp"' in lowered
    ):
        return True
    return False
