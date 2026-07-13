from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Target(StrEnum):
    MEMORY = "memory"
    WIKI = "wiki"
    SKILL = "skill"
    POLICY = "policy"
    IGNORE = "ignore"


class Scope(StrEnum):
    PRIVATE = "private"
    WORKSPACE = "workspace"
    PROJECT = "project"
    PUBLIC = "public"


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Evidence:
    uri: str
    excerpt: str
    line_start: int | None = None
    line_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    target: Target
    title: str
    body: str
    confidence: float
    scope: Scope = Scope.WORKSPACE
    risk: Risk = Risk.LOW
    evidence: list[Evidence] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    claim_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = self.target.value
        data["scope"] = self.scope.value
        data["risk"] = self.risk.value
        return data
