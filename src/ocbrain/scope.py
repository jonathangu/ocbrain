from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCOPE_TYPES = {
    "global",
    "project",
    "repo",
    "client",
    "personal_finance",
    "task",
    "session",
    "legacy_unscoped",
}
VISIBILITIES = {"public", "internal", "confidential", "secret"}
EGRESS_POLICIES = {"hosted_ok", "local_only", "approval_required", "prohibited"}

LOCAL_MODEL_TARGET = "local_model"
HOSTED_MODEL_TARGET = "hosted_model"
DELIVERY_TARGETS = {LOCAL_MODEL_TARGET, HOSTED_MODEL_TARGET}

DEFAULT_GLOBAL_SCOPE_ID = "global:doctrine"


@dataclass(frozen=True)
class ScopeTag:
    scope_type: str
    scope_id: str
    visibility: str = "internal"
    egress_policy: str = "local_only"
    provenance: str = "explicit"

    def __post_init__(self) -> None:
        if self.scope_type not in SCOPE_TYPES:
            raise ValueError(f"invalid scope_type: {self.scope_type}")
        if not self.scope_id:
            raise ValueError("scope_id is required")
        if self.visibility not in VISIBILITIES:
            raise ValueError(f"invalid visibility: {self.visibility}")
        if self.egress_policy not in EGRESS_POLICIES:
            raise ValueError(f"invalid egress_policy: {self.egress_policy}")

    @property
    def confidential(self) -> bool:
        return self.visibility in {"confidential", "secret"} or self.scope_type == "client"

    @property
    def hosted_egress_allowed(self) -> bool:
        return self.egress_policy == "hosted_ok" and not self.confidential

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ScopeTag:
        if not data:
            return legacy_unscoped_scope()
        return cls(
            scope_type=str(data.get("scope_type") or data.get("tier") or "legacy_unscoped"),
            scope_id=str(data.get("scope_id") or inferred_scope_id(data)),
            visibility=str(data.get("visibility") or default_visibility(data)),
            egress_policy=str(data.get("egress_policy") or default_egress_policy(data)),
            provenance=str(data.get("provenance") or "explicit"),
        )


@dataclass(frozen=True)
class ScopeContext:
    project: str | None = None
    repo: str | None = None
    client: str | None = None
    task: str | None = None
    session: str | None = None
    runtime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ScopeContext:
        if not data:
            return cls()
        return cls(
            project=optional_str(data.get("project")),
            repo=optional_str(data.get("repo")),
            client=optional_str(data.get("client")),
            task=optional_str(data.get("task")),
            session=optional_str(data.get("session")),
            runtime=optional_str(data.get("runtime")),
        )

    def compatible_scope_ids(self) -> set[str]:
        ids: set[str] = {DEFAULT_GLOBAL_SCOPE_ID}
        if self.project:
            ids.add(f"project:{self.project}")
        if self.repo:
            ids.add(f"repo:{self.repo}")
        if self.client:
            ids.add(f"client:{self.client}")
        if self.task:
            ids.add(f"task:{self.task}")
        if self.session:
            ids.add(f"session:{self.session}")
        return ids


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def global_scope(*, hosted_ok: bool = True) -> ScopeTag:
    return ScopeTag(
        scope_type="global",
        scope_id=DEFAULT_GLOBAL_SCOPE_ID,
        visibility="internal",
        egress_policy="hosted_ok" if hosted_ok else "local_only",
        provenance="explicit",
    )


def legacy_unscoped_scope() -> ScopeTag:
    return ScopeTag(
        scope_type="legacy_unscoped",
        scope_id="legacy:unscoped",
        visibility="internal",
        egress_policy="local_only",
        provenance="quarantined",
    )


def resolve_write_scope(
    context: ScopeContext | None = None,
    explicit: ScopeTag | dict[str, Any] | None = None,
) -> ScopeTag:
    if isinstance(explicit, ScopeTag):
        return explicit
    if isinstance(explicit, dict):
        return ScopeTag.from_dict(explicit)
    context = context or ScopeContext()
    confidential = context.client is not None

    def inferred(scope_type: str, scope_id: str) -> ScopeTag:
        return ScopeTag(
            scope_type,
            scope_id,
            visibility="confidential" if confidential else "internal",
            egress_policy="local_only" if confidential else "approval_required",
            provenance="inferred",
        )

    if context.task:
        return inferred("task", f"task:{context.task}")
    if context.session:
        return inferred("session", f"session:{context.session}")
    if context.repo:
        return inferred("repo", f"repo:{context.repo}")
    if context.client:
        return ScopeTag(
            "client",
            f"client:{context.client}",
            visibility="confidential",
            egress_policy="local_only",
            provenance="inferred",
        )
    if context.project:
        return inferred("project", f"project:{context.project}")
    return legacy_unscoped_scope()


def scope_match(
    scope: ScopeTag,
    context: ScopeContext | None = None,
    *,
    cross_scope: bool = False,
) -> float:
    context = context or ScopeContext()
    if scope.scope_type == "global":
        return 1.0
    if scope.scope_type == "legacy_unscoped":
        return 0.05 if cross_scope else 0.0
    if scope.scope_id in context.compatible_scope_ids():
        return 1.25
    if scope.confidential:
        return 0.0
    return 0.15 if cross_scope else 0.0


def normalize_delivery_target(
    target: str | None,
    *,
    default: str = LOCAL_MODEL_TARGET,
) -> str:
    resolved = default if target is None else target
    if not isinstance(resolved, str) or resolved not in DELIVERY_TARGETS:
        allowed = ", ".join(sorted(DELIVERY_TARGETS))
        raise ValueError(f"delivery_target must be one of: {allowed}")
    return resolved


def egress_allowed(
    scope: ScopeTag,
    context: ScopeContext,
    target: str,
    *,
    cross_scope: bool = False,
) -> tuple[bool, str]:
    match = scope_match(scope, context, cross_scope=cross_scope)
    if match == 0:
        return False, "scope_mismatch"
    if target == LOCAL_MODEL_TARGET:
        return scope.egress_policy != "prohibited", "allowed_local"
    if target in {HOSTED_MODEL_TARGET, "hosted_teacher"}:
        if scope.hosted_egress_allowed:
            return True, "allowed_hosted"
        return False, f"egress_policy:{scope.egress_policy};visibility:{scope.visibility}"
    if target == "human_export":
        if scope.egress_policy in {"hosted_ok", "approval_required"}:
            return True, "allowed_export"
        return False, f"egress_policy:{scope.egress_policy}"
    return False, f"unknown_target:{target}"


def inferred_scope_id(data: dict[str, Any]) -> str:
    tier = str(data.get("scope_type") or data.get("tier") or "legacy_unscoped")
    project = data.get("project") or data.get("scope_project")
    if tier == "global":
        return DEFAULT_GLOBAL_SCOPE_ID
    if project:
        return f"{tier}:{project}"
    return "legacy:unscoped"


def default_visibility(data: dict[str, Any]) -> str:
    if (
        data.get("confidential")
        or data.get("scope_type") == "client"
        or data.get("tier") == "confidential"
    ):
        return "confidential"
    return "internal"


def default_egress_policy(data: dict[str, Any]) -> str:
    if default_visibility(data) in {"confidential", "secret"}:
        return "local_only"
    if data.get("scope_type") == "global" or data.get("tier") == "global":
        return "hosted_ok"
    return "local_only"
