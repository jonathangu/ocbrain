"""ocbrain v0.2 configuration surface.

One config module for every v0.2 tunable (spec §3, resolution R1). The public
entry point is :func:`load_config`, which layers, in order:

1. hard-coded defaults (the section dataclasses below),
2. an optional JSON file at ``$OCBRAIN_CONFIG`` (default ``data/ocbrain.config.json``),
3. ``OCBRAIN_<SECTION>_<FIELD>`` environment overrides.

``DatasetConfig`` is the ``dataset`` section here — there is deliberately no
separate ``dataset/config.py`` (R1). The single shared ``correction.threshold``
key (R1/R2) lives on :class:`CorrectionConfig`.

Secrets are never stored: ``JudgeConfig.api_key_env`` holds the *name* of an
environment variable, never its value.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("OCBRAIN_CONFIG", "data/ocbrain.config.json")
).expanduser()


@dataclass(frozen=True)
class AutopilotConfig:
    lock_path: str = "data/autopilot.lock"
    snapshot_dir: str = "data/snapshots/"
    snapshot_keep: int = 3
    stage_budget_seconds: int = 300
    runtimes_excerpt: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewConfig:
    settle_minutes: int = 30
    min_tool_calls_success: int = 5
    session_roots: list[str] = field(
        default_factory=lambda: [
            "~/.openclaw/agents",
            "~/.claude/projects",
            "~/.codex",
        ]
    )


@dataclass(frozen=True)
class CorrectionConfig:
    # Shared threshold: review's user_correction signal AND DPO pair mining (R1/R2).
    threshold: float = 0.6


@dataclass(frozen=True)
class LabelsConfig:
    half_life_days: float = 30.0
    good_threshold: float = 0.35
    bad_threshold: float = -0.35
    min_mass: float = 0.6
    hard_bad_weight: float = 0.9


@dataclass(frozen=True)
class QuarantineConfig:
    bad_feedback_count: int = 2
    bad_feedback_window_days: int = 7
    thrash_count: int = 3
    thrash_window_days: int = 14


@dataclass(frozen=True)
class PromoteConfig:
    min_confidence: float = 0.6
    max_injected: int = 40
    max_chars: int = 6000
    decay_days: int = 30
    bootstrap_min_confidence: float = 0.85


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool = True
    api_key_env: str = "OPENAI_API_KEY"  # variable NAME only; value never persisted
    model: str = "gpt-5-mini"
    daily_usd_cap: float = 0.50
    batch_size: int = 20
    per_run_item_cap: int = 100
    signal_weight: float = 0.4
    # {model: {"prompt": usd_per_mtok, "completion": usd_per_mtok}}; supplied via
    # config JSON so no price is baked into source.
    price_per_mtok: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig:
    sft_min_assistant_chars: int = 80
    sft_max_context_turns: int = 12
    sft_max_context_chars: int = 16000
    dpo_side_chars: list[int] = field(default_factory=lambda: [40, 8000])
    include_tool_turns: bool = False
    tool_result_truncate: int = 500
    persona_author_ids: list[str] = field(
        default_factory=lambda: ["8518484672", "jongugu"]
    )
    persona_direct_agents: list[str] = field(default_factory=lambda: ["main"])
    persona_git_repos: list[str] = field(default_factory=list)
    persona_git_authors: list[str] = field(
        default_factory=lambda: ["Jonathan Gu", "jonathangu@gmail.com"]
    )
    persona_authored_globs: list[str] = field(default_factory=list)
    persona_system_prompt: str = "You are Jonathan Gu. Reply as Jonathan would."
    export_dir: str = "data/datasets"
    export_min_scope: str = "workspace"
    export_min_label: str = "good"
    learning_db: str = "~/.openclaw/learning.db"
    commitments_path: str = "~/.openclaw/commitments/commitments.json"
    cron_state_path: str = "~/.openclaw/cron/jobs-state.json"


@dataclass(frozen=True)
class OcbrainConfig:
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    quarantine: QuarantineConfig = field(default_factory=QuarantineConfig)
    promote: PromoteConfig = field(default_factory=PromoteConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def _coerce(current: Any, incoming: Any) -> Any:
    """Coerce an incoming (JSON/env) value to the type of the current default."""
    if isinstance(current, bool):
        if isinstance(incoming, str):
            return incoming.strip().lower() in {"1", "true", "yes", "on"}
        return bool(incoming)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(incoming)
    if isinstance(current, float):
        return float(incoming)
    if isinstance(current, list):
        if isinstance(incoming, str):
            parsed = json.loads(incoming)
            return list(parsed) if isinstance(parsed, list) else [parsed]
        return list(incoming)
    if isinstance(current, dict):
        if isinstance(incoming, str):
            return dict(json.loads(incoming))
        return dict(incoming)
    return incoming


def _apply_section_overrides(section: Any, overrides: dict[str, Any]) -> Any:
    """Return a copy of a frozen section dataclass with ``overrides`` applied."""
    valid = {f.name for f in fields(section)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid:
            continue
        changes[key] = _coerce(getattr(section, key), value)
    return replace(section, **changes) if changes else section


def _env_overrides(section_name: str, section: Any) -> dict[str, Any]:
    """Collect ``OCBRAIN_<SECTION>_<FIELD>`` env vars for one section."""
    overrides: dict[str, Any] = {}
    for f in fields(section):
        env_key = f"OCBRAIN_{section_name.upper()}_{f.name.upper()}"
        if env_key in os.environ:
            overrides[f.name] = os.environ[env_key]
    return overrides


def load_config(
    path: Path | str | None = None, *, env: dict[str, str] | None = None
) -> OcbrainConfig:
    """Load config from defaults + optional JSON file + env overrides.

    ``path`` defaults to ``$OCBRAIN_CONFIG`` / ``data/ocbrain.config.json``. A
    missing file is fine (defaults win). ``env`` defaults to ``os.environ``.
    """
    if env is not None:
        # Temporarily consult the provided mapping for env overrides.
        saved = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(env)
            return _load_config_from_environ(path)
        finally:
            os.environ.clear()
            os.environ.update(saved)
    return _load_config_from_environ(path)


def _load_config_from_environ(path: Path | str | None) -> OcbrainConfig:
    config_path = (
        Path(path).expanduser()
        if path is not None
        else Path(
            os.environ.get("OCBRAIN_CONFIG", "data/ocbrain.config.json")
        ).expanduser()
    )
    file_data: dict[str, Any] = {}
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            file_data = loaded

    cfg = OcbrainConfig()
    section_changes: dict[str, Any] = {}
    for f in fields(cfg):
        section = getattr(cfg, f.name)
        if not is_dataclass(section):
            continue
        overrides: dict[str, Any] = {}
        from_file = file_data.get(f.name)
        if isinstance(from_file, dict):
            overrides.update(from_file)
        overrides.update(_env_overrides(f.name, section))
        if overrides:
            section_changes[f.name] = _apply_section_overrides(section, overrides)
    return replace(cfg, **section_changes) if section_changes else cfg
