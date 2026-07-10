"""ocbrain v0.3 configuration surface.

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
    # Per-stage wall-clock overrides (seconds). A stage named here uses its own
    # budget; every other budget-aware stage falls back to stage_budget_seconds.
    # e.g. {"dataset_mine": 900}. Set via config JSON / OCBRAIN_AUTOPILOT_STAGE_BUDGETS.
    stage_budgets: dict[str, int] = field(default_factory=dict)
    runtimes_excerpt: list[str] = field(default_factory=list)
    # Named stage sequences the autopilot driver can run instead of a hard-coded
    # list. ``light`` is the fast 30-min-timer-safe cycle; ``heavy`` is the full
    # fold+mine+export cycle. The driver picks a profile per run (v0.3).
    profiles: dict[str, list[str]] = field(
        default_factory=lambda: {
            "light": [
                "migrate",
                "review",
                "autolabel",
                "tripwires",
                "promote",
                "excerpt_render",
                "maintain",
            ],
            "heavy": [
                "snapshot",
                "migrate",
                "harvest",
                "injection_scan",
                "review",
                "compile",
                "autolabel",
                "tripwires",
                "promote",
                "excerpt_render",
                "maintain",
                "dataset_mine",
                "dataset_export",
            ],
        }
    )
    # Locking discipline across profiles. ``shared`` == light and heavy runs
    # contend for the same autopilot lock so they never overlap.
    profile_locks: str = "shared"
    # Reclaim a large WAL only after dataset mining has committed every bounded
    # writer batch. Small WALs are left to SQLite's normal autocheckpoint path.
    checkpoint_after_dataset_mine: bool = True
    checkpoint_wal_min_bytes: int = 64 * 1024 * 1024


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
    # One-time human-authored seeding of the injectable memory set. ``sources``
    # names the harvest origins (e.g. curated ``memory_file`` doctrine) whose
    # high-confidence rows may be bootstrapped into memory up to ``cap`` (v0.3).
    human_bootstrap: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "sources": ["memory_file"],
            "cap": 15,
        }
    )


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
    # Candidate-filtering knobs (v0.3). ``sources`` whitelists which knowledge
    # origins the judge grades; ``exclude_catalog_docs`` keeps the 101k-file
    # catalog backlog out of the graded set so spend is not wasted on it.
    targeting: dict[str, Any] = field(
        default_factory=lambda: {
            "sources": ["retrieval_touched", "lesson", "session_derived"],
            "exclude_catalog_docs": True,
        }
    )


@dataclass(frozen=True)
class DatasetConfig:
    sft_min_assistant_chars: int = 80
    sft_max_context_turns: int = 12
    sft_max_context_chars: int = 16000
    dpo_side_chars: list[int] = field(default_factory=lambda: [40, 8000])
    include_tool_turns: bool = False
    tool_result_truncate: int = 500
    # Identity-bearing persona selectors (telegram sender ids / usernames, git
    # author name+email strings, and the persona system prompt) ship EMPTY /
    # generic. This repo is public; no real ids, usernames, emails, or names may
    # live in committed code. The operator supplies real values via the config
    # JSON file or OCBRAIN_DATASET_* env overrides (never committed).
    persona_author_ids: list[str] = field(default_factory=list)
    # Founder feedback authors: telegram sender ids whose corrections / approvals /
    # thanks carry extra weight in the label fold and get author-provenance stamped
    # on mined DPO pairs. Each entry is a ``{"id": "<sender_id>", "weight": <float>}``
    # dict supplied by the LOCAL config JSON (never committed — this repo is public).
    # Ships EMPTY: an author absent from this list is a generic user (weight 1.0).
    # Membership here does NOT admit an author into the persona/voice stream; that is
    # governed solely by ``persona_author_ids`` (a founder like a co-founder can be a
    # feedback author WITHOUT ever becoming a persona target).
    founder_feedback_authors: list[dict] = field(default_factory=list)
    persona_direct_agents: list[str] = field(default_factory=lambda: ["main"])
    persona_git_repos: list[str] = field(default_factory=list)
    persona_git_authors: list[str] = field(default_factory=list)
    persona_authored_globs: list[str] = field(default_factory=list)
    persona_system_prompt: str = "You are the operator. Reply as they would."
    export_dir: str = "data/datasets"
    export_min_scope: str = "workspace"
    export_min_label: str = "good"
    # Optional local-LLM grade threshold. ``None`` preserves the v0.3 export
    # behavior; when set, ungraded rows and rows below the threshold stay local
    # but are withheld from the training export.
    export_min_grade: float | None = None
    learning_db: str = "~/.openclaw/learning.db"
    commitments_path: str = "~/.openclaw/commitments/commitments.json"
    cron_state_path: str = "~/.openclaw/cron/jobs-state.json"
    # Curated memory / identity / doctrine files to harvest as ``memory_file``
    # evidence, in addition to the transcript session_roots. Absolute paths or
    # globs; ships EMPTY (public repo). The operator points these at high-value
    # doctrine outside the session roots — e.g. per-workspace MEMORY.md / IDENTITY
    # files that the transcript harvest never reaches.
    memory_globs: list[str] = field(default_factory=list)
    # Relax the DPO structural pair gate (v0.3). The strict gate rejected both
    # real founder corrections in the overnight run; when true, mining admits a
    # pair on softer structural evidence. Defaults on for v0.3.
    dpo_relaxed_gate: bool = True
    # Mining never holds SQLite's single-writer lock across an entire corpus.
    # Commit after either bound and record wait/hold telemetry in the stage
    # result. Smaller batches make MCP feedback and stallcheck writes responsive.
    write_batch_size: int = 50
    write_batch_seconds: float = 2.0


@dataclass(frozen=True)
class DatasetGradingConfig:
    """Privacy-preserving local LLM grading.

    Dataset text is more sensitive than ordinary knowledge metadata and the
    corpus is contractually local-only. The grader therefore accepts loopback
    HTTP endpoints only; remote/hosted URLs are rejected in code.
    """

    endpoint: str = "http://127.0.0.1:11434/api/chat"
    model: str = ""
    timeout_seconds: int = 180
    per_run_item_cap: int = 100
    daily_item_cap: int = 500
    prompt_version: str = "dataset-rubric-v1"


@dataclass(frozen=True)
class ArchiveConfig:
    # Maintenance-lane archival of never-referenced catalog docs (v0.3). A catalog
    # doc untouched by any retrieval for ``catalog_never_referenced_days`` is
    # eligible for archival, up to ``batch_cap`` rows per pass.
    enabled: bool = True
    catalog_never_referenced_days: int = 14
    batch_cap: int = 5000


@dataclass(frozen=True)
class EmbedConfig:
    # Semantic embedding of knowledge rows for vector attribution (v0.3),
    # replacing FTS-only attribution. Secrets are never stored: ``api_key_env``
    # holds the NAME of an env var, never its value. ``daily_usd_cap`` bounds spend.
    enabled: bool = True
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    daily_usd_cap: float = 0.25
    batch_size: int = 128
    api_key_env: str = "OPENAI_API_KEY"  # variable NAME only; value never persisted
    price_per_mtok: dict[str, float] = field(
        default_factory=lambda: {"text-embedding-3-small": 0.02}
    )


@dataclass(frozen=True)
class ExcerptRenderConfig:
    # Autopilot ``excerpt_render`` stage (v0.3): render the injectable memory view
    # into the managed block of runtime files after ``promote``. ``targets`` is a
    # list of file paths whose ``BEGIN/END OCBRAIN MANAGED BLOCK`` is written or
    # updated each cycle; content OUTSIDE the markers is never touched, and an
    # unchanged block is not rewritten (mtime preserved). Ships EMPTY (public
    # repo) — the operator points ``targets`` at real runtime files (e.g.
    # per-workspace ``MEMORY.md``) via the LOCAL config JSON. ``scope`` / ``limit``
    # bound what is rendered; the char budget comes from ``promote.max_chars``.
    targets: list[str] = field(default_factory=list)
    scope: str | None = None
    limit: int = 40


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
    dataset_grading: DatasetGradingConfig = field(default_factory=DatasetGradingConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    excerpt_render: ExcerptRenderConfig = field(default_factory=ExcerptRenderConfig)


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


# --------------------------------------------------------------------------- #
# Founder feedback helpers
# --------------------------------------------------------------------------- #
def founder_ids(cfg: OcbrainConfig) -> list[str]:
    """Return the configured founder-feedback author ids (attribution only).

    These ids let the transcript parser stamp ``authored_by`` on a founder's turns
    even when the founder is not a persona author, so their corrections/approvals
    can be weighted and their DPO pairs tagged. Being here never admits an author
    into the persona/voice stream (that is ``persona_author_ids`` only).
    """
    out: list[str] = []
    for entry in cfg.dataset.founder_feedback_authors:
        if isinstance(entry, dict):
            ident = str(entry.get("id") or "").strip()
        else:
            ident = str(entry or "").strip()
        if ident:
            out.append(ident)
    return out


def founder_weight(cfg: OcbrainConfig, author_id: str | None) -> float:
    """Weight multiplier for a signal authored by ``author_id`` (1.0 == generic).

    A founder in ``founder_feedback_authors`` carries their configured weight; an
    author absent from the list (or ``None``) is a generic user at 1.0. A present
    entry with a missing/invalid ``weight`` also falls back to 1.0.
    """
    if not author_id:
        return 1.0
    target = str(author_id).strip()
    for entry in cfg.dataset.founder_feedback_authors:
        if not isinstance(entry, dict):
            if str(entry or "").strip() == target:
                return 1.0
            continue
        if str(entry.get("id") or "").strip() == target:
            try:
                weight = float(entry.get("weight", 1.0))
            except (TypeError, ValueError):
                return 1.0
            return weight if weight > 0 else 1.0
    return 1.0
