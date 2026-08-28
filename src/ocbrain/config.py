"""ocbrain configuration surface.

Five sections, one per thing that actually reads configuration: ``retrieval``
(the ``search_core_v1`` ranking gates), ``scopes`` (scope folding and the alias
table), ``curator`` (``scripts/wiki-curator.py``), ``deslop`` (the write-time
closeout gate), and ``supersede`` (how much authority a runtime supersession
carries). There were seventeen; thirteen configured subsystems that were
deleted or were never read at all.

The public entry point is :func:`load_config`, which layers, in order:

1. hard-coded defaults (the section dataclasses below),
2. an optional JSON file at ``$OCBRAIN_CONFIG`` (default
   ``~/.ocbrain/ocbrain.config.json``, with legacy checkout fallback),
3. ``OCBRAIN_<SECTION>_<FIELD>`` environment overrides.

A section or key the file names but this module does not define is skipped, not
an error. Operator config files outlive the fields they were written against,
and a brain that refuses to start because of a retired key is worse than one
that ignores it.

Secrets are never stored: where a section names an API key it holds the *name*
of an environment variable, never its value.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

# Config lives beside the data it configures, not beside the code. The historical
# default was the *relative* ``data/ocbrain.config.json``, which made three things
# true at once: resolution depended on the working directory, a `git clean -xfd`
# or fresh clone silently discarded operator settings, and a test suite run from a
# checkout inherited whatever that checkout happened to have. A brain whose
# curator settings vanished that way keeps exiting 0 while promoting nothing.
#
# ``~/.ocbrain/ocbrain.config.json`` is checked first and is the documented home.
# The old checkout-relative path is still honored when it exists and the new one
# does not, so an existing install keeps working until it moves.
USER_CONFIG_PATH = Path("~/.ocbrain/ocbrain.config.json").expanduser()
LEGACY_CONFIG_PATH = Path("data/ocbrain.config.json")


class ConfigError(RuntimeError):
    """The operator config file exists but cannot be used.

    Deliberately ``RuntimeError`` and not ``ValueError``: ``json.JSONDecodeError``
    *is* a ``ValueError``, and the curator's per-claim guard catches
    ``ValueError`` to mean "this target was previously tombstoned". A malformed
    config riding that channel once reported every claim in a run as blocked --
    a silent supersession outage whose log said nothing about the file that
    caused it. A config problem must name itself, once, loudly.

    Note the asymmetry this preserves: a *retired key* in the file is tolerated
    by design (see the module docstring); a file that is not JSON is a different
    class of problem and is never tolerated.
    """


def default_config_path() -> Path:
    """Resolve the config path: ``$OCBRAIN_CONFIG``, then user, then legacy."""
    if override := os.environ.get("OCBRAIN_CONFIG"):
        return Path(override).expanduser()
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    if LEGACY_CONFIG_PATH.exists():
        return LEGACY_CONFIG_PATH
    return USER_CONFIG_PATH


# Retained for compatibility. Previously this read $OCBRAIN_CONFIG at *import*
# time, so it went stale the moment the environment changed; call
# ``default_config_path()`` instead.
DEFAULT_CONFIG_PATH = USER_CONFIG_PATH


@dataclass(frozen=True)
class RetrievalConfig:
    # Hybrid ranking gates for ``search_core_v1``. These were module constants
    # until an operator needed to tune serving precision without editing source.
    # Defaults match the shipped constants in ``core_v1``; raising the floors
    # trades recall for precision, lowering them does the reverse.
    #
    # ``min_dense_cosine`` is the floor for a candidate the lexical arm also
    # found; ``min_dense_only_cosine`` is the stricter floor for a candidate only
    # the dense arm found. ``require_dense_support`` additionally holds lexical
    # hits to ``min_dense_cosine`` when the dense arm is healthy, which is what
    # keeps a shared generic token from serving an unrelated belief.
    hybrid_rrf_k: int = 60
    min_dense_cosine: float = 0.30
    min_dense_only_cosine: float = 0.55
    min_lexical_query_term_matches: int = 2
    min_redundant_lexical_strength_ratio: float = 0.50
    require_dense_support: bool = True
    # Retrieval feedback shifts a belief's score by ``1 + boost``. The boost is
    # ``avg_signal * weight``, damped by observation count and clamped to
    # +/-``feedback_clamp``. Set ``feedback_weight`` to 0 to ignore feedback.
    feedback_weight: float = 0.125
    feedback_clamp: float = 0.25
    feedback_prior_observations: float = 3.0


@dataclass(frozen=True)
class ScopesConfig:
    """Operator vocabulary for scope ids: folding and an alias table.

    Callers name their own scope. The same project therefore arrives spelled a
    dozen ways ("Coframe Brain", "coframe-brain", "coframe_brain__v2"), and scope
    matching is exact string equality, so every spelling that is not the stored
    one reaches nothing. ``fold_enabled`` collapses case and separator noise;
    ``aliases`` handles the rest, where two genuinely different names mean the
    same scope.

    ``aliases`` maps a FOLDED, fully prefixed scope id to the canonical fully
    prefixed id, e.g. ``{"project:coframe-brain": "project:coframe"}``. An alias
    may rename a scope but never re-type it: a mapping whose target carries a
    different ``type:`` prefix is ignored, so the table can never be used to
    promote a project belief into ``global:doctrine`` behind the ledger's back.

    Ships EMPTY. This repo is public and real project names are operator data;
    an empty table reproduces today's exact-match behavior.
    """

    aliases: dict[str, str] = field(default_factory=dict)
    fold_enabled: bool = True


@dataclass(frozen=True)
class CuratorConfig:
    # Which evidence the wiki curator may send to a model, and which model.
    #
    # `egress_policies` is the operator's standing declaration of intent. It
    # ships as `hosted_ok` only, so a fresh install sends nothing it was not
    # explicitly given. `local_only` is rejected: using that material with this
    # hosted curator requires an explicit reclassification before selection.
    #
    # Two things are NOT configurable and are enforced in code regardless:
    # `prohibited` egress and `secret` visibility are never eligible. Those are
    # the floor, not a default.
    #
    # Every applied run records an egress audit naming exactly what was sent.
    egress_policies: list[str] = field(default_factory=lambda: ["hosted_ok"])
    visibilities: list[str] = field(default_factory=lambda: ["internal"])
    provider: str = "anthropic"
    model: str = ""  # empty means the provider's default
    max_beliefs: int = 24
    current_ttl_days: int = 90
    # Which project scopes the scheduled curator compiles, in order. A single
    # pinned project is a wiki that freezes the moment work moves to a second
    # scope, and the evidence keeps arriving regardless: one real brain had 574
    # eligible objects spread over ~40 project scopes while the pin curated 19.
    #
    # Ships as the historical single pin. Real project names are operator data
    # and this repo is public, so the list an operator actually wants is set in
    # their own config file, never here.
    projects: list[str] = field(default_factory=lambda: ["workspace"])
    # A project with fewer eligible objects than this is skipped, and reported as
    # skipped, rather than spending a hosted call on a handful of rows. Set to 1
    # to curate every project with any eligible evidence at all.
    min_evidence_per_project: int = 3


@dataclass(frozen=True)
class DeslopConfig:
    # Whether a client's closeout is refused when its summary trips an enforced
    # slop rule. Off by default, and the default is the point: a rejected
    # closeout loses the client's work, and the closeout-to-evidence path is the
    # single largest supply of curator-eligible evidence. Findings ride along in
    # the receipt as `slop_findings` so the writer sees them either way; turn
    # this on once you trust the rules against your own corpus.
    reject_closeout_slop: bool = False


@dataclass(frozen=True)
class SupersedeConfig:
    """How much authority one runtime ``brain.supersede`` call carries.

    ``tier`` picks the routing predicate, never which code paths exist -- both
    the direct and the pending path are always compiled and always reachable.

    ``project``
        The default. A project/repo/client/task-scoped, unpinned target under
        the rate cap is replaced immediately; a pinned target, anything in a
        ``global:*`` scope, and every call over the cap becomes an undecided
        proposal for an admin instead.
    ``pending_all``
        Every supersession becomes a proposal. For an operator who wants a
        human between an agent and the serving corpus at all times.

    ``direct_cap`` bounds how many direct supersessions one caller may land in
    a trailing 24 hours. Overflow is *routed*, never refused: an agent that
    hits the cap still gets its correction recorded, as a proposal.

    ``curator_direct`` exempts the scheduled wiki curator from that cap for an
    ordinary belief -- unpinned, and outside ``global:*``. A per-caller budget
    sized for a runtime agent is the wrong instrument for a process that
    recompiles the whole corpus hourly: past its eighth correction the curator
    pends everything, and because a proposal does not change the input that
    produced it, the next cycle proposes the same supersessions again. The live
    core reached 283 undecided proposals over 33 beliefs this way in 18 hours.
    Turning this off restores the all-pending behaviour exactly. What bounds the
    curator instead is the margin rule -- a claim more than 0.05 below the
    confidence of the belief it would retire is still deferred -- and the digest
    gate, neither of which can be configured away. Pinned and doctrine targets
    still pend, and the cap still binds every other caller unchanged.
    """

    tier: str = "project"
    direct_cap: int = 8
    curator_direct: bool = True


@dataclass(frozen=True)
class OcbrainConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scopes: ScopesConfig = field(default_factory=ScopesConfig)
    curator: CuratorConfig = field(default_factory=CuratorConfig)
    deslop: DeslopConfig = field(default_factory=DeslopConfig)
    supersede: SupersedeConfig = field(default_factory=SupersedeConfig)


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

    ``path`` defaults to :func:`default_config_path`. A missing file is fine
    (defaults win). ``env`` defaults to ``os.environ``. A file that exists but
    is not valid JSON raises :class:`ConfigError` -- defaults apply only when
    the file is absent, never as a silent stand-in for one an operator wrote.
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


def _read_config_file(config_path: Path) -> dict[str, Any]:
    """Parse the operator config, refusing loudly what cannot be parsed.

    Without this, every subsystem that lazily consults the config met a broken
    file in its own place, in its own exception shape, hours apart -- and the
    one inside the curator's per-claim loop surfaced as "every claim blocked".
    Name the file and the position once, here, for all of them.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"config file {config_path} is unreadable: {exc}") from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"config file {config_path} is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg}); fix or remove "
            "it -- defaults apply only when the file is absent"
        ) from exc
    return loaded if isinstance(loaded, dict) else {}


# One parse per config state per process. ``load_config`` sits inside per-claim
# and per-row paths (the supersede router consults it for every curated claim),
# and an uncached read is both a real cost and the mechanism that let a file
# edited mid-run change the answer between two claims of the same cycle. The key
# mirrors ``scope._scopes_cache_key``: a stat of the resolved file plus every
# environment variable that can change the layered result.
_CONFIG_CACHE: tuple[Any, OcbrainConfig] | None = None


def _env_override_names() -> tuple[str, ...]:
    names: list[str] = []
    defaults = OcbrainConfig()
    for f in fields(defaults):
        section = getattr(defaults, f.name)
        if not is_dataclass(section):
            continue
        for sf in fields(section):
            names.append(f"OCBRAIN_{f.name.upper()}_{sf.name.upper()}")
    return tuple(names)


_ENV_OVERRIDE_NAMES = _env_override_names()


def _config_cache_key(config_path: Path) -> Any:
    stamp: tuple[str, int, int] | tuple[str] = (str(config_path),)
    try:
        info = config_path.stat()
        stamp = (str(config_path), info.st_mtime_ns, info.st_size)
    except OSError:
        pass  # absent file: the path alone identifies the (defaults-only) state
    env_state = tuple(
        (name, os.environ[name]) for name in _ENV_OVERRIDE_NAMES if name in os.environ
    )
    return (stamp, os.environ.get("OCBRAIN_CONFIG"), env_state)


def _load_config_from_environ(path: Path | str | None) -> OcbrainConfig:
    global _CONFIG_CACHE
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    key = _config_cache_key(config_path)
    if _CONFIG_CACHE is not None and _CONFIG_CACHE[0] == key:
        return _CONFIG_CACHE[1]
    file_data: dict[str, Any] = {}
    if config_path.exists():
        file_data = _read_config_file(config_path)

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
    resolved = replace(cfg, **section_changes) if section_changes else cfg
    _CONFIG_CACHE = (key, resolved)
    return resolved


def describe_config(path: Path | str | None = None) -> dict[str, Any]:
    """Report the effective config and where every value came from.

    A layered config is only usable if an operator can see which layer won. This
    labels each field ``default``, ``file``, or ``env`` and names the file it
    resolved, so "why is the curator sending nothing" is one command rather than
    an archaeology exercise.
    """
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    file_data: dict[str, Any] = {}
    if config_path.exists():
        file_data = _read_config_file(config_path)
    effective = load_config(config_path)
    defaults = OcbrainConfig()

    sections: dict[str, Any] = {}
    for f in fields(effective):
        section = getattr(effective, f.name)
        if not is_dataclass(section):
            continue
        default_section = getattr(defaults, f.name)
        from_file = file_data.get(f.name) if isinstance(file_data.get(f.name), dict) else {}
        env_keys = set(_env_overrides(f.name, section))
        entries: dict[str, Any] = {}
        for field_def in fields(section):
            if field_def.name in env_keys:
                source = "env"
            elif field_def.name in from_file:
                source = "file"
            else:
                source = "default"
            entries[field_def.name] = {
                "value": getattr(section, field_def.name),
                "source": source,
                "default": getattr(default_section, field_def.name),
            }
        sections[f.name] = entries
    return {
        "config_path": str(config_path),
        "config_path_exists": config_path.exists(),
        "user_config_path": str(USER_CONFIG_PATH),
        "legacy_config_path": str(LEGACY_CONFIG_PATH),
        "env_override_pattern": "OCBRAIN_<SECTION>_<FIELD>",
        "sections": sections,
    }


# --------------------------------------------------------------------------- #
# Founder feedback helpers
# --------------------------------------------------------------------------- #
