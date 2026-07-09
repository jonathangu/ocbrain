"""v0.3 config surface: new knobs, sections, and their layering (defaults + JSON + env).

Other lanes consume these exact names, so these tests pin the field names, default
shapes, and override plumbing for the v0.3 additions:

* ``judge.targeting`` (candidate-filtering knobs),
* ``archive`` section (never-referenced catalog archival),
* ``embed`` section (semantic embedding + daily cap; api_key_env is a NAME only),
* ``autopilot.profiles`` + ``autopilot.profile_locks`` (light/heavy cycles),
* ``promote.human_bootstrap`` (one-time memory-file seeding),
* ``dataset.dpo_relaxed_gate`` (relaxed DPO structural gate).
"""

from __future__ import annotations

import json
from pathlib import Path

from ocbrain.config import (
    ArchiveConfig,
    EmbedConfig,
    OcbrainConfig,
    load_config,
)


def _write_cfg(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_defaults_present_on_empty_config(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert isinstance(cfg, OcbrainConfig)
    assert isinstance(cfg.archive, ArchiveConfig)
    assert isinstance(cfg.embed, EmbedConfig)


def test_judge_targeting_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.judge.targeting["sources"] == [
        "retrieval_touched",
        "lesson",
        "session_derived",
    ]
    assert cfg.judge.targeting["exclude_catalog_docs"] is True


def test_archive_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.archive.enabled is True
    assert cfg.archive.catalog_never_referenced_days == 14
    assert cfg.archive.batch_cap == 5000


def test_embed_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.embed.enabled is True
    assert cfg.embed.provider == "openai"
    assert cfg.embed.model == "text-embedding-3-small"
    assert cfg.embed.daily_usd_cap == 0.25
    assert cfg.embed.batch_size == 128
    assert cfg.embed.api_key_env == "OPENAI_API_KEY"
    assert cfg.embed.price_per_mtok == {"text-embedding-3-small": 0.02}


def test_autopilot_profiles_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.autopilot.profiles["light"] == [
        "migrate",
        "review",
        "autolabel",
        "tripwires",
        "promote",
        "maintain",
    ]
    assert cfg.autopilot.profiles["heavy"] == [
        "snapshot",
        "migrate",
        "harvest",
        "injection_scan",
        "review",
        "compile",
        "autolabel",
        "tripwires",
        "promote",
        "maintain",
        "dataset_mine",
        "dataset_export",
    ]
    assert cfg.autopilot.profile_locks == "shared"


def test_promote_human_bootstrap_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.promote.human_bootstrap == {
        "enabled": True,
        "sources": ["memory_file"],
        "cap": 15,
    }


def test_dataset_dpo_relaxed_gate_default(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.dataset.dpo_relaxed_gate is True


def test_embed_api_key_env_is_a_name_not_a_secret(tmp_path: Path) -> None:
    # The field must hold the NAME of an env var; it must never look like a value.
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.embed.api_key_env.isupper() or "_" in cfg.embed.api_key_env
    assert not cfg.embed.api_key_env.startswith("sk-")


# --------------------------------------------------------------------------- #
# JSON file overrides
# --------------------------------------------------------------------------- #
def test_json_overrides_new_sections(tmp_path: Path) -> None:
    path = _write_cfg(
        tmp_path,
        {
            "archive": {"catalog_never_referenced_days": 30, "enabled": False},
            "embed": {"daily_usd_cap": 1.5, "model": "text-embedding-3-large"},
            "dataset": {"dpo_relaxed_gate": False},
            "promote": {"human_bootstrap": {"enabled": False, "cap": 0, "sources": []}},
        },
    )
    cfg = load_config(path)
    assert cfg.archive.catalog_never_referenced_days == 30
    assert cfg.archive.enabled is False
    assert cfg.archive.batch_cap == 5000  # untouched default preserved
    assert cfg.embed.daily_usd_cap == 1.5
    assert cfg.embed.model == "text-embedding-3-large"
    assert cfg.dataset.dpo_relaxed_gate is False
    assert cfg.promote.human_bootstrap == {"enabled": False, "cap": 0, "sources": []}


def test_json_overrides_judge_targeting_and_profiles(tmp_path: Path) -> None:
    path = _write_cfg(
        tmp_path,
        {
            "judge": {"targeting": {"sources": ["lesson"], "exclude_catalog_docs": False}},
            "autopilot": {"profiles": {"light": ["review"]}, "profile_locks": "isolated"},
        },
    )
    cfg = load_config(path)
    assert cfg.judge.targeting == {"sources": ["lesson"], "exclude_catalog_docs": False}
    assert cfg.autopilot.profiles == {"light": ["review"]}
    assert cfg.autopilot.profile_locks == "isolated"


# --------------------------------------------------------------------------- #
# Env overrides
# --------------------------------------------------------------------------- #
def test_env_override_embed_scalar(tmp_path: Path) -> None:
    cfg = load_config(
        tmp_path / "missing.json",
        env={"OCBRAIN_EMBED_DAILY_USD_CAP": "0.75", "OCBRAIN_EMBED_ENABLED": "false"},
    )
    assert cfg.embed.daily_usd_cap == 0.75
    assert cfg.embed.enabled is False


def test_env_override_dict_field_json(tmp_path: Path) -> None:
    cfg = load_config(
        tmp_path / "missing.json",
        env={"OCBRAIN_JUDGE_TARGETING": json.dumps({"sources": ["x"], "exclude_catalog_docs": 0})},
    )
    assert cfg.judge.targeting == {"sources": ["x"], "exclude_catalog_docs": 0}


def test_env_beats_json(tmp_path: Path) -> None:
    path = _write_cfg(tmp_path, {"archive": {"batch_cap": 100}})
    cfg = load_config(path, env={"OCBRAIN_ARCHIVE_BATCH_CAP": "999"})
    assert cfg.archive.batch_cap == 999
