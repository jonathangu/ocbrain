"""The ops manifest: deployed wiring becomes assertable state.

Everything here is synthetic -- plists written into tmp_path, hooks copied
between tmp dirs, a launchctl that is a lambda. Nothing reads the real
~/Library or ~/.ocbrain, because a test that depends on the machine it runs on
is the exact disease the module treats.
"""

from __future__ import annotations

import json
import plistlib
from pathlib import Path

from ocbrain.opscheck import OPS_MANIFEST_SCHEMA, ops_check, write_ops_manifest


def _plist(directory: Path, label: str, env: dict[str, str], program: str) -> Path:
    path = directory / f"{label}.plist"
    with path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": label,
                "ProgramArguments": ["/bin/bash", "-lc", program],
                "EnvironmentVariables": env,
                "StartInterval": 3600,
            },
            handle,
        )
    return path


def _estate(tmp_path: Path) -> dict[str, Path]:
    """One synthetic machine: repo with an example hook, agents dir, hooks dir."""
    repo = tmp_path / "repo"
    (repo / "examples" / "harness").mkdir(parents=True)
    (repo / "data").mkdir()
    source = repo / "examples" / "harness" / "session-start.sh"
    source.write_text("#!/bin/sh\necho briefing\n", encoding="utf-8")
    (repo / "data" / "active-core.path").write_text(
        str(tmp_path / "core.sqlite"), encoding="utf-8"
    )
    (tmp_path / "core.sqlite").write_bytes(b"")

    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    _plist(
        agents,
        "test.ocbrain-promote",
        {"OCBRAIN_HYGIENE_APPLY": "1", "OCBRAIN_PROCMINE": "1"},
        str(repo / "scripts" / "brain-promote.sh"),
    )
    _plist(agents, "test.unrelated-job", {}, "/usr/bin/true")

    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / source.name).write_bytes(source.read_bytes())
    (hooks / source.name).chmod(0o755)

    return {"repo": repo, "agents": agents, "hooks": hooks, "manifest": tmp_path / "m.json"}


def _write(estate: dict[str, Path]) -> dict:
    return write_ops_manifest(
        estate["manifest"],
        launch_agents_dir=estate["agents"],
        hooks_dirs=(estate["hooks"],),
        repo_root=estate["repo"],
    )


def _check(estate: dict[str, Path], loaded: str = "test.ocbrain-promote") -> dict:
    return ops_check(estate["manifest"], run_launchctl=lambda: loaded)


def test_write_manifest_discovers_only_ocbrain_jobs_and_installed_hooks(tmp_path: Path):
    estate = _estate(tmp_path)
    manifest = _write(estate)
    assert manifest["schema_version"] == OPS_MANIFEST_SCHEMA
    assert [job["label"] for job in manifest["jobs"]] == ["test.ocbrain-promote"]
    assert manifest["jobs"][0]["env"]["OCBRAIN_PROCMINE"] == "1"
    assert len(manifest["hooks"]) == 1
    assert manifest["files"] and manifest["files"][0].endswith("active-core.path")


def test_a_freshly_written_manifest_verifies_clean(tmp_path: Path):
    estate = _estate(tmp_path)
    _write(estate)
    result = _check(estate)
    assert result["healthy"] is True
    assert result["findings"] == []
    assert result["checked"] == {"jobs": 1, "hooks": 1, "files": 1}


def test_an_env_key_changed_on_the_machine_is_a_finding(tmp_path: Path):
    """The procmine class: intended-on, actually-off, and nothing noticed."""
    estate = _estate(tmp_path)
    _write(estate)
    _plist(
        estate["agents"],
        "test.ocbrain-promote",
        {"OCBRAIN_HYGIENE_APPLY": "1"},  # OCBRAIN_PROCMINE silently gone
        "whatever",
    )
    result = _check(estate)
    assert result["healthy"] is False
    assert any(
        f["check"] == "job_env" and "OCBRAIN_PROCMINE" in f["detail"] for f in result["findings"]
    )


def test_an_unloaded_job_is_a_finding_even_with_the_plist_present(tmp_path: Path):
    """The Aug 6-18 class: file on disk, nothing running, every DB check green."""
    estate = _estate(tmp_path)
    _write(estate)
    result = _check(estate, loaded="some.other.job")
    assert result["healthy"] is False
    assert any("not loaded" in f["detail"] for f in result["findings"])


def test_a_drifted_hook_is_a_finding(tmp_path: Path):
    """The --repo class: the installed copy quietly stops matching the repo."""
    estate = _estate(tmp_path)
    _write(estate)
    hook = estate["hooks"] / "session-start.sh"
    hook.write_text("#!/bin/sh\necho briefing --repo somewhere\n", encoding="utf-8")
    result = _check(estate)
    assert result["healthy"] is False
    assert any(f["check"] == "hook" and "drifted" in f["detail"] for f in result["findings"])


def test_a_dangling_active_core_pointer_is_a_finding(tmp_path: Path):
    estate = _estate(tmp_path)
    _write(estate)
    (estate["repo"] / "data" / "active-core.path").write_text(
        str(tmp_path / "gone.sqlite"), encoding="utf-8"
    )
    result = _check(estate)
    assert result["healthy"] is False
    assert any("does not exist" in f["detail"] for f in result["findings"])


def test_no_manifest_is_a_warning_with_instructions_not_a_failure(tmp_path: Path):
    result = ops_check(tmp_path / "absent.json", run_launchctl=lambda: "")
    assert result["healthy"] is True
    assert result["status"] == "no_manifest"
    assert "--write-manifest" in result["warnings"][0]


def test_a_corrupt_manifest_is_a_failure_that_names_itself(tmp_path: Path):
    path = tmp_path / "m.json"
    path.write_text("{broken", encoding="utf-8")
    result = ops_check(path, run_launchctl=lambda: "")
    assert result["healthy"] is False
    assert result["findings"][0]["check"] == "manifest"


def test_launchctl_unavailable_skips_loaded_checks_with_a_warning(tmp_path: Path):
    estate = _estate(tmp_path)
    _write(estate)
    result = ops_check(estate["manifest"], run_launchctl=lambda: None)
    assert result["healthy"] is True  # plist + env still verified
    assert any("skipped" in w for w in result["warnings"])


def test_an_env_key_added_on_the_machine_but_not_in_the_manifest_is_a_finding(
    tmp_path: Path,
):
    """Drift is bidirectional: an unmanifested flag is a decision nobody recorded."""
    estate = _estate(tmp_path)
    _write(estate)
    _plist(
        estate["agents"],
        "test.ocbrain-promote",
        {"OCBRAIN_HYGIENE_APPLY": "1", "OCBRAIN_PROCMINE": "1", "OCBRAIN_NEW_FLAG": "1"},
        "whatever",
    )
    result = _check(estate)
    assert result["healthy"] is False
    assert any("OCBRAIN_NEW_FLAG" in f["detail"] for f in result["findings"])


def test_manifest_roundtrip_is_json_stable(tmp_path: Path):
    estate = _estate(tmp_path)
    _write(estate)
    on_disk = json.loads(estate["manifest"].read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == OPS_MANIFEST_SCHEMA
    assert on_disk["jobs"][0]["label"] == "test.ocbrain-promote"
