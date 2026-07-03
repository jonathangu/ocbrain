"""Tests for scripts/install-skills, the standalone human-run skill installer.

The installer is deliberately NOT an ocbrain subcommand: ocbrain-the-CLI never
installs skills. These tests drive the bash script via subprocess with HOME
pointed at a temp dir, using stdlib only.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "install-skills"
SKILL_SRC = REPO_ROOT / "skills" / "ocbrain" / "SKILL.md"


def run_installer(home: Path, *args: str, env_extra: dict[str, str] | None = None):
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def claude_copy(home: Path) -> Path:
    return home / ".claude" / "skills" / "ocbrain" / "SKILL.md"


def openclaw_copy(home: Path) -> Path:
    return home / ".openclaw" / "skills" / "ocbrain" / "SKILL.md"


def test_fresh_install_lands_both_copies(tmp_path: Path) -> None:
    result = run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    expected = SKILL_SRC.read_text()
    assert claude_copy(tmp_path).read_text() == expected
    assert openclaw_copy(tmp_path).read_text() == expected
    assert "installed" in result.stdout


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    assert run_installer(tmp_path).returncode == 0
    before = {p: p.stat().st_mtime_ns for p in (claude_copy(tmp_path), openclaw_copy(tmp_path))}
    result = run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "up-to-date" in result.stdout
    for path, mtime_ns in before.items():
        assert path.stat().st_mtime_ns == mtime_ns  # unchanged copy was not rewritten
        assert path.read_text() == SKILL_SRC.read_text()


def test_rerun_overwrites_drifted_copy(tmp_path: Path) -> None:
    assert run_installer(tmp_path).returncode == 0
    claude_copy(tmp_path).write_text("locally edited drift\n")
    result = run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "update (content differs)" in result.stdout
    assert claude_copy(tmp_path).read_text() == SKILL_SRC.read_text()


def test_uninstall_removes_exactly_ours(tmp_path: Path) -> None:
    other_claude = tmp_path / ".claude" / "skills" / "other-skill" / "SKILL.md"
    other_openclaw = tmp_path / ".openclaw" / "skills" / "other-skill" / "SKILL.md"
    for sibling in (other_claude, other_openclaw):
        sibling.parent.mkdir(parents=True)
        sibling.write_text("---\nname: other-skill\ndescription: keep me\n---\n")
    assert run_installer(tmp_path).returncode == 0
    result = run_installer(tmp_path, "--uninstall")
    assert result.returncode == 0, result.stderr
    assert not claude_copy(tmp_path).parent.exists()
    assert not openclaw_copy(tmp_path).parent.exists()
    assert other_claude.read_text().startswith("---")
    assert other_openclaw.exists()


def test_uninstall_when_absent_is_clean(tmp_path: Path) -> None:
    result = run_installer(tmp_path, "--uninstall")
    assert result.returncode == 0, result.stderr
    assert "nothing installed" in result.stdout


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = run_installer(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "plan " in result.stdout and "dry-run: no changes made" in result.stdout
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".openclaw").exists()


def test_dry_run_uninstall_removes_nothing(tmp_path: Path) -> None:
    assert run_installer(tmp_path).returncode == 0
    result = run_installer(tmp_path, "--uninstall", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert claude_copy(tmp_path).exists()
    assert openclaw_copy(tmp_path).exists()


def test_env_overrides_redirect_targets(tmp_path: Path) -> None:
    claude_root = tmp_path / "custom-claude"
    openclaw_root = tmp_path / "custom-openclaw"
    result = run_installer(
        tmp_path,
        env_extra={
            "CLAUDE_SKILLS_DIR": str(claude_root),
            "OPENCLAW_SKILLS_DIR": str(openclaw_root),
        },
    )
    assert result.returncode == 0, result.stderr
    assert (claude_root / "ocbrain" / "SKILL.md").exists()
    assert (openclaw_root / "ocbrain" / "SKILL.md").exists()
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".openclaw").exists()


def test_target_only_flags(tmp_path: Path) -> None:
    result = run_installer(tmp_path, "--claude-only")
    assert result.returncode == 0, result.stderr
    assert claude_copy(tmp_path).exists()
    assert not (tmp_path / ".openclaw").exists()

    result = run_installer(tmp_path, "--openclaw-only")
    assert result.returncode == 0, result.stderr
    assert openclaw_copy(tmp_path).exists()


def test_conflicting_only_flags_fail(tmp_path: Path) -> None:
    result = run_installer(tmp_path, "--claude-only", "--openclaw-only")
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    assert lines[0] == "---", "SKILL.md must open with YAML frontmatter"
    end = lines.index("---", 1)
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        # OpenClaw's frontmatter parser supports single-line `key: value` only.
        assert sep == ":", f"frontmatter line is not single-line key: value: {line!r}"
        fields[key.strip()] = value.strip()
    return fields


def test_skill_frontmatter_parses(tmp_path: Path) -> None:
    assert run_installer(tmp_path).returncode == 0
    for installed in (claude_copy(tmp_path), openclaw_copy(tmp_path)):
        fields = parse_frontmatter(installed.read_text())
        assert fields["name"] == "ocbrain"
        assert fields["description"].strip()
