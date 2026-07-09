"""Public-safety scanner tests. Synthetic fixtures only -- every violation is
built in a throwaway tmp git repo or a denylist we create here, so nothing in
the real tracked tree ever contains a real private identifier or secret."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ocbrain import publicsafety as ps

# A synthetic denylist entry (NOT a real Jonathan identifier).
FAKE_DENY = "acme-private-marker-xyz"
# A planted fake secret that trips text.py's openai_key pattern.
FAKE_SECRET = "sk-" + "A1b2C3d4E5f6G7h8J9k0"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "README.md").write_text("# clean repo\nnothing private here\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _write_denylist(root: Path, entries: list[str]) -> None:
    (root / "data").mkdir(exist_ok=True)
    (root / ps.DENYLIST_REL).write_text("\n".join(entries) + "\n", encoding="utf-8")


# --- low-level scanner units ---------------------------------------------- #


def test_count_denylist_hits_case_insensitive() -> None:
    assert ps.count_denylist_hits("see CoFrAmE here", ["coframe"]) == 1
    assert ps.count_denylist_hits("nothing to see", ["coframe"]) == 0


def test_private_path_flags_non_allowlisted_segment() -> None:
    text = "path /Users/bob/code/secret-employer-repo/main.py here"
    assert ps.private_path_segments(text, {"ocbrain"}) == ["secret-employer-repo"]


def test_private_path_allowlists_this_repo() -> None:
    text = "see /Users/guclaw/.openclaw/workspace/ocbrain/src/x.py"
    assert ps.private_path_segments(text, ps.WORKSPACE_ALLOWLIST) == []


def test_forbidden_path_matches_data_logs_and_artifacts() -> None:
    assert ps.is_forbidden_tracked_path("data/ocbrain.sqlite")
    assert ps.is_forbidden_tracked_path("logs/autopilot.log")
    assert ps.is_forbidden_tracked_path("exports/train.jsonl")
    assert not ps.is_forbidden_tracked_path("src/ocbrain/cli.py")


# --- clean tree passes ---------------------------------------------------- #


def test_clean_repo_passes(repo: Path) -> None:
    _write_denylist(repo, [FAKE_DENY])
    result = ps.scan(repo)
    assert result.ok, result.report()
    assert result.denylist_present


def test_missing_denylist_warns_but_continues(repo: Path) -> None:
    result = ps.scan(repo)  # no denylist written
    assert result.ok
    assert not result.denylist_present
    assert any("absent" in w for w in result.warnings)


# --- scanner catches each violation class --------------------------------- #


def test_catches_denylist_hit(repo: Path) -> None:
    _write_denylist(repo, [FAKE_DENY])
    (repo / "notes.md").write_text(f"leaking {FAKE_DENY} into a doc\n", encoding="utf-8")
    _git(repo, "add", "notes.md")
    _git(repo, "commit", "-q", "-m", "add note")
    result = ps.scan(repo)
    assert not result.ok
    deny = [f for f in result.findings if f.rule == "denylist"]
    assert deny and deny[0].path == "notes.md"
    # Value must NEVER appear in the finding output.
    assert FAKE_DENY not in result.report()


def test_catches_tracked_data_file(repo: Path) -> None:
    (repo / "data").mkdir()
    (repo / "data" / "ocbrain.sqlite").write_bytes(b"SQLite format 3\x00fake")
    # Force-add past a .gitignore-free tmp repo (data/ isn't ignored here).
    _git(repo, "add", "-f", "data/ocbrain.sqlite")
    _git(repo, "commit", "-q", "-m", "oops db")
    result = ps.scan(repo)
    assert not result.ok
    assert any(f.rule == "tracked_data_artifact" for f in result.findings)


def test_catches_planted_secret_in_diff_range(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "app.py").write_text(f'API = "{FAKE_SECRET}"\n', encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-q", "-m", "add secret")
    head = _git(repo, "rev-parse", "HEAD").strip()
    result = ps.scan(repo, diff_range=f"{base}..{head}")
    assert not result.ok
    leaks = [f for f in result.findings if f.rule == "secret_leak"]
    assert leaks and leaks[0].path == "app.py"
    # The secret value itself must never be echoed in the report.
    assert FAKE_SECRET not in result.report()


def test_catches_private_path_in_tracked_file(repo: Path) -> None:
    (repo / "doc.md").write_text(
        "build at /Users/bob/code/other-private-repo/build.sh\n", encoding="utf-8"
    )
    _git(repo, "add", "doc.md")
    _git(repo, "commit", "-q", "-m", "leak path")
    result = ps.scan(repo)
    assert not result.ok
    assert any(f.rule == "private_path" for f in result.findings)


def test_test_dir_excluded_from_content_scans(repo: Path) -> None:
    # Adversarial fixtures under tests/ are grandfathered (secret/deny/path).
    _write_denylist(repo, [FAKE_DENY])
    (repo / "tests").mkdir()
    (repo / "tests" / "fixture.py").write_text(
        f'S = "{FAKE_SECRET}"  # {FAKE_DENY} /Users/x/code/foo/y\n', encoding="utf-8"
    )
    _git(repo, "add", "tests/fixture.py")
    _git(repo, "commit", "-q", "-m", "fixture")
    result = ps.scan(repo)
    assert result.ok, result.report()


def test_diff_range_ignores_removed_lines(repo: Path) -> None:
    # Removing a secret must not be flagged as adding one.
    (repo / "old.py").write_text(f'X = "{FAKE_SECRET}"\n', encoding="utf-8")
    _git(repo, "add", "old.py")
    _git(repo, "commit", "-q", "-m", "seed")
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "old.py").write_text("X = 1\n", encoding="utf-8")
    _git(repo, "commit", "-qa", "-m", "scrub")
    head = _git(repo, "rev-parse", "HEAD").strip()
    result = ps.scan(repo, diff_range=f"{base}..{head}")
    assert not any(f.rule == "secret_leak" for f in result.findings), result.report()
