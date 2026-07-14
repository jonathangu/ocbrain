"""Public-safety scanner tests. Synthetic fixtures only -- every violation is
built in a throwaway tmp git repo or a denylist we create here, so nothing in
the real tracked tree ever contains a real private identifier or secret."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from ocbrain_ops import publicsafety as ps

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
    text = "see /Users/example/.openclaw/workspace/ocbrain/src/x.py"
    assert ps.private_path_segments(text, ps.WORKSPACE_ALLOWLIST) == []


def test_forbidden_path_matches_data_logs_and_artifacts() -> None:
    assert ps.is_forbidden_tracked_path("data/ocbrain.sqlite")
    assert ps.is_forbidden_tracked_path("logs/autopilot.log")
    assert ps.is_forbidden_tracked_path("exports/train.jsonl")
    assert not ps.is_forbidden_tracked_path("src/ocbrain/cli.py")


def test_source_distribution_explicitly_excludes_runtime_private_roots() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = set(config["tool"]["hatch"]["build"]["exclude"])
    assert {"/data/**", "/logs/**", "/uv.lock"} <= excluded
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in ignored and "logs/" in ignored
    assert ps.content_scan_excluded("packages/ops/src/ocbrain_ops/publicsafety.py")


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


# --- assigned_secret precision (ruling 1a) -------------------------------- #


def _commit_added_line(repo: Path, rel: str, line: str) -> str:
    """Commit ``rel`` containing ``line`` and return ``base..head`` for scan()."""
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / rel).write_text(line + "\n", encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", f"add {rel}")
    head = _git(repo, "rev-parse", "HEAD").strip()
    return f"{base}..{head}"


def test_assigned_secret_fires_on_quoted_literal(repo: Path) -> None:
    # RHS is a quoted literal of plausible secret length -> real leak, caught.
    rng = _commit_added_line(repo, "conf.py", 'password = "hunter2plausiblelen"')
    result = ps.scan(repo, diff_range=rng)
    leaks = [f for f in result.findings if f.rule == "secret_leak"]
    assert leaks and leaks[0].path == "conf.py", result.report()
    assert "assigned_secret" in leaks[0].detail


def test_assigned_secret_ignores_env_lookup(repo: Path) -> None:
    # The exact embed.py false-positive shape: env lookup, not a secret.
    rng = _commit_added_line(
        repo, "embed.py", "    api_key = resolved_env.get(cfg.embed.api_key_env)"
    )
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "secret_leak" for f in result.findings), result.report()


def test_assigned_secret_ignores_identifier_and_annotation(repo: Path) -> None:
    for line in (
        "    api_key: str,",  # type annotation
        "    response = call(payload, api_key=api_key, model=model)",  # identifier
        "    secret = self.config.secret  # attribute access",  # attribute access
    ):
        rng = _commit_added_line(repo, "mod.py", line)
        result = ps.scan(repo, diff_range=rng)
        assert not any(f.rule == "secret_leak" for f in result.findings), (
            f"{line!r} -> {result.report()}"
        )


def test_refine_secret_leaks_unit() -> None:
    # env lookup: assigned_secret dropped.
    assert ps.refine_secret_leaks("api_key = resolved_env.get(x)", ["assigned_secret"]) == []
    # quoted literal: assigned_secret kept.
    assert ps.refine_secret_leaks('token = "xoxb-plausible-length"', []) == ["assigned_secret"]
    # unrelated format leak passes through untouched.
    assert ps.refine_secret_leaks("k = v", ["openai_key"]) == ["openai_key"]


# --- plist entropy / private-path exemption (ruling 1b) ------------------- #


def test_plist_skips_entropy_and_private_path(repo: Path) -> None:
    # A launchd plist whose <string> carries wrapper + workspace log paths.
    (repo / "ops").mkdir()
    plist = (
        "<plist><array>\n"
        "  <string>/Users/bob/other-private/service-env/run-wrapper.sh</string>\n"
        "  <string>/Users/bob/code/employer-secret-repo/logs/out.log</string>\n"
        "</array></plist>\n"
    )
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "ops" / "svc.plist").write_text(plist, encoding="utf-8")
    _git(repo, "add", "ops/svc.plist")
    _git(repo, "commit", "-q", "-m", "add plist")
    head = _git(repo, "rev-parse", "HEAD").strip()
    result = ps.scan(repo, diff_range=f"{base}..{head}")
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()
    assert not any(f.rule == "private_path" for f in result.findings), result.report()


def test_plist_still_subject_to_placement_and_denylist(repo: Path) -> None:
    # Placement (a) and denylist (b) STILL apply to plists.
    _write_denylist(repo, [FAKE_DENY])
    (repo / "ops").mkdir()
    (repo / "ops" / "svc.plist").write_text(
        f"<plist><string>marker {FAKE_DENY} here</string></plist>\n", encoding="utf-8"
    )
    (repo / "logs").mkdir()
    (repo / "logs" / "job.plist").write_text("<plist/>\n", encoding="utf-8")
    _git(repo, "add", "-f", "ops/svc.plist", "logs/job.plist")
    _git(repo, "commit", "-q", "-m", "plist with deny + bad placement")
    result = ps.scan(repo)
    assert any(f.rule == "denylist" for f in result.findings), result.report()
    assert any(
        f.rule == "tracked_data_artifact" and f.path == "logs/job.plist" for f in result.findings
    ), result.report()


def test_entropy_pathcheck_excluded_unit() -> None:
    assert ps.entropy_pathcheck_excluded("ops/com.jonathangu.ocbrain_ops.autopilot.light.plist")
    assert not ps.entropy_pathcheck_excluded("src/ocbrain/cli.py")


def test_explicit_public_git_commit_is_not_an_entropy_finding(repo: Path) -> None:
    public_revision = "a790972f0f844d81067ed45c28b524220a10c019"
    rng = _commit_added_line(
        repo,
        "version.py",
        f'MLX_LM_GIT_COMMIT = "{public_revision}"',
    )
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_unlabeled_full_hex_value_remains_an_entropy_finding(repo: Path) -> None:
    suspicious_value = "a790972f0f844d81067ed45c28b524220a10c019"
    rng = _commit_added_line(repo, "payload.py", f'VALUE = "{suspicious_value}"')
    result = ps.scan(repo, diff_range=rng)
    assert any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_labeled_sha256_is_not_an_entropy_finding(repo: Path) -> None:
    digest = "28d8cd1b4287d12eb9bc21c67a7d916877c79dd330dffdff52b1ae8423e74d82"
    rng = _commit_added_line(repo, "manifest.json", f'  "sha256": "{digest}"')
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_unlabeled_sha256_remains_an_entropy_finding(repo: Path) -> None:
    digest = "28d8cd1b4287d12eb9bc21c67a7d916877c79dd330dffdff52b1ae8423e74d82"
    rng = _commit_added_line(repo, "manifest.json", f'  "value": "{digest}"')
    result = ps.scan(repo, diff_range=rng)
    assert any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_public_repo_url_is_not_an_entropy_finding(repo: Path) -> None:
    rng = _commit_added_line(
        repo,
        "SECURITY.md",
        "Report at https://github.com/jonathangu/ocbrain/security/advisories/new",
    )
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_public_repo_url_query_token_remains_an_entropy_finding(repo: Path) -> None:
    suspicious = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEf"
    rng = _commit_added_line(
        repo,
        "SECURITY.md",
        "Report at https://github.com/jonathangu/ocbrain/security/advisories/new"
        f"?token={suspicious}",
    )
    result = ps.scan(repo, diff_range=rng)
    assert any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_python_identifier_is_not_an_entropy_finding(repo: Path) -> None:
    rng = _commit_added_line(
        repo,
        "review.py",
        "max_operations = REVIEW_BATCH_MAX_OPERATIONS",
    )
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_quoted_python_entropy_still_fails(repo: Path) -> None:
    suspicious = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEf"
    rng = _commit_added_line(repo, "settings.py", f'VALUE = "{suspicious}"')
    result = ps.scan(repo, diff_range=rng)
    assert any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_explicit_human_readable_python_version_is_not_entropy(repo: Path) -> None:
    rng = _commit_added_line(
        repo,
        "settings.py",
        'prompt_version: str = "dataset-rubric-v3-human-calibration-anchors"',
    )
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_version_exception_does_not_hide_random_or_unscoped_entropy(repo: Path) -> None:
    suspicious = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEf"
    for line in (
        f'prompt_version = "{suspicious}"',
        'VALUE = "dataset-rubric-v3-human-calibration-anchors"',
        'other_version = "dataset-rubric-v3-human-calibration-anchors"',
        'api_version = "leaked-v2-private-secret-credential-material"',
        'conversion = "internal-v7-customer-private-access-material"',
    ):
        rng = _commit_added_line(repo, "versioned.py", line)
        result = ps.scan(repo, diff_range=rng)
        assert any(f.rule == "high_entropy" for f in result.findings), result.report()


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
