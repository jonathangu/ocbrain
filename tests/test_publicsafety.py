"""Public-safety scanner tests. Synthetic fixtures only -- every violation is
built in a throwaway tmp git repo or a denylist we create here, so nothing in
the real tracked tree ever contains a real private identifier or secret."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from ocbrain import publicsafety as ps

# A synthetic denylist entry (NOT a real Jonathan identifier).
FAKE_DENY = "acme-private-marker-xyz"
# A planted fake secret that trips text.py's openai_key pattern.
FAKE_SECRET = "sk-" + "A1b2C3d4E5f6G7h8J9k0"


def _user_path(*parts: str) -> str:
    return "/".join(("", "Users", "example", *parts))


def _ipv4(*octets: str) -> str:
    return ".".join(octets)


def _oslogin_user() -> str:
    return "_".join(("alice", "example", "com"))


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
    text = f"path {_user_path('code', 'secret-employer-repo', 'main.py')} here"
    assert ps.private_path_segments(text, {"ocbrain"}) == ["secret-employer-repo"]


def test_private_path_allowlists_this_repo() -> None:
    text = f"see {_user_path('.openclaw', 'workspace', 'ocbrain', 'src', 'x.py')}"
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
    assert ps.content_scan_excluded("src/ocbrain/publicsafety.py")
    assert ps.builtin_scan_excluded("scripts/procmine/normalize.py")
    assert not ps.content_scan_excluded("scripts/procmine/normalize.py")
    assert not ps.content_scan_excluded("tests/test_publicsafety.py")


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


def test_redaction_module_remains_subject_to_private_denylist(repo: Path) -> None:
    """A heuristic exemption must not weaken the local denylist baseline."""
    _write_denylist(repo, [FAKE_DENY])
    path = repo / "scripts" / "procmine" / "normalize.py"
    path.parent.mkdir(parents=True)
    path.write_text(f'REDACTION_EXAMPLE = "{FAKE_DENY}"\n', encoding="utf-8")
    _git(repo, "add", "scripts/procmine/normalize.py")
    _git(repo, "commit", "-q", "-m", "add redactor")
    result = ps.scan(repo)
    findings = [f for f in result.findings if f.rule == "denylist"]
    assert len(findings) == 1, result.report()
    assert findings[0].path == "scripts/procmine/normalize.py"


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
        f"build at {_user_path('code', 'other-private-repo', 'build.sh')}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "doc.md")
    _git(repo, "commit", "-q", "-m", "leak path")
    result = ps.scan(repo)
    assert not result.ok
    assert any(f.rule == "private_path" for f in result.findings)


def test_test_dir_is_scanned_for_every_content_violation(repo: Path) -> None:
    """Generated test fixtures get no blanket privacy or credential bypass."""
    _write_denylist(repo, [FAKE_DENY])
    (repo / "tests").mkdir()
    fake_secret = "sk-" + "".join(("A1b2", "C3d4", "E5f6", "G7h8", "J9k0"))
    entropy_token = "".join(("AbCdEfGh", "IjKlMnOp", "QrStUvWx", "Yz012345", "6789AbCd"))
    private_path = "/".join(("", "Users", "example", "code", "synthetic-private", "x.py"))
    private_host = ".".join(("10", "21", "31", "41"))
    (repo / "tests" / "fixture.py").write_text(
        f'S = "{fake_secret}"; VALUE = "{entropy_token}" '
        f"# {FAKE_DENY} {private_path} host={private_host}\n",
        encoding="utf-8",
    )
    base = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "add", "tests/fixture.py")
    _git(repo, "commit", "-q", "-m", "fixture")
    head = _git(repo, "rev-parse", "HEAD").strip()
    result = ps.scan(repo, diff_range=f"{base}..{head}")
    rules = {finding.rule for finding in result.findings}
    assert {"denylist", "private_path", "infra_identifier", "secret_leak", "high_entropy"} <= rules


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
        f"  <string>{_user_path('other-private', 'service-env', 'run-wrapper.sh')}</string>\n"
        f"  <string>{_user_path('code', 'employer-secret-repo', 'logs', 'out.log')}</string>\n"
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


def test_plist_still_subject_to_infrastructure_identifier_scan(repo: Path) -> None:
    (repo / "ops").mkdir()
    private_host = _ipv4("10", "11", "12", "13")
    (repo / "ops" / "svc.plist").write_text(
        f'<plist><string>host="{private_host}"</string></plist>\n', encoding="utf-8"
    )
    _git(repo, "add", "ops/svc.plist")
    _git(repo, "commit", "-q", "-m", "plist with private host")
    result = ps.scan(repo)
    findings = [f for f in result.findings if f.rule == "infra_identifier"]
    assert len(findings) == 1, result.report()
    assert private_host not in result.report()


def test_entropy_pathcheck_excluded_unit() -> None:
    assert ps.entropy_pathcheck_excluded("ops/com.example.agent.plist")
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
    suspicious_value = "".join(("a790972f", "0f844d81", "067ed45c", "28b52422", "0a10c019"))
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
    suspicious = "".join(("AbCdEfGh", "IjKlMnOp", "QrStUvWx", "Yz012345", "6789AbCdEf"))
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


def test_multiline_python_declaration_is_not_an_entropy_finding(repo: Path) -> None:
    name = "test_" + "_".join(
        ("a", "long", "declaration", "identifier", "is", "structure", "not", "a", "secret")
    )
    rng = _commit_added_line(
        repo,
        "test_long_name.py",
        f"def {name}(",
    )
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_quoted_python_entropy_still_fails(repo: Path) -> None:
    suspicious = "".join(("AbCdEfGh", "IjKlMnOp", "QrStUvWx", "Yz012345", "6789AbCdEf"))
    rng = _commit_added_line(repo, "settings.py", f'VALUE = "{suspicious}"')
    result = ps.scan(repo, diff_range=rng)
    assert any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_shell_pytest_node_identifier_is_not_an_entropy_finding(repo: Path) -> None:
    node = "test_" + "_".join(
        ("the", "long", "public", "identifier", "names", "the", "behavior", "under", "test")
    )
    rng = _commit_added_line(
        repo,
        "mutation-proof.sh",
        f'"$T::{node}"',
    )
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


def test_shell_pytest_node_exception_does_not_hide_other_entropy(repo: Path) -> None:
    suspicious = "".join(("AbCdEfGh", "IjKlMnOp", "QrStUvWx", "Yz012345", "6789AbCdEf"))
    node = "test_" + "_".join(("the", "long", "public", "identifier", "names", "behavior"))
    rng = _commit_added_line(
        repo,
        "mutation-proof.sh",
        f'"$T::{node}" "{suspicious}"',
    )
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
    suspicious = "".join(("AbCdEfGh", "IjKlMnOp", "QrStUvWx", "Yz012345", "6789AbCdEf"))
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


def test_sql_ddl_identifier_is_not_an_entropy_finding(repo: Path) -> None:
    identifier = "idx_" + "_".join(
        ("retrieval", "uses", "knowledge", "outcome", "served")
    )
    rng = _commit_added_line(repo, "schema.sql", f"CREATE INDEX {identifier}")
    result = ps.scan(repo, diff_range=rng)
    assert not any(f.rule == "high_entropy" for f in result.findings), result.report()


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


# --------------------------------------------------------------------------- #
# The detector shapes the 2026-08-24 leaks proved missing
# --------------------------------------------------------------------------- #


def test_private_path_flags_developer_and_documents_containers():
    hits = ps.private_path_segments(
        "see "
        + _user_path("Developer", "secret-repo", "src", "x.py")
        + " and "
        + _user_path("Documents", "private-notes", "a.md"),
        ps.WORKSPACE_ALLOWLIST,
    )
    assert hits == ["secret-repo", "private-notes"]


def test_private_path_still_allowlists_this_repo_under_developer():
    assert (
        ps.private_path_segments(
            _user_path("Developer", "ocbrain", "src", "x.py"), ps.WORKSPACE_ALLOWLIST
        )
        == []
    )


def test_private_path_flags_a_home_dotdir():
    """An absolute home dot-dir path always reveals a username; `~/…` does not."""
    hits = ps.private_path_segments(
        f"wrote {_user_path('.privatebrain', 'logs', 'run.log')}", ps.WORKSPACE_ALLOWLIST
    )
    assert hits == [".privatebrain"]
    assert ps.private_path_segments("wrote ~/.privatebrain/logs/run.log", set()) == []


def test_infra_identifiers_flag_a_routable_ipv4_but_not_documentation_ranges():
    private_host = _ipv4("10", "11", "12", "13")
    assert ps.infrastructure_identifier_kinds(f'host="{private_host}"', None) == [
        "routable IPv4 literal"
    ]
    for exempt in ("127.0.0.1", "0.0.0.0", "192.0.2.7", "198.51.100.9", "203.0.113.4"):
        assert ps.infrastructure_identifier_kinds(f"connect to {exempt}", None) == []
    # Not an address at all: an octet over 255 is a version or a counter.
    assert ps.infrastructure_identifier_kinds("build 300.1.1.1", None) == []


def test_infra_identifiers_flag_the_oslogin_spelling_but_not_snake_case_code():
    assert ps.infrastructure_identifier_kinds(f'user="{_oslogin_user()}"', None) == [
        "OS Login account spelling"
    ]
    # The live tree's proven false-positive shapes must stay silent.
    assert ps.infrastructure_identifier_kinds("search_documents_ai trigger", None) == []
    assert ps.infrastructure_identifier_kinds("run_stage_dev pipeline", None) == []


def test_local_account_pattern_guards_against_generic_and_short_names():
    assert ps.local_account_pattern("runner") is None
    assert ps.local_account_pattern("root") is None
    assert ps.local_account_pattern("bob") is None
    pattern = ps.local_account_pattern("localx")
    assert pattern is not None
    assert ps.infrastructure_identifier_kinds(
        "log at /Users/localx/.openclaw/run.log", pattern
    ) == ["this machine's account name"]


def test_infra_findings_withhold_the_matched_value(repo):
    """CI logs on a public repo are public; a finding must not re-leak its match."""
    (repo / "docs").mkdir()
    private_host = _ipv4("10", "11", "12", "13")
    (repo / "docs" / "note.md").write_text(
        f'the box is at host="{private_host}"\n', encoding="utf-8"
    )
    _git(repo, "add", "docs/note.md")
    _git(repo, "commit", "-qm", "add note")
    result = ps.scan(repo)
    findings = [f for f in result.findings if f.rule == "infra_identifier"]
    assert len(findings) == 1
    assert private_host not in findings[0].detail
    assert "routable IPv4 literal" in findings[0].detail


def test_private_path_findings_withhold_the_matched_segment(repo):
    """A private-path failure must not reprint the identifier into CI logs."""
    segment = "synthetic" + "-private-segment"
    path = "/".join(("", "Users", "example", "code", segment, "note.md"))
    (repo / "note.md").write_text(f"built at {path}\n", encoding="utf-8")
    _git(repo, "add", "note.md")
    _git(repo, "commit", "-qm", "add private path")
    result = ps.scan(repo)
    findings = [finding for finding in result.findings if finding.rule == "private_path"]
    assert len(findings) == 1, result.report()
    assert segment not in findings[0].detail
    assert segment not in result.report()
    assert "value withheld" in findings[0].detail


def test_ci_public_safety_uses_a_validated_real_diff_range() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    assert "--diff-range" in workflow
    assert "git cat-file -e" in workflow
    assert 'base_commit="${{ github.event.before }}"' in workflow
    assert 'base_commit="${{ github.event.pull_request.base.sha }}"' in workflow
