"""Public-safety scanner for the TRACKED git tree.

Jonathan's standing directive, enforced here as a repo default: push anything
safe to see to the public repo, but never let private memories or private-repo
material reach it. This module is the mechanism. It reads *git* (the tracked
tree and diffs), never the runtime brain DB.

Checks
------
(a) placement   -- no tracked file under ``data/`` or ``logs/`` and no tracked
                   ``*.sqlite`` / ``*.jsonl`` dataset artifact.
(b) denylist    -- no hits against a LOCAL, gitignored denylist of Jonathan's
                   private identifiers (``data/public-safety-denylist.txt``).
                   Matched case-insensitively; if the file is absent we
                   warn-and-continue with the built-in secret patterns only.
                   Values are NEVER printed -- findings report counts/locations.
(c) new secrets -- no high-entropy candidate secrets, and no ``text.py`` leak
                   patterns, in *newly added* diff lines.
(d) private path-- no absolute ``/Users/`` paths that reveal a repo/container
                   segment outside a small allowlist (this repo + orchestrator
                   infra). Pragmatic and low-false-positive: only project
                   container segments (``workspace/``, ``code/`` ...) are read.

The secret/entropy scanners (c) run only on added diff lines because, as the
real tree proves, they false-positive on ordinary source (``api_key = env``)
and on documentation hashes. Placement/denylist/private-path (a,b,d) run
tree-wide. Test fixtures, lockfiles, the denylist file, and this module's own
source are excluded from *content* scans because they necessarily contain
adversarial-looking or definitional patterns; every other tracked file -- all
product source, docs, ops, scripts -- is scanned.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ocbrain.text import find_high_entropy_spans, find_probable_secret_leaks

# --- configuration -------------------------------------------------------- #

# LOCAL + gitignored (all of data/ is gitignored). Never tracked.
DENYLIST_REL = "data/public-safety-denylist.txt"

# Repo / project-container segments that are safe to reveal in an absolute
# /Users/ path: this repo itself plus the orchestrator infra dirs that already
# live in the tracked tree. Everything else in a container segment is flagged.
WORKSPACE_ALLOWLIST: set[str] = {"ocbrain", "task-artifacts", "task-status"}

# Tracked paths that must never exist (placement check (a)).
_FORBIDDEN_PREFIXES = ("data/", "logs/")
_FORBIDDEN_SUFFIXES = (".jsonl",)

# Files excluded from *content* scans (b/c/d) — see module docstring.
_CONTENT_SKIP_EXACT = {"uv.lock", "src/ocbrain/publicsafety.py", DENYLIST_REL}

# Absolute /Users/ path tokens, and the project-container segment inside them.
_USERS_PATH_RE = re.compile(r"/Users/[^\s:\"'()\[\]<>|]+")
_CONTAINER_RE = re.compile(r"/(?:workspace|code|repos|projects|git)/([A-Za-z0-9][A-Za-z0-9._-]*)")

# ``assigned_secret`` (text.py) fires on keyword-assignment to an UNQUOTED RHS,
# which is exactly the false-positive class the real tree proves out:
# ``api_key = resolved_env.get(...)``, ``api_key: str``, ``api_key=api_key`` --
# an env lookup, a type annotation, an identifier, never a secret. In the
# tracked-tree guard we DISCARD text.py's assigned_secret hit and re-derive it
# ourselves, firing ONLY when the RHS is a quoted string literal of plausible
# secret length. This narrows false positives without weakening detection: the
# format-anchored (sk-/ghp_/xox…) and high-entropy scanners still catch real
# random secrets whether quoted or not, and this now ALSO catches a plausible
# quoted literal the source pattern structurally could not.
_QUOTED_SECRET_ASSIGN_RE = re.compile(
    r"""(?i)(api[_-]?key|secret|token|password|credential)\s*[:=]\s*["'][^"']{8,}["']"""
)

# A full public Git object id is reproducibility metadata, not a credential,
# but only suppress the entropy finding when the same line explicitly labels
# it as a Git commit/revision. An unlabeled 40-hex value remains suspicious.
_FULL_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_REVISION_CONTEXT_RE = re.compile(
    r"(?i)(?:git.{0,24}(?:commit|revision)|(?:commit|revision).{0,24}git)"
)


# --- findings ------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    """One public-safety violation. ``detail`` NEVER contains a secret value
    or a denylist entry -- only counts, locations, and safe type names."""

    rule: str
    path: str
    detail: str
    line: int | None = None

    def render(self) -> str:
        where = f"{self.path}:{self.line}" if self.line is not None else self.path
        return f"[{self.rule}] {where} -- {self.detail}"


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    denylist_present: bool = False
    denylist_size: int = 0
    tracked_count: int = 0
    diff_range: str | None = None

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "diff_range": self.diff_range,
            "tracked_count": self.tracked_count,
            "denylist_present": self.denylist_present,
            "denylist_size": self.denylist_size,
            "warnings": list(self.warnings),
            "findings": [
                {"rule": f.rule, "path": f.path, "line": f.line, "detail": f.detail}
                for f in self.findings
            ],
        }

    def report(self) -> str:
        lines: list[str] = []
        deny = (
            f"present ({self.denylist_size} entries)"
            if self.denylist_present
            else "ABSENT (built-in patterns only)"
        )
        lines.append(
            f"public-safety-check: tracked={self.tracked_count} denylist={deny} "
            f"range={self.diff_range or 'tree-only'}"
        )
        for warning in self.warnings:
            lines.append(f"  warn: {warning}")
        if self.ok:
            lines.append("  OK: no public-safety violations found.")
        else:
            lines.append(f"  FAIL: {len(self.findings)} violation(s):")
            for finding in self.findings:
                lines.append(f"    - {finding.render()}")
        return "\n".join(lines)


# --- low-level scanners (pure, unit-testable) ----------------------------- #


def load_denylist(root: Path) -> tuple[list[str], bool]:
    """Return (entries, present). Blank lines and ``#`` comments are ignored.
    Contents are never logged by callers."""

    path = root / DENYLIST_REL
    if not path.exists():
        return [], False
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        entry = raw.strip()
        if entry and not entry.startswith("#"):
            entries.append(entry)
    return entries, True


def count_denylist_hits(text: str, denylist: list[str]) -> int:
    """Number of DISTINCT denylist entries appearing (case-insensitive) in
    ``text``. Returns a count only -- never the matched value."""

    low = text.lower()
    return sum(1 for entry in denylist if entry and entry.lower() in low)


def private_path_segments(text: str, allowlist: set[str]) -> list[str]:
    """Non-allowlisted project-container segments inside absolute /Users/
    paths in ``text`` (check (d))."""

    hits: list[str] = []
    for match in _USERS_PATH_RE.finditer(text):
        token = match.group(0)
        for seg in _CONTAINER_RE.finditer(token):
            name = seg.group(1)
            if name not in allowlist:
                hits.append(name)
    return hits


def is_forbidden_tracked_path(rel: str) -> bool:
    """True if a tracked path violates placement rule (a)."""

    low = rel.lower()
    if any(low.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
        return True
    if low.endswith(_FORBIDDEN_SUFFIXES):
        return True
    # Any sqlite dataset artifact anywhere (.sqlite, .sqlite-wal, .sqlite3 ...).
    if ".sqlite" in low:
        return True
    return False


def content_scan_excluded(rel: str) -> bool:
    return (
        rel in _CONTENT_SKIP_EXACT
        or rel.startswith("tests/")
        or rel.endswith(".lock")
    )


def entropy_pathcheck_excluded(rel: str) -> bool:
    """True for files exempt from ONLY the two heuristic content checks --
    high_entropy (c) and private_path (d). launchd plists (``ops/*.plist``)
    legitimately embed absolute wrapper/log path strings that read as long
    high-entropy runs and workspace path segments. Placement (a) and denylist
    (b) STILL apply to plists; only these two heuristics are relaxed."""

    return rel.endswith(".plist")


def filter_public_git_revision_spans(line: str, spans: list[str]) -> list[str]:
    """Drop an explicitly labeled full Git object id from entropy findings."""
    if not _GIT_REVISION_CONTEXT_RE.search(line):
        return spans
    return [span for span in spans if not _FULL_GIT_OBJECT_RE.fullmatch(span)]


def refine_secret_leaks(content: str, leaks: list[str]) -> list[str]:
    """Re-derive the heuristic ``assigned_secret`` finding under the guard's
    stricter, quoted-literal rule. Every other (format/entropy-anchored) leak
    name passes through untouched; ``assigned_secret`` is kept only when the
    line carries a quoted secret literal of plausible length."""

    refined = [name for name in leaks if name != "assigned_secret"]
    if _QUOTED_SECRET_ASSIGN_RE.search(content):
        refined.append("assigned_secret")
    return refined


# --- git plumbing --------------------------------------------------------- #


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def git_tracked_files(root: Path) -> list[str]:
    out = _git(root, "ls-files", "-z")
    return [p for p in out.split("\0") if p]


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def git_added_lines(root: Path, diff_range: str) -> list[tuple[str, int, str]]:
    """Parse ``git diff --unified=0 <range>`` into (path, new_lineno, content)
    for each added line. Deletions and context are ignored."""

    out = _git(root, "diff", "--unified=0", "--no-color", diff_range)
    added: list[tuple[str, int, str]] = []
    path: str | None = None
    new_lineno = 0
    for line in out.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            path = None if target == "/dev/null" else target[2:]  # strip "b/"
            continue
        if line.startswith("@@"):
            match = _HUNK_RE.match(line)
            if match:
                new_lineno = int(match.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if path is not None:
                added.append((path, new_lineno, line[1:]))
            new_lineno += 1
        elif not line.startswith("-"):
            # context line (shouldn't appear at -U0, but stay safe)
            new_lineno += 1
    return added


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except (OSError, IsADirectoryError):
        return None
    if b"\0" in data:  # binary
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --- high-level scan ------------------------------------------------------ #


def scan(root: Path, *, diff_range: str | None = None) -> ScanResult:
    result = ScanResult(diff_range=diff_range)
    denylist, present = load_denylist(root)
    result.denylist_present = present
    result.denylist_size = len(denylist)
    if not present:
        result.warnings.append(
            f"denylist {DENYLIST_REL} absent; using built-in secret patterns only"
        )

    tracked = git_tracked_files(root)
    result.tracked_count = len(tracked)

    for rel in tracked:
        # (a) placement -- path only, applies to every tracked file.
        if is_forbidden_tracked_path(rel):
            result.findings.append(
                Finding(
                    "tracked_data_artifact",
                    rel,
                    "tracked file under data/ or logs/ or a dataset artifact (never public)",
                )
            )
        if content_scan_excluded(rel):
            continue
        text = _read_text(root / rel)
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            # (b) denylist -- tree-wide, count only.
            if denylist:
                hits = count_denylist_hits(line, denylist)
                if hits:
                    result.findings.append(
                        Finding(
                            "denylist",
                            rel,
                            f"{hits} denylisted private identifier(s) present on this line",
                            line=i,
                        )
                    )
            # (d) private /Users/ path -- tree-wide, except plists (see
            # entropy_pathcheck_excluded: wrapper/log paths are legitimate).
            if not entropy_pathcheck_excluded(rel):
                for seg in private_path_segments(line, WORKSPACE_ALLOWLIST):
                    result.findings.append(
                        Finding(
                            "private_path",
                            rel,
                            f"absolute /Users/ path reveals non-allowlisted segment '{seg}'",
                            line=i,
                        )
                    )

    # (c) new secrets -- diff-scoped, high false-positive tree-wide so added
    # lines only. Reuses the text.py secret/leak + entropy scanners.
    if diff_range:
        for rel, lineno, content in git_added_lines(root, diff_range):
            if content_scan_excluded(rel):
                continue
            leaks = refine_secret_leaks(content, find_probable_secret_leaks(content))
            if leaks:
                result.findings.append(
                    Finding(
                        "secret_leak",
                        rel,
                        f"added line matches secret pattern(s): {','.join(sorted(set(leaks)))}",
                        line=lineno,
                    )
                )
            # high_entropy (c) skips plists -- their absolute path strings read as
            # long high-entropy runs (see entropy_pathcheck_excluded).
            if not entropy_pathcheck_excluded(rel):
                spans = filter_public_git_revision_spans(
                    content, find_high_entropy_spans(content)
                )
                if spans:
                    result.findings.append(
                        Finding(
                            "high_entropy",
                            rel,
                            f"added line has high-entropy token (len {max(len(s) for s in spans)})",
                            line=lineno,
                        )
                    )

    return result
