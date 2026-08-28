"""Assert the deployed wiring, not just the database.

`ocbrain doctor` has always answered "is the core healthy" -- integrity,
foreign keys, a stdio smoke test. It has never answered "is the *machine*
wired the way the operator intended", and that gap has a defect ledger: a
launchd agent that was enabled but plistless ran nothing for twelve days while
every DB-level check stayed green; a whole mining subsystem shipped, tested,
and dark behind an env var set nowhere; a scheduled loop that ran for two days
without a flag its curator needed. Nine of the eleven defects logged against
this system in its first four production days lived in state *outside the
repo* -- plists, env blocks, hand-copied hooks, untracked pointer files --
which no test suite sees and no doctor checked.

The fix is not more checks hardcoded to one machine's job names. It is a
**manifest**: a machine-local JSON file (`~/.ocbrain/ops-manifest.json`,
untracked, like everything else under `~/.ocbrain/`) that records what this
machine is *supposed* to have -- which launchd jobs, with which environment,
which hooks copied from which repo examples, which control files present.
``ocbrain doctor --ops --write-manifest`` bootstraps it from the current state
(deployment day is the one day the current state is the intended state), and
``ocbrain doctor --ops`` thereafter reports every drift: a job unloaded, an
env key changed, a hook that no longer matches its source, a pointer file
gone. Intent lives on the machine; the checker lives here.
"""

from __future__ import annotations

import json
import plistlib
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPS_MANIFEST_SCHEMA = "ocbrain.ops-manifest.v1"
DEFAULT_MANIFEST_PATH = Path("~/.ocbrain/ops-manifest.json")
DEFAULT_LAUNCH_AGENTS_DIR = Path("~/Library/LaunchAgents")
DEFAULT_HOOKS_DIRS = (Path("~/.claude/hooks"),)

# Control files worth asserting when present at bootstrap time. Both are
# untracked by design, which is exactly why nothing else notices them missing:
# the pointer file silently reroutes every launcher to a stale checkout DB when
# absent, and the denylist degrades the public-safety scan to built-ins only.
_BOOTSTRAP_FILES = ("data/active-core.path", "data/public-safety-denylist.txt")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _expand(value: str | Path) -> Path:
    return Path(value).expanduser()


def _read_plist(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            loaded = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _default_launchctl() -> str | None:
    """The loaded-jobs listing, or None where launchctl does not exist."""
    try:
        proc = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _mentions_ocbrain(plist: dict[str, Any]) -> bool:
    label = str(plist.get("Label") or "")
    args = " ".join(str(part) for part in plist.get("ProgramArguments") or [])
    return "ocbrain" in label.lower() or "ocbrain" in args.lower()


def write_ops_manifest(
    manifest_path: Path | str | None = None,
    *,
    launch_agents_dir: Path | str | None = None,
    hooks_dirs: tuple[Path, ...] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Snapshot the current wiring as the intended wiring.

    Discovery, not configuration: every LaunchAgent whose label or program
    mentions ocbrain, every installed hook that matches an example shipped in
    ``examples/harness/``, and the untracked control files that exist today.
    Nothing machine-specific is hardcoded here -- the operator's job names come
    from the operator's plists.
    """
    target = _expand(manifest_path or DEFAULT_MANIFEST_PATH)
    agents_dir = _expand(launch_agents_dir or DEFAULT_LAUNCH_AGENTS_DIR)
    hook_dirs = tuple(_expand(d) for d in (hooks_dirs or DEFAULT_HOOKS_DIRS))
    root = _expand(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    jobs: list[dict[str, Any]] = []
    if agents_dir.is_dir():
        for plist_path in sorted(agents_dir.glob("*.plist")):
            plist = _read_plist(plist_path)
            if not plist or not _mentions_ocbrain(plist):
                continue
            env = plist.get("EnvironmentVariables")
            jobs.append(
                {
                    "label": str(plist.get("Label") or plist_path.stem),
                    "plist": str(plist_path),
                    "env": {str(k): str(v) for k, v in env.items()}
                    if isinstance(env, dict)
                    else {},
                }
            )

    hooks: list[dict[str, str]] = []
    examples = root / "examples" / "harness"
    if examples.is_dir():
        for source in sorted(examples.glob("*.sh")):
            for hook_dir in hook_dirs:
                installed = hook_dir / source.name
                if installed.is_file():
                    hooks.append({"installed": str(installed), "source": str(source)})

    files = [str(root / rel) for rel in _BOOTSTRAP_FILES if (root / rel).is_file()]

    manifest = {
        "schema_version": OPS_MANIFEST_SCHEMA,
        "written_at": _now_iso(),
        "repo_root": str(root),
        "jobs": jobs,
        "hooks": hooks,
        "files": files,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def ops_check(
    manifest_path: Path | str | None = None,
    *,
    run_launchctl: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Verify the machine against its manifest. Every drift is a finding.

    Severity is binary on purpose. A drifted hook and an unloaded job are the
    same class of problem -- the machine no longer matches what the operator
    decided -- and the history here is of "minor" drift compounding silently.
    An absent manifest is a warning, not a failure: a fresh install has nothing
    to assert yet, and the report says how to bootstrap.
    """
    target = _expand(manifest_path or DEFAULT_MANIFEST_PATH)
    findings: list[dict[str, str]] = []
    warnings: list[str] = []

    def finding(check: str, detail: str) -> None:
        findings.append({"check": check, "detail": detail})

    if not target.is_file():
        return {
            "action": "ops-check",
            "status": "no_manifest",
            "healthy": True,
            "manifest": str(target),
            "findings": [],
            "warnings": [
                f"no ops manifest at {target}; run "
                "`ocbrain doctor --ops --write-manifest` once the machine is "
                "wired the way you intend"
            ],
        }

    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "action": "ops-check",
            "status": "failed",
            "healthy": False,
            "manifest": str(target),
            "findings": [{"check": "manifest", "detail": f"unreadable manifest: {exc}"}],
            "warnings": [],
        }
    if manifest.get("schema_version") != OPS_MANIFEST_SCHEMA:
        finding(
            "manifest",
            f"schema {manifest.get('schema_version')!r} is not {OPS_MANIFEST_SCHEMA!r}",
        )

    loaded_jobs: str | None = None
    jobs = manifest.get("jobs") or []
    if jobs:
        loaded_jobs = (run_launchctl or _default_launchctl)()
        if loaded_jobs is None:
            warnings.append("launchctl unavailable; job-loaded checks skipped")
    for job in jobs:
        label = str(job.get("label") or "")
        plist_path = _expand(str(job.get("plist") or ""))
        if not plist_path.is_file():
            finding("job", f"{label}: plist missing at {plist_path}")
            continue
        plist = _read_plist(plist_path)
        if plist is None:
            finding("job", f"{label}: plist at {plist_path} does not parse")
            continue
        actual_env = {
            str(k): str(v)
            for k, v in (
                plist.get("EnvironmentVariables") or {}
                if isinstance(plist.get("EnvironmentVariables"), dict)
                else {}
            ).items()
        }
        for key, expected in (job.get("env") or {}).items():
            if key not in actual_env:
                finding("job_env", f"{label}: env {key} missing (expected {expected!r})")
            elif actual_env[key] != str(expected):
                finding(
                    "job_env",
                    f"{label}: env {key} is {actual_env[key]!r}, manifest says {expected!r}",
                )
        for key in actual_env:
            if key not in (job.get("env") or {}):
                finding(
                    "job_env",
                    f"{label}: env {key} set on the machine but absent from the "
                    "manifest -- re-run --write-manifest if intended",
                )
        if loaded_jobs is not None and label and label not in loaded_jobs:
            finding("job", f"{label}: not loaded (launchctl list does not show it)")

    for hook in manifest.get("hooks") or []:
        installed = _expand(str(hook.get("installed") or ""))
        source = _expand(str(hook.get("source") or ""))
        if not installed.is_file():
            finding("hook", f"hook missing at {installed}")
            continue
        if not source.is_file():
            finding("hook", f"hook source missing at {source} (repo moved?)")
            continue
        if installed.read_bytes() != source.read_bytes():
            finding(
                "hook",
                f"{installed} drifted from {source} -- re-copy it or re-run "
                "--write-manifest if the drift is intended",
            )
        elif not installed.stat().st_mode & 0o100:
            finding("hook", f"{installed} is not executable")

    for entry in manifest.get("files") or []:
        path = _expand(str(entry))
        if not path.is_file():
            finding("file", f"expected file missing: {path}")
            continue
        if path.name == "active-core.path":
            pointer = path.read_text(encoding="utf-8").strip()
            if pointer and not _expand(pointer).exists():
                finding("file", f"{path} points at {pointer}, which does not exist")

    healthy = not findings
    return {
        "action": "ops-check",
        "status": "ok" if healthy else "failed",
        "healthy": healthy,
        "manifest": str(target),
        "checked": {
            "jobs": len(jobs),
            "hooks": len(manifest.get("hooks") or []),
            "files": len(manifest.get("files") or []),
        },
        "findings": findings,
        "warnings": warnings,
    }
