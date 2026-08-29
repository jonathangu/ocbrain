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
machine is *supposed* to have -- which launchd jobs, with salted hashes of
their environment values, which hooks copied from which repo examples, which
control files present. Raw environment values never enter the manifest or
findings.
``ocbrain doctor --ops --write-manifest`` bootstraps it from the current state
(deployment day is the one day the current state is the intended state), and
``ocbrain doctor --ops`` thereafter reports every drift: a job unloaded, an
env key changed, a hook that no longer matches its source, a pointer file
gone. Intent lives on the machine; the checker lives here.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import secrets
import subprocess
import tempfile
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


def _display_path(path: Path) -> str:
    """Keep local diagnostics useful without printing the operator's home path."""
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        return str(path)
    return "~" if not relative.parts else f"~/{relative}"


def _env_digest(*, salt: str, label: str, key: str, value: str) -> str:
    payload = "\0".join((salt, label, key, value)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_private_json(path: Path, payload: dict[str, Any], *, replace: bool) -> None:
    """Publish an owner-only manifest atomically, refusing overwrite by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"ops manifest already exists at {_display_path(path)}; "
                    "pass --replace-manifest only after reviewing the intended wiring change"
                ) from exc
            temporary.unlink()
        os.chmod(path, 0o600, follow_symlinks=False)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


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


def _loaded_job_labels(output: str) -> set[str]:
    return {parts[-1] for line in output.splitlines() if (parts := line.split())}


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
    replace: bool = False,
) -> dict[str, Any]:
    """Snapshot the current wiring as the intended wiring.

    Discovery, not configuration: every LaunchAgent whose label or program
    mentions ocbrain, every installed hook that matches an example shipped in
    ``examples/harness/``, and the untracked control files that exist today.
    Nothing machine-specific is hardcoded here -- the operator's job names come
    from the operator's plists. Environment values are salted and hashed so a
    manifest or doctor report cannot become a credential-disclosure path. An
    existing manifest is replaced only when the caller explicitly opts in.
    """
    target = _expand(manifest_path or DEFAULT_MANIFEST_PATH)
    agents_dir = _expand(launch_agents_dir or DEFAULT_LAUNCH_AGENTS_DIR)
    hook_dirs = tuple(_expand(d) for d in (hooks_dirs or DEFAULT_HOOKS_DIRS))
    root = _expand(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    env_hash_salt = secrets.token_hex(16)

    jobs: list[dict[str, Any]] = []
    if agents_dir.is_dir():
        for plist_path in sorted(agents_dir.glob("*.plist")):
            plist = _read_plist(plist_path)
            if not plist or not _mentions_ocbrain(plist):
                continue
            env = plist.get("EnvironmentVariables")
            label = str(plist.get("Label") or plist_path.stem)
            jobs.append(
                {
                    "label": label,
                    "plist": str(plist_path),
                    "env_sha256": {
                        str(key): _env_digest(
                            salt=env_hash_salt,
                            label=label,
                            key=str(key),
                            value=str(value),
                        )
                        for key, value in env.items()
                    }
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
        "env_hash_salt": env_hash_salt,
        "jobs": jobs,
        "hooks": hooks,
        "files": files,
    }
    _atomic_private_json(target, manifest, replace=replace)
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
            "manifest": _display_path(target),
            "findings": [],
            "warnings": [
                f"no ops manifest at {_display_path(target)}; run "
                "`ocbrain doctor --ops --write-manifest` once the machine is "
                "wired the way you intend"
            ],
        }

    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "action": "ops-check",
            "status": "failed",
            "healthy": False,
            "manifest": _display_path(target),
            "findings": [{"check": "manifest", "detail": f"unreadable manifest: {exc}"}],
            "warnings": [],
        }
    if not isinstance(manifest, dict):
        return {
            "action": "ops-check",
            "status": "failed",
            "healthy": False,
            "manifest": _display_path(target),
            "findings": [{"check": "manifest", "detail": "manifest root is not an object"}],
            "warnings": [],
        }
    try:
        mode = target.stat().st_mode & 0o777
    except OSError as exc:
        finding("manifest", f"manifest permissions could not be read: {exc}")
    else:
        if mode & 0o077:
            finding("manifest", f"manifest permissions are {mode:04o}, expected owner-only 0600")
    if manifest.get("schema_version") != OPS_MANIFEST_SCHEMA:
        finding(
            "manifest",
            f"schema {manifest.get('schema_version')!r} is not {OPS_MANIFEST_SCHEMA!r}",
        )

    loaded_jobs: str | None = None
    jobs_value = manifest.get("jobs") or []
    if not isinstance(jobs_value, list) or not all(isinstance(job, dict) for job in jobs_value):
        finding("manifest", "jobs must be a list of objects")
        jobs: list[dict[str, Any]] = []
    else:
        jobs = jobs_value
    if jobs:
        loaded_jobs = (run_launchctl or _default_launchctl)()
        if loaded_jobs is None:
            warnings.append("launchctl unavailable; job-loaded checks skipped")
    for job in jobs:
        label = str(job.get("label") or "")
        plist_path = _expand(str(job.get("plist") or ""))
        if not plist_path.is_file():
            finding("job", f"{label}: plist missing at {_display_path(plist_path)}")
            continue
        plist = _read_plist(plist_path)
        if plist is None:
            finding("job", f"{label}: plist at {_display_path(plist_path)} does not parse")
            continue
        actual_env = {
            str(k): str(v)
            for k, v in (
                plist.get("EnvironmentVariables") or {}
                if isinstance(plist.get("EnvironmentVariables"), dict)
                else {}
            ).items()
        }
        expected_env = job.get("env_sha256") or {}
        if "env" in job:
            finding(
                "manifest",
                f"{label}: plaintext env mapping is unsupported; rewrite the manifest",
            )
        if not isinstance(expected_env, dict):
            finding("manifest", f"{label}: env_sha256 must be an object")
            expected_env = {}
        env_hash_salt = manifest.get("env_hash_salt")
        if expected_env and not isinstance(env_hash_salt, str):
            finding("manifest", f"{label}: env hash salt is missing")
            expected_env = {}
        for key, expected in expected_env.items():
            if key not in actual_env:
                finding("job_env", f"{label}: env {key} missing")
            elif _env_digest(
                salt=env_hash_salt,
                label=label,
                key=str(key),
                value=actual_env[key],
            ) != str(expected):
                finding("job_env", f"{label}: env {key} differs from the manifest")
        for key in actual_env:
            if key not in expected_env:
                finding(
                    "job_env",
                    f"{label}: env {key} set on the machine but absent from the "
                    "manifest -- re-run --write-manifest if intended",
                )
        if loaded_jobs is not None and label and label not in _loaded_job_labels(loaded_jobs):
            finding("job", f"{label}: not loaded (launchctl list does not show it)")

    hooks_value = manifest.get("hooks") or []
    if not isinstance(hooks_value, list) or not all(
        isinstance(hook, dict) for hook in hooks_value
    ):
        finding("manifest", "hooks must be a list of objects")
        hooks: list[dict[str, Any]] = []
    else:
        hooks = hooks_value
    for hook in hooks:
        installed = _expand(str(hook.get("installed") or ""))
        source = _expand(str(hook.get("source") or ""))
        if not installed.is_file():
            finding("hook", f"hook missing at {_display_path(installed)}")
            continue
        if not source.is_file():
            finding("hook", f"hook source missing at {_display_path(source)} (repo moved?)")
            continue
        if installed.read_bytes() != source.read_bytes():
            finding(
                "hook",
                f"{_display_path(installed)} drifted from {_display_path(source)} -- "
                "re-copy it or re-run "
                "--write-manifest if the drift is intended",
            )
        elif not installed.stat().st_mode & 0o100:
            finding("hook", f"{_display_path(installed)} is not executable")

    files_value = manifest.get("files") or []
    if not isinstance(files_value, list) or not all(
        isinstance(entry, str) for entry in files_value
    ):
        finding("manifest", "files must be a list of paths")
        files: list[str] = []
    else:
        files = files_value
    for entry in files:
        path = _expand(str(entry))
        if not path.is_file():
            finding("file", f"expected file missing: {_display_path(path)}")
            continue
        if path.name == "active-core.path":
            pointer = path.read_text(encoding="utf-8").strip()
            if pointer and not _expand(pointer).exists():
                finding("file", f"{_display_path(path)} points at a target that does not exist")

    healthy = not findings
    return {
        "action": "ops-check",
        "status": "ok" if healthy else "failed",
        "healthy": healthy,
        "manifest": _display_path(target),
        "checked": {
            "jobs": len(jobs),
            "hooks": len(hooks),
            "files": len(files),
        },
        "findings": findings,
        "warnings": warnings,
    }
