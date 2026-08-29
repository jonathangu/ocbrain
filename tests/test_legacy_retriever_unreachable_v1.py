"""The legacy blend retriever must stay unreachable from a v1 core.

The repo carries two independently tuned rankers. ``core_v1`` serves the live
path (FTS5 bm25 + a dense sidecar fused with weighted RRF and multiplicative
scope, confidence, quality, recency, and feedback terms). ``retrieve.py`` is the retired
legacy ranker: a flat ``relevance * scope_weight * confidence * pinned *
catalog_stub`` product with a repo-FTS fallback. Both are still imported by
``mcp.py``, ``cli.py`` and ``shared_context.py``, each behind an ``is_core_v1``
early return.

Nothing enforced that separation. A single ``retrieve(...)`` call added outside
one of those guards would silently mix the two formulas on the live core, and
every existing test would still pass. This gate measures the boundary instead of
assuming it: it drives the complete advertised tool surface and counts how many
``retrieve.py`` functions actually execute.

The second test is the gate's own mutation proof. It runs the identical driver
against a legacy core, where the count MUST be non-zero. If the tracer stops
firing, the driver stops dispatching, or the tool table goes stale, that test
fails and the zero in the first test is exposed as an empty measurement rather
than a passing one.
"""

from __future__ import annotations

import ast
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest
from test_mcp_v1 import _seed_v1

from ocbrain.cli import main as cli_main
from ocbrain.db import connect, init_db
from ocbrain.mcp import (
    ADMIN_PROFILE,
    RUNTIME_TOOLS,
    call_tool,
    tools_for_profile,
)
from ocbrain.shared_context import build_context

LEGACY_PACKAGE = "ocbrain"
LEGACY_LEAF = "retrieve"
LEGACY_MODULE = f"{LEGACY_PACKAGE}.{LEGACY_LEAF}"

_WORKTREE_RETRIEVE = (Path(__file__).resolve().parents[1] / "src/ocbrain/retrieve.py").resolve()
# Derived from the module actually imported, not from the repo layout. Under a
# bare interpreter ``ocbrain`` resolves through the editable install rather than
# this tree, so a tracer keyed on this tree's path would match no frame and
# report a clean zero for the wrong reason.
# The tracer-key regression test below pins the two together.
_RETRIEVE_SOURCE = str(Path(importlib.import_module(LEGACY_MODULE).__file__).resolve())

# Every call site of the legacy ranker in the package, per module, keyed by path
# relative to ``src/ocbrain``. Measured on this tree; provenance and the guard
# each one sits behind are recorded in docs/THRESHOLDS.md §H.
LEGACY_RETRIEVE_CALL_SITES = {"mcp.py": 2, "cli.py": 1, "shared_context.py": 1}

# One representative argument set per advertised tool. Write tools are included
# deliberately: a refusal still runs the dispatch branch that would host a stray
# legacy call. ``test_tool_coverage_is_complete`` fails if a tool is added to the
# server without being added here, so this table cannot silently go stale.
TOOL_ARGUMENTS: dict[str, dict[str, object]] = {
    "brain.briefing": {"context": {"project": "ocbrain"}},
    "brain.closeout": {"status": "completed", "summary": "gate probe"},
    "brain.context": {"query": "shared context", "limit": 5},
    "brain.correct": {"target": "belief:shared-context", "layer": "body", "op": "replace"},
    "brain.digest": {},
    "brain.egress_preview": {},
    "brain.feedback": {"retrieval_use_id": "missing", "outcome": "used"},
    "brain.forget": {"target": "belief:shared-context"},
    "brain.get": {"id": "belief:shared-context"},
    "brain.goal_close": {
        "goal_id": "missing",
        "status": "completed",
        "verifier_uri": "repo://ocbrain/pytest",
        "verifier_status": "pass",
    },
    "brain.goal_open": {
        "objective": "gate probe",
        "finish_line": "pytest -q",
        "source_path": "docs/THRESHOLDS.md",
    },
    "brain.ingest": {"body": "gate probe evidence"},
    "brain.ledger": {},
    "brain.preview": {"query": "shared context", "limit": 5},
    "brain.proposal_decide": {"proposal_event_id": "missing", "decision": "approve"},
    "brain.proposals": {},
    "brain.search": {"query": "shared context", "limit": 5},
    "brain.source": {"id": "missing"},
    "brain.supersede": {"target": "belief:shared-context", "body": "x", "reason": "gate probe"},
}

# Scoped and cross-scope variants: the legacy blend is reached through the
# scoped branch of brain.search, so an unscoped-only driver would miss it.
SCOPED_VARIANTS: tuple[tuple[str, dict[str, object]], ...] = (
    ("brain.context", {"query": "shared context", "limit": 5, "context": {"project": "ocbrain"}}),
    ("brain.search", {"query": "shared context", "limit": 5, "context": {"project": "ocbrain"}}),
    ("brain.search", {"query": "shared context", "limit": 5, "cross_scope": True}),
    ("brain.preview", {"query": "shared context", "limit": 5, "context": {"project": "ocbrain"}}),
)


# The read-side CLI surface. The MCP driver below reaches ``call_tool`` only, so
# without this the CLI's own legacy call site (``cmd_preview``) is measured by
# nothing and the CHANGELOG's "the read-side CLI reaches none of them either" is
# prose with no instrument under it.
READ_SIDE_CLI: tuple[tuple[str, ...], ...] = (
    ("status",),
    ("briefing", "--project", "ocbrain"),
    ("digest", "--project", "ocbrain"),
    ("search", "shared context", "--limit", "5"),
    ("search", "shared context", "--limit", "5", "--project", "ocbrain"),
    ("preview", "shared context", "--limit", "5"),
    ("preview", "shared context", "--limit", "5", "--project", "ocbrain"),
    ("preview", "shared context", "--limit", "5", "--cross-scope"),
)


def _trace_retrieve(run) -> set[str]:
    """Run ``run()`` and return the ``retrieve.py`` functions that executed.

    One tracer for both drivers on purpose: two copies is how one of them gets
    fixed and the other stays blind.
    """
    executed: set[str] = set()

    def _tracer(frame, event, _arg):
        # Returning nothing declines local tracing: one record per call event is
        # all this needs, and per-line tracing over a 20 MB dispatch is not.
        if event == "call" and frame.f_code.co_filename == _RETRIEVE_SOURCE:
            executed.add(frame.f_code.co_name)

    previous = sys.gettrace()
    sys.settrace(_tracer)
    try:
        run()
    finally:
        sys.settrace(previous)
    return executed


def _drive_every_tool(conn: sqlite3.Connection) -> tuple[set[str], int]:
    """Dispatch the whole advertised surface, tracing retrieve.py.

    Returns the set of ``retrieve.py`` function names that executed and the
    number of tool dispatches attempted.
    """
    calls = [(name, dict(args)) for name, args in sorted(TOOL_ARGUMENTS.items())]
    calls.extend((name, dict(args)) for name, args in SCOPED_VARIANTS)
    dispatched = 0

    def _run() -> None:
        nonlocal dispatched
        for name, arguments in calls:
            try:
                call_tool(conn, {"name": name, "arguments": arguments}, profile=ADMIN_PROFILE)
            except Exception:  # noqa: BLE001 - a refusal still exercised the branch
                pass
            dispatched += 1

    executed = _trace_retrieve(_run)
    return executed, dispatched


def _drive_read_side_cli(db_path: Path) -> tuple[set[str], int]:
    """Run every read-side CLI command against ``db_path``, tracing retrieve.py."""
    dispatched = 0

    def _run() -> None:
        nonlocal dispatched
        for argv in READ_SIDE_CLI:
            try:
                cli_main(["--db", str(db_path), *argv])
            except (Exception, SystemExit):  # noqa: BLE001 - a refusal still ran the branch
                pass
            dispatched += 1

    executed = _trace_retrieve(_run)
    return executed, dispatched


V1_DB_NAME = "core-v1.sqlite"  # written by test_mcp_v1._seed_v1
LEGACY_DB_NAME = "legacy.sqlite"


def _seed_legacy(tmp_path: Path) -> sqlite3.Connection:
    """A legacy core with one current belief, so the legacy ranker has input."""
    conn = connect(tmp_path / LEGACY_DB_NAME)
    init_db(conn)
    conn.execute(
        """
        INSERT INTO current_beliefs (
          belief_id, body, scope_type, scope_id, visibility, egress_policy,
          confidence, confidence_band, evidence_ids, status, pinned,
          approved_event_id, last_event_id, last_compiled_at
        ) VALUES (
          'belief:shared-context',
          'Shared context is the stable bridge across every runtime.',
          'project', 'project:ocbrain', 'internal', 'local_only',
          0.9, 'high', '[]', 'current', 0,
          'evt_a', 'evt_a', '2026-07-10T00:00:00+00:00'
        )
        """
    )
    conn.commit()
    return conn


def test_the_tracer_is_keyed_to_the_module_under_test() -> None:
    """The tracer must watch the ``retrieve.py`` in this worktree.

    Outside pytest's rootdir ``ocbrain`` resolves through the editable install
    instead, and every gate in this file would then trace a file nothing
    executes and report a clean zero for the wrong reason.
    """
    assert _RETRIEVE_SOURCE == str(_WORKTREE_RETRIEVE), (
        "ocbrain.retrieve resolves outside this worktree, so every trace in this "
        f"file measures nothing. imported={_RETRIEVE_SOURCE} worktree={_WORKTREE_RETRIEVE}"
    )


def test_tool_coverage_is_complete() -> None:
    """The driver must cover every tool the server advertises."""
    advertised = tools_for_profile(ADMIN_PROFILE)
    assert advertised == set(TOOL_ARGUMENTS), (
        "TOOL_ARGUMENTS drifted from the advertised tool surface; "
        f"missing={sorted(advertised - set(TOOL_ARGUMENTS))} "
        f"extra={sorted(set(TOOL_ARGUMENTS) - advertised)}"
    )
    assert RUNTIME_TOOLS <= advertised


def test_legacy_retriever_never_runs_on_a_v1_core(tmp_path: Path) -> None:
    conn = _seed_v1(tmp_path)
    try:
        executed, dispatched = _drive_every_tool(conn)
    finally:
        conn.close()

    # A driver that dispatched nothing would report an empty set for the wrong
    # reason. Pin the count to the table so an emptied table cannot pass.
    assert dispatched == len(TOOL_ARGUMENTS) + len(SCOPED_VARIANTS) == 23
    assert executed == set(), (
        "a v1 core reached the retired legacy blend retriever; "
        f"retrieve.py functions executed: {sorted(executed)}"
    )


def test_the_same_driver_does_reach_the_legacy_retriever(tmp_path: Path) -> None:
    """Mutation proof for the gate above: the instrument must report dirty."""
    conn = _seed_legacy(tmp_path)
    try:
        executed, dispatched = _drive_every_tool(conn)
    finally:
        conn.close()

    assert dispatched == len(TOOL_ARGUMENTS) + len(SCOPED_VARIANTS) == 23
    assert "retrieve" in executed, (
        "the driver no longer reaches retrieve.py on a legacy core, so the "
        "zero measured on a v1 core is an empty check rather than a passing one"
    )
    # Measured on this fixture: the legacy path runs 9 distinct retrieve.py
    # functions. Documented in docs/THRESHOLDS.md §H.
    assert len(executed) >= 5, sorted(executed)


def test_the_read_side_cli_never_runs_the_legacy_retriever_on_a_v1_core(
    tmp_path: Path, capsys
) -> None:
    """The CLI hosts its own legacy call site, and no tool dispatch reaches it."""
    _seed_v1(tmp_path).close()
    executed, dispatched = _drive_read_side_cli(tmp_path / V1_DB_NAME)
    capsys.readouterr()

    assert dispatched == len(READ_SIDE_CLI)
    assert executed == set(), (
        "the read-side CLI reached the retired legacy blend retriever on a v1 "
        f"core; retrieve.py functions executed: {sorted(executed)}"
    )


def test_the_same_cli_driver_does_reach_the_legacy_retriever(tmp_path: Path, capsys) -> None:
    """Mutation proof for the CLI gate: the instrument must report dirty.

    Without this, deleting every command from ``READ_SIDE_CLI`` — or pointing
    the driver at a database it cannot open — would leave the zero above
    passing on an empty set.
    """
    _seed_legacy(tmp_path).close()
    executed, dispatched = _drive_read_side_cli(tmp_path / LEGACY_DB_NAME)
    capsys.readouterr()

    assert dispatched == len(READ_SIDE_CLI)
    assert "retrieve" in executed, (
        "the CLI driver no longer reaches retrieve.py on a legacy core, so the "
        "zero measured on a v1 core is an empty check rather than a passing one"
    )
    assert len(executed) >= 5, sorted(executed)


def test_build_context_refuses_a_v1_core_itself(tmp_path: Path) -> None:
    """The guard has to live in the function that makes the legacy call.

    ``shared_context.build_context`` calls the legacy ranker unconditionally;
    the only ``is_core_v1`` check is one frame up in ``mcp.py``. That makes the
    containment a property of today's call graph, not of the function: a second
    caller adds a live unguarded path without changing the frozen call-site
    count, so neither gate in this file would see it.
    """
    conn = _seed_v1(tmp_path)
    try:
        with pytest.raises(ValueError, match="v1"):
            build_context(conn, "shared context", limit=5)
    finally:
        conn.close()


def _dotted_name(node: ast.expr) -> str | None:
    """``a.b.c`` as a string, or None for anything not a plain attribute chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _legacy_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Names bound to ``retrieve.retrieve``, and names bound to the module.

    ``ast.walk`` rather than a top-level scan, so a function-local or otherwise
    lazy import binds exactly like a module-level one.
    """
    functions: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            relative = node.level > 0
            if module == LEGACY_MODULE or (relative and module == LEGACY_LEAF):
                # from ocbrain.retrieve import retrieve [as x] / from .retrieve import ...
                for alias in node.names:
                    if alias.name in {LEGACY_LEAF, "*"}:
                        functions.add(alias.asname or LEGACY_LEAF)
            elif module == LEGACY_PACKAGE or (relative and not module):
                # from ocbrain import retrieve [as x] / from . import retrieve
                for alias in node.names:
                    if alias.name == LEGACY_LEAF:
                        modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == LEGACY_MODULE:
                    # `import ocbrain.retrieve` binds `ocbrain`; the call reads
                    # the full dotted path, so record that path.
                    modules.add(alias.asname or alias.name)
    return functions - modules, modules


def _legacy_call_sites(root: Path) -> dict[str, int]:
    """Count references to the legacy ranker in every module under ``root``.

    Resolves bindings through the AST rather than matching text, and counts
    every *reference* to the bound name, not only a direct call: binding the
    legacy ranker to another name is a call site one indirection away, and a
    gate that only sees ``retrieve(`` would report it clean.
    """
    call_sites: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "retrieve.py" or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        functions, modules = _legacy_bindings(tree)
        if not functions and not modules:
            continue
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load) and node.id in functions:
                    count += 1
            elif isinstance(node, ast.Attribute):
                if not isinstance(node.ctx, ast.Load) or node.attr != LEGACY_LEAF:
                    continue
                dotted = _dotted_name(node)
                if dotted is not None and dotted.rsplit(".", 1)[0] in modules:
                    count += 1
        if count:
            call_sites[path.relative_to(root).as_posix()] = count
    return call_sites


# Every evasive binding form the scanner must resolve, as (module name, source,
# expected count). Each one is a way of reaching ``ocbrain.retrieve.retrieve``
# that a line-oriented scanner cannot see. This table is the scanner's own
# mutation proof: it is written against forms that are legal Python, not against
# forms the current tree happens to use.
_EVASIVE_MODULES: tuple[tuple[str, str, int], ...] = (
    (
        "function_local_import.py",
        "def shim(conn, q):\n"
        "    from ocbrain.retrieve import retrieve\n"
        "    return retrieve(conn, q, limit=5)\n",
        1,
    ),
    (
        "relative_import.py",
        "from .retrieve import retrieve\n\n\ndef shim(conn, q):\n    return retrieve(conn, q)\n",
        1,
    ),
    (
        "aliased_import.py",
        "from ocbrain.retrieve import retrieve as _blend\n\n\n"
        "def shim(conn, q):\n    return _blend(conn, q)\n",
        1,
    ),
    (
        "dotted_module_import.py",
        "import ocbrain.retrieve\n\n\ndef shim(conn, q):\n"
        "    return ocbrain.retrieve.retrieve(conn, q)\n",
        1,
    ),
    (
        "aliased_module_import.py",
        "import ocbrain.retrieve as legacy\n\n\ndef shim(conn, q):\n"
        "    return legacy.retrieve(conn, q)\n",
        1,
    ),
    (
        "package_relative_module.py",
        "from . import retrieve\n\n\ndef shim(conn, q):\n    return retrieve.retrieve(conn, q)\n",
        1,
    ),
    (
        "package_absolute_module.py",
        "from ocbrain import retrieve as legacy_mod\n\n\ndef shim(conn, q):\n"
        "    return legacy_mod.retrieve(conn, q)\n",
        1,
    ),
    (
        "parenthesized_import.py",
        "from ocbrain.retrieve import (\n    retrieve,\n)\n\n\n"
        "def a(conn, q):\n    return retrieve(conn, q)\n\n\n"
        "def b(conn, q):\n    return retrieve(conn, q, cross_scope=True)\n",
        2,
    ),
    (
        "indirect_reference.py",
        "from ocbrain.retrieve import retrieve\n\nRANKER = retrieve\n\n\n"
        "def shim(conn, q):\n    return RANKER(conn, q)\n",
        1,
    ),
)

# Modules that must NOT be counted: a same-named local function and a same-named
# method attribute. A scanner that fails these is matching text, not bindings.
_INNOCENT_MODULES: tuple[tuple[str, str], ...] = (
    (
        "own_retrieve.py",
        "def retrieve(conn, q):\n    return {'items': []}\n\n\n"
        "def shim(conn, q):\n    return retrieve(conn, q)\n",
    ),
    (
        "method_named_retrieve.py",
        "class Cache:\n    def retrieve(self, key):\n        return None\n\n\n"
        "def shim(cache, key):\n    return cache.retrieve(key)\n",
    ),
)


def test_the_call_site_scanner_resolves_every_evasive_binding_form(tmp_path: Path) -> None:
    """The scanner's own gate: it must see call sites this tree does not use.

    The dynamic driver only sees call sites it dispatches, so this scanner is
    the declared backstop for the rest of the package. A backstop that only
    matches the import form the current tree happens to use is an allow-list,
    and a call site written in any other legal form walks past it. Every form
    below is planted, counted, and required to be found.
    """
    package = tmp_path / "ocbrain"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "retrieve.py").write_text("def retrieve(conn, q, **kw):\n    return {'items': []}\n")
    expected: dict[str, int] = {}
    for name, source, count in _EVASIVE_MODULES:
        (package / name).write_text(source)
        expected[name] = count
    for name, source in _INNOCENT_MODULES:
        (package / name).write_text(source)

    found = _legacy_call_sites(package)
    assert found == expected, (
        "the call-site scanner is an allow-list: it missed or miscounted a legal "
        f"way of reaching ocbrain.retrieve.retrieve. found={found} expected={expected}"
    )


def test_the_legacy_retriever_has_exactly_the_known_call_sites() -> None:
    """Freeze the legacy ranker's blast radius across the whole package.

    The dynamic gate above can only see call sites the tool driver reaches. A
    fourth ``retrieve(...)`` added in a module the driver never dispatches would
    slip past it. This counts the call sites directly, so any new one has to be
    added here and its guard re-proved by hand.
    """
    src = Path(__file__).resolve().parents[1] / "src/ocbrain"
    call_sites = _legacy_call_sites(src)

    assert call_sites == LEGACY_RETRIEVE_CALL_SITES, (
        "the legacy blend retriever gained or lost a call site; each one must sit "
        "behind an is_core_v1 early return, and the dynamic gate above must be "
        f"re-proved before this table is updated. found={call_sites}"
    )
