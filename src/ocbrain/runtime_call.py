"""One-shot runtime-profile fallback for a broken MCP stdio connection.

This is intentionally not a second server. It executes exactly one normal
runtime MCP tool call against the same database and exits. Its purpose is to
preserve feedback and closeout receipts when a client reports ``Transport
closed`` and cannot reconnect its per-task stdio transport.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ocbrain.db import DEFAULT_DB_PATH, connect
from ocbrain.mcp import RUNTIME_PROFILE, RUNTIME_TOOLS, handle_request
from ocbrain.scope import DELIVERY_TARGETS, LOCAL_MODEL_TARGET


def invoke(
    db_path: Path,
    tool: str,
    arguments: dict[str, Any],
    *,
    delivery_target: str = LOCAL_MODEL_TARGET,
) -> Any:
    if tool not in RUNTIME_TOOLS:
        raise PermissionError(f"one-shot fallback permits runtime tools only: {tool}")
    conn = connect(db_path)
    try:
        response = handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            },
            profile=RUNTIME_PROFILE,
            delivery_target=delivery_target,
        )
    finally:
        conn.close()
    if not isinstance(response, dict):
        raise RuntimeError("OCBrain returned no response")
    error = response.get("error")
    if isinstance(error, dict):
        raise RuntimeError(str(error.get("message") or error))
    content = ((response.get("result") or {}).get("content") or [])
    if not content or not isinstance(content[0], dict):
        raise RuntimeError("OCBrain returned an invalid tool result")
    text = content[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError("OCBrain returned a non-text tool result")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Execute one OCBrain runtime tool when a client's MCP stdio transport is closed"
        )
    )
    result.add_argument("tool", choices=sorted(RUNTIME_TOOLS))
    result.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    result.add_argument(
        "--delivery-target",
        choices=sorted(DELIVERY_TARGETS),
        default=LOCAL_MODEL_TARGET,
        help="Server-controlled delivery policy for this one-shot call",
    )
    result.add_argument(
        "--arguments-file",
        type=Path,
        help="JSON object containing tool arguments; omit to read JSON from stdin",
    )
    return result


def _arguments(path: Path | None) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8") if path is not None else sys.stdin.read()
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        payload = invoke(
            args.db.expanduser().resolve(),
            args.tool,
            _arguments(args.arguments_file),
            delivery_target=args.delivery_target,
        )
    except Exception as exc:  # noqa: BLE001 - command must return a stable failure
        print(
            json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
