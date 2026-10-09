from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "brain-promote.sh"


def _curate_argv(**overrides: str) -> list[str]:
    env = os.environ.copy()
    for name in (
        "OCBRAIN_PROMOTE_MAX_TOKENS",
        "OCBRAIN_PROMOTE_PROJECT",
        "OCBRAIN_PROMOTE_PROVIDER",
        "OCBRAIN_PROMOTE_BASE_URL",
        "OCBRAIN_PROMOTE_MODEL",
        "OCBRAIN_PROMOTE_API_KEY_ENV",
        "OCBRAIN_PROMOTE_ENV_FILE",
    ):
        env.pop(name, None)
    env.update(
        {
            "OCBRAIN_DB": "/tmp/ocbrain-shell-test.sqlite",
            "OCBRAIN_PYTHON": "/tmp/ocbrain-shell-test-python",
            "OCBRAIN_ROOT": str(ROOT),
            **overrides,
        }
    )
    completed = subprocess.run(  # noqa: S603 - fixed repository script
        ["/bin/bash", str(SCRIPT), "--print-curate-argv"],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
    )
    assert completed.stdout.endswith(b"\0")
    return [part.decode() for part in completed.stdout[:-1].split(b"\0")]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("anthropic", "16000"),
        ("openai", "16000"),
        ("moonshot", "8000"),
    ],
)
def test_provider_selects_safe_scheduled_token_default(
    provider: str, expected: str
) -> None:
    argv = _curate_argv(OCBRAIN_PROMOTE_PROVIDER=provider)

    assert argv[argv.index("--max-tokens") + 1] == expected


@pytest.mark.parametrize("provider", ["anthropic", "openai", "moonshot"])
def test_explicit_token_override_wins_for_every_provider(provider: str) -> None:
    argv = _curate_argv(
        OCBRAIN_PROMOTE_PROVIDER=provider,
        OCBRAIN_PROMOTE_MAX_TOKENS="12345",
    )

    assert argv[argv.index("--max-tokens") + 1] == "12345"


def test_curate_argv_preserves_quoting_and_forwards_all_arguments() -> None:
    argv = _curate_argv(
        OCBRAIN_DB="/tmp/db path/brain.sqlite",
        OCBRAIN_PYTHON="/tmp/python path/python",
        OCBRAIN_ROOT="/tmp/repo path",
        OCBRAIN_PROMOTE_PROVIDER="moonshot",
        OCBRAIN_PROMOTE_PROJECT="project with spaces",
        OCBRAIN_PROMOTE_MAX_BELIEFS="17",
        OCBRAIN_PROMOTE_MAX_TOKENS="7000",
        OCBRAIN_WIKI_DIR="/tmp/wiki path",
    )

    assert argv == [
        "/tmp/python path/python",
        "/tmp/repo path/scripts/wiki-curator.py",
        "--db",
        "/tmp/db path/brain.sqlite",
        "--provider",
        "moonshot",
        "--project",
        "project with spaces",
        "--wiki-dir",
        "/tmp/wiki path",
        "--max-beliefs",
        "17",
        "--max-tokens",
        "7000",
        "--apply",
    ]


_ENDPOINT_FLAGS = ("--base-url", "--model", "--api-key-env", "--env-file")


def test_endpoint_flags_absent_when_unset() -> None:
    argv = _curate_argv()

    assert not set(_ENDPOINT_FLAGS) & set(argv)
    assert argv[-1] == "--apply"


def test_endpoint_flags_forwarded_when_set() -> None:
    argv = _curate_argv(
        OCBRAIN_PROMOTE_BASE_URL="https://llm.example/v1",
        OCBRAIN_PROMOTE_MODEL="some model",
        OCBRAIN_PROMOTE_API_KEY_ENV="LLM_KEY",
        OCBRAIN_PROMOTE_ENV_FILE="/tmp/env file",
    )

    assert argv[argv.index("--base-url") + 1] == "https://llm.example/v1"
    assert argv[argv.index("--model") + 1] == "some model"
    assert argv[argv.index("--api-key-env") + 1] == "LLM_KEY"
    assert argv[argv.index("--env-file") + 1] == "/tmp/env file"
