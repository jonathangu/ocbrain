from __future__ import annotations

import pytest

from ocbrain.text import find_probable_secret_leaks, redact_secrets


@pytest.mark.parametrize("key", ["PASSWORD", "DB_PASSWORD", "api_key", "clientSecret"])
@pytest.mark.parametrize("value", ["test" + "pass", "abc" + "123", "simpleword"])
def test_short_explicit_assignments_remain_protected(key: str, value: str) -> None:
    text = f"{key}={value}"
    assert "assigned_secret" in find_probable_secret_leaks(text)
    assert value not in redact_secrets(text)
    assert find_probable_secret_leaks(redact_secrets(text)) == []
