from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import pytest

import ocbrain.curator
from ocbrain.core_v1 import append_core_event, init_core_v1, record_core_v1_evidence
from ocbrain.curator import ELIGIBLE_KINDS, select_evidence
from ocbrain.db import connect
from ocbrain.mcp_v1 import decide_proposal_v1, pending_supersede_count
from ocbrain.scope import ScopeTag
from ocbrain.wiki import current_wiki_beliefs


def _curator_module():
    path = Path(__file__).parents[1] / "scripts" / "wiki-curator.py"
    spec = importlib.util.spec_from_file_location("wiki_curator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("provider", "model", "expected_field", "unsupported_field"),
    [
        ("openai", "gpt-5-mini", "max_completion_tokens", "max_tokens"),
        ("moonshot", "moonshot-v1-32k", "max_tokens", "max_completion_tokens"),
    ],
)
def test_openai_compatible_provider_uses_supported_token_budget_field(
    monkeypatch, provider, model, expected_field, unsupported_field
):
    captured: dict[str, bytes] = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["request"] = request.data
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"beliefs": []})},
                }
            ]
        }
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(ocbrain.curator.urllib.request, "urlopen", fake_urlopen)
    result = ocbrain.curator.request_claims(
        provider=provider,
        api_key="test-key-never-sent",
        base_url="https://provider.invalid/v1",
        model=model,
        evidence=[],
        existing=[],
        max_beliefs=1,
        max_tokens=1_234,
    )

    assert result == {"beliefs": []}
    payload = json.loads(captured["request"])
    assert payload[expected_field] == 1_234
    assert unsupported_field not in payload


def _seed_wiki_belief(
    conn,
    *,
    belief_id: str,
    body: str,
    project: str = "test",
    visibility: str = "internal",
    egress_policy: str = "hosted_ok",
) -> None:
    proposal_id = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "wiki_fact",
            "body": body,
            "evidence_ids": [],
            "scope": ScopeTag(
                "project",
                f"project:{project}",
                visibility=visibility,
                egress_policy=egress_policy,
                provenance="test",
            ).to_dict(),
            "confidence": 0.9,
            "attributes": {
                "key": belief_id.removeprefix("belief:"),
                "title": body,
                "category": "system",
            },
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal_id,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="test seed",
    )


def test_selector_enforces_visibility_egress_and_kind_boundaries(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    cases = (
        ("internal hosted", "audit_finding", "internal", "hosted_ok", "test"),
        (
            "internal approval required",
            "audit_finding",
            "internal",
            "approval_required",
            "test",
        ),
        ("internal local", "audit_finding", "internal", "local_only", "test"),
        (
            "confidential approval required",
            "audit_finding",
            "confidential",
            "approval_required",
            "test",
        ),
        (
            "confidential hosted",
            "audit_finding",
            "confidential",
            "hosted_ok",
            "test",
        ),
        ("internal prohibited", "audit_finding", "internal", "prohibited", "test"),
        ("raw transcript", "codex_history_file", "internal", "hosted_ok", "test"),
        ("other project", "audit_finding", "internal", "hosted_ok", "other"),
    )
    ids: dict[str, str] = {}
    for body, kind, visibility, egress_policy, project in cases:
        evidence_id, _ = record_core_v1_evidence(
            conn,
            body=body,
            kind=kind,
            scope=ScopeTag(
                "project",
                f"project:{project}",
                visibility=visibility,
                egress_policy=egress_policy,
            ),
            writer="test",
        )
        ids[body] = evidence_id
    conn.commit()

    default_ids = {
        row["evidence_id"] for row in select_evidence(conn, limit=20, project="test")
    }
    acknowledged_ids = {
        row["evidence_id"]
        for row in select_evidence(
            conn, limit=20, allow_hosted_egress=True, project="test"
        )
    }

    assert default_ids == {ids["internal hosted"]}
    assert acknowledged_ids == {
        ids["internal hosted"],
        ids["internal approval required"],
    }
    assert ids["internal local"] not in acknowledged_ids
    assert ids["confidential approval required"] not in acknowledged_ids
    assert ids["confidential hosted"] not in acknowledged_ids
    assert ids["internal prohibited"] not in acknowledged_ids
    assert ids["raw transcript"] not in acknowledged_ids
    assert ids["other project"] not in acknowledged_ids
    with pytest.raises(ValueError, match="project is required"):
        select_evidence(conn, limit=20)
    conn.close()


def test_hosted_existing_wiki_gate_never_admits_local_or_confidential(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    cases = (
        ("belief:hosted", "hosted belief", "internal", "hosted_ok", "test"),
        (
            "belief:approval",
            "approval belief",
            "internal",
            "approval_required",
            "test",
        ),
        ("belief:local", "local belief", "internal", "local_only", "test"),
        ("belief:prohibited", "prohibited belief", "internal", "prohibited", "test"),
        (
            "belief:confidential",
            "confidential belief",
            "confidential",
            "hosted_ok",
            "test",
        ),
        ("belief:secret", "secret belief", "secret", "approval_required", "test"),
        ("belief:other", "other project belief", "internal", "hosted_ok", "other"),
    )
    for belief_id, body, visibility, egress_policy, project in cases:
        _seed_wiki_belief(
            conn,
            belief_id=belief_id,
            body=body,
            project=project,
            visibility=visibility,
            egress_policy=egress_policy,
        )

    default_ids = {
        row["belief_id"]
        for row in current_wiki_beliefs(
            conn,
            project="test",
            hosted_egress=True,
        )
    }
    acknowledged_ids = {
        row["belief_id"]
        for row in current_wiki_beliefs(
            conn,
            project="test",
            hosted_egress=True,
            allow_approval_required=True,
        )
    }

    assert default_ids == {"belief:hosted"}
    assert acknowledged_ids == {"belief:hosted", "belief:approval"}
    assert "belief:local" not in acknowledged_ids
    assert "belief:prohibited" not in acknowledged_ids
    assert "belief:confidential" not in acknowledged_ids
    assert "belief:secret" not in acknowledged_ids
    assert "belief:other" not in acknowledged_ids
    with pytest.raises(ValueError, match="requires hosted_egress"):
        current_wiki_beliefs(conn, allow_approval_required=True)
    with pytest.raises(ValueError, match="project is required"):
        current_wiki_beliefs(conn, hosted_egress=True)
    conn.close()


def test_hosted_prompt_excludes_local_and_confidential_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    evidence_cases = (
        ("HOSTED EVIDENCE SAFE QUOTE", "internal", "hosted_ok"),
        ("APPROVAL EVIDENCE SAFE QUOTE", "internal", "approval_required"),
        ("LOCAL EVIDENCE MUST NEVER EGRESS", "internal", "local_only"),
        ("CONFIDENTIAL EVIDENCE MUST NEVER EGRESS", "confidential", "hosted_ok"),
    )
    evidence_ids: dict[str, str] = {}
    for body, visibility, egress_policy in evidence_cases:
        evidence_id, _ = record_core_v1_evidence(
            conn,
            body=body,
            kind="audit_finding",
            scope=ScopeTag(
                "project",
                "project:test",
                visibility=visibility,
                egress_policy=egress_policy,
            ),
            writer="test",
        )
        evidence_ids[body] = evidence_id
    belief_cases = (
        ("belief:prompt-hosted", "HOSTED BELIEF SAFE", "internal", "hosted_ok"),
        (
            "belief:prompt-approval",
            "APPROVAL BELIEF SAFE",
            "internal",
            "approval_required",
        ),
        (
            "belief:prompt-local",
            "LOCAL BELIEF MUST NEVER EGRESS",
            "internal",
            "local_only",
        ),
        (
            "belief:prompt-confidential",
            "CONFIDENTIAL BELIEF MUST NEVER EGRESS",
            "confidential",
            "hosted_ok",
        ),
    )
    for belief_id, body, visibility, egress_policy in belief_cases:
        _seed_wiki_belief(
            conn,
            belief_id=belief_id,
            body=body,
            visibility=visibility,
            egress_policy=egress_policy,
        )
    conn.commit()
    conn.close()

    captured: dict[str, bytes] = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["request"] = request.data
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "beliefs": [
                                    {
                                        "key": "hosted-evidence-safe",
                                        "title": "Hosted evidence is safe",
                                        "body": (
                                            "The hosted evidence passed every outbound "
                                            "scope and privacy gate."
                                        ),
                                        "category": "system",
                                        "lifecycle": "current",
                                        "confidence": 0.9,
                                        "supports": [
                                            {
                                                "evidence_id": evidence_ids[
                                                    "HOSTED EVIDENCE SAFE QUOTE"
                                                ],
                                                "quote": "HOSTED EVIDENCE SAFE QUOTE",
                                            }
                                        ],
                                    }
                                ]
                            }
                        )
                    },
                }
            ]
        }
        return io.BytesIO(json.dumps(response).encode())

    curator = _curator_module()
    monkeypatch.setattr(ocbrain.curator.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("KIMI_API_KEY", "test-key-never-sent")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki-curator.py",
            "--provider",
            "moonshot",
            "--db",
            str(db_path),
            "--wiki-dir",
            str(tmp_path / "wiki"),
            "--project",
            "test",
            # This fixture is about the egress boundary, not the thinness gate.
            "--min-evidence-per-project",
            "1",
            "--allow-hosted-egress",
            "--apply",
            "--force",
        ],
    )

    assert curator.main() == 0
    payload = json.loads(captured["request"])
    prompt = payload["messages"][1]["content"]
    assert "HOSTED EVIDENCE SAFE QUOTE" in prompt
    assert "APPROVAL EVIDENCE SAFE QUOTE" in prompt
    assert "HOSTED BELIEF SAFE" in prompt
    assert "APPROVAL BELIEF SAFE" in prompt
    assert "LOCAL EVIDENCE MUST NEVER EGRESS" not in prompt
    assert "CONFIDENTIAL EVIDENCE MUST NEVER EGRESS" not in prompt
    assert "LOCAL BELIEF MUST NEVER EGRESS" not in prompt
    assert "CONFIDENTIAL BELIEF MUST NEVER EGRESS" not in prompt


def test_selection_policy_floor_cannot_be_configured_away() -> None:
    """Hosted curation never treats a local-only label as operator consent."""
    from ocbrain.curator import resolve_selection_policy

    egress, visibility = resolve_selection_policy(
        egress_policies=["hosted_ok", "prohibited"],
        visibilities=["internal", "confidential", "secret"],
    )
    assert "prohibited" not in egress
    assert "secret" not in visibility
    assert "confidential" in visibility

    with pytest.raises(ValueError, match="local_only.*hosted curator"):
        resolve_selection_policy(egress_policies=["hosted_ok", "local_only"])

    # Shipped default stays narrow.
    assert resolve_selection_policy() == (("hosted_ok",), ("internal",))
    # A policy that admits nothing is an error, not a silent empty selection.
    with pytest.raises(ValueError, match="admits nothing"):
        resolve_selection_policy(egress_policies=["prohibited"])
    with pytest.raises(ValueError, match="admits nothing"):
        resolve_selection_policy(visibilities=["secret"])


def test_selection_policy_materializes_one_shot_iterables_once() -> None:
    """Policy resolution must not consume an advertised Iterable twice."""
    from ocbrain.curator import resolve_selection_policy

    class OneShotIterable:
        def __init__(self, values: tuple[str, ...]) -> None:
            self.values = values
            self.iterations = 0

        def __iter__(self):
            if self.iterations:
                raise AssertionError("selection-policy iterable was consumed twice")
            self.iterations += 1
            return iter(self.values)

    egress = OneShotIterable(("hosted_ok",))
    visibility = OneShotIterable(("internal", "confidential"))
    assert resolve_selection_policy(
        egress_policies=egress,
        visibilities=visibility,
        allow_hosted_egress=True,
    ) == (("approval_required", "hosted_ok"), ("confidential", "internal"))
    assert egress.iterations == visibility.iterations == 1

    one_shot_local = (policy for policy in ("hosted_ok", "local_only"))
    with pytest.raises(ValueError, match="local_only.*hosted curator"):
        resolve_selection_policy(
            egress_policies=one_shot_local,
            allow_hosted_egress=True,
        )


def test_local_only_policy_is_rejected_before_evidence_selection(tmp_path: Path) -> None:
    """Local-only evidence must be reclassified before any hosted selection."""
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    cases = (
        ("LOCAL ONLY INTERNAL BODY", "internal", "local_only"),
        ("HOSTED OK INTERNAL BODY", "internal", "hosted_ok"),
        ("PROHIBITED CONFIDENTIAL BODY", "confidential", "prohibited"),
        ("SECRET LOCAL BODY", "secret", "local_only"),
    )
    for body, visibility, egress_policy in cases:
        record_core_v1_evidence(
            conn,
            body=body,
            kind="audit_finding",
            scope=ScopeTag(
                "project", "project:test", visibility=visibility, egress_policy=egress_policy
            ),
            writer="test",
        )
    conn.commit()

    default_bodies = {
        row["body"] for row in select_evidence(conn, limit=20, project="test")
    }
    assert default_bodies == {"HOSTED OK INTERNAL BODY"}

    with pytest.raises(ValueError, match="local_only.*hosted curator"):
        select_evidence(
            conn,
            limit=20,
            project="test",
            egress_policies=["hosted_ok", "local_only"],
            visibilities=["internal", "confidential"],
        )
    assert conn.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 0


def test_curation_egress_is_audited_before_the_send(tmp_path: Path) -> None:
    """Hosted-cleared evidence remains auditable before every send."""
    from ocbrain.curator import record_curation_egress

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    evidence_id, _ = record_core_v1_evidence(
        conn,
        body="AUDITED EVIDENCE BODY",
        kind="audit_finding",
        scope=ScopeTag(
            "project", "project:test", visibility="internal", egress_policy="hosted_ok"
        ),
        writer="test",
    )
    conn.commit()
    rows = select_evidence(
        conn, limit=20, project="test", egress_policies=["hosted_ok"]
    )
    assert len(rows) == 1

    audit_id = record_curation_egress(
        conn,
        evidence=rows,
        provider="anthropic",
        model="claude-sonnet-5",
        project="test",
        egress_policies=("hosted_ok",),
    )
    audit = conn.execute(
        "SELECT target, context_json, included_json, payload_hash FROM egress_audits WHERE id=?",
        (audit_id,),
    ).fetchone()
    assert audit["target"] == "anthropic:claude-sonnet-5"
    assert evidence_id in audit["included_json"]
    assert "wiki_curation" in audit["context_json"]
    assert "hosted_ok" in audit["context_json"]
    # The body itself is not copied into the audit, only its identity and size.
    assert "AUDITED EVIDENCE BODY" not in audit["included_json"]
    assert audit["payload_hash"]


def test_curator_updates_a_restated_fact_instead_of_minting_a_second(tmp_path: Path) -> None:
    """A reworded claim must update the belief that already states the fact.

    belief_id derives from the topic key the model chose, so a later run that
    rewords the same fact under a new key used to create a second served belief.
    Exact-body dedup never catches it, and every run added another phrasing.
    """
    from ocbrain.curator import apply_claims

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    first = [
        {
            "key": "hermes-runtime-config",
            "title": "Hermes runtime",
            "body": "Hermes runs as the launchd service ai.hermes.gateway with auto-start "
            "and restart, delivering to Telegram.",
            "category": "system",
            "lifecycle": "durable",
            "confidence": 0.9,
            "evidence_ids": [],
        }
    ]
    applied = apply_claims(conn, first, model="test", project="test")
    assert len(applied["applied"]) == 1
    original_id = applied["applied"][0]

    # Same fact, different key and wording — the shape a later run produces.
    second = [
        {
            "key": "hermes-runtime-service",
            "title": "Hermes runtime",
            "body": "Hermes runs as launchd service ai.hermes.gateway with auto-start and "
            "auto-restart, delivering to Telegram.",
            "category": "system",
            "lifecycle": "durable",
            "confidence": 0.9,
            "evidence_ids": [],
        }
    ]
    reapplied = apply_claims(conn, second, model="test", project="test")

    # It updated the existing belief rather than adding a second one.
    assert reapplied["applied"] == [original_id]
    serving = conn.execute(
        "SELECT belief_id, body FROM current_beliefs WHERE serve=1 AND status='current'"
    ).fetchall()
    assert len(serving) == 1
    assert str(serving[0]["belief_id"]) == original_id
    assert "auto-restart" in str(serving[0]["body"])
    conn.close()


def test_curator_still_adds_a_genuinely_different_fact(tmp_path: Path) -> None:
    """Restatement collapsing must not swallow distinct knowledge."""
    from ocbrain.curator import apply_claims

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    def claim(key: str, body: str) -> dict:
        return {
            "key": key,
            "title": key,
            "body": body,
            "category": "system",
            "lifecycle": "durable",
            "confidence": 0.9,
            "evidence_ids": [],
        }

    apply_claims(
        conn,
        [claim("hermes-runtime", "Hermes runs as the launchd service ai.hermes.gateway.")],
        model="test",
        project="test",
    )
    apply_claims(
        conn,
        [
            claim(
                "clickhouse-access",
                "Production ClickHouse access is SELECT-only and the live host rotates.",
            )
        ],
        model="test",
        project="test",
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM current_beliefs WHERE serve=1 AND status='current'"
        ).fetchone()[0]
        == 2
    )
    conn.close()


def test_durable_preference_claims_compile_to_global_scope(tmp_path: Path) -> None:
    """A durable preference is doctrine, not a fact about whoever ran the curator.

    Stamped into `project:<runner>` it is invisible everywhere else, which is how
    a brain accumulates preferences no other project can reach.
    """
    from ocbrain.curator import apply_claims, claim_scope

    assert claim_scope({"category": "preference", "lifecycle": "durable"}, project="hermes") == (
        "global",
        "global:doctrine",
    )
    assert claim_scope({"category": "preference", "lifecycle": "current"}, project="hermes") == (
        "project",
        "project:hermes",
    )
    assert claim_scope({"category": "system", "lifecycle": "durable"}, project="hermes") == (
        "project",
        "project:hermes",
    )

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    applied = apply_claims(
        conn,
        [
            {
                "key": "reply-style",
                "title": "Reply style",
                "body": "Answers open with the result and never restate the question.",
                "category": "preference",
                "lifecycle": "durable",
                "confidence": 0.9,
                "evidence_ids": [],
            },
            {
                "key": "sotu-mart",
                "title": "SOTU mart",
                "body": "The sandbox_sotu mart is rebuilt by a launchd job every morning.",
                "category": "project",
                "lifecycle": "durable",
                "confidence": 0.9,
                "evidence_ids": [],
            },
        ],
        model="test",
        project="hermes",
    )
    assert len(applied["applied"]) == 2

    scopes = {
        str(row["belief_id"]): (str(row["scope_type"]), str(row["scope_id"]))
        for row in conn.execute(
            "SELECT belief_id, scope_type, scope_id FROM current_beliefs WHERE serve=1"
        )
    }
    by_scope = set(scopes.values())
    assert ("global", "global:doctrine") in by_scope
    assert ("project", "project:hermes") in by_scope
    # Wider reach is not wider egress: the curator still writes local_only.
    policies = conn.execute("SELECT egress_policy FROM current_beliefs").fetchall()
    assert {str(row["egress_policy"]) for row in policies} == {"local_only"}
    conn.close()


def test_global_belief_dedups_against_project_restatement(tmp_path: Path) -> None:
    """A globally promoted fact is updated, not re-minted once per project.

    The dedup lookup used to see only the running project's scope, so the first
    curator run after a promotion would state the same thing a second time.
    """
    from ocbrain.curator import apply_claims

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    def claim(key: str, body: str) -> dict:
        return {
            "key": key,
            "title": "Reply style",
            "body": body,
            "category": "preference",
            "lifecycle": "durable",
            "confidence": 0.9,
            "evidence_ids": [],
        }

    first = apply_claims(
        conn,
        [claim("reply-style", "Answers open with the result and never restate the question.")],
        model="test",
        project="hermes",
    )
    original_id = first["applied"][0]

    # A different project, a reworded restatement of the same durable preference.
    second = apply_claims(
        conn,
        [
            claim(
                "answer-style",
                "Answers open with the result and never restate the question asked.",
            )
        ],
        model="test",
        project="codex",
    )

    assert second["applied"] == [original_id]
    serving = conn.execute(
        "SELECT belief_id, scope_id FROM current_beliefs WHERE serve=1 AND status='current'"
    ).fetchall()
    assert len(serving) == 1
    assert str(serving[0]["scope_id"]) == "global:doctrine"
    conn.close()
# --------------------------------------------------------------------------- #
# Multi-project curation
#
# A curator pinned to one project scope curates one spelling of one project. On
# a real brain that was 5 of 574 eligible objects, and the wiki froze for six
# days while evidence kept arriving in ~40 other project scopes. These pin the
# repair and, just as importantly, the cost discipline that has to survive it:
# a project only bills a hosted call when its own evidence changed.
# --------------------------------------------------------------------------- #

_EVIDENCE_BLOCK_RE = re.compile(
    r'<evidence id="([^"]+)"[^>]*>\n(.*?)\n</evidence>', re.DOTALL
)


def _seed_project_evidence(conn, project: str, count: int, *, marker: str = "") -> None:
    """Give ``project`` ``count`` curation-eligible objects, tagged so a stubbed
    provider can tell which project's prompt it is answering."""
    for index in range(count):
        record_core_v1_evidence(
            conn,
            body=(
                f"PROJECT {project} FACT {marker}{index}: the {project} runner writes "
                f"its receipt to /var/{project}/receipt-{marker}{index}.json every run."
            ),
            kind="task_closeout_summary",
            scope=ScopeTag(
                "project",
                f"project:{project}",
                visibility="internal",
                egress_policy="hosted_ok",
            ),
            writer="test",
        )
    conn.commit()


def _stub_provider(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer every curation request locally, and record the prompts sent.

    A test must never reach a hosted provider, and the prompt is also the only
    place the project under compilation is visible, so recording it is what lets
    a test assert which projects billed a call.
    """
    prompts: list[str] = []

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data)
        prompt = payload["messages"][1]["content"]
        prompts.append(prompt)
        match = _EVIDENCE_BLOCK_RE.search(prompt)
        assert match is not None, "curation prompt carried no evidence"
        evidence_id, body = match.group(1), match.group(2)
        belief = {
            "key": f"receipt-path-{len(prompts)}",
            "title": "Runner receipt path",
            "body": body[:400],
            "category": "system",
            "lifecycle": "current",
            "confidence": 0.9,
            "supports": [{"evidence_id": evidence_id, "quote": body[:60]}],
        }
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"beliefs": [belief]})},
                }
            ]
        }
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(ocbrain.curator.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("KIMI_API_KEY", "test-key-never-sent")
    return prompts


def _run_curator(monkeypatch: pytest.MonkeyPatch, db_path: Path, wiki_dir: Path, *args):
    curator = _curator_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki-curator.py",
            "--provider",
            "moonshot",
            "--db",
            str(db_path),
            "--wiki-dir",
            str(wiki_dir),
            *args,
        ],
    )
    return curator.main()


def _projects_in(prompts: list[str]) -> list[str]:
    """Which project each recorded prompt was compiled for."""
    found = []
    for prompt in prompts:
        match = re.search(r"PROJECT (\S+) FACT", prompt)
        assert match is not None, "prompt carried no project marker"
        found.append(match.group(1))
    return found


def test_select_evidence_includes_alias_variant_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection matches a project by canonical spelling, not exact string.

    Stored rows keep whatever spelling the client used and are never rewritten,
    so equality against one spelling leaves the rest of the project unreachable.
    """
    monkeypatch.setenv(
        "OCBRAIN_SCOPES_ALIASES",
        json.dumps({"project:coframe-brain-v2": "project:coframe-brain"}),
    )
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    stored = {
        "canonical": "project:coframe-brain",
        "spaced and cased": "project:Coframe Brain",
        "underscored": "project:coframe_brain",
        "aliased": "project:coframe-brain-v2",
        "unrelated": "project:personalization-headroom",
    }
    for body, scope_id in stored.items():
        record_core_v1_evidence(
            conn,
            body=f"EVIDENCE FROM THE {body} SCOPE SPELLING",
            kind="task_closeout_summary",
            scope=ScopeTag(
                "project", scope_id, visibility="internal", egress_policy="hosted_ok"
            ),
            writer="test",
        )
    conn.commit()

    reached = {
        str(row["scope_id"])
        for row in select_evidence(conn, limit=50, project="Coframe Brain")
    }
    assert reached == {
        "project:coframe-brain",
        "project:Coframe Brain",
        "project:coframe_brain",
        "project:coframe-brain-v2",
    }
    # Widening only ever adds spellings of the project the caller named.
    assert "project:personalization-headroom" not in reached
    conn.close()


def test_history_file_kinds_are_never_eligible(tmp_path: Path) -> None:
    """Raw transcripts must stay ineligible however far the scope gate widens.

    Curating more projects only stays defensible if the thing that never leaves
    the machine still never leaves it. The kind allow-list is the whole gate, so
    it is asserted directly rather than through one sampled transcript kind.
    """
    assert not [kind for kind in ELIGIBLE_KINDS if kind.endswith("_history_file")]

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    for kind in (
        "claude_history_file",
        "codex_history_file",
        "cursor_history_file",
        "hermes_history_file",
        "openclaw_history_file",
        "unknown_history_file",
    ):
        record_core_v1_evidence(
            conn,
            body=f"RAW TRANSCRIPT FROM {kind} THAT MUST NEVER EGRESS",
            kind=kind,
            scope=ScopeTag(
                "project", "project:test", visibility="internal", egress_policy="hosted_ok"
            ),
            writer="test",
        )
    conn.commit()
    assert (
        select_evidence(
            conn,
            limit=50,
            project="test",
            egress_policies=["hosted_ok", "approval_required"],
            visibilities=["internal", "confidential"],
        )
        == []
    )
    conn.close()


def test_cli_rejects_local_only_egress_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    curator = _curator_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki-curator.py",
            "--db",
            str(tmp_path / "core.sqlite"),
            "--project",
            "test",
            "--egress-policy",
            "local_only",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        curator.main()
    assert exc_info.value.code == 2
    assert "invalid choice: 'local_only'" in capsys.readouterr().err


def test_multi_project_state_digest_is_per_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One state file, one digest per project.

    A shell loop over ``--project`` cannot do this: state.json is one file per
    wiki dir, so the second project would overwrite the first project's digest
    and every scheduled cycle would re-bill a call for every project.
    """
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    _seed_project_evidence(conn, "alpha", 4)
    _seed_project_evidence(conn, "beta", 4)
    conn.close()

    prompts = _stub_provider(monkeypatch)
    wiki_dir = tmp_path / "wiki"
    assert (
        _run_curator(
            monkeypatch,
            db_path,
            wiki_dir,
            "--project",
            "alpha",
            "--project",
            "beta",
            "--apply",
        )
        == 0
    )

    assert sorted(_projects_in(prompts)) == ["alpha", "beta"]
    state = json.loads((wiki_dir / "state.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == ocbrain.curator.WIKI_STATE_SCHEMA
    assert sorted(state["projects"]) == ["alpha", "beta"]
    digests = {name: entry["input_digest"] for name, entry in state["projects"].items()}
    assert digests["alpha"] != digests["beta"]
    assert all(digests.values())


def test_unchanged_project_digest_skips_api_call_while_changed_project_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost discipline: only the project whose evidence moved bills a call."""
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    _seed_project_evidence(conn, "alpha", 4)
    _seed_project_evidence(conn, "beta", 4)
    conn.close()

    prompts = _stub_provider(monkeypatch)
    wiki_dir = tmp_path / "wiki"
    scopes = ("--project", "alpha", "--project", "beta", "--apply")
    assert _run_curator(monkeypatch, db_path, wiki_dir, *scopes) == 0
    assert sorted(_projects_in(prompts)) == ["alpha", "beta"]

    # A completely quiet cycle costs nothing at all.
    prompts.clear()
    assert _run_curator(monkeypatch, db_path, wiki_dir, *scopes) == 0
    assert prompts == []

    # New evidence in one project bills that project, and only that project.
    conn = connect(db_path)
    _seed_project_evidence(conn, "beta", 2, marker="new-")
    conn.close()
    assert _run_curator(monkeypatch, db_path, wiki_dir, *scopes) == 0
    assert _projects_in(prompts) == ["beta"]

    state = json.loads((wiki_dir / "state.json").read_text(encoding="utf-8"))
    assert sorted(state["projects"]) == ["alpha", "beta"]


def test_legacy_flat_state_json_migrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing install keeps its short-circuit across the upgrade.

    Pre-multi-project runs wrote one flat ``input_digest`` for the pinned
    ``workspace`` project. Reading it as anything other than workspace's digest
    would re-bill a hosted call for a project that is already up to date.
    """
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    _seed_project_evidence(conn, "workspace", 4)
    conn.close()

    prompts = _stub_provider(monkeypatch)
    wiki_dir = tmp_path / "wiki"
    assert _run_curator(monkeypatch, db_path, wiki_dir, "--project", "workspace", "--apply") == 0
    assert len(prompts) == 1
    state = json.loads((wiki_dir / "state.json").read_text(encoding="utf-8"))
    digest = state["projects"]["workspace"]["input_digest"]

    # Rewrite state.json in the pre-upgrade shape and confirm it still gates.
    (wiki_dir / "state.json").write_text(
        json.dumps({"schema_version": "ocbrain.wiki-state.v1", "input_digest": digest}),
        encoding="utf-8",
    )
    prompts.clear()
    assert _run_curator(monkeypatch, db_path, wiki_dir, "--project", "workspace", "--apply") == 0
    assert prompts == []

    # A legacy digest belongs to workspace alone; another project still runs.
    conn = connect(db_path)
    _seed_project_evidence(conn, "beta", 4)
    conn.close()
    (wiki_dir / "state.json").write_text(
        json.dumps({"schema_version": "ocbrain.wiki-state.v1", "input_digest": digest}),
        encoding="utf-8",
    )
    assert (
        _run_curator(
            monkeypatch,
            db_path,
            wiki_dir,
            "--project",
            "workspace",
            "--project",
            "beta",
            "--apply",
        )
        == 0
    )
    assert _projects_in(prompts) == ["beta"]


def test_thin_project_is_skipped_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A scope too thin to justify a hosted call is skipped out loud."""
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    _seed_project_evidence(conn, "thin", 2)
    _seed_project_evidence(conn, "thick", 4)
    conn.close()

    prompts = _stub_provider(monkeypatch)
    assert (
        _run_curator(
            monkeypatch,
            db_path,
            tmp_path / "wiki",
            "--project",
            "thin",
            "--project",
            "thick",
            "--min-evidence-per-project",
            "3",
            "--apply",
        )
        == 0
    )

    assert _projects_in(prompts) == ["thick"]
    lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    skipped = [line for line in lines if line.get("status") == "skipped_thin_project"]
    assert [line["project"] for line in skipped] == ["thin"]
    assert skipped[0]["eligible_evidence"] == 2
    assert skipped[0]["min_evidence_per_project"] == 3
    rollup = next(line for line in lines if line["action"] == "wiki-curate-rollup")
    assert rollup["projects_by_status"] == {
        "completed": ["thick"],
        "skipped_thin_project": ["thin"],
    }
    assert rollup["hosted_calls"] == 1


def test_rewording_a_doctrine_fact_does_not_demote_it(tmp_path: Path) -> None:
    """An update to a doctrine fact must not move it into a project scope.

    An approved proposal writes its scope onto the belief. A claim that
    `claim_scope` types as project-scoped — anything that is not a durable
    preference — would therefore drag the `global:doctrine` fact it restates
    down with it, undoing a promotion that required a named approver.
    """
    from ocbrain.curator import apply_claims

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    doctrine_id = "belief:doctrine-gcloud"
    proposal_id = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": doctrine_id,
            "belief_type": "wiki_fact",
            "body": "For Coframe production infrastructure, use the gcloud CLI rather "
            "than kubectl.",
            "evidence_ids": [],
            "scope": ScopeTag(
                "global",
                "global:doctrine",
                visibility="internal",
                egress_policy="hosted_ok",
                provenance="test",
            ).to_dict(),
            "confidence": 0.95,
            "attributes": {"key": "gcloud-over-kubectl", "category": "workflow"},
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal_id,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="doctrine seed",
    )
    conn.commit()

    restatement = [
        {
            "key": "gcloud-not-kubectl",
            "title": "gcloud over kubectl",
            "body": "Use the gcloud CLI rather than kubectl for Coframe production "
            "infrastructure.",
            "category": "workflow",
            "lifecycle": "durable",
            "confidence": 0.9,
            "evidence_ids": [],
        }
    ]
    # The first project rewords the doctrine fact in place; the second finds it
    # already saying exactly that and writes nothing at all.
    first = apply_claims(conn, restatement, model="test", project="coframe")
    assert first["applied"] == [doctrine_id]
    second = apply_claims(
        conn, restatement, model="test", project="coframe-personalization"
    )
    assert second["applied"] == []
    assert second["unchanged"] == [doctrine_id]

    served = conn.execute(
        "SELECT belief_id, scope_type, scope_id FROM current_beliefs "
        "WHERE belief_type='wiki_fact' AND serve=1 AND status='current'"
    ).fetchall()
    assert len(served) == 1
    assert str(served[0]["belief_id"]) == doctrine_id
    # A rewording is not a demotion: only a scope_promoted event moves a belief.
    assert str(served[0]["scope_type"]) == "global"
    assert str(served[0]["scope_id"]) == "global:doctrine"
    conn.close()


def test_same_key_in_another_scope_never_mints_a_second_copy(tmp_path: Path) -> None:
    """A key already served anywhere names THE fact, whatever the wording.

    Restatement similarity is a heuristic and it missed on a live corpus: a
    coframe-scoped run reworded the doctrine-scoped asa2 fact below the
    threshold and minted a second serving copy of the same key, which wiki-lint
    then flagged as conflicting-key. Key equality is the identity test wiki-lint
    already enforces corpus-wide, so minting must honor it too.

    What the collision *does* changed with the contradiction cascade. A key
    collision carrying a different statement is a correction, and this target is
    doctrine, which is never replaced unattended: the supersession is recorded
    as a pending proposal and the doctrine fact keeps serving its own words
    until an operator decides. What must not happen either way is a second
    serving belief under the key, or doctrine quietly demoted to the running
    project's scope.
    """
    from ocbrain.curator import apply_claims

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    first = apply_claims(
        conn,
        [
            {
                "key": "research-vm-live",
                "title": "Live research VM",
                "body": "The live analysis VM is asa2; asa1 is terminated.",
                "category": "preference",
                "lifecycle": "durable",
                "confidence": 0.9,
                "evidence_ids": [],
            }
        ],
        model="test",
        project="workspace",
    )
    doctrine_id = first["applied"][0]

    # Same key from another project, worded far past any restatement threshold,
    # and typed so claim_scope would stamp it project-scoped if it minted fresh.
    second = apply_claims(
        conn,
        [
            {
                "key": "research-vm-live",
                "title": "Live research VM",
                "body": (
                    "For applied-science infrastructure, use asa2 going forward; "
                    "the earlier primary box was decommissioned."
                ),
                "category": "system",
                "lifecycle": "durable",
                "confidence": 0.85,
                "evidence_ids": [],
            }
        ],
        model="test",
        project="coframe",
    )

    assert second["applied"] == []
    assert second["deferred"] == [doctrine_id]
    rows = conn.execute(
        "SELECT belief_id, scope_id, body FROM current_beliefs "
        "WHERE serve=1 AND json_extract(attributes_json,'$.key')='research-vm-live'"
    ).fetchall()
    assert len(rows) == 1
    assert str(rows[0]["belief_id"]) == doctrine_id
    assert str(rows[0]["scope_id"]) == "global:doctrine"
    assert "asa1 is terminated" in str(rows[0]["body"])
    assert pending_supersede_count(conn) == 1
    conn.close()


def test_same_key_same_body_elsewhere_is_unchanged(tmp_path: Path) -> None:
    """An identical claim under a key served elsewhere makes no proposal."""
    from ocbrain.curator import apply_claims

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    body = "The live analysis VM is asa2; asa1 is terminated."
    claim = {
        "key": "research-vm-live",
        "title": "Live research VM",
        "body": body,
        "category": "preference",
        "lifecycle": "durable",
        "confidence": 0.9,
        "evidence_ids": [],
    }
    first = apply_claims(conn, [claim], model="test", project="workspace")
    second = apply_claims(
        conn,
        [{**claim, "category": "system"}],
        model="test",
        project="coframe",
    )
    assert second["applied"] == []
    assert second["unchanged"] == [first["applied"][0]]
    conn.close()


def test_a_restatement_never_renames_a_key_another_belief_holds(tmp_path: Path) -> None:
    """Key identity outranks body similarity when choosing the update target.

    Writing a claim's attributes onto a restatement target renames that
    target's key. If the claim's key is already served by a different belief,
    that rename produces two serving beliefs sharing one key — which is how
    `hermes-auxiliary-routing-fix` collided on the live corpus, a belief minted
    under one key being rekeyed onto another belief's key an hour later.
    """
    from ocbrain.curator import apply_claims

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    def claim(key: str, body: str) -> dict:
        return {
            "key": key,
            "title": "Routing defect",
            "body": body,
            "category": "system",
            "lifecycle": "durable",
            "confidence": 0.85,
            "evidence_ids": ["evd_shared"],
        }

    # An older belief under its own key, phrased close to what comes next.
    older = apply_claims(
        conn,
        [
            claim(
                "aux-api-mode-defect",
                "Auxiliary routing sends calls without an explicit api_mode.",
            )
        ],
        model="test",
        project="coframe",
    )["applied"][0]
    # A belief that owns the key the next claim will use.
    owner = apply_claims(
        conn,
        [claim("aux-routing-fix", "Routing was repaired by pinning the auxiliary api_mode.")],
        model="test",
        project="coframe",
    )["applied"][0]
    assert older != owner

    # Same key as the owner, worded as a restatement of the OLDER belief.
    apply_claims(
        conn,
        [
            claim(
                "aux-routing-fix",
                "Auxiliary routing sends calls without an explicit api_mode set.",
            )
        ],
        model="test",
        project="coframe",
    )

    keys = conn.execute(
        "SELECT json_extract(attributes_json,'$.key') k, COUNT(*) n FROM current_beliefs "
        "WHERE serve=1 GROUP BY 1 HAVING n > 1"
    ).fetchall()
    assert keys == [], f"a key is served by more than one belief: {[tuple(r) for r in keys]}"
    conn.close()
