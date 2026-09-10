from __future__ import annotations

from ocbrain.entities import extract_entities

VOCABULARY = {
    "asa2": ["asa2", "applied-science-analytics-2"],
    "recs worker": ["recs-api", "recs.cofra.me"],
}


def _kinds(pairs: list[tuple[str, str]], kind: str) -> list[str]:
    return [entity for entity, entity_kind in pairs if entity_kind == kind]


def test_vocabulary_names_and_aliases_resolve_to_the_canonical_name() -> None:
    assert extract_entities("the asa2 headroom leg", VOCABULARY) == [("asa2", "vocab")]
    assert extract_entities("APPLIED-SCIENCE-ANALYTICS-2 lag", VOCABULARY) == [("asa2", "vocab")]
    assert extract_entities("recs-api is slow", VOCABULARY) == [("recs worker", "vocab")]


def test_vocabulary_matches_on_word_boundaries() -> None:
    assert extract_entities("asa2x is a different thing", VOCABULARY) == []


def test_vocabulary_never_emits_a_name_shorter_than_three_characters() -> None:
    assert extract_entities("ab cd ef", {"ab": ["ab"]}) == []


def test_pull_request_references_are_lowercased() -> None:
    assert extract_entities("coframe/coframe#3695 landed", VOCABULARY) == [
        ("coframe/coframe#3695", "pr")
    ]
    assert extract_entities("see #3695", VOCABULARY) == [("#3695", "pr")]
    assert extract_entities("FIX #12", VOCABULARY) == [("#12", "pr")]


def test_a_one_digit_pull_request_reference_is_not_an_entity() -> None:
    assert extract_entities("see #1 for the change", VOCABULARY) == []


def test_record_ids_are_extracted_for_every_stable_prefix() -> None:
    text = (
        "belief_0123456789abcdef evd_0123456789abcdef close_0123456789abcdef "
        "ret_0123456789abcdef evt_0123456789abcdef"
    )
    assert _kinds(extract_entities(text, VOCABULARY), "id") == [
        "belief_0123456789abcdef",
        "close_0123456789abcdef",
        "evd_0123456789abcdef",
        "evt_0123456789abcdef",
        "ret_0123456789abcdef",
    ]


def test_hosts_are_extracted_from_urls_and_from_standing_text() -> None:
    assert ("recs.cofra.me", "host") in extract_entities(
        "https://recs.cofra.me/health is up", VOCABULARY
    )
    assert ("asa2.example.com", "host") in extract_entities(
        "ASA2.EXAMPLE.COM timed out", VOCABULARY
    )


def test_a_filename_is_not_a_host() -> None:
    assert extract_entities("see README.md and config.json", VOCABULARY) == []
    assert extract_entities("see notes.txt", VOCABULARY) == []


def test_a_hex_digest_is_an_entity_and_a_shorter_run_is_not() -> None:
    assert extract_entities("6ecf3ed", VOCABULARY) == [("6ecf3ed", "digest")]
    assert extract_entities("abcdef", VOCABULARY) == []


def test_dates_are_entities_only_in_their_dated_shapes() -> None:
    assert ("2026-09-10", "date") in extract_entities("shipped 2026-09-10", VOCABULARY)
    assert ("20260910", "date") in extract_entities("shipped 20260910T120000", VOCABULARY)
    assert _kinds(extract_entities("the count was 20260910 units", VOCABULARY), "date") == []


def test_generic_words_are_not_entities() -> None:
    assert extract_entities("the cadence of the retrain", VOCABULARY) == []


def test_extraction_is_deterministic() -> None:
    text = "asa2#3695 recs.cofra.me 6ecf3ed 2026-09-10 belief_0123456789abcdef"
    assert extract_entities(text, VOCABULARY) == extract_entities(text, VOCABULARY)


def test_an_empty_vocabulary_still_extracts_code_shaped_entities() -> None:
    assert extract_entities("asa2 headroom", {}) == []
    assert extract_entities("#3695", None) == [("#3695", "pr")]


def test_abbreviations_and_version_strings_are_not_hosts() -> None:
    found = extract_entities("see e.g. python 3.11.16 or i.e. recs.cofra.me")
    hosts = [entity for entity, kind in found if kind == "host"]
    assert hosts == ["recs.cofra.me"]
