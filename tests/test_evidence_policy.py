import json

import pytest

from deepresearch.llm import Message

from deepresearch.research import (
    EvidencePolicy,
    EvidencePolicyConfig,
    ResearchState,
)


def add_search(
    state: ResearchState,
    *,
    query: str = "LangGraph",
    results: list[dict] | None = None,
) -> None:

    if results is None:

        results = [
            {
                "source_id":
                    "S_11111111",
                "title":
                    "Source A",
                "url":
                    "https://example.com/a",
                "content":
                    "Snippet A",
                "score":
                    0.4,
            },
            {
                "source_id":
                    "S_22222222",
                "title":
                    "Source B",
                "url":
                    "https://example.com/b",
                "content":
                    "Snippet B",
                "score":
                    0.95,
            },
            {
                "source_id":
                    "S_33333333",
                "title":
                    "Source C",
                "url":
                    "https://example.com/c",
                "content":
                    "Snippet C",
                "score":
                    0.7,
            },
        ]

    message = Message(
        role="tool",
        name="web_search",
        tool_call_id="search_call",
        content=json.dumps(
            {
                "query": query,
                "results": results,
            }
        ),
    )

    assert state.ingest_message(
        message
    )


def add_page(
    state: ResearchState,
    *,
    source_id: str,
    url: str,
) -> None:

    message = Message(
        role="tool",
        name="web_page_reader",
        tool_call_id=(
            f"reader_{source_id}"
        ),
        content=json.dumps(
            {
                "source_id":
                    source_id,
                "title":
                    "Full page",
                "url":
                    url,
                "content":
                    "Full page evidence",
                "content_type":
                    "text/html",
                "truncated":
                    False,
            }
        ),
    )

    assert state.ingest_message(
        message
    )


def test_invalid_policy_config_is_rejected() -> None:

    with pytest.raises(
        ValueError
    ):
        EvidencePolicyConfig(
            min_sources=0
        )

    with pytest.raises(
        ValueError
    ):
        EvidencePolicyConfig(
            max_recommended_reads=0
        )


def test_empty_state_requires_search() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    policy = EvidencePolicy()

    decision = policy.evaluate(
        state
    )

    assert (
        decision.action
        == "search_more"
    )

    assert decision.ready is False


def test_insufficient_sources_requires_search() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    add_search(
        state,
        results=[
            {
                "source_id":
                    "S_11111111",
                "title":
                    "Source A",
                "url":
                    "https://example.com/a",
                "content":
                    "Snippet A",
                "score":
                    0.9,
            }
        ],
    )

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
        )
    )

    decision = policy.evaluate(
        state
    )

    assert (
        decision.action
        == "search_more"
    )


def test_enough_sources_but_no_pages_requires_read() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    add_search(
        state
    )

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
            max_recommended_reads=2,
        )
    )

    decision = policy.evaluate(
        state
    )

    assert (
        decision.action
        == "read_more"
    )

    assert (
        decision.recommended_source_ids
        == (
            "S_22222222",
            "S_33333333",
        )
    )


def test_sufficient_evidence_allows_synthesis() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    add_search(
        state
    )

    add_page(
        state,
        source_id="S_22222222",
        url="https://example.com/b",
    )

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
        )
    )

    decision = policy.evaluate(
        state
    )

    assert (
        decision.action
        == "synthesize"
    )

    assert decision.ready


def test_no_unread_sources_falls_back_to_search() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    add_search(
        state,
        results=[
            {
                "source_id":
                    "S_11111111",
                "title":
                    "Source A",
                "url":
                    "https://example.com/a",
                "content":
                    "Snippet A",
                "score":
                    0.9,
            }
        ],
    )

    add_page(
        state,
        source_id="S_11111111",
        url="https://example.com/a",
    )

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=1,
            min_full_pages=2,
        )
    )

    decision = policy.evaluate(
        state
    )

    assert (
        decision.action
        == "search_more"
    )

    assert (
        decision.recommended_source_ids
        == ()
    )