
import json

import pytest

from deepresearch.llm import Message

from deepresearch.research import (
    SourceAuthorityPolicy,
    EvidencePolicy,
    EvidencePolicyConfig,
    ResearchState,
)


OFFICIAL_URL = (
    "https://docs.langchain.com/"
    "oss/python/langgraph/overview"
)

BLOG_URL = "https://example.com/blog"
OTHER_URL = "https://example.org/article"


def create_state(
    *,
    include_official: bool = True,
) -> ResearchState:

    state = ResearchState(
        task="Research LangGraph official docs"
    )

    results = [
        {
            "source_id": "S_11111111",
            "title": "Community Blog",
            "url": BLOG_URL,
            "content": "Blog snippet",
            "score": 0.99,
        },
        {
            "source_id": "S_22222222",
            "title": "Other Article",
            "url": OTHER_URL,
            "content": "Article snippet",
            "score": 0.85,
        },
    ]

    if include_official:

        results.append(
            {
                "source_id": "S_33333333",
                "title": "LangGraph Official Docs",
                "url": OFFICIAL_URL,
                "content": "Official snippet",
                "score": 0.65,
            }
        )

    else:

        results.append(
            {
                "source_id": "S_44444444",
                "title": "Third Party",
                "url": "https://example.net/research",
                "content": "Third-party snippet",
                "score": 0.75,
            }
        )

    message = Message(
        role="tool",
        name="web_search",
        tool_call_id="search_001",
        content=json.dumps(
            {
                "query": "LangGraph docs",
                "results": results,
            }
        ),
    )

    assert state.ingest_message(message)

    return state


def read_page(
    state: ResearchState,
    *,
    source_id: str,
    url: str,
) -> None:

    message = Message(
        role="tool",
        name="web_page_reader",
        tool_call_id=f"reader_{source_id}",
        content=json.dumps(
            {
                "source_id": source_id,
                "title": "Full Page",
                "url": url,
                "content": "Full page evidence",
                "content_type": "text/html",
                "truncated": False,
            }
        ),
    )

    assert state.ingest_message(message)


def create_policy(
    *,
    required_hosts: tuple[str, ...] = (
        "docs.langchain.com",
    ),
) -> EvidencePolicy:

    return EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
            min_search_queries=1,
            required_hosts=required_hosts,
            min_required_host_full_pages=1,
        )
    )


def test_authority_matches_exact_hostname() -> None:

    authority = SourceAuthorityPolicy(
        required_hosts=(
            "docs.langchain.com",
        )
    )

    assert authority.matches(OFFICIAL_URL)

    assert authority.matches(
        "https://DOCS.LANGCHAIN.COM/overview"
    )

    assert not authority.matches(
        "https://docs.langchain.com.evil.com/page"
    )

    assert not authority.matches(
        "https://docs.langchain.com@evil.com/page"
    )

    assert not authority.matches(
        "file:///docs.langchain.com/page"
    )


def test_invalid_authority_config_is_rejected() -> None:

    with pytest.raises(ValueError):

        EvidencePolicyConfig(
            required_hosts=(
                "https://docs.langchain.com",
            )
        )


def test_no_official_source_requires_search() -> None:

    state = create_state(
        include_official=False
    )

    decision = create_policy().evaluate(state)

    assert decision.action == "search_more"

    assert not decision.ready


def test_official_snippet_requires_page_read() -> None:

    state = create_state()

    decision = create_policy().evaluate(state)

    assert decision.action == "read_more"

    # Official source must be recommended even though
    # its search score is below the blog's score.
    assert decision.recommended_source_ids == (
        "S_33333333",
    )


def test_reading_blog_does_not_satisfy_authority() -> None:

    state = create_state()

    read_page(
        state,
        source_id="S_11111111",
        url=BLOG_URL,
    )

    assert state.read_source_count == 1

    decision = create_policy().evaluate(state)

    assert decision.action == "read_more"

    assert decision.recommended_source_ids == (
        "S_33333333",
    )


def test_reading_official_page_allows_synthesis() -> None:

    state = create_state()

    read_page(
        state,
        source_id="S_33333333",
        url=OFFICIAL_URL,
    )

    decision = create_policy().evaluate(state)

    assert decision.action == "synthesize"

    assert decision.ready


def test_authority_can_be_disabled() -> None:

    state = create_state()

    # Only a blog has been read.
    read_page(
        state,
        source_id="S_11111111",
        url=BLOG_URL,
    )

    # Without required_hosts, the existing quantity
    # policy should retain its original behavior.
    policy = create_policy(
        required_hosts=()
    )

    decision = policy.evaluate(state)

    assert decision.action == "synthesize"
