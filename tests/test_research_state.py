import json

from deepresearch.llm import Message

from deepresearch.research import (
    ResearchState,
)


def make_search_message(
    *,
    query: str = "LangGraph",
    source_id: str = "S_12345678",
    content: str = "Search snippet",
    score: float = 0.9,
) -> Message:

    payload = {
        "query": query,
        "results": [
            {
                "source_id":
                    source_id,
                "title":
                    "LangGraph docs",
                "url":
                    "https://example.com/langgraph",
                "content":
                    content,
                "score":
                    score,
            }
        ],
    }

    return Message(
        role="tool",
        name="web_search",
        tool_call_id="search_001",
        content=json.dumps(
            payload
        ),
    )


def make_reader_message(
    *,
    source_id: str = "S_12345678",
    content: str = "Full page evidence",
) -> Message:

    payload = {
        "source_id":
            source_id,
        "title":
            "LangGraph docs",
        "url":
            "https://example.com/langgraph",
        "content":
            content,
        "content_type":
            "text/html",
        "truncated":
            False,
    }

    return Message(
        role="tool",
        name="web_page_reader",
        tool_call_id="reader_001",
        content=json.dumps(
            payload
        ),
    )


def test_ingest_search_result() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    assert state.ingest_message(
        make_search_message()
    )

    assert state.source_count == 1

    assert state.read_source_count == 0

    assert state.unread_source_count == 1

    assert state.search_queries == [
        "LangGraph"
    ]

    record = state.sources[
        "S_12345678"
    ]

    assert (
        record.discovered_by_search
        is True
    )

    assert (
        record.read_full_page
        is False
    )


def test_page_read_enriches_search_source() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    state.ingest_message(
        make_search_message()
    )

    state.ingest_message(
        make_reader_message()
    )

    assert state.source_count == 1

    assert state.read_source_count == 1

    record = state.sources[
        "S_12345678"
    ]

    assert (
        record.source.content
        == "Full page evidence"
    )

    # Search relevance should survive
    # full-page enrichment.
    assert (
        record.source.score
        == 0.9
    )

    assert record.read_full_page

    assert (
        record.content_type
        == "text/html"
    )

    assert record.truncated is False


def test_search_does_not_overwrite_full_page() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    state.ingest_message(
        make_search_message()
    )

    state.ingest_message(
        make_reader_message()
    )

    state.ingest_message(
        make_search_message(
            query="LangGraph agents",
            content="Shorter new snippet",
            score=0.95,
        )
    )

    record = state.sources[
        "S_12345678"
    ]

    assert (
        record.source.content
        == "Full page evidence"
    )

    assert (
        record.source.score
        == 0.95
    )

    assert record.search_queries == [
        "LangGraph",
        "LangGraph agents",
    ]


def test_repeated_query_is_deduplicated() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    state.ingest_message(
        make_search_message()
    )

    state.ingest_message(
        make_search_message()
    )

    assert state.search_queries == [
        "LangGraph"
    ]

    assert (
        state.sources[
            "S_12345678"
        ].search_queries
        == ["LangGraph"]
    )


def test_invalid_tool_payload_is_ignored() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    message = Message(
        role="tool",
        name="web_search",
        tool_call_id="bad_001",
        content="not-json",
    )

    assert (
        state.ingest_message(
            message
        )
        is False
    )

    assert state.source_count == 0


def test_reconstruct_state_from_messages() -> None:

    messages = [
        Message(
            role="user",
            content="Research LangGraph",
        ),
        make_search_message(),
        make_reader_message(),
    ]

    state = (
        ResearchState.from_messages(
            task="Research LangGraph",
            messages=messages,
        )
    )

    assert state.source_count == 1

    assert state.read_source_count == 1

    assert (
        state.tool_observation_count
        == 2
    )

def test_citation_sources_expose_best_evidence() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    state.ingest_message(
        make_search_message()
    )

    state.ingest_message(
        make_reader_message(
            content="Full page evidence",
        )
    )

    sources = (
        state.citation_sources
    )

    assert (
        sources[
            "S_12345678"
        ].content
        == "Full page evidence"
    )

    assert (
        sources[
            "S_12345678"
        ].score
        == 0.9
    )