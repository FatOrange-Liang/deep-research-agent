import json

from deepresearch.llm import Message

from deepresearch.research import (
    Source,
    collect_sources,
    extract_citation_ids,
    validate_citations,
)


def test_extract_citation_ids() -> None:

    answer = (
        "Claim one [S_12345678]. "
        "Claim two [S_abcdef12]."
    )

    assert extract_citation_ids(
        answer
    ) == [
        "S_12345678",
        "S_abcdef12",
    ]


def test_duplicate_citations_are_deduplicated() -> None:

    answer = (
        "A [S_12345678]. "
        "B [S_12345678]."
    )

    assert extract_citation_ids(
        answer
    ) == [
        "S_12345678"
    ]


def test_valid_citation() -> None:

    sources = {
        "S_12345678": Source(
            source_id="S_12345678",
            title="Example",
            url="https://example.com",
            content="Example content",
        )
    }

    result = validate_citations(
        "This is supported [S_12345678].",
        sources,
    )

    assert result.is_valid

    assert result.invalid_source_ids == []


def test_invalid_citation() -> None:

    sources = {
        "S_12345678": Source(
            source_id="S_12345678",
            title="Example",
            url="https://example.com",
            content="Example content",
        )
    }

    result = validate_citations(
        "Unsupported claim [S_deadbeef].",
        sources,
    )

    assert result.is_valid is False

    assert result.invalid_source_ids == [
        "S_deadbeef"
    ]


def test_collect_sources_from_tool_message() -> None:

    content = json.dumps(
        {
            "query": "LangGraph",
            "results": [
                {
                    "source_id":
                        "S_12345678",
                    "title":
                        "LangGraph docs",
                    "url":
                        "https://example.com",
                    "content":
                        "LangGraph content",
                    "score":
                        0.9,
                }
            ],
        }
    )

    messages = [
        Message(
            role="tool",
            name="web_search",
            tool_call_id="call_001",
            content=content,
        )
    ]

    sources = collect_sources(
        messages
    )

    assert (
        "S_12345678"
        in sources
    )

    assert (
        sources[
            "S_12345678"
        ].title
        == "LangGraph docs"
    )

def test_malformed_citation_is_detected() -> None:

    sources = {
        "S_4f78c9e4": Source(
            source_id="S_4f78c9e4",
            title="LangGraph docs",
            url="https://example.com",
            content="LangGraph documentation",
        )
    }

    result = validate_citations(
        (
            "LangGraph supports "
            "human-in-the-loop "
            "[S_4f78c9e]."
        ),
        sources,
    )

    assert result.is_valid is False

    assert result.malformed_source_ids == [
        "S_4f78c9e"
    ]

    assert result.invalid_source_ids == []