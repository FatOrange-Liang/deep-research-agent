import json

from deepresearch.llm import (
    LLMResponse,
    Message,
    MockLLM,
)

from deepresearch.research import (
    CitationGuard,
    ResearchState,
)


def create_state() -> ResearchState:

    state = ResearchState(
        task="Research LangGraph"
    )

    source_content = json.dumps(
        {
            "query": "LangGraph",
            "results": [
                {
                    "source_id":
                        "S_12345678",
                    "title":
                        "LangGraph official docs",
                    "url":
                        "https://example.com/langgraph",
                    "content":
                        (
                            "LangGraph is a "
                            "stateful agent "
                            "orchestration framework."
                        ),
                    "score":
                        0.9,
                }
            ],
        }
    )

    message = Message(
        role="tool",
        name="web_search",
        tool_call_id="call_001",
        content=source_content,
    )

    state.ingest_message(
        message
    )

    return state


def test_valid_answer_requires_no_repair() -> None:

    llm = MockLLM(
        responses=[]
    )

    guard = CitationGuard(
        llm=llm
    )

    result = guard.check(
        answer=(
            "LangGraph is stateful "
            "[S_12345678]."
        ),
        state=create_state(),
    )

    assert result.passed

    assert result.repaired is False

    assert result.attempts == 0

    assert (
        llm.remaining_responses
        == 0
    )


def test_invalid_citation_is_repaired() -> None:

    llm = MockLLM(
        responses=[
            LLMResponse(
                content=(
                    "LangGraph is stateful "
                    "[S_12345678]."
                )
            )
        ]
    )

    guard = CitationGuard(
        llm=llm
    )

    result = guard.check(
        answer=(
            "LangGraph is stateful "
            "[S_deadbeef]."
        ),
        state=create_state(),
    )

    assert result.passed

    assert result.repaired

    assert result.attempts == 1

    assert (
        result.validation
        .invalid_source_ids
        == []
    )


def test_missing_citation_is_repaired() -> None:

    llm = MockLLM(
        responses=[
            LLMResponse(
                content=(
                    "LangGraph is stateful "
                    "[S_12345678]."
                )
            )
        ]
    )

    guard = CitationGuard(
        llm=llm
    )

    result = guard.check(
        answer=(
            "LangGraph is stateful."
        ),
        state=create_state(),
    )

    assert result.passed

    assert result.repaired

    assert result.attempts == 1


def test_guard_stops_after_max_attempts() -> None:

    llm = MockLLM(
        responses=[
            LLMResponse(
                content=(
                    "Wrong [S_deadbeef]."
                )
            ),
            LLMResponse(
                content=(
                    "Still wrong "
                    "[S_deadbeef]."
                )
            ),
        ]
    )

    guard = CitationGuard(
        llm=llm,
        max_repair_attempts=2,
    )

    result = guard.check(
        answer=(
            "Wrong [S_deadbeef]."
        ),
        state=create_state(),
    )

    assert result.passed is False

    assert result.repaired

    assert result.attempts == 2

    assert (
        result.validation
        .invalid_source_ids
        == ["S_deadbeef"]
    )


def test_malformed_citation_is_repaired() -> None:

    llm = MockLLM(
        responses=[
            LLMResponse(
                content=(
                    "LangGraph is stateful "
                    "[S_12345678]."
                )
            )
        ]
    )

    guard = CitationGuard(
        llm=llm
    )

    result = guard.check(
        answer=(
            "LangGraph is stateful "
            "[S_1234567]."
        ),
        state=create_state(),
    )

    assert result.passed

    assert result.repaired

    assert result.attempts == 1

    assert (
        result.validation
        .malformed_source_ids
        == []
    )