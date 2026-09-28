import json

from deepresearch.agent import (
    AgentResult,
)

from deepresearch.llm import (
    LLMResponse,
    Message,
    MockLLM,
)

from deepresearch.research import (
    CitationGuard,
)


def create_result(
    answer: str,
) -> AgentResult:

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

    messages = [
        Message(
            role="user",
            content="Research LangGraph",
        ),
        Message(
            role="tool",
            name="web_search",
            tool_call_id="call_001",
            content=source_content,
        ),
        Message(
            role="assistant",
            content=answer,
        ),
    ]

    return AgentResult(
        answer=answer,
        stop_reason="completed",
        messages=messages,
        steps=[],
    )


def test_valid_answer_requires_no_repair() -> None:

    llm = MockLLM(
        responses=[]
    )

    guard = CitationGuard(
        llm=llm
    )

    result = guard.check(
        create_result(
            (
                "LangGraph is stateful "
                "[S_12345678]."
            )
        )
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
        create_result(
            (
                "LangGraph is stateful "
                "[S_deadbeef]."
            )
        )
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
        create_result(
            "LangGraph is stateful."
        )
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
        create_result(
            "Wrong [S_deadbeef]."
        )
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
        create_result(
            (
                "LangGraph is stateful "
                "[S_1234567]."
            )
        )
    )

    assert result.passed

    assert result.repaired

    assert result.attempts == 1

    assert (
        result.validation
        .malformed_source_ids
        == []
    )