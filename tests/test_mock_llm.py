import pytest

from deepresearch.llm import (
    LLMResponse,
    Message,
    MockLLM,
    ToolCall,
)


def test_mock_llm_returns_responses_in_order() -> None:
    llm = MockLLM(
        responses=[
            LLMResponse(content="first"),
            LLMResponse(content="second"),
        ]
    )

    messages = [
        Message(
            role="user",
            content="hello",
        )
    ]

    response_1 = llm.chat(messages)
    response_2 = llm.chat(messages)

    assert response_1.content == "first"
    assert response_2.content == "second"


def test_mock_llm_tracks_calls() -> None:
    llm = MockLLM(
        responses=[
            LLMResponse(content="done"),
        ]
    )

    messages = [
        Message(
            role="user",
            content="hello",
        )
    ]

    llm.chat(messages)

    assert len(llm.calls) == 1

    recorded_messages, recorded_tools = llm.calls[0]

    assert recorded_messages[0].content == "hello"
    assert recorded_tools is None


def test_mock_llm_tool_call_response() -> None:
    tool_call = ToolCall(
        id="call_001",
        name="calculator",
        arguments={
            "expression": "12 * 8",
        },
    )

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[tool_call],
            ),
        ]
    )

    response = llm.chat(
        [
            Message(
                role="user",
                content="Calculate 12 * 8",
            )
        ]
    )

    assert response.has_tool_calls
    assert response.tool_calls[0].name == "calculator"


def test_mock_llm_raises_when_empty() -> None:
    llm = MockLLM(responses=[])

    with pytest.raises(
        RuntimeError,
        match="no responses remaining",
    ):
        llm.chat(
            [
                Message(
                    role="user",
                    content="hello",
                )
            ]
        )


def test_remaining_responses() -> None:
    llm = MockLLM(
        responses=[
            LLMResponse(content="one"),
            LLMResponse(content="two"),
        ]
    )

    assert llm.remaining_responses == 2

    llm.chat(
        [
            Message(
                role="user",
                content="hello",
            )
        ]
    )

    assert llm.remaining_responses == 1